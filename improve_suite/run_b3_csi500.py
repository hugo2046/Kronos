"""B3 跨池验证（csi500）——计划 §4.3 阶段 2。

载入冻结 ``B3_best.pt``（**不重训不调参**），对 csi500 两窗逐日出截面得分。
输入口径与训练一致（窗口 z-score + clip5；冻结 Kronos 主干取隐状态——复用
``cross_section_kda`` 推理路径，import 不复制）。

引擎参数**跑前冻结**：k=100 / n=10 / min_hold=5 / 单边 15bp；双基准 = 000905.SH
指数 + 同池等权。

判据：B3 在 csi500 的 oos1 窗 AER(等权) > 0 且 AER(指数) > 0 → "双截面存活，
待前向确认"；否则降级"csi300 单窗偶然"，阶段 5 的 Mamba 时序头取消立项。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_b3_csi500
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import DATA_DIR as BL_DATA_DIR
from baseline_suite.common import BaselineConfig, ensure_dirs as bl_ensure_dirs
from baseline_suite.signal import build_px_tradeable
from improve_suite.common import DATA_DIR, FIG_DIR
from paper_replication.benchmark import build_pool_equal_weight_benchmark
from paper_replication.engine import EngineConfig, attach_benchmark, compute_perf, run_portfolio

REPO_ROOT = Path(__file__).resolve().parents[1]
CSI500_INDEX = "000905.SH"
KDA_DATA_DIR = REPO_ROOT / "cross_section_kda" / "data"


def _b3_signals(provider, rebalances, *, pool: str, device: str) -> pd.DataFrame:
    """载入冻结 B3，逐日出 csi500 截面得分宽表。"""
    import torch
    from cross_section_kda import B3KronosKdaHead, KronosFrozenBackbone
    from cross_section_kda.data import build_daily_samples
    from model import Kronos, KronosTokenizer

    cfg = ImproveConfig_load_for_b3()
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name).to(device)
    kronos = Kronos.from_pretrained(cfg.model_name).to(device)
    backbone = KronosFrozenBackbone(tokenizer=tokenizer, kronos=kronos, device=device)
    m = B3KronosKdaHead(backbone).to(device)
    ckpt_path = KDA_DATA_DIR / "B3_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"B3 checkpoint 缺失：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    logger.info(f"载入冻结 B3 checkpoint：{ckpt_path.name}")

    rows = []
    for i, d in enumerate(rebalances):
        ds = d.strftime("%Y-%m-%d")
        b = build_daily_samples(provider, date=ds, pool=pool)
        if b is None:
            rows.append({})
            continue
        with torch.no_grad():
            score = m(b.x_norm.to(device), b.stamp.to(device)).cpu().numpy()
        rows.append({c: float(s) for c, s in zip(b.codes, score)})
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"B3 csi500 [{i + 1}/{len(rebalances)}] {ds}: {len(b.codes)} 只")
    wide = pd.DataFrame(rows, index=rebalances)
    logger.info(f"B3 csi500 信号：{wide.shape[0]} 日 × 平均 {wide.notna().sum(axis=1).mean():.0f} 只/日")
    return wide


def ImproveConfig_load_for_b3():
    from improve_suite.common import ImproveConfig

    return ImproveConfig.load(window="paper")  # 仅取 model/tokenizer 名


def _seg_cfg(window: str) -> BaselineConfig:
    """csi500 引擎配置：k=100/n=10/min_hold=5/15bp（跑前冻结，§4.3）。"""
    base = BaselineConfig.load(window=window)
    return replace(base, pool="csi500", top_k=100, drop_n=10)


def backtest_b3_csi500(window: str, *, device: str) -> dict:
    """单窗 B3 csi500 过引擎 + 双基准（000905.SH + 同池等权）。"""
    from kronos_qlib import QlibProvider

    cfg = _seg_cfg(window)
    start, end = cfg.backtest_start, cfg.backtest_end
    provider = QlibProvider("csi500", start, end)
    rebalances = provider.trading_days(start, end)

    sig = _b3_signals(provider, rebalances, pool="csi500", device=device)
    all_cols = sorted(sig.columns)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)

    # 双基准：000905.SH 指数 + 同池等权。
    # ⚠️ 不能用 probe_index_benchmark——它内部硬编码 000300.SH（csi300），csi500 须显式取 000905.SH。
    bench_idx = _csi500_idx(provider, start, end)
    bench_ew = build_pool_equal_weight_benchmark(px, trd)

    ec = EngineConfig(top_k=cfg.top_k, drop_n=cfg.drop_n, min_hold=cfg.min_hold, cost_bps=cfg.cost_bps)
    sig_a = sig.reindex(index=px.index, columns=px.columns)
    daily_ret, _, trades = run_portfolio(sig_a, px, trd, cfg=ec)
    excess_idx = attach_benchmark(daily_ret, bench_idx)
    excess_ew = attach_benchmark(daily_ret, bench_ew)
    perf_idx = compute_perf(excess_idx, trades, name=f"B3_csi500_{window}")
    perf_ew = compute_perf(excess_ew, trades, name=f"B3_csi500_{window}")
    logger.info(
        f"[B3 csi500 {window}] AER(指数={CSI500_INDEX})={perf_idx.aer:+.2%} IR={perf_idx.ir:+.3f} | "
        f"AER(等权)={perf_ew.aer:+.2%} IR={perf_ew.ir:+.3f} (n={perf_idx.n_days})"
    )
    return {
        "perf_idx": perf_idx.to_dict(),
        "perf_ew": perf_ew.to_dict(),
        "daily_ret": daily_ret,
    }


def _csi500_idx(provider, start, end) -> pd.Series:
    """取 000905.SH（csi500）指数日收益。

    不能用 ``probe_index_benchmark``——它内部硬编码 000300.SH（csi300）。
    """
    from kronos_qlib import QlibProvider

    p = QlibProvider([CSI500_INDEX], start, end)
    df = p.fetch(["$close"], freq="day")
    if len(df) == 0:
        raise RuntimeError(f"{CSI500_INDEX} 在 {start}~{end} 无数据")
    if "instrument" in df.index.names:
        df = df.xs(CSI500_INDEX, level="instrument")
    close = df["close"].sort_index()
    return close.pct_change(fill_method=None).dropna()


def main() -> None:
    bl_ensure_dirs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"

    logger.info("==== 阶段 2 B3 csi500 跨池验证（计划 §4.3）====")
    paper = backtest_b3_csi500("paper", device=device)
    oos = backtest_b3_csi500("oos", device=device)

    # 判据（跑前冻结）：oos1 AER(等权)>0 且 AER(指数)>0 → 双截面存活
    oos_ew_pos = oos["perf_ew"]["aer"] > 0
    oos_idx_pos = oos["perf_idx"]["aer"] > 0
    survives = oos_ew_pos and oos_idx_pos
    if survives:
        verdict = "B3 双截面存活（csi300+csi500），待前向确认——阶段 5 Mamba 时序头可立项"
    else:
        verdict = "B3 csi500 样本外未双正——降级为 csi300 单窗偶然，阶段 5 Mamba 时序头取消"

    out = {
        "stage": "2_b3_csi500",
        "engine": {"top_k": 100, "drop_n": 10, "min_hold": 5, "cost_bps": 15},
        "benchmarks": {"index": CSI500_INDEX, "equal_weight": "同池等权"},
        "results": {"paper": {"perf_idx": paper["perf_idx"], "perf_ew": paper["perf_ew"]},
                    "oos": {"perf_idx": oos["perf_idx"], "perf_ew": oos["perf_ew"]}},
        "verdict": {
            "oos_aer_ew": oos["perf_ew"]["aer"], "oos_aer_idx": oos["perf_idx"]["aer"],
            "survives": bool(survives), "verdict": verdict,
        },
    }
    out_path = DATA_DIR / "stage2_b3_csi500_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"==== B3 csi500 判读：{verdict} ====")
    logger.info(f"结果落盘 {out_path}")


if __name__ == "__main__":
    main()
