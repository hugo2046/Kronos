"""G5 阶段 0：归因分析（纯诊断，零训练，计划 §1，20260817）。

**问题**：G1 的 backtest 窗 mean 聚合 AER(等权) +14.33% 是不是动量换皮？

**预承诺（计划 §1，先于本脚本任何计算冻结——引用原文）**：
    - 若 backtest 窗逐日持仓重合度 mean(|G1_top50 ∩ M_top50|/50) > 60% →
      归因判定"G1 疑似动量代理"（结算文档须引用并按"冗余候选"口径解读；
      G5 各头增设"对 M 正交化残差的组合表现"报告项）；
    - 若 ≤ 60% → "G1 含非动量增量"（正交化报告项仍做，对称性）。

指标（0.1~0.3）：
    0.1 逐日持仓重合度（G1_mean vs M / G1_mean vs F0_mean，backtest + 2025H2 双窗，
        canonical 引擎口径重放，replay_holdings 与引擎 TradeLog 逐位对拍过）；
    0.2 信号截面秩相关：corr(G1_mean, M) / corr(G1_mean, −R) / corr(G1_mean, F0_mean)
        逐日 Spearman 均值（双窗）；
    0.3 市值画像：DDB 无市值字段（候选字段全零，探查记录入 JSON）→ 按计划披露跳过。

披露：G1@2025H2 信号由 g5_head/gen_g1_2025h2.py 在预承诺提交后、归因计算前补生成
（canonical 协议零自由度；计划"零新推理"为撰写时的事实性误设，仓库无 G1@2025H2）。

落盘：g5_head/data/g5_attribution_results.json（+ 控制台判定）。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m g5_head.run_attribution
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import BaselineConfig
from baseline_suite.signal import build_px_tradeable
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from finetune_suite.run_g2_signals import WINDOW_DEFS
from g5_head.holdings import daily_overlap, replay_holdings
from paper_replication.engine import EngineConfig

PKG_DIR = Path(__file__).resolve().parent
OUT_DIR = PKG_DIR / "data"
R4 = PKG_DIR.parent / "finetune_suite" / "data"

# 预承诺阈值（计划 §1，冻结）
OVERLAP_THRESHOLD = 0.60
TOP_K = 50

# 双窗信号路径（G1_mean / M / F0_mean 需过引擎重放；R 仅秩相关）
SIGNAL_PATHS = {
    "backtest": {
        "G1_mean": R4 / "g1" / "daily_signals_backtest_G1_mean.parquet",
        "M": R4 / "daily_signals_backtest_M.parquet",
        "F0_mean": R4 / "daily_signals_backtest_F0_mean.parquet",
        "R": PKG_DIR.parent / "baseline_suite" / "data" / "daily_signals_oos_R.parquet",
    },
    "2025h2": {
        "G1_mean": OUT_DIR / "daily_signals_2025h2_G1_mean.parquet",
        "M": R4 / "g0" / "daily_signals_2025h2_M.parquet",
        "F0_mean": R4 / "g0" / "daily_signals_2025h2_F0_mean.parquet",
        "R": PKG_DIR.parent / "baseline_suite" / "data" / "daily_signals_oos_R.parquet",
    },
}
WINDOW_BOUNDS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": WINDOW_DEFS["2025h2"],
}

# 0.3 市值字段候选（qlib-DDB 桥对未知字段回填 0，全零=无该数据）
MVCAP_CANDIDATES = [
    "$market_value", "$mktcap", "$total_mv", "$float_mv",
    "$negotiable_mv", "$cap", "$total_share", "$float_share", "$marketcap",
]


def _load_window(window: str) -> dict[str, pd.DataFrame]:
    start, end = WINDOW_BOUNDS[window]
    out: dict[str, pd.DataFrame] = {}
    for name, path in SIGNAL_PATHS[window].items():
        if not path.exists():
            raise FileNotFoundError(f"[{window}] {name} 信号缺失：{path}")
        df = pd.read_parquet(path)
        if name == "R":  # oos 全表按窗切
            df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        out[name] = df
    ref = out["M"].index
    for name, df in out.items():
        assert df.index.equals(ref), (
            f"[{window}] {name} 索引与 M 不一致：{df.index.min()}~{df.index.max()} "
            f"vs {ref.min()}~{ref.max()}"
        )
    logger.info(f"[{window}] 信号就绪：{len(ref)} 日（{ref.min().date()}~{ref.max().date()}）")
    return out


def _holdings_overlap(signals: dict, cfg: BaselineConfig, window: str) -> dict:
    """0.1：canonical 引擎口径重放三信号持仓，算 G1_mean 与 M / F0_mean 的重合度。

    px/可交易表必须按窗构造（cfg 的 backtest 边界重绑到当前窗）——否则
    ``build_px_tradeable`` 会取 oos 全窗 260 日，信号 reindex 后首日落在信号窗外，
    引擎首日建仓在空信号上执行、此后永无买入（重放全空，实测已抓）。
    """
    from dataclasses import replace

    from kronos_qlib import QlibProvider

    start, end = WINDOW_BOUNDS[window]
    wcfg = replace(cfg, window=f"g5_attr_{window}", backtest_start=start, backtest_end=end)
    provider = QlibProvider(cfg.pool, start, end)
    all_cols = sorted(set().union(*[set(s.columns) for s in signals.values()]))
    rebalances = pd.DatetimeIndex(signals["M"].index)
    px, trd = build_px_tradeable(provider, wcfg, rebalances, all_cols)
    ec = EngineConfig(top_k=cfg.top_k, drop_n=cfg.drop_n, min_hold=cfg.min_hold, cost_bps=cfg.cost_bps)

    held: dict[str, dict] = {}
    for name in ("G1_mean", "M", "F0_mean"):
        sig = signals[name].reindex(index=px.index, columns=px.columns)
        holdings, _ = replay_holdings(sig, px, trd, cfg=ec)
        held[name] = holdings
        sizes = [len(v) for v in holdings.values()]
        assert np.mean(sizes) > 0.5 * cfg.top_k, (
            f"[{window}] {name} 重放持仓日均 {np.mean(sizes):.1f} 只异常（应≈{cfg.top_k}）")
        logger.info(f"[{window}] {name} 持仓重放：{len(holdings)} 日，"
                    f"日均 {np.mean(sizes):.1f} 只（top_k={cfg.top_k}）")

    return {
        "G1_mean_vs_M": daily_overlap(held["G1_mean"], held["M"], k=TOP_K),
        "G1_mean_vs_F0_mean": daily_overlap(held["G1_mean"], held["F0_mean"], k=TOP_K),
    }


def _rank_corr(a: pd.DataFrame, b: pd.DataFrame, label: str) -> float:
    """0.2：逐日截面 Spearman 的均值（公共非 NaN 列，<5 只跳过该日）。"""
    rhos: list[float] = []
    for d in a.index.intersection(b.index):
        xa, xb = a.loc[d], b.loc[d]
        common = xa.notna() & xb.notna()
        if int(common.sum()) < 5:
            continue
        rho, _ = stats.spearmanr(xa[common], xb[common])
        if np.isfinite(rho):
            rhos.append(float(rho))
    val = float(np.mean(rhos)) if rhos else float("nan")
    logger.info(f"  corr({label})：{len(rhos)} 日均值 {val:+.4f}")
    return val


def _probe_market_cap(cfg: BaselineConfig) -> dict:
    """0.3：探查 DDB 市值字段（未知字段被 qlib-DDB 桥回填 0——全零视为无数据）。"""
    from kronos_qlib import QlibProvider

    start, end = WINDOW_BOUNDS["backtest"]
    probe_days = pd.bdate_range(start, periods=5)
    p = QlibProvider(cfg.pool, start, end)
    results = {}
    for f in MVCAP_CANDIDATES:
        try:
            df = p.fetch([f], freq="day")
            df = df[(df.index.get_level_values(0).isin(probe_days))]
            nonzero = int((df.iloc[:, 0] != 0).sum()) if len(df) else 0
            results[f] = {"rows": int(len(df)), "nonzero": nonzero}
        except Exception as e:  # noqa: BLE001
            results[f] = {"error": str(e)[:100]}
    has_cap = any(v.get("nonzero", 0) > 0 for v in results.values())
    return {"probe": results, "has_market_cap": bool(has_cap),
            "skipped": not has_cap}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = BaselineConfig.load(window="oos")  # 引擎/池口径（top_k=50 等）
    logger.info(f"引擎口径：csi300 top_k={cfg.top_k} drop_n={cfg.drop_n} "
                f"min_hold={cfg.min_hold} cost_bps={cfg.cost_bps}")

    out: dict = {
        "experiment": "g5_stage0_attribution",
        "date": "2026-08-17",
        "precommitment": {
            "threshold": OVERLAP_THRESHOLD,
            "metric": "backtest 窗 mean(|G1_top50 ∩ M_top50|/50)",
            "source": "docs/G5预测头魔改实验计划_20260817.md §1（提交 8aa6564 冻结）",
        },
        "windows": {w: list(b) for w, b in WINDOW_BOUNDS.items()},
        "overlap": {}, "rank_corr": {},
    }

    for window in ("backtest", "2025h2"):
        logger.info("=" * 70)
        logger.info(f"窗口 {window}：{WINDOW_BOUNDS[window]}")
        signals = _load_window(window)

        logger.info(f"[{window}] 0.1 持仓重合度（引擎重放，top_k={TOP_K}）")
        ov = _holdings_overlap(signals, cfg, window)
        for k, v in ov.items():
            logger.info(f"[{window}] 重合度 {k}：{v:.4f}（{v:.1%}）")
        out["overlap"][window] = ov

        logger.info(f"[{window}] 0.2 信号截面秩相关（逐日 Spearman 均值）")
        rc = {
            "corr_G1_vs_M": _rank_corr(signals["G1_mean"], signals["M"], "G1_mean, M"),
            "corr_G1_vs_negR": _rank_corr(signals["G1_mean"], -signals["R"], "G1_mean, −R"),
            "corr_G1_vs_F0_mean": _rank_corr(signals["G1_mean"], signals["F0_mean"], "G1_mean, F0_mean"),
        }
        out["rank_corr"][window] = rc

    logger.info("=" * 70)
    logger.info("0.3 市值字段探查（provider 无市值则披露跳过）")
    out["market_cap"] = _probe_market_cap(cfg)
    if out["market_cap"]["skipped"]:
        logger.info("0.3 跳过：DDB 无市值字段（候选全零）——按计划披露跳过")

    # —— 预承诺判定（对照冻结阈值，一次性）——
    key_overlap = out["overlap"]["backtest"]["G1_mean_vs_M"]
    if key_overlap > OVERLAP_THRESHOLD:
        verdict = "G1 疑似动量代理"
        note = (f"backtest 窗持仓重合度 {key_overlap:.1%} > 60% → C2 切换按'冗余候选'口径"
                "解读；G5 各头增设对 M 正交化残差组合表现报告项（仅报告，不进判据）")
    else:
        verdict = "G1 含非动量增量"
        note = (f"backtest 窗持仓重合度 {key_overlap:.1%} ≤ 60% → C2 按原设计解读；"
                "正交化报告项仍做（对称性）")
    out["verdict"] = {"metric": key_overlap, "threshold": OVERLAP_THRESHOLD,
                      "verdict": verdict, "note": note}

    path = OUT_DIR / "g5_attribution_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    logger.info(f"归因结果落盘 → {path}")

    print("\n==== G5 阶段 0 归因判定（对照 §1 预承诺）====")
    print(f"backtest G1_mean vs M 持仓重合度 = {key_overlap:.4f}（阈值 {OVERLAP_THRESHOLD:.0%}）")
    for w in ("backtest", "2025h2"):
        print(f"[{w}] 重合度 G1~M {out['overlap'][w]['G1_mean_vs_M']:.4f} | "
              f"G1~F0 {out['overlap'][w]['G1_mean_vs_F0_mean']:.4f}")
        rc = out["rank_corr"][w]
        print(f"[{w}] 秩相关 G1~M {rc['corr_G1_vs_M']:+.4f} | "
              f"G1~(−R) {rc['corr_G1_vs_negR']:+.4f} | G1~F0 {rc['corr_G1_vs_F0_mean']:+.4f}")
    print(f"0.3 市值画像：{'跳过（DDB 无市值字段）' if out['market_cap']['skipped'] else '有数据'}")
    print(f"判定：{verdict}——{note}")


if __name__ == "__main__":
    main()
