"""阶段 2 分布信号 S1~S3 过引擎 + R3 门控回填（计划 §4.2 / §3-R3）。

前置：canonical 逐路径已落盘（``run_canonical_paths``）。

三条分布信号（冻结）：对每 (date, code)，由 N 路径 H 日平均收益
``r_i = mean(path_i)/close_t − 1`` 计算 S1 ``neg_std`` / S2 ``sharpe_like`` / S3 ``q10``，
分别过引擎（k=50/n=5/min_hold=5/15bp，双基准）。

R3 回填：路径离散度低分位（截面均 path std 过去 60 日分位 < 0.8）→ canonical mean，
否则退守（NaN 信号 = 冻结交易，top-k 引擎无法真等权，此为最近似操作化）。
**门控在 paper+oos 拼接序列上计算**——oos 段借 paper 历史预热，避免人为冷启动冻结。

判据（跑前冻结）：任一 S 两窗 AER(等权) 均 > 0 且 oos1 AER(等权) > canonical mean
oos1（−15.10%）+10pp → 分布信息有效；R3 独立判据：两子窗 AER(等权) > 纯 mean。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_stage2_dist
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import DATA_DIR as BL_DATA_DIR
from baseline_suite.common import BaselineConfig, ensure_dirs as bl_ensure_dirs
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from improve_suite.common import DATA_DIR, ImproveConfig
from improve_suite.dist_signals import run_dist_signals
from improve_suite.path_store import read_paths
from improve_suite.regime_switch import build_switch_signal

CANONICAL_MEAN_OOS_EW = -0.1510  # §0 已确认事实：canonical mean oos AER(等权)


def _load_paths(window: str):
    cfg = ImproveConfig.load(window=window)
    label = cfg.canonical_label()
    paths = read_paths(DATA_DIR / f"paths_{window}_{label}.parquet")
    last_close = pd.read_parquet(DATA_DIR / f"last_close_{window}.parquet")
    last_close.index = pd.to_datetime(last_close.index)
    last_close = last_close[~last_close.index.duplicated(keep="last")]  # 防御旧 checkpoint 重复
    return paths, last_close


def _compute_dist(window: str) -> dict[str, pd.DataFrame]:
    """单窗分布信号宽表（neg_std/sharpe_like/q10）。"""
    from kronos_qlib import QlibProvider

    cfg = BaselineConfig.load(window=window)
    paths, last_close = _load_paths(window)
    rebalances = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end).trading_days(
        cfg.backtest_start, cfg.backtest_end
    )
    return run_dist_signals(paths, last_close, rebalances)


def _dispersion_gate_full(neg_std_paper, neg_std_oos, *, q=0.8, lookback=60):
    """在 paper+oos 拼接的 neg_std 上算离散度门控，再切回两窗。

    daily_disp = (-neg_std).mean(axis=1)（截面均 path std）；
    门控 = 过去 lookback 日分位 < q → True（低心虚）。
    """
    full_neg = pd.concat([neg_std_paper, neg_std_oos]).sort_index()
    daily_disp = (-full_neg).mean(axis=1).sort_index()
    roll_q = daily_disp.rolling(lookback, min_periods=max(lookback // 2, 10)).rank(pct=True)
    gate_full = (roll_q < q)
    # 切回两窗
    p_dates = neg_std_paper.index
    o_dates = neg_std_oos.index
    return gate_full.reindex(p_dates), gate_full.reindex(o_dates)


def _backtest_window(window: str, dist: dict, gate: pd.Series) -> dict:
    """单窗：R3 + S1~S3 + mean 过双基准引擎。"""
    from kronos_qlib import QlibProvider

    cfg = BaselineConfig.load(window=window)
    start, end = cfg.backtest_start, cfg.backtest_end
    rebalances = QlibProvider(cfg.pool, start, end).trading_days(start, end)

    mean_sig = pd.read_parquet(BL_DATA_DIR / f"daily_signals_{window}_mean.parquet").reindex(rebalances)
    nan_table = pd.DataFrame(np.nan, index=mean_sig.index, columns=mean_sig.columns, dtype=float)
    R3 = build_switch_signal(mean_sig, nan_table, gate)
    n_true = int(np.where(pd.isna(gate.to_numpy()), False, gate.to_numpy()).sum())
    logger.info(f"[{window}] R3 门控：{n_true}/{len(rebalances)} 日低离散（True→mean，余冻结）")

    signals = {"R3": R3, "mean": mean_sig}
    signals.update({k: dist[k].reindex(rebalances) for k in ("neg_std", "sharpe_like", "q10")})

    all_cols = sorted(set().union(*[set(s.columns) for s in signals.values()]))
    provider = QlibProvider(cfg.pool, start, end)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
    bench_idx, bench_ew, _ = build_dual_benchmarks(provider, cfg, px, trd)

    results = {}
    for name in ("R3", "neg_std", "sharpe_like", "q10", "mean"):
        pi, pe, dr, _, _ = run_group(signals[name], px, trd, bench_idx, bench_ew, cfg=cfg, name=name)
        results[name] = {"perf_idx": pi, "perf_ew": pe, "daily_ret": dr}
    return results


def _serialize(res: dict) -> dict:
    return {
        k: {
            "perf_idx": v["perf_idx"].to_dict() if hasattr(v["perf_idx"], "to_dict") else v["perf_idx"],
            "perf_ew": v["perf_ew"].to_dict() if hasattr(v["perf_ew"], "to_dict") else v["perf_ew"],
        }
        for k, v in res.items()
    }


def main() -> None:
    bl_ensure_dirs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("==== 阶段 2 分布信号 S1~S3 + R3 回填（计划 §4.2 / §3-R3）====")

    paper_dist = _compute_dist("paper")
    oos_dist = _compute_dist("oos")

    # R3 门控在拼接序列上算（oos 借 paper 预热）
    gate_p, gate_o = _dispersion_gate_full(paper_dist["neg_std"], oos_dist["neg_std"])

    paper = _backtest_window("paper", paper_dist, gate_p)
    oos = _backtest_window("oos", oos_dist, gate_o)

    # —— S1~S3 判据（跑前冻结）——
    s_verdicts, any_effective = {}, False
    for s in ("neg_std", "sharpe_like", "q10"):
        p_ew, o_ew = paper[s]["perf_ew"].aer, oos[s]["perf_ew"].aer
        both_pos = (p_ew > 0) and (o_ew > 0)
        beats = o_ew > CANONICAL_MEAN_OOS_EW + 0.10
        eff = both_pos and beats
        any_effective = any_effective or eff
        s_verdicts[s] = {"paper_ew": p_ew, "oos_ew": o_ew, "both_positive": bool(both_pos),
                         "beats_mean_oos_plus10pp": bool(beats), "effective": bool(eff)}
    s_summary = "分布信息有效，待前向确认" if any_effective else "路径分布统计不含增量 alpha（如实记录）"

    # —— R3 判据（独立，§3 criterion 4）——
    r3_p, r3_o = paper["R3"]["perf_ew"].aer, oos["R3"]["perf_ew"].aer
    mean_p, mean_o = paper["mean"]["perf_ew"].aer, oos["mean"]["perf_ew"].aer
    r3_both = (r3_p > mean_p) and (r3_o > mean_o)
    r3_verdict = ("心虚度含门控信息，待前向确认" if r3_both
                  else "心虚度门控未两窗都强于纯 mean（不构成候选发现）")

    logger.info(f"==== 分布信号判读：{s_summary} ====")
    for s, v in s_verdicts.items():
        logger.info(f"  {s}: paper {v['paper_ew']:+.2%} / oos {v['oos_ew']:+.2%} → effective={v['effective']}")
    logger.info(f"==== R3 判读：{r3_verdict}（R3 paper {r3_p:+.2%}/oos {r3_o:+.2%} vs mean {mean_p:+.2%}/{mean_o:+.2%}）====")

    out = {
        "stage": "2_dist_r3",
        "signals": {"S1": "neg_std", "S2": "sharpe_like", "S3": "q10"},
        "canonical_mean_oos_ew": CANONICAL_MEAN_OOS_EW,
        "results": {"paper": _serialize(paper), "oos": _serialize(oos)},
        "dist_verdicts": s_verdicts, "dist_summary": s_summary,
        "r3_verdict": {"r3_paper_ew": r3_p, "r3_oos_ew": r3_o, "mean_paper_ew": mean_p,
                       "mean_oos_ew": mean_o, "both_strong": bool(r3_both), "verdict": r3_verdict},
    }
    out_path = DATA_DIR / "stage2_dist_r3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"结果落盘 {out_path}")


if __name__ == "__main__":
    main()
