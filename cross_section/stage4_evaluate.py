"""阶段 4：因子评估（计划 §5）。

对 Kronos 信号做标准单因子检验，**同时对基线因子跑完全相同的评估代码**。
基线（同口径同池同调仓日，成本几乎为零）：
    1. 10 日动量：``close[t]/close[t-10]-1``；
    2. 10 日反转：动量取负（A 股日频反转通常更强，Kronos 必须跨过的门槛）。

指标：
    1. 逐调仓日 RankIC → 均值、ICIR、t 值（111 期）；
    2. 分 5 组等权，组合收益单调性 + 多空净值（年化、最大回撤）；
    3. 多空收益扣单边 15bp 近似成本后的净值。

用法：
    /home/user/miniconda3/envs/quant/bin/python -m cross_section.stage4_evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from cross_section.baselines import compute_baseline_signals
from cross_section.common import DATA_DIR, ExperimentConfig
from cross_section.evaluate import evaluate_factor
from cross_section.rebalance import build_rebalance_dates
from kronos_qlib import QlibProvider

SIGNALS_PATH = DATA_DIR / "signals.parquet"
EVAL_CACHE_PATH = DATA_DIR / "signals_with_baselines.parquet"
EVAL_JSON_PATH = DATA_DIR / "eval_metrics.json"


def build_merged_signals(cfg: ExperimentConfig) -> pd.DataFrame:
    """合并 Kronos 信号 + 基线因子（内连接到共同可评估行）。"""
    signals = pd.read_parquet(SIGNALS_PATH)
    p = QlibProvider(cfg.pool, cfg.backtest_start, cfg.data_end)
    rebalances = build_rebalance_dates(p, cfg)
    baselines = compute_baseline_signals(p, cfg, rebalances)

    merged = signals.merge(baselines, on=["date", "code"], how="inner")
    logger.info(
        f"合并：Kronos {len(signals)} 行 × 基线 {len(baselines)} 行 → "
        f"内连接 {len(merged)} 行（{merged['date'].nunique()} 期）"
    )
    merged.to_parquet(EVAL_CACHE_PATH, index=False)
    return merged


def run_evaluation(cfg: ExperimentConfig) -> dict:
    """跑三个因子的完整 IC + 分组评估，返回 metrics 字典。"""
    merged = build_merged_signals(cfg)

    factors = {"kronos": "signal", "momentum": "momentum_10d", "reversal": "reversal_10d"}
    metrics: dict[str, dict] = {}
    ic_dump: dict[str, list] = {}
    grp_dump: dict[str, dict] = {}
    for label, col in factors.items():
        logger.info("=" * 70)
        logger.info(f"因子评估：{label}（列 {col}）")
        logger.info("=" * 70)
        ic, grp = evaluate_factor(merged, col, cfg)
        metrics[label] = {
            "factor_col": col,
            "n_periods": ic.n_periods,
            "rankic_mean": ic.rankic_mean,
            "rankic_std": ic.rankic_std,
            "icir": ic.icir,
            "t_stat": ic.t_stat,
            "p_value": ic.p_value,
            "rankic_positive_ratio": ic.rankic_positive_ratio,
            "group_mean_returns": grp.group_mean_returns.round(5).to_dict(),
            "long_short_annualized_net": grp.annualized,
            "long_short_max_drawdown_net": grp.max_drawdown,
        }
        ic_dump[label] = [
            {"date": d.strftime("%Y-%m-%d"), "rankic": (None if pd.isna(v) else float(v))}
            for d, v in ic.rankic_series.items()
        ]
        grp_dump[label] = {
            "long_short_series": [
                {"date": d.strftime("%Y-%m-%d"), "ret": float(r)}
                for d, r in grp.long_short_series.items()
            ],
        }

    out = {"metrics": metrics, "rankic_series": ic_dump, "long_short_series": grp_dump}
    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"评估指标写入 {EVAL_JSON_PATH}")
    return out


def main() -> int:
    cfg = ExperimentConfig.load()
    logger.info(
        f"阶段4 因子评估：n_groups={cfg.n_groups} cost_bps={cfg.cost_bps} "
        f"（单边，双边换手上限 200%）"
    )
    run_evaluation(cfg)
    logger.info("✅ 阶段4 评估完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
