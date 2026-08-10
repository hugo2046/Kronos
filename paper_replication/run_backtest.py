"""阶段 3 入口：四组信号同引擎回测 + 预注册判读 + 结果落盘。

用法（解释器 /home/user/miniconda3/envs/quant/bin/python）::

    python -m paper_replication.run_backtest

前置：K/M/R/P 信号已落盘（run_signals kronos / baselines 先跑）。
产出 ``paper_replication/data/backtest_results.json``，支撑结果文档全部数字。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from paper_replication.benchmark import (
    build_pool_equal_weight_benchmark,
    probe_index_benchmark,
)
from paper_replication.common import DATA_DIR, ReplicationConfig, ensure_data_dir
from paper_replication.pipeline import judge, run_group
from paper_replication.signal import build_px_tradeable


def main() -> None:
    cfg = ReplicationConfig.load()
    ensure_data_dir()

    # —— 1. 加载四组信号 ——
    sigs = {}
    for tag in ("K", "M", "R", "P"):
        path = DATA_DIR / f"daily_signals_{tag}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"信号缺失：{path}（先跑 run_signals {tag.lower()}）")
        sigs[tag] = pd.read_parquet(path)
        logger.info(f"载入 {tag} 信号：{sigs[tag].shape[0]} 日 × {sigs[tag].shape[1]} 列")

    # —— 2. 取数：价格/可交易宽表 + 双基准 ——
    from kronos_qlib import QlibProvider

    all_cols = sorted(set().union(*[set(s.columns) for s in sigs.values()]))
    rebalances = pd.DatetimeIndex(sigs["M"].index)
    provider = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
    bench_idx = probe_index_benchmark(provider, cfg.backtest_start, cfg.backtest_end)
    bench_ew = build_pool_equal_weight_benchmark(px, trd)
    # beta_gap：指数累计 − 等权累计（结构性等权-beta 溢价）
    common = bench_idx.index.intersection(bench_ew.index)
    beta_gap = float((1 + bench_idx.loc[common]).prod() - (1 + bench_ew.loc[common]).prod())
    logger.info(f"等权-beta 溢价 beta_gap = {beta_gap:+.2%}")

    # —— 3. 四组同引擎回测 ——
    results = {}
    for tag in ("P", "M", "R", "K"):  # 先跑 P（门禁）→ M/R（基线）→ K（被试）
        sig = sigs[tag]
        pi, pe, dr, ex = run_group(
            sig, px, trd, bench_idx, bench_ew, cfg=cfg, name=tag
        )
        results[tag] = {
            "perf_idx": pi.to_dict(),
            "perf_ew": pe.to_dict(),
        }

    # —— 4. 预注册判读 ——
    verdict = judge(
        perf_k_idx=_dict_to_perf(results["K"]["perf_idx"]),
        perf_k_ew=_dict_to_perf(results["K"]["perf_ew"]),
        perf_m_idx=_dict_to_perf(results["M"]["perf_idx"]),
        perf_r_idx=_dict_to_perf(results["R"]["perf_idx"]),
        perf_p_ew=_dict_to_perf(results["P"]["perf_ew"]),
        beta_gap=beta_gap,
    )

    # —— 5. 落盘 ——
    out = {
        "config": {
            "pool": cfg.pool,
            "window": [cfg.backtest_start, cfg.backtest_end],
            "N": cfg.sample_count,
            "top_k": cfg.top_k,
            "drop_n": cfg.drop_n,
            "min_hold": cfg.min_hold,
            "cost_bps": cfg.cost_bps,
        },
        "anchors": {
            "paper_base_aer": 0.1911,
            "paper_base_ir": 1.3782,
            "source": "arXiv:2508.02739 Table 10 CSI300 (阶段 0 提取)",
        },
        "benchmarks": {
            "index": "000300.SH csi300 市值加权",
            "equal_weight": "同池等权（剥离 beta）",
            "beta_gap": beta_gap,
        },
        "groups": results,
        "verdict": verdict,
    }
    out_path = DATA_DIR / "backtest_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"回测结果落盘 {out_path}")
    logger.info(f"=== 判定：{verdict['verdict']} ===")


def _dict_to_perf(d: dict):
    """dict → PerfStats（judge 需要 PerfStats 对象）。"""
    from paper_replication.engine import PerfStats

    return PerfStats(**d)


if __name__ == "__main__":
    main()
