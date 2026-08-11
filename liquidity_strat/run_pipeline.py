"""分析+回测+判读 一条龙入口（Kronos 信号跑完后调用）。

用法::

    python -m liquidity_strat.run_pipeline          # exst 主轨
    python -m liquidity_strat.run_pipeline --all    # 双轨（exst + withst）

前置：``daily_signals_K_union.parquet`` 已是 502 日全量（``run_signals kronos`` 跑完）。
产出：``analysis_signal_layer.json`` / ``analysis_portfolio_layer.json`` / ``judgements.json``。
"""
from __future__ import annotations

import argparse

from loguru import logger

from liquidity_strat.common import ST_TRACKS, LiquidityConfig, ensure_dirs
from liquidity_strat.analysis import run_analysis
from liquidity_strat.backtest import run_backtest_all
from liquidity_strat.judge import run_judge


def main() -> None:
    parser = argparse.ArgumentParser(description="流动性分层 分析+回测+判读")
    parser.add_argument("--all", action="store_true", help="双轨（exst + withst），默认仅 exst 主轨")
    args = parser.parse_args()
    ensure_dirs()
    cfg = LiquidityConfig.load()
    tracks = ST_TRACKS if args.all else ("exst",)
    logger.info(f"=== 信号层分析（analyze_factor）tracks={tracks} ===")
    run_analysis(cfg, tracks=tracks)
    logger.info(f"=== 组合层回测（qlib_bt）tracks={tracks} ===")
    run_backtest_all(cfg, tracks=tracks)
    logger.info("=== 五条预注册判读（主轨 exst）===")
    run_judge()
    logger.info("pipeline 完成。")


if __name__ == "__main__":
    main()
