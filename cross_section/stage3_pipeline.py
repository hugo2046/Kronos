"""阶段 3：全量信号生成（计划 §4）。

沪深300 全池，111 期（区间与调仓日按 §3.1 确定），产出
``cross_section/data/signals.parquet``：``date, code, signal, fwd_ret_10d``。

无未来函数自查（§4）：
    - signal 只用 <= t 的数据（``run_inference_one_period`` 内已断言窗口末值 <= 调仓日）；
    - fwd_ret 只用于事后评估，绝不回流入信号。

支持断点续跑：逐期写 checkpoint 到 ``cross_section/data/.pipeline_checkpoint.parquet``，
中断后重跑会跳过已完成期。

用法：
    /home/user/miniconda3/envs/quant/bin/python -m cross_section.stage3_pipeline
"""
from __future__ import annotations

import sys
import time

import pandas as pd
from loguru import logger

from cross_section.common import DATA_DIR, ExperimentConfig, ensure_data_dir
from cross_section.rebalance import evaluability_boundary
from cross_section.signal import run_inference_one_period
from cross_section.stage2_pilot import load_predictor
from kronos_qlib import QlibProvider

CHECKPOINT_PATH = DATA_DIR / ".pipeline_checkpoint.parquet"
SIGNALS_PATH = DATA_DIR / "signals.parquet"


def run_pipeline(cfg: ExperimentConfig, *, resume: bool = True) -> pd.DataFrame:
    """全量信号生成。

    :param cfg: 实验配置。
    :param resume: 是否断点续跑（跳过 checkpoint 中已完成期）。
    :returns: 全量 signal 长表 ``[date, code, signal, fwd_ret_10d]``。
    """
    ensure_data_dir()
    p = QlibProvider(cfg.pool, cfg.backtest_start, cfg.data_end)
    predictor = load_predictor(cfg)

    # §3.1 可评估边界硬门禁（重申，全量运行前必须打印）
    last_eval, _ = evaluability_boundary(
        p, predict_len=cfg.predict_len, data_end=cfg.data_end
    )
    from cross_section.rebalance import build_rebalance_dates

    rebalances = build_rebalance_dates(p, cfg)
    n_periods = len(rebalances)
    logger.info(
        f"§3.1 硬门禁：数据末日={cfg.data_end} / 最后可评估调仓日="
        f"{last_eval.date()} / 实际调仓日数={n_periods} / "
        f"首末调仓日={rebalances[0].date()}~{rebalances[-1].date()}"
    )

    # 断点续跑
    done_dates: set[pd.Timestamp] = set()
    if resume and CHECKPOINT_PATH.exists():
        done_df = pd.read_parquet(CHECKPOINT_PATH)
        done_dates = set(done_df["date"].unique())
        logger.info(f"断点续跑：已完成 {len(done_dates)} / {n_periods} 期")

    all_rows: list[pd.DataFrame] = []
    t0 = time.time()
    n_done = len(done_dates)
    for i, d in enumerate(rebalances):
        d_ts = pd.Timestamp(d)
        if d_ts in done_dates:
            continue
        ds = d_ts.strftime("%Y-%m-%d")
        ts = time.time()
        period_df, stats = run_inference_one_period(predictor, cfg, p, ds)
        if len(period_df) == 0:
            logger.warning(f"  {ds}: 空结果，跳过（{stats}）")
            continue
        all_rows.append(period_df)
        n_done += 1
        elapsed = time.time() - t0
        rate = (n_done - len(done_dates)) / max(elapsed, 1e-6)
        logger.info(
            f"  [{n_done}/{n_periods}] {ds}: kept={stats['n_kept']} "
            f"halt={stats['skipped_halt']} short={stats['skipped_short']} "
            f"用时 {time.time()-ts:.1f}s（{rate:.2f} 期/s）"
        )
        # 每 5 期落一次 checkpoint
        if n_done % 5 == 0:
            _save_checkpoint(all_rows, done_dates)

    _save_checkpoint(all_rows, done_dates)
    signals = pd.read_parquet(CHECKPOINT_PATH)
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

    # 写最终 signals.parquet（checkpoint 保留用于复跑）
    signals.to_parquet(SIGNALS_PATH, index=False)
    logger.info(
        f"✅ 阶段3 完成：{signals['date'].nunique()} 期 × "
        f"{signals.groupby('date')['code'].count().mean():.0f} 只/期，"
        f"共 {len(signals)} 行 → {SIGNALS_PATH}"
    )
    _report_summary(signals)
    return signals


def _save_checkpoint(new_rows: list[pd.DataFrame], done_dates: set) -> None:
    """合并新结果与既有 checkpoint，落盘。"""
    frames = []
    if CHECKPOINT_PATH.exists():
        frames.append(pd.read_parquet(CHECKPOINT_PATH))
    if new_rows:
        frames.append(pd.concat(new_rows, ignore_index=True))
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "code"], keep="last")
    combined.to_parquet(CHECKPOINT_PATH, index=False)


def _report_summary(signals: pd.DataFrame) -> None:
    """全量信号摘要（有限性 / fwd_ret 覆盖率 / 分布）。"""
    import numpy as np

    logger.info(f"  signal 非有限值数: {(~np.isfinite(signals['signal'])).sum()}")
    logger.info(f"  fwd_ret_10d 非有限值数: {(~np.isfinite(signals['fwd_ret_10d'])).sum()}")
    logger.info(
        f"  signal 全期均值={signals['signal'].mean():.5f} "
        f"std={signals['signal'].std():.5f}"
    )
    logger.info(
        f"  fwd_ret_10d 全期均值={signals['fwd_ret_10d'].mean():.5f} "
        f"std={signals['fwd_ret_10d'].std():.5f}"
    )


def main() -> int:
    cfg = ExperimentConfig.load()
    logger.info(
        f"阶段3 全量信号：pool={cfg.pool} {cfg.backtest_start}~{cfg.backtest_end} "
        f"L={cfg.lookback} H={cfg.predict_len} sample_count={cfg.sample_count}"
    )
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
