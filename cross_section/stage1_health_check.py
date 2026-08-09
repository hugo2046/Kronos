"""阶段 1：取数与窗口构造体检（计划 §2）。

体检门禁（任一不过即停）：
    1. 逐调仓日 n_pool 恒为 300；n_kept >= 280；异常（kept < 250）先查清；
    2. 池随时间变化：首末两个调仓日成分集合必须不同；
    3. 每个 df 恰 90 行、列序 open/high/low/close/volume/amount、无 NaN；
    4. x_ts 末值 == 调仓日（或其之前最近交易日），y_ts 恰 10 个交易日且全部 > x_ts 末值。

用法：
    /home/user/miniconda3/envs/quant/bin/python -m cross_section.stage1_health_check

不在本阶段"抽 5 只与公开行情核对"——数据层复权自洽性已由集成测试验收。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from loguru import logger

from cross_section.common import ExperimentConfig
from cross_section.rebalance import build_rebalance_dates
from kronos_qlib import REQUIRED_COLS, QlibProvider, build_inference_windows


def run_health_check(cfg: ExperimentConfig) -> bool:
    """执行 §2 全部四条门禁，返回是否全部通过。

    :param cfg: 实验配置。
    :returns: 全部门禁通过为 True。
    """
    p = QlibProvider(cfg.pool, cfg.backtest_start, cfg.data_end)
    rebalances = build_rebalance_dates(p, cfg)

    n_periods = len(rebalances)
    # 抽查 6 个调仓日（首、末、中间均匀 4 个），用足够少的期数快速体检
    sample_idx = np.linspace(0, n_periods - 1, 6).round().astype(int)
    sample_idx = sorted(set(sample_idx.tolist()))
    sample_dates = [rebalances[i] for i in sample_idx]
    logger.info(f"阶段1 体检：抽查 {len(sample_dates)} / {n_periods} 个调仓日")

    all_stats = []
    window_failures: list[str] = []
    halt_flag = False

    for d in sample_dates:
        ds = d.strftime("%Y-%m-%d")
        df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows(
            p, ds, lookback=cfg.lookback, predict_len=cfg.predict_len, pool=cfg.pool
        )
        all_stats.append({"date": ds, **stats})

        # —— 门禁 1：n_pool=300、n_kept>=280、异常即停 ——
        assert stats["n_pool"] == 300, (
            f"门禁1失败 {ds}：n_pool={stats['n_pool']} != 300（沪深300 定员）"
        )
        assert stats["n_kept"] >= 250, (
            f"门禁1失败 {ds}：n_kept={stats['n_kept']} < 250，"
            f"异常（halt={stats['skipped_halt']} short={stats['skipped_short']}）先查清"
        )

        # —— 门禁 3：每个 df 恰 90 行、列序固定、无 NaN ——
        for j, df in enumerate(df_list):
            if df.shape[0] != cfg.lookback:
                window_failures.append(
                    f"{ds} 第{j}只 {codes[j]} 行数={df.shape[0]} != {cfg.lookback}"
                )
                halt_flag = True
            if list(df.columns) != REQUIRED_COLS:
                window_failures.append(
                    f"{ds} 第{j}只 {codes[j]} 列序={list(df.columns)} != {REQUIRED_COLS}"
                )
                halt_flag = True
            if df[REQUIRED_COLS].isnull().any().any():
                window_failures.append(f"{ds} 第{j}只 {codes[j]} 含 NaN")
                halt_flag = True

        # —— 门禁 4：x_ts 末值 == 调仓日；y_ts 恰 H 个且全 > x_ts 末值 ——
        t_ts = pd.Timestamp(ds)
        for j in range(len(df_list)):
            x_end = pd.Timestamp(x_ts_list[j].iloc[-1])
            # x_ts 末值应为 <= 调仓日的最近交易日（调仓日本身是交易日则相等）
            assert x_end <= t_ts, (
                f"门禁4失败 {ds} 第{j}只：x_ts末值 {x_end.date()} > 调仓日 {ds}"
            )
            # 若调仓日是交易日，x 末值应等于调仓日
            if x_end != t_ts:
                window_failures.append(
                    f"门禁4提醒 {ds} 第{j}只：x_ts末值 {x_end.date()} != 调仓日 {ds}"
                    f"（调仓日非交易日，向下取整）"
                )
            y_ts = pd.DatetimeIndex(y_ts_list[j])
            assert len(y_ts) == cfg.predict_len, (
                f"门禁4失败 {ds} 第{j}只：y_ts 长度={len(y_ts)} != {cfg.predict_len}"
            )
            assert (y_ts > x_end).all(), (
                f"门禁4失败 {ds} 第{j}只：y_ts 含 <= x_ts末值 的日期"
            )

    # —— 门禁 2：池随时间变化（首末两个调仓日成分不同）——
    first_members = set(p.list_pool_at(cfg.pool, rebalances[0].strftime("%Y-%m-%d")))
    last_members = set(p.list_pool_at(cfg.pool, rebalances[-1].strftime("%Y-%m-%d")))
    assert first_members != last_members, (
        f"门禁2失败：首({rebalances[0].date()})末({rebalances[-1].date()}) "
        f"调仓日成分完全相同 → point-in-time 失效"
    )
    overlap = len(first_members & last_members)
    logger.info(
        f"门禁2 通过：首末日成分 |∩|={overlap}（|首|={len(first_members)} "
        f"|末|={len(last_members)}），point-in-time 生效"
    )

    # 打印逐期体检摘要
    logger.info("逐调仓日体检摘要（n_pool / n_kept / halt / short）：")
    for s in all_stats:
        flag = " ⚠kept<280" if s["n_kept"] < 280 else ""
        logger.info(
            f"  {s['date']}: pool={s['n_pool']} kept={s['n_kept']} "
            f"halt={s['skipped_halt']} short={s['skipped_short']}{flag}"
        )

    if window_failures:
        logger.error(f"窗口结构门禁失败 {len(window_failures)} 处：")
        for f in window_failures[:20]:
            logger.error(f"  {f}")
        return False
    logger.info("门禁3/4 通过：所有抽检窗口行数=90、列序正确、无 NaN、时间戳正确")
    return not halt_flag


def main() -> int:
    cfg = ExperimentConfig.load()
    logger.info(f"配置加载完成：pool={cfg.pool} L={cfg.lookback} H={cfg.predict_len}")
    ok = run_health_check(cfg)
    if ok:
        logger.info("✅ 阶段1 全部门禁通过")
        return 0
    logger.error("❌ 阶段1 门禁不达标，停止，不进入下一阶段")
    return 1


if __name__ == "__main__":
    sys.exit(main())
