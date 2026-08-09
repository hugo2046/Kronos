"""调仓日序列与可评估边界硬门禁（计划 §3.1）。

日历延伸到 2040-12-31，真实数据止于更早日期（实测 2026-08-07）。
``build_inference_windows`` 只在 2040 日历用尽时报错，**不会**替你拦住
"预测得出、真实收益取不到"的调仓日。故必须显式把调仓日约束在可评估区间内。

    最后一个可评估调仓日 = 交易日历中 <= 数据末日的倒数第 (H+1) 个交易日
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from cross_section.common import ExperimentConfig


def evaluability_boundary(
    provider, *, predict_len: int, data_end: str
) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    """计算可评估边界。

    :param provider: :class:`kronos_qlib.QlibProvider`。
    :param predict_len: 预测长度 H。
    :param data_end: 真实数据末日 ``YYYY-MM-DD``。
    :returns: ``(last_evaluable_rebalance, calendar_le_data_end)``。
    """
    cal = provider.trading_days()
    data_end_ts = pd.Timestamp(data_end)
    cal_le = cal[cal <= data_end_ts]
    if len(cal_le) < predict_len + 1:
        raise ValueError(
            f"可评估边界：数据末日 {data_end} 之前交易日不足 {predict_len + 1} 个"
        )
    last_evaluable = cal_le[-(predict_len + 1)]
    return last_evaluable, cal_le


def build_rebalance_dates(
    provider, cfg: ExperimentConfig
) -> pd.DatetimeIndex:
    """构造调仓日序列（每 ``rebalance_freq`` 个交易日一次）。

    断言最后一个调仓日 + H 个交易日 <= 数据末日（计划 §3.1 硬门禁）。

    :param provider: :class:`kronos_qlib.QlibProvider`。
    :param cfg: 实验配置。
    :returns: 调仓日 ``pd.DatetimeIndex``。
    """
    last_evaluable, cal_le = evaluability_boundary(
        provider, predict_len=cfg.predict_len, data_end=cfg.data_end
    )

    start_ts = pd.Timestamp(cfg.backtest_start)
    end_ts = pd.Timestamp(cfg.backtest_end)
    if end_ts > last_evaluable:
        raise ValueError(
            f"配置 backtest_end={cfg.backtest_end} 晚于最后可评估调仓日 "
            f"{last_evaluable.date()}（§3.1 硬门禁）"
        )

    cal_full = provider.trading_days()
    mask = (cal_full >= start_ts) & (cal_full <= end_ts)
    cal_bt = cal_full[mask]
    rebalances = cal_bt[:: cfg.rebalance_freq]

    # §3.1 硬门禁：最后一个调仓日 + H 个交易日 <= 数据末日
    last_idx = cal_full.get_loc(rebalances[-1])
    last_plus_H = cal_full[last_idx + cfg.predict_len]
    data_end_ts = pd.Timestamp(cfg.data_end)
    assert last_plus_H <= data_end_ts, (
        f"§3.1 硬门禁失败：最后调仓日 {rebalances[-1].date()} + {cfg.predict_len} "
        f"个交易日 = {last_plus_H.date()} > 数据末日 {cfg.data_end}"
    )

    logger.info(
        f"可评估边界：数据末日={cfg.data_end} / "
        f"最后可评估调仓日={last_evaluable.date()} / "
        f"实际调仓日数={len(rebalances)} / "
        f"首末调仓日={rebalances[0].date()}~{rebalances[-1].date()}"
    )
    return rebalances
