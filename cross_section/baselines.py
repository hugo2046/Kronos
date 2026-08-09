"""基线因子（计划 §5）：10 日动量 / 10 日反转。

同口径、同池、同调仓日——成本几乎为零的真实门槛。
A 股日频反转通常更强，这是 Kronos 必须跨过的真实门槛。

    - 10 日动量：``close[t] / close[t-10] - 1``
    - 10 日反转：动量取负
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from cross_section.common import ExperimentConfig
from kronos_qlib import QlibProvider


def compute_baseline_signals(
    provider: QlibProvider,
    cfg: ExperimentConfig,
    rebalances: pd.DatetimeIndex,
) -> pd.DataFrame:
    """逐调仓日算动量 / 反转基线信号（同池同调仓日）。

    :param provider: :class:`kronos_qlib.QlibProvider`。
    :param cfg: 实验配置。
    :param rebalances: 调仓日序列。
    :returns: 长表 ``[date, code, momentum_10d, reversal_10d]``。
    """
    rows = []
    for d in rebalances:
        ds = d.strftime("%Y-%m-%d")
        # 用 build_inference_windows 的同口径池（point-in-time + 停牌跳过），
        # 这样基线与 Kronos 信号的截面完全可比。
        from kronos_qlib import build_inference_windows

        _, _, _, codes, stats = build_inference_windows(
            provider, ds, lookback=cfg.lookback, predict_len=cfg.predict_len, pool=cfg.pool
        )
        if len(codes) == 0:
            continue

        # 取 [t-lookback, t] 的后复权 close，算动量 close[t]/close[t-10]-1
        t_ts = pd.Timestamp(ds)
        orig_start = provider._start_date
        orig_end = provider._end_date
        orig_inst = provider.instruments_
        fetch_start = (t_ts - pd.Timedelta(days=cfg.lookback * 2)).strftime("%Y-%m-%d")
        try:
            provider._start_date = fetch_start
            provider._end_date = ds
            provider.instruments_ = codes
            df = provider.fetch(["$close"], freq="day")
        finally:
            provider._start_date = orig_start
            provider._end_date = orig_end
            provider.instruments_ = orig_inst

        for code in codes:
            try:
                sub = (
                    df.xs(code, level="instrument")
                    if "instrument" in df.index.names
                    else df
                )
                sub = sub.sort_index().loc[:t_ts]
            except KeyError:
                continue
            if len(sub) < 11 or t_ts not in sub.index:
                continue
            close_t = sub.loc[t_ts, "close"]
            close_t_10 = sub.iloc[-11]["close"]
            mom = float(close_t / close_t_10 - 1.0)
            rows.append(
                {
                    "date": t_ts,
                    "code": code,
                    "momentum_10d": mom,
                    "reversal_10d": -mom,
                }
            )
    out = pd.DataFrame(rows)
    logger.info(
        f"基线因子：{out['date'].nunique()} 期 × "
        f"{out.groupby('date')['code'].count().mean():.0f} 只/期"
    )
    return out
