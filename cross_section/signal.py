"""Kronos 预测 → 截面信号 + 前向收益（计划 §0 信号定义 / §4 全量产物）。

信号定义（§0，逐字一致，不可漂移）::

    signal_t = mean(pred_close[t+1..t+H]) / close_t - 1

前向收益（§4，事后评估用，绝不回流入信号）::

    fwd_ret_10d = close[t+H] / close[t] - 1   （后复权 close-to-close）

无未来函数自查（§4）：信号只用 <= t 的数据；fwd_ret 只用于事后评估。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from cross_section.common import ExperimentConfig


def compute_signal_from_preds(
    pred_close_path: pd.Series,
    last_close: float,
) -> float:
    """从一条预测 close 路径算 H 日平均预期收益率信号。

    :param pred_close_path: ``predict_batch`` 返回的 close 列（长度 H）。
    :param last_close: 输入窗口末值（调仓日 t 的真实后复权 close）。
    :returns: ``signal_t = mean(pred_close[t+1..t+H]) / close_t - 1``。
    """
    return float(np.mean(pred_close_path.values) / last_close - 1.0)


def build_signal_frame(
    rows: list[dict],
) -> pd.DataFrame:
    """把逐调仓日逐股票的信号行汇总为长表。

    :param rows: 每行含 ``date / code / signal / fwd_ret_10d``。
    :returns: 列为 ``[date, code, signal, fwd_ret_10d]`` 的 DataFrame。
    """
    df = pd.DataFrame(rows)
    df = df[["date", "code", "signal", "fwd_ret_10d"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_fwd_ret(
    provider,
    cfg: ExperimentConfig,
    code: str,
    t: str,
    y_dates: pd.DatetimeIndex,
) -> float:
    """算单只股票的前向 10 日收益（事后评估用，不进信号）。

    :param provider: :class:`kronos_qlib.QlibProvider`。
    :param cfg: 实验配置。
    :param code: 股票代码。
    :param t: 调仓日 ``YYYY-MM-DD``。
    :param y_dates: 调仓日之后 H 个交易日（来自 ``build_inference_windows``）。
    :returns: ``close[t+H] / close[t] - 1``（后复权 close-to-close）。
    """
    fetch_start = pd.Timestamp(t)
    fetch_end = y_dates[-1]
    # 复用 provider 取数：临时改区间与 instruments（_fetch_via 同款最小侵入）
    orig_start = provider._start_date
    orig_end = provider._end_date
    orig_inst = provider.instruments_
    try:
        provider._start_date = fetch_start.strftime("%Y-%m-%d")
        provider._end_date = fetch_end.strftime("%Y-%m-%d")
        provider.instruments_ = [code]
        df = provider.fetch(["$close"], freq="day")
    finally:
        provider._start_date = orig_start
        provider._end_date = orig_end
        provider.instruments_ = orig_inst

    # df: MultiIndex(datetime, instrument)，level 0 = datetime
    sub = df.xs(code, level="instrument") if "instrument" in df.index.names else df
    sub = sub.sort_index()
    if pd.Timestamp(t) not in sub.index or y_dates[-1] not in sub.index:
        return np.nan
    close_t = sub.loc[pd.Timestamp(t), "close"]
    close_end = sub.loc[y_dates[-1], "close"]
    return float(close_end / close_t - 1.0)


def run_inference_one_period(
    predictor,
    cfg: ExperimentConfig,
    provider,
    rebalance_date: str,
) -> tuple[pd.DataFrame, dict]:
    """单调仓日全链路：构窗 → predict_batch → signal。

    :param predictor: :class:`model.KronosPredictor`。
    :param cfg: 实验配置。
    :param provider: :class:`kronos_qlib.QlibProvider`。
    :param rebalance_date: 调仓日 ``YYYY-MM-DD``。
    :returns: ``(period_df, stats)``，period_df 含
        ``[date, code, signal, fwd_ret_10d]``。
    """
    import torch

    from kronos_qlib import build_inference_windows

    df_list, x_ts_list, y_ts_list, codes, win_stats = build_inference_windows(
        provider,
        rebalance_date,
        lookback=cfg.lookback,
        predict_len=cfg.predict_len,
        pool=cfg.pool,
    )
    if len(df_list) == 0:
        logger.warning(f"{rebalance_date}: 无可用股票（{win_stats}）")
        return pd.DataFrame(), win_stats

    last_closes = [df["close"].iloc[-1] for df in df_list]
    # §0：全程 torch.manual_seed(42)；predict_batch 内部对 sample_count 次采样取均值
    torch.manual_seed(cfg.seed)
    preds = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_ts_list,
        y_timestamp_list=y_ts_list,
        pred_len=cfg.predict_len,
        T=cfg.T,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        sample_count=cfg.sample_count,
        verbose=False,
    )

    rows = []
    for j, pred_df in enumerate(preds):
        signal = compute_signal_from_preds(pred_df[cfg.signal_field], last_closes[j])
        # 前向收益：用真实后复权 close 算，事后评估用
        y_dates = pd.DatetimeIndex(y_ts_list[j])
        fwd_ret = compute_fwd_ret(provider, cfg, codes[j], rebalance_date, y_dates)
        rows.append(
            {
                "date": pd.Timestamp(rebalance_date),
                "code": codes[j],
                "signal": signal,
                "fwd_ret_10d": fwd_ret,
            }
        )
    period_df = pd.DataFrame(rows)
    return period_df, win_stats
