"""四组信号生成（计划 §3 / §4）。

四组信号（同引擎、同区间、同池，控制变量）：

    - **K**（Kronos-base, N=20）：``run_kronos_signals`` 复用
      ``kronos_qlib.build_inference_windows`` + ``KronosPredictor.predict_batch``，
      ``signal_t = mean(pred_close[t+1..t+H]) / close_t - 1``（与 cross_section 同口径）。
    - **M**（10 日动量）：``close[t]/close[t-10] - 1``。
    - **R**（10 日反转）：``-动量``。
    - **P**（随机占位, seed=42）：``numpy.random.default_rng(42)`` 每日每股一个标准正态。

宽表约定：``index=date, columns=code, values=signal``（等权组合引擎消费）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from paper_replication.common import ReplicationConfig


def predict_batch_chunked(
    predictor,
    df_list,
    x_ts_list,
    y_ts_list,
    *,
    pred_len: int,
    T: float,
    top_k: int,
    top_p: float,
    sample_count: int,
    chunk_size: int = 32,
    verbose: bool = False,
):
    """对 ``predict_batch`` 做显存友好的分块包装。

    ``predict_batch`` 内部把 B 只股票 × sample_count 次采样一次性堆成
    (B, seq, feat) 张量送 GPU，N=20 / B=299 时显存峰值超 RTX 5090 32GB（实测 OOM）。
    按 ``chunk_size`` 切片逐块推理、块间清缓存，把单次显存峰值压到 chunk_size×N。

    :param chunk_size: 单块股票数（默认 32；N=20 下 32×20=640 序列，实测安全）。
    :returns: 与输入顺序一致的 ``pred_df`` 列表（与 ``predict_batch`` 同构）。
    """
    import torch

    n = len(df_list)
    out: list = []
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        preds_chunk = predictor.predict_batch(
            df_list=df_list[s:e],
            x_timestamp_list=x_ts_list[s:e],
            y_timestamp_list=y_ts_list[s:e],
            pred_len=pred_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        out.extend(preds_chunk)
        # 块间释放显存碎片（N=20 下连续块会累积 reserved-but-unallocated）
        torch.cuda.empty_cache()
    return out


def compute_signal_from_preds(pred_close_path: pd.Series, last_close: float) -> float:
    """从一条预测 close 路径算 H 日平均预期收益率信号（与 cross_section/signal.py 同口径）。

        signal_t = mean(pred_close[t+1..t+H]) / close_t - 1
    """
    return float(np.mean(pred_close_path.values) / last_close - 1.0)


def run_kronos_signals(
    predictor,
    provider,
    cfg: ReplicationConfig,
    rebalances: pd.DatetimeIndex,
    *,
    progress_every: int = 10,
) -> pd.DataFrame:
    """逐交易日生成 Kronos 信号宽表（K 组）。

    复用 ``build_inference_windows`` + ``predict_batch``（与 cross_section/signal.py
    同链路），区别仅在于调仓日为**每日**（论文口径）而非每 10 日。

    :param predictor: :class:`model.KronosPredictor`。
    :param provider: :class:`kronos_qlib.QlibProvider`。
    :param cfg: 复现配置。
    :param rebalances: 调仓日序列（每日）。
    :param progress_every: 每 N 日打印一次进度。
    :returns: 信号宽表 ``index=date, columns=code, values=signal``。
    """
    import torch

    from kronos_qlib import build_inference_windows

    rows = []  # 每日一行 dict{code: signal}
    for i, d in enumerate(rebalances):
        ds = d.strftime("%Y-%m-%d")
        df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows(
            provider,
            ds,
            lookback=cfg.lookback,
            predict_len=cfg.predict_len,
            pool=cfg.pool,
        )
        if len(df_list) == 0:
            logger.warning(f"{ds}: 无可用股票（{stats}）")
            rows.append({})
            continue

        last_closes = [df["close"].iloc[-1] for df in df_list]
        torch.manual_seed(cfg.seed)
        preds = predict_batch_chunked(
            predictor,
            df_list,
            x_ts_list,
            y_ts_list,
            pred_len=cfg.predict_len,
            T=cfg.T,
            top_k=cfg.sample_top_k,
            top_p=cfg.top_p,
            sample_count=cfg.sample_count,
        )
        day_signals = {}
        for j, pred_df in enumerate(preds):
            day_signals[codes[j]] = compute_signal_from_preds(
                pred_df[cfg.signal_field], last_closes[j]
            )
        rows.append(day_signals)

        if (i + 1) % progress_every == 0 or i == 0:
            logger.info(
                f"Kronos 信号 [{i + 1}/{len(rebalances)}] {ds}: "
                f"{len(codes)} 只，信号 std={np.std(list(day_signals.values())):.4f}"
            )

    wide = pd.DataFrame(rows, index=rebalances)
    logger.info(f"Kronos 信号宽表：{wide.shape[0]} 日 × 平均 {wide.notna().sum(axis=1).mean():.0f} 只/日")
    return wide


def run_momentum_reversal(
    provider, cfg: ReplicationConfig, rebalances: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐交易日算 10 日动量 / 反转信号（M / R 组）。

    与 cross_section/baselines.py 同口径：``momentum = close[t]/close[t-10] - 1``，
    ``reversal = -momentum``。一次性取全区间 close 再逐日切片，避免重复 fetch。

    :returns: ``(mom_wide, rev_wide)``。
    """
    fetch_start = (
        pd.Timestamp(cfg.backtest_start) - pd.Timedelta(days=cfg.lookback * 2)
    ).strftime("%Y-%m-%d")
    orig_start = provider._start_date
    orig_end = provider._end_date
    orig_inst = provider.instruments_
    try:
        provider._start_date = fetch_start
        provider._end_date = cfg.backtest_end
        provider.instruments_ = cfg.pool
        df = provider.fetch(["$close"], freq="day")
    finally:
        provider._start_date = orig_start
        provider._end_date = orig_end
        provider.instruments_ = orig_inst

    if "instrument" in df.index.names:
        wide = df["close"].unstack("instrument")
    else:
        wide = df["close"]
    wide = wide.sort_index()

    mom_rows, rev_rows = [], []
    for d in rebalances:
        if d not in wide.index:
            mom_rows.append({})
            rev_rows.append({})
            continue
        # 需要 t 和 t-10（10 个交易日前）的 close
        loc = wide.index.get_loc(d)
        if loc < 10:
            mom_rows.append({})
            rev_rows.append({})
            continue
        close_t = wide.iloc[loc]
        close_t_10 = wide.iloc[loc - 10]
        mom = close_t / close_t_10 - 1.0
        mom_rows.append(mom.to_dict())
        rev_rows.append((-mom).to_dict())

    mom_wide = pd.DataFrame(mom_rows, index=rebalances)
    rev_wide = pd.DataFrame(rev_rows, index=rebalances)
    logger.info(
        f"M/R 信号：{mom_wide.shape[0]} 日 × 平均 {mom_wide.notna().sum(axis=1).mean():.0f} 只/日"
    )
    return mom_wide, rev_wide


def run_placeholder(
    cfg: ReplicationConfig,
    rebalances: pd.DatetimeIndex,
    columns: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """随机占位信号（P 组）。

    每日每股一个标准正态，``numpy.random.default_rng(seed)``。与收益完全独立，
    用于引擎零点门禁（§4 规则1）。列与 K/M/R 对齐（取并集，缺失填 NaN）。

    :param columns: 列名（来自 K/M/R 的并集，保证四组同池可比）。
    :param seed: 随机种子。
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(rebalances), len(columns)))
    wide = pd.DataFrame(data, index=rebalances, columns=columns)
    logger.info(f"占位信号 P（seed={seed}）：{wide.shape[0]} 日 × {wide.shape[1]} 列")
    return wide


def build_px_tradeable(
    provider,
    cfg: ReplicationConfig,
    rebalances: pd.DatetimeIndex,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """取全区间后复权 close + 可交易掩码（停牌=不可交易）宽表。

    引擎用 ``px_wide`` 算逐日收益与换手额，用 ``tradeable`` 跳过当日停牌股。

    :param columns: 股票池列（与信号对齐）。
    :returns: ``(px_wide, tradeable)``：
        ``px_wide`` = 后复权 close 宽表；
        ``tradeable`` = bool 宽表（``tradestatuscode==1`` 为正常交易）。
    """
    orig_start = provider._start_date
    orig_end = provider._end_date
    orig_inst = provider.instruments_
    try:
        provider._start_date = cfg.backtest_start
        provider._end_date = cfg.backtest_end
        provider.instruments_ = columns
        df = provider.fetch(["$close", "$tradestatuscode"], freq="day")
    finally:
        provider._start_date = orig_start
        provider._end_date = orig_end
        provider.instruments_ = orig_inst

    if "instrument" in df.index.names:
        px = df["close"].unstack("instrument")
        tsc = df["tradestatuscode"].unstack("instrument")
    else:
        px = df["close"]
        tsc = df["tradestatuscode"]
    px = px.sort_index()
    tsc = tsc.sort_index().reindex_like(px)
    # 本 DDB tradestatuscode 取值域（论文窗口实测）：-1 正常交易（72041），
    # 0 / 2 停牌（155 / 393），4 罕见特殊状态（11）。可交易 = 仅 -1。
    # 注：cross_section/windows.py 用 ``==0`` 判停牌（只捕获 0，漏 2），
    # 那是既有代码的口径，本模块按**真实数据语义**用 ``==-1`` 为可交易，更严格。
    tradeable = (tsc == -1).fillna(False)
    tradeable = tradeable & px.notna()
    logger.info(
        f"价格/可交易宽表：{px.shape[0]} 日 × {px.shape[1]} 列；"
        f"日均不可交易 {(~tradeable).sum(axis=1).mean():.1f} 只"
    )
    return px, tradeable
