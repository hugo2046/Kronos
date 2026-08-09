"""构造 Kronos 推理窗口。

核心函数 :func:`build_inference_windows` 产出可直接喂给
``KronosPredictor.predict_batch``（model/kronos.py:562）的四元组
``df_list / x_timestamp_list / y_timestamp_list / codes``。

7 条语义规定（每条都是正确性要求，非风格偏好，详见计划 §2.2）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# 与 KronosPredictor.price_cols + vol_col + amt_vol（model/kronos.py:489-491）
# 严格一致的列顺序。predict 内部按 z-score + clip 5 自行归一化，这里**不预归一化**。
REQUIRED_COLS: list[str] = ["open", "high", "low", "close", "volume", "amount"]

# 数据层额外拉取、仅用于跳过判定、不进窗口的辅助字段
_AUX_COLS: list[str] = ["preclose", "tradestatuscode"]


def build_inference_windows(
    provider,
    rebalance_date: str,
    *,
    lookback: int = 90,
    predict_len: int = 10,
    pool: str = "csi300",
    filter_pipe: list | None = None,
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.Series], list[str], dict]:
    """构造 ``predict_batch`` 所需的四元组 + 统计字典。

    :param provider: :class:`QlibProvider` 实例（或 duck-typing 等价物，
        需实现 ``fetch / trading_days / list_pool_at``）。
    :param rebalance_date: 调仓日 ``t``，``YYYY-MM-DD``。
    :param lookback: 窗口长度 L（输入历史交易日数）。
    :param predict_len: 预测长度 H（未来交易日数）。
    :param pool: 市场字符串（``csi300`` 等）。
    :param filter_pipe: 可选过滤器（如 ``STFilter``），默认不启用。**陷阱 5**：
        启用 filter_pipe 时 provider 构造时 instruments 必须传 str 市场名，
        传 list 会被静默丢弃。
    :returns: ``(df_list, x_timestamp_list, y_timestamp_list, codes, stats)``，
        前 4 项顺序一一对应；``stats`` 含
        ``n_pool / n_kept / skipped_short / skipped_halt``。

    语义（计划 §2.2）：

    1. 池按 t 时点取（point-in-time 成分，无幸存者偏差）。
    2. 窗口取 ≤ t 的最后 L 个交易日；行数不足 L 整只跳过。
    3. 窗口内含 ``tradestatuscode == 0``（停牌）→ 跳过；**不前向填充**。
    4. 列顺序固定 :data:`REQUIRED_COLS`。
    5. 不做归一化（predict 内部自做 z-score + clip 5）。
    6. y_timestamp 取 t 之后的 H 个交易日，**来自 ``D.calendar``**（节假日不可
       用 ``date + n`` 推），且须落在真实数据覆盖区间内（陷阱 3）。
    7. ST 过滤留成 filter_pipe 可选参数，默认不启用。
    """
    t = pd.Timestamp(rebalance_date)

    # —— ① 池按 t 时点取（point-in-time 成分）———————————————
    members = provider.list_pool_at(pool, rebalance_date)
    if len(members) == 0:
        raise ValueError(
            f"build_inference_windows：{rebalance_date} 时点 {pool} 成分为空"
        )

    # —— ⑥ 先确定交易日窗口，约束在数据覆盖内（陷阱 3）————————
    full_cal = provider.trading_days()
    # 真实数据覆盖末日：日历里截至"今天"的最新交易日（日历延伸到 2040，
    # 但真实行情止于更早日期）。以 full_cal 中 <= t 的部分确定 t 在日历中的
    # 位置，再向后切 H 个交易日作为 y_timestamp。
    cal_le_t = full_cal[full_cal <= t]
    if len(cal_le_t) == 0:
        raise ValueError(
            f"build_inference_windows：日历中无 <= {rebalance_date} 的交易日"
        )
    # t 应是交易日；若不是，向下取到 <= t 的最近交易日作为有效调仓日
    t_effective = cal_le_t[-1]
    t_pos = len(cal_le_t) - 1  # t_effective 在 full_cal 中的下标

    # x/y 窗口在日历上的起止
    x_start_pos = max(0, t_pos - lookback + 1)
    x_window_cal = full_cal[x_start_pos : t_pos + 1]  # ≤ t 的最多 L 个交易日
    y_window_cal = full_cal[t_pos + 1 : t_pos + 1 + predict_len]
    if len(y_window_cal) < predict_len:
        raise ValueError(
            f"build_inference_windows：{rebalance_date} 之后交易日不足 "
            f"{predict_len} 个（日历末端），无法构造 y_timestamp"
        )

    # —— ② 拉取区间数据（一次拉全区间，逐股票切片）——————————
    fetch_start = x_window_cal[0].strftime("%Y-%m-%d")
    fetch_end = y_window_cal[-1].strftime("%Y-%m-%d")
    # 计划陷阱 5：list + filter_pipe 会被静默丢弃。无 filter_pipe 时传 list
    # 精准取成分；启用 filter_pipe 时改传 str 市场名（让过滤器生效）。
    instruments_arg = pool if filter_pipe is not None else members

    fields = [f"${c}" for c in REQUIRED_COLS + _AUX_COLS]
    raw = _fetch_via(
        provider, instruments_arg, fetch_start, fetch_end, fields, filter_pipe
    )

    # raw: MultiIndex(datetime, instrument)（swap_level=True 下 level 0 是
    # datetime）。**直接在堆叠表上逐股票 xs**，避免 unstack 把稀疏的
    # (日期, 股票) 网格补成 NaN——NaN 填充会让"行数不足"检查失效（看似有 L
    # 行实则多半 NaN）。qlib QlibDataLoader.load 返回的就是稀疏堆叠表。
    if "instrument" not in raw.index.names:
        raise ValueError(
            f"fetch 返回的 DataFrame 缺少 instrument 索引层，实际 index.names="
            f"{raw.index.names}（应为 MultiIndex(datetime, instrument)）"
        )
    available = raw.index.get_level_values("instrument").unique()

    df_list: list[pd.DataFrame] = []
    x_ts_list: list[pd.Series] = []
    y_ts_list: list[pd.Series] = []
    codes: list[str] = []
    stats = {
        "n_pool": len(members),
        "n_kept": 0,
        "skipped_short": 0,
        "skipped_halt": 0,
    }

    y_window_idx = pd.DatetimeIndex(y_window_cal)

    for code in members:
        if code not in available:
            # 该股票在拉取区间内无任何数据 → 行数不足，跳过
            stats["skipped_short"] += 1
            continue
        # 取该股票全部历史，截到 ≤ t_effective（按 level 名 xs，避免位置歧义）
        sub = raw.xs(code, level="instrument")
        sub = sub.loc[:t_effective]

        # —— ② 行数不足 L 整只跳过（新上市/长期停牌）————————
        if len(sub) < lookback:
            stats["skipped_short"] += 1
            continue

        window = sub.iloc[-lookback:]

        # —— ③ 停牌剔除：窗口内含 tradestatuscode==0 → 跳过，不前向填充 ——
        if "tradestatuscode" in window.columns and (
            (window["tradestatuscode"] == 0)
        ).any():
            stats["skipped_halt"] += 1
            continue

        # —— ④ 列顺序固定（只取 REQUIRED_COLS，丢弃辅助字段）—————
        window_df = window[REQUIRED_COLS].copy()

        # —— ⑤ 不归一化（predict 内部自做）——————————————
        # x_timestamp 来自窗口的实际交易日（稀疏，已剔除无数据日），
        # 而非日历投影——保证与 df 行数严格一致。
        x_ts = pd.Series(window_df.index)
        y_ts = pd.Series(y_window_idx)

        df_list.append(window_df)
        x_ts_list.append(x_ts)
        y_ts_list.append(y_ts)
        codes.append(code)
        stats["n_kept"] += 1

    logger.info(
        f"build_inference_windows({rebalance_date}, pool={pool}, "
        f"lookback={lookback}, predict_len={predict_len}): "
        f"pool={stats['n_pool']} kept={stats['n_kept']} "
        f"skipped_short={stats['skipped_short']} "
        f"skipped_halt={stats['skipped_halt']}"
    )
    return df_list, x_ts_list, y_ts_list, codes, stats


def _fetch_via(provider, instruments_arg, start, end, fields, filter_pipe):
    """在 provider 的拉数区间与 instruments 上调一次 fetch。

    单独成函数便于单测注入 FakeProvider：FakeProvider.fetch 只看 instruments_arg
    / start / end / fields / filter_pipe 即可返回固定 MultiIndex。
    """
    # 临时改 provider 的区间与 instruments，让 fetch 用到本次窗口的精确边界。
    orig_start, orig_end, orig_inst = (
        provider._start_date,
        provider._end_date,
        provider.instruments_,
    )
    try:
        provider._start_date = start
        provider._end_date = end
        provider.instruments_ = instruments_arg
        return provider.fetch(fields, filter_pipe=filter_pipe, freq="day")
    finally:
        provider._start_date = orig_start
        provider._end_date = orig_end
        provider.instruments_ = orig_inst
