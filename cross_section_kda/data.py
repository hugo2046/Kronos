"""数据构造与预注册切分（计划 §2）。

要点（全部预注册，不许事后挪动）：

- 池 csi300，point-in-time 成分（复用 ``kronos_qlib``）；
- 样本：**每个交易日**逐股票一条（扩大训练集，不限于调仓日）；
- 特征窗 ≤t 的 90 个交易日，6 列 OHLCVA；剔除规则与
  ``kronos_qlib.windows.build_inference_windows`` 完全一致（行数不足 / 窗口内停牌
  整只剔除，不前向填充）；
- 标签 ``close[t+10]/close[t]-1``，**按日截面 z-score** 后做回归目标；
- 特征做窗口 z-score + clip5（与 ``KronosPredictor`` 同口径），保证 B1
  「预训练表示 vs 原始特征」是唯一变量；
- 切分含 10 交易日 purge（防标签窗口重叠泄露）：train 2022-01-04~2023-12-15、
  early-stop 2024-01-02~2024-06-14、final 2024-07-01~2026-07-22（50 期，封盘）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

# —— 预注册切分边界（计划 §2，不许事后挪动）——
TRAIN_START, TRAIN_END = "2022-01-04", "2023-12-15"
EARLY_STOP_START, EARLY_STOP_END = "2024-01-02", "2024-06-14"
FINAL_START = "2024-07-01"
PURGE = 10  # 标签窗口 H=10 的 purge 间隔

# 与 build_inference_windows 一致的 6 列（OHLCVA）顺序
REQUIRED_COLS: list[str] = ["open", "high", "low", "close", "volume", "amount"]
_AUX_COLS: list[str] = ["preclose", "tradestatuscode"]

LOOKBACK = 90
PREDICT_LEN = 10
CLIP = 5  # 与 KronosPredictor.clip 一致


# ============================================================
# 切分（含 purge 间隔断言）
# ============================================================


@dataclass(frozen=True)
class Splits:
    """预注册切分边界 + 日历引用（供 purge 间隔断言）。

    ``train`` / ``early_stop`` 为 ``(start, end)`` 字符串对；``final_start``
    为最终验证段起点（终点由 B0 网格决定，见 :func:`build_final_grid`）。
    """

    calendar: pd.DatetimeIndex
    train: tuple[str, str]
    early_stop: tuple[str, str]
    final_start: str
    purge: int = PURGE


def _cal_index_gap(cal: pd.DatetimeIndex, a: str, b: str) -> int:
    """a 到 b 之间的交易日数（不含 a，含 b 前的中间日）。"""
    ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    between = cal[(cal > ta) & (cal < tb)]
    return len(between)


def assert_purge_intervals(splits: Splits) -> None:
    """断言 train/early-stop、early-stop/final 两段间隔 >= purge 个交易日。

    防标签窗口（H=10）重叠泄露：train 末样本的标签用 [end, end+10]，
    不能与 early-stop 首样本的特征窗重叠。
    """
    train_end = splits.train[1]
    es_start = splits.early_stop[0]
    es_end = splits.early_stop[1]
    final_start = splits.final_start

    g1 = _cal_index_gap(splits.calendar, train_end, es_start)
    g2 = _cal_index_gap(splits.calendar, es_end, final_start)
    assert g1 >= splits.purge, (
        f"purge 间隔不足：train_end={train_end} → early_stop_start={es_start} "
        f"仅 {g1} 个交易日 < {splits.purge}"
    )
    assert g2 >= splits.purge, (
        f"purge 间隔不足：early_stop_end={es_end} → final_start={final_start} "
        f"仅 {g2} 个交易日 < {splits.purge}"
    )
    logger.info(
        f"purge 间隔 OK：train→es {g1} 日、es→final {g2} 日（均 ≥ {splits.purge}）"
    )


def make_splits(provider) -> Splits:
    """从真实交易日历构造切分（调 assert_purge_intervals 前先有日历）。"""
    cal = provider.trading_days()
    return Splits(
        calendar=cal,
        train=(TRAIN_START, TRAIN_END),
        early_stop=(EARLY_STOP_START, EARLY_STOP_END),
        final_start=FINAL_START,
        purge=PURGE,
    )


def build_final_grid(b0_signals_path: Path | str, *, final_start: str = FINAL_START) -> list[pd.Timestamp]:
    """final 段调仓日网格 = B0 signals parquet 中 >= final_start 的全部日期。

    保证五臂共用同一日期网格（与 B0 的 50 期完全相同，计划 §3）。
    """
    df = pd.read_parquet(b0_signals_path)
    dates = sorted(
        pd.Timestamp(d) for d in df["date"].unique()
        if pd.Timestamp(d) >= pd.Timestamp(final_start)
    )
    if len(dates) == 0:
        raise ValueError(f"B0 parquet 无 >= {final_start} 的调仓日，无法构造 final 网格")
    return dates


# ============================================================
# 样本构造（每个交易日逐股票一条）
# ============================================================


@dataclass
class SampleTensorBatch:
    """一个 batch / 一个调仓日的张量化样本。

    ``x_norm`` 已做窗口 z-score + clip5；``stamp`` 为 5 列时间特征；
    ``y_z`` 为按日截面 z-score 的标签（仅训练/早停段用）；``fwd_ret_raw`` 为
    未截面化的原始前向收益（评估时做 Spearman，绝不回流入特征）。
    """

    date: pd.Timestamp
    codes: list[str]
    x_norm: torch.Tensor        # [N, T, 6]
    stamp: torch.Tensor         # [N, T, 5]
    fwd_ret_raw: np.ndarray     # [N] 原始 close[t+10]/close[t]-1（截面 z-score 前）
    y_z: np.ndarray             # [N] 按日截面 z-score（回归目标）


def _calc_time_stamps(ts_index: pd.DatetimeIndex) -> np.ndarray:
    """复刻 model/kronos.py 的 calc_time_stamps，返回 [T, 5] float32。"""
    df = pd.DataFrame(index=ts_index)
    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["day"] = df.index.day
    df["month"] = df.index.month
    return df.values.astype(np.float32)


def _window_zscore_clip(x: np.ndarray) -> np.ndarray:
    """窗口 z-score + clip5，与 KronosPredictor.predict_batch 内部一致。

    :param x: [T, 6] 原始 OHLCVA。
    :returns: [T, 6] 归一化。
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    z = (x - mean) / (std + 1e-5)
    return np.clip(z, -CLIP, CLIP).astype(np.float32)


def build_daily_samples(
    provider,
    *,
    date: str,
    pool: str = "csi300",
    lookback: int = LOOKBACK,
    predict_len: int = PREDICT_LEN,
) -> SampleTensorBatch | None:
    """构造单个交易日 t 的全部样本（逐 point-in-time 成分股票一条）。

    剔除规则与 ``build_inference_windows`` 一致：行数不足 lookback / 窗口内
    含停牌（``tradestatuscode==0``）整只剔除，不前向填充。

    :param provider: :class:`kronos_qlib.QlibProvider`。
    :param date: 交易日 ``YYYY-MM-DD``。
    :param pool: 市场字符串。
    :returns: :class:`SampleTensorBatch`；该日无可用样本时返回 None。
    """
    t = pd.Timestamp(date)
    members = provider.list_pool_at(pool, date)
    if len(members) == 0:
        return None

    full_cal = provider.trading_days()
    cal_le_t = full_cal[full_cal <= t]
    if len(cal_le_t) < lookback + predict_len + 1:
        return None
    t_pos = len(cal_le_t) - 1
    x_start_pos = max(0, t_pos - lookback + 1)
    x_window_cal = full_cal[x_start_pos : t_pos + 1]
    y_end_pos = t_pos + predict_len
    y_window_cal = full_cal[t_pos + 1 : y_end_pos + 1]
    if len(y_window_cal) < predict_len:
        return None

    fetch_start = x_window_cal[0].strftime("%Y-%m-%d")
    fetch_end = y_window_cal[-1].strftime("%Y-%m-%d")
    orig_start, orig_end, orig_inst = (
        provider._start_date, provider._end_date, provider.instruments_,
    )
    try:
        provider._start_date = fetch_start
        provider._end_date = fetch_end
        provider.instruments_ = members
        fields = [f"${c}" for c in REQUIRED_COLS + _AUX_COLS]
        raw = provider.fetch(fields, freq="day")
    finally:
        provider._start_date = orig_start
        provider._end_date = orig_end
        provider.instruments_ = orig_inst

    if "instrument" not in raw.index.names:
        raise ValueError(
            f"fetch 返回缺 instrument 索引层，got {raw.index.names}"
        )
    available = raw.index.get_level_values("instrument").unique()
    y_idx = pd.DatetimeIndex(y_window_cal)

    xs: list[np.ndarray] = []
    stamps: list[np.ndarray] = []
    codes: list[str] = []
    fwd: list[float] = []
    x_ts_ref = pd.DatetimeIndex(x_window_cal)
    n_short = n_halt = 0

    for code in members:
        if code not in available:
            n_short += 1
            continue
        sub = raw.xs(code, level="instrument").sort_index()
        sub_x = sub.loc[:t]
        if len(sub_x) < lookback:
            n_short += 1
            continue
        window = sub_x.iloc[-lookback:]
        if "tradestatuscode" in window.columns and (window["tradestatuscode"] == 0).any():
            n_halt += 1
            continue
        # 标签 close[t+10]/close[t]-1（后复权 close-to-close）
        if t not in sub.index or y_window_cal[-1] not in sub.index:
            n_short += 1
            continue
        close_t = float(sub.loc[t, "close"])
        close_end = float(sub.loc[y_window_cal[-1], "close"])
        fwd_ret = close_end / close_t - 1.0

        x_raw = window[REQUIRED_COLS].values.astype(np.float32)
        xs.append(_window_zscore_clip(x_raw))
        stamps.append(_calc_time_stamps(window.index))
        codes.append(code)
        fwd.append(fwd_ret)

    if not xs:
        logger.warning(f"build_daily_samples({date}): 无可用样本 short={n_short} halt={n_halt}")
        return None

    x_norm = torch.from_numpy(np.stack(xs))             # [N, T, 6]
    stamp = torch.from_numpy(np.stack(stamps))          # [N, T, 5]
    fwd_arr = np.array(fwd, dtype=np.float64)           # [N]
    # 按日截面 z-score：样本内（同日截面）标准化，回归目标 MSE
    mu, sd = fwd_arr.mean(), fwd_arr.std()
    y_z = ((fwd_arr - mu) / (sd + 1e-8)).astype(np.float32)

    logger.debug(
        f"build_daily_samples({date}): kept={len(xs)} short={n_short} halt={n_halt} "
        f"fwd_ret mean={fwd_arr.mean():.5f} std={fwd_arr.std():.5f}"
    )
    return SampleTensorBatch(
        date=t, codes=codes, x_norm=x_norm, stamp=stamp,
        fwd_ret_raw=fwd_arr, y_z=y_z,
    )


__all__ = [
    "Splits",
    "assert_purge_intervals",
    "make_splits",
    "build_final_grid",
    "build_daily_samples",
    "SampleTensorBatch",
    "TRAIN_START", "TRAIN_END",
    "EARLY_STOP_START", "EARLY_STOP_END",
    "FINAL_START", "PURGE",
    "REQUIRED_COLS", "LOOKBACK", "PREDICT_LEN", "CLIP",
]
