"""MH1 数据构造：四期限标签、purge、同日截面与成对采样（计划 §4）。

数据源两路（与 H1 同口径，协议记录）：

- **训练段**：G1 同源 union pkl（``h1_readout.corpus_loader``，csi300 PIT 并集
  2014-01-02~2025-06-30），90 行窗口完整 + 四期限标签全部有限且终点 ≤
  ``TRAIN_LABEL_END``；日线历 = 语料全体符号日期并集（H1 同构，自包含不触 DDB）。
- **验证 / W3 / W4**：qlib PIT csi300（``QlibProvider``，t 日成员资格），
  x 窗语义与 ``cross_section_kda.data.build_daily_samples`` 逐字一致（行数
  不足/窗口内 ``tradestatuscode==0`` 整只剔除，不前向填充）；标签按交易所
  日历定位 t+k，终点缺行保持 NaN。

泄漏门禁：所有行情查询与标签终点上界 ≤ ``FORWARD_CUTOFF``（2026-07-24），
任何路径越界立即断言失败。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger

from mh1_multihorizon.config import (
    BATCH, CLIP, FORWARD_CUTOFF, HORIZONS, LOOKBACK, MIN_TRAIN_CROSS, POOL,
)

_FEATURES = ["open", "high", "low", "close", "vol", "amt"]
# qlib 侧列名（build_daily_samples 同款）
_REQUIRED_QLIB = ["open", "high", "low", "close", "volume", "amount"]
_AUX_QLIB = ["preclose", "tradestatuscode"]


# ============================================================
# 核心标签逻辑（先按交易所日历对齐，再定位未来价格）
# ============================================================


def horizon_labels(
    sym_dates: pd.DatetimeIndex,
    sym_close: np.ndarray,
    calendar: pd.DatetimeIndex,
    cal_pos: dict[pd.Timestamp, int],
    i: int,
    horizons: Sequence[int],
) -> np.ndarray:
    """单样本四期限标签（核心日历定位，计划 §4.1）。

    决策日 = ``sym_dates[i]`` 在交易所日历上的位置 ``c``；第 k 期限终点 =
    ``calendar[c+k]``，**必须在该符号的行日期中恰有此日**才有标签；缺行保持
    NaN（不前向/后向填充，不用个股下一条记录代替下一交易日）；分母须有限
    且 > 0。

    :param sym_dates: 符号行日期（升序，交易所日历子集）。
    :param sym_close: 符号后复权收盘 ``[n]``。
    :param calendar: 交易所（或语料并集）交易日历。
    :param cal_pos: 日历日期 → 下标。
    :param i: 决策日行号。
    :param horizons: 期限元组。
    :returns: ``[len(horizons)]`` float64 标签（缺失为 NaN）。
    """
    t = sym_dates[i]
    c = cal_pos[t]
    n_cal = len(calendar)
    date_pos = {d: j for j, d in enumerate(sym_dates)}
    base = sym_close[i]
    out = np.full(len(horizons), np.nan, dtype=np.float64)
    if not (np.isfinite(base) and base > 0):
        return out
    for hi, k in enumerate(horizons):
        if c + k >= n_cal:
            continue
        j = date_pos.get(calendar[c + k])
        if j is None or j <= i:
            continue
        nxt = sym_close[j]
        if np.isfinite(nxt) and nxt > 0:
            out[hi] = float(nxt) / float(base) - 1.0
    return out


# ============================================================
# union pkl 表（训练段）
# ============================================================


@dataclass(frozen=True)
class SymbolTable:
    """pkl 同构符号表：OHLCVA ``[n,6]`` float32 + 行日期。"""

    vals: np.ndarray
    dates: pd.DatetimeIndex

    @property
    def close(self) -> np.ndarray:
        return self.vals[:, 3].astype(np.float64)


def load_union_tables(pool: str = POOL) -> dict[str, SymbolTable]:
    """train+val pkl 拼接 → {code: SymbolTable}（H1 ``_load_symbol_tables`` 同构）。

    只经 ``h1_readout.corpus_loader`` 受限 Unpickler 读取，去重排序，不修改源。
    """
    from h1_readout.corpus_loader import load_corpus_split

    tr = load_corpus_split(pool, "train")
    va = load_corpus_split(pool, "val")
    out: dict[str, SymbolTable] = {}
    for code in sorted(set(tr) | set(va)):
        parts = [d for d in (tr.get(code), va.get(code)) if d is not None and len(d)]
        if not parts:
            continue
        df = pd.concat(parts) if len(parts) > 1 else parts[0]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        out[code] = SymbolTable(
            vals=df[_FEATURES].values.astype(np.float32),
            dates=pd.DatetimeIndex(df.index))
    return out


def union_calendar(tables: dict[str, SymbolTable]) -> pd.DatetimeIndex:
    """语料全体符号日期并集（H1 同构交易日历，自包含）。"""
    cal = sorted(set().union(*(t.dates for t in tables.values())))
    return pd.DatetimeIndex(cal)


def _calc_stamps(dates: pd.DatetimeIndex) -> np.ndarray:
    """5 列时间特征 [T,5]（minute/hour/weekday/day/month，官方同口径）。"""
    df = pd.DataFrame(index=dates)
    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["day"] = df.index.day
    df["month"] = df.index.month
    return df.values.astype(np.float32)


def _window_zscore_clip(x: np.ndarray) -> np.ndarray:
    """窗口 z-score + clip5（KronosPredictor 同口径；只用本窗口统计量）。"""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    z = (x - mean) / (std + 1e-5)
    return np.clip(z, -CLIP, CLIP).astype(np.float32)


# ============================================================
# 训练段扫描（向量化；20 日跨段剔除内建于决策日上界）
# ============================================================


@dataclass
class TrainDay:
    """训练决策日：截面成员 (code, 行号) + 四期限原始标签 ``[N,4]``。"""

    date: pd.Timestamp
    codes: list[str]
    rows: np.ndarray     # [N] 决策日行号（窗口末行）
    labels: np.ndarray   # [N,4] raw close[t+k]/close[t]-1（全部有限）


@dataclass
class ScanStats:
    """扫描统计（计划 §4.2/§4.3：每日报告排除数，缺失终点单独统计）。"""

    n_days: int = 0
    n_samples: int = 0
    skipped_days_lt_min: int = 0
    excluded_missing_label_endpoint: int = 0
    excluded_short_window: int = 0
    excluded_bad_denominator: int = 0
    decision_min: str | None = None
    decision_max: str | None = None
    label_endpoint_min: str | None = None
    label_endpoint_max: str | None = None


def scan_train_days(
    tables: dict[str, SymbolTable],
    calendar: pd.DatetimeIndex,
    *,
    start: str,
    label_end: str,
    horizons: Sequence[int] = HORIZONS,
    min_cross: int = MIN_TRAIN_CROSS,
    lookback: int = LOOKBACK,
) -> tuple[list[TrainDay], ScanStats]:
    """扫描训练决策日（H1 ``_scan`` 向量化同构 + 四期限完整标签）。

    样本须同时满足：决策日 ≥ ``start``；**所有**期限终点（含 +20）按日历
    ≤ ``label_end``（跨段剔除）；四期限价格行齐全且有限（分母 > 0）；
    90 行窗口完整（行不足 = 停牌/缺数 → 跳过，不前向填充）。日截面 ≥
    ``min_cross`` 才成日，否则整日跳过计入统计。
    """
    n_cal = len(calendar)
    cal_pos = {d: i for i, d in enumerate(calendar)}
    max_k = max(horizons)
    start_c = int(calendar.searchsorted(pd.Timestamp(start), side="left"))
    # 决策日日历上界：c + max_k 仍在日历内且 ≤ label_end
    lab_c = int(calendar.searchsorted(pd.Timestamp(label_end), side="right")) - 1
    cap_c = min(lab_c - max_k, n_cal - 1 - max_k)
    if cap_c < start_c:
        return [], ScanStats()

    stats = ScanStats()
    per_day: dict[int, list[tuple[str, int, np.ndarray]]] = {}

    for code, tab in tables.items():
        close = tab.close
        n = len(tab.dates)
        if n == 0:
            continue
        pos = calendar.searchsorted(tab.dates)  # 行 → 日历位（dates ⊆ calendar）
        rowidx = np.full(n_cal, -1, np.int64)
        rowidx[pos] = np.arange(n, dtype=np.int64)
        # 候选决策行：日历位 ∈ [start_c, cap_c] 且行号 ≥ lookback-1
        rows = np.arange(n)
        mask = (pos >= start_c) & (pos <= cap_c) & (rows >= lookback - 1)
        cand = rows[mask]
        if len(cand) == 0:
            continue
        c = pos[cand]
        # 90 行窗口完整：窗口首行的日历位恰为 c-lookback+1（即行连续对齐日历）
        w_first_row = rowidx[c - lookback + 1]
        complete = w_first_row == (cand - lookback + 1)
        stats.excluded_short_window += int((~complete).sum())
        cand, c = cand[complete], c[complete]
        if len(cand) == 0:
            continue
        base = close[cand]
        good_denom = np.isfinite(base) & (base > 0)
        stats.excluded_bad_denominator += int((~good_denom).sum())
        cand, c, base = cand[good_denom], c[good_denom], base[good_denom]
        if len(cand) == 0:
            continue
        # 四期限标签：终点 = calendar[c+k]，行须恰存在且在决策行之后
        labs = np.full((len(cand), len(horizons)), np.nan)
        all_ok = np.ones(len(cand), dtype=bool)
        for hi, k in enumerate(horizons):
            r = rowidx[c + k]
            ok_pos = np.clip(r, 0, None)
            fin = np.isfinite(close[ok_pos]) & (close[ok_pos] > 0)
            ok = (r > cand) & fin
            labs[:, hi] = np.where(ok, close[ok_pos] / base - 1.0, np.nan)
            all_ok &= ok
        stats.excluded_missing_label_endpoint += int((~all_ok).sum())
        for j in np.flatnonzero(all_ok):
            per_day.setdefault(int(c[j]), []).append((code, int(cand[j]), labs[j]))

    days: list[TrainDay] = []
    for c in sorted(per_day):
        items = per_day[c]
        if len(items) < min_cross:
            stats.skipped_days_lt_min += 1
            continue
        codes = [it[0] for it in items]
        rows_arr = np.array([it[1] for it in items], dtype=np.int64)
        labels = np.stack([it[2] for it in items])
        days.append(TrainDay(calendar[c], codes, rows_arr, labels))
        stats.n_samples += len(items)
    stats.n_days = len(days)
    if days:
        stats.decision_min = str(days[0].date.date())
        stats.decision_max = str(days[-1].date.date())
        endpoints = [calendar[cal_pos[d.date] + max_k] for d in days]
        stats.label_endpoint_min = str(endpoints[0].date())
        stats.label_endpoint_max = str(endpoints[-1].date())
    return days, stats


def build_train_batch(
    tables: dict[str, SymbolTable],
    day: TrainDay,
    idx: np.ndarray,
    calendar: pd.DatetimeIndex,
    *,
    lookback: int = LOOKBACK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按截面成员下标物化一批 (x_norm [B,90,6], stamp [B,90,5], labels [B,4])。

    stamp 为该决策日 90 交易日日历窗的 [90,5]（批内共享，expand 到 B）——
    与 H1 ``build_train_batch`` 同构。
    """
    cal_pos = {d: i for i, d in enumerate(calendar)}
    xs = [_window_zscore_clip(
        tables[day.codes[k]].vals[day.rows[k] - lookback + 1: day.rows[k] + 1])
        for k in idx]
    x = torch.from_numpy(np.stack(xs))
    labels = torch.from_numpy(day.labels[idx].astype(np.float32))
    c = cal_pos[day.date]
    win_cal = calendar[c - lookback + 1: c + 1]
    stamp = torch.from_numpy(_calc_stamps(win_cal))[None, :, :].expand(
        x.shape[0], -1, -1)
    return x, stamp.contiguous(), labels


# ============================================================
# 成对采样器（S/M 同批；退化截面跳过 + 确定性补足）
# ============================================================


@dataclass
class SampledBatch:
    """一次成功采样的元信息（S/M 配对键）。"""

    date: pd.Timestamp
    batch_codes: list[str]
    rows: np.ndarray


class PairedDailySampler:
    """同日截面无放回采样（计划 §4.2：S/M 完全相同样本集合）。

    独立 ``random.Random(seed)``（官方 ``finetune/dataset.py`` 同款，不干扰
    模型初始化随机流）；同一种子的 S/M 两次运行产生逐位相同的批次序列。
    抽样后任一期限标签方差为零 → 该批**两臂共同跳过**，RNG 流继续以相同
    确定性序列补足更新数。
    """

    def __init__(self, tables: dict[str, SymbolTable], train_days: list[TrainDay],
                 calendar: pd.DatetimeIndex, *, seed: int,
                 batch_size: int = BATCH) -> None:
        self.tables = tables
        self.days = train_days
        self.calendar = calendar
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.n_skipped_degenerate = 0

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SampledBatch]:
        """抽一批（不检查退化；退化检查在 :meth:`updates` 内做）。"""
        day = self.rng.choice(self.days)
        n = len(day.codes)
        k = min(self.batch_size, n)
        idx = np.array(self.rng.sample(range(n), k), dtype=np.int64)
        x, stamp, labels = build_train_batch(self.tables, day, idx, self.calendar)
        info = SampledBatch(day.date, [day.codes[j] for j in idx], day.rows[idx])
        return x, stamp, labels, info

    def updates(self, n_updates: int) -> Iterator[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, SampledBatch]]:
        """产出恰 ``n_updates`` 个有效更新（退化批跳过不计数）。"""
        yielded = 0
        while yielded < n_updates:
            x, stamp, labels, info = self.sample()
            if not np.isfinite(labels.numpy()).all():
                self.n_skipped_degenerate += 1
                continue
            if float((labels.numpy().std(axis=0) == 0).any()):
                self.n_skipped_degenerate += 1
                continue
            yielded += 1
            yield x, stamp, labels, info


# ============================================================
# PIT 评估日（验证选点段 / W3 / W4；qlib 同口径）
# ============================================================


@dataclass
class PitDay:
    """PIT 评估决策日：物化张量 + 四期限标签（NaN = 终点缺行）。"""

    date: pd.Timestamp
    codes: list[str]
    x_norm: torch.Tensor     # [N,90,6]
    stamp: torch.Tensor      # [N,90,5]
    labels: np.ndarray       # [N,4] raw fwd（NaN 允许，IC 时按可用性匹配）


@dataclass
class PitStats:
    """PIT 构造统计（排除原因分桶 + 每日样本数）。"""

    n_days: int = 0
    per_day: dict = field(default_factory=dict)   # date → {pool, kept, short, halt, missing_label}
    decision_min: str | None = None
    decision_max: str | None = None


def build_pit_days(
    provider,
    *,
    start: str,
    end: str,
    label_end: str | None,
    pool: str = POOL,
    horizons: Sequence[int] = HORIZONS,
    lookback: int = LOOKBACK,
    require_complete_labels: bool = False,
) -> tuple[list[PitDay], PitStats]:
    """构造 PIT 评估日（x 窗语义与 ``build_daily_samples`` 逐字一致）。

    资格只依据 t 日 PIT 成员（§4.3）——退池股若仍挂牌，其标签照常计算；
    不以"t+20 仍在指数"作资格。``label_end`` 给定时，决策日上界 = 最后一个
    使全部期限终点 ≤ ``label_end`` 的交易日（§4.7 每段终点留在段内）。
    ``require_complete_labels=True``（验证选点段）时缺任一期限终点的股票整只
    剔除并单独计数；False（W3/W4）保留打分（标签 NaN，IC 时匹配）。

    :raises AssertionError: 任何行情查询越过 ``FORWARD_CUTOFF``。
    """
    max_k = max(horizons)
    # 日历须含 90 日回看历史（仅上界封 FORWARD_CUTOFF，回看不设下界——
    # 过去数据非 forward）
    calendar = provider.trading_days(end=FORWARD_CUTOFF)
    n_cal = len(calendar)
    cap_c = (min(int(calendar.searchsorted(pd.Timestamp(label_end),
                                           side="right")) - 1, n_cal - 1) - max_k
             if label_end is not None else n_cal - 1 - max_k)
    first_c = int(calendar.searchsorted(pd.Timestamp(start), side="left"))
    last_c = min(int(calendar.searchsorted(pd.Timestamp(end), side="right")) - 1,
                 cap_c)
    days_idx = list(range(first_c, last_c + 1))
    stats = PitStats()
    out: list[PitDay] = []
    for c in days_idx:
        if c < lookback - 1:      # 回看窗不足 90 交易日 → 该日无法构样
            continue
        t = calendar[c]
        ds = str(t.date())
        members = provider.list_pool_at(pool, ds)
        if not members:
            continue
        if c + max_k >= n_cal:
            break
        x_cal = calendar[c - lookback + 1: c + 1]
        endpoints = [calendar[c + k] for k in horizons]
        fetch_start = str(x_cal[0].date())
        fetch_end = str(max(endpoints).date())
        assert fetch_end <= FORWARD_CUTOFF, (
            f"行情查询越过 forward 封存线：{ds} → {fetch_end}")

        orig = (provider._start_date, provider._end_date, provider.instruments_)
        try:
            provider._start_date = fetch_start
            provider._end_date = fetch_end
            provider.instruments_ = members
            fields = [f"${col}" for col in _REQUIRED_QLIB + _AUX_QLIB]
            raw = provider.fetch(fields, freq="day")
        finally:
            provider._start_date, provider._end_date, provider.instruments_ = orig

        available = (raw.index.get_level_values("instrument").unique()
                     if len(raw) else [])
        xs, stamps, codes, lab_rows = [], [], [], []
        n_short = n_halt = n_missing = 0
        for code in members:
            if code not in available:
                n_short += 1
                continue
            sub = raw.xs(code, level="instrument").sort_index()
            sub_x = sub.loc[:t]
            if len(sub_x) < lookback or t not in sub.index:
                n_short += 1
                continue
            window = sub_x.iloc[-lookback:]
            if "tradestatuscode" in window.columns and (
                    window["tradestatuscode"] == 0).any():
                n_halt += 1
                continue
            close_t = float(sub.loc[t, "close"])
            if not (np.isfinite(close_t) and close_t > 0):
                n_short += 1
                continue
            row = np.full(len(horizons), np.nan)
            for hi, k in enumerate(horizons):
                ep = endpoints[hi]
                if ep in sub.index:
                    nxt = float(sub.loc[ep, "close"])
                    if np.isfinite(nxt) and nxt > 0:
                        row[hi] = nxt / close_t - 1.0
            if require_complete_labels and not np.isfinite(row).all():
                n_missing += 1
                continue
            xs.append(_window_zscore_clip(window[_REQUIRED_QLIB].values.astype(np.float32)))
            stamps.append(_calc_stamps(window.index))
            codes.append(code)
            lab_rows.append(row)
        if not codes:
            stats.per_day[ds] = {"pool": len(members), "kept": 0, "short": n_short,
                                 "halt": n_halt, "missing_label": n_missing}
            continue
        out.append(PitDay(
            date=t, codes=codes,
            x_norm=torch.from_numpy(np.stack(xs)),
            stamp=torch.from_numpy(np.stack(stamps)),
            labels=np.stack(lab_rows)))
        stats.per_day[ds] = {"pool": len(members), "kept": len(codes),
                             "short": n_short, "halt": n_halt,
                             "missing_label": n_missing}
    stats.n_days = len(out)
    if out:
        stats.decision_min = str(out[0].date.date())
        stats.decision_max = str(out[-1].date.date())
        logger.info(f"[PIT {start}~{end}] {len(out)} 日（决策 "
                    f"{stats.decision_min}~{stats.decision_max}），末日样本 "
                    f"{len(out[-1].codes)} 只")
    return out, stats


__all__ = [
    "horizon_labels", "SymbolTable", "load_union_tables", "union_calendar",
    "TrainDay", "ScanStats", "scan_train_days", "build_train_batch",
    "SampledBatch", "PairedDailySampler", "PitDay", "PitStats", "build_pit_days",
]
