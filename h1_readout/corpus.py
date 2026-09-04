"""H1 全语料日截面子集构造（计划 §1，数据窗冻结）。

从 G1 同源 pkl（受限 Unpickler 装载，train+val 拼接出每股完整历史 2014-01-02~
2025-06-30）构造**日对齐截面语料**，语义逐字对齐 R1 的
``cross_section_kda.data.build_daily_samples``（R1 读出惯例）：

- 样本 = (决策日 t, 股票)：x = t 截止的 90 个**交易日**日历窗内恰有 90 行的
  OHLCVA（行不足=停牌/缺数→跳过，不前向填充）→ 窗口 z-score + clip5；
- 标签 = close[t+10 交易日]/close[t] − 1（后复权，交易日历推进；t+10 日缺行→
  跳过），按日截面 z-score（y_z），原始 fwd 留存（早停 RankIC 用）；
- stamp = 5 列时间特征（minute/hour/weekday/day/month，日频下前两列恒 0——
  官方 ``finetune/dataset.py`` 同口径）；
- 交易日历 = 语料全体符号日期并集（自包含，不触 DDB）。

**purge 冻结（计划 §1）**：训练决策日 ∈ [2014-01-02, **2024-12-17**]
（t+10 交易日 ≤ 2024-12-31 = train pkl 末，标签绝不入 2025）；早停段决策日 ∈
**2025-01-01~2025-06-30**（G1 同 val 窗；x 窗可回看 2024Q4——train 期特征，
与 R1 早停段 x 窗回看 train 期同构，无标签泄露）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

# —— 数据窗（计划 §1 冻结）——
TRAIN_START = "2014-01-02"
TRAIN_LABEL_END = "2024-12-17"     # 最后训练决策日（t+10 ≤ 2024-12-31）
ES_START = "2025-01-01"
ES_END = "2025-06-30"

# —— 几何（R1 逐字）——
LOOKBACK = 90
PREDICT_LEN = 10
CLIP = 5.0

# 截面退化保护：有效样本 < 64 只的决策日整日跳过（IC 损失噪声地板）
MIN_CROSS_SECTION = 64

_FEATURES = ["open", "high", "low", "close", "vol", "amt"]


@dataclass
class TrainDay:
    """训练决策日：截面成员的 (symbol, 行号) 索引 + 批内 z-score 标签。

    ``entries`` 与 ``y_z`` / ``fwd_raw`` / ``codes`` 一一对应；x 窗在采样时
    从 ``corpus.symbols[code]`` 的行切片即时构造（在线，不物化全天张量）。
    """

    date: pd.Timestamp
    codes: list[str]
    rows: np.ndarray            # [N] 每样本在其符号数组中的行号（窗口末行）
    y_z: np.ndarray             # [N] 按日截面 z-score
    fwd_raw: np.ndarray         # [N] 原始 close[t+10]/close[t]-1


@dataclass
class EvalDay:
    """早停/评估决策日：物化张量（一次构造，多次消费）。"""

    date: pd.Timestamp
    codes: list[str]
    x_norm: torch.Tensor        # [N, 90, 6]
    stamp: torch.Tensor         # [N, 90, 5]
    y_z: np.ndarray
    fwd_raw: np.ndarray


@dataclass
class Corpus:
    pool: str
    calendar: pd.DatetimeIndex
    cal_pos: dict = field(default_factory=dict)   # date → calendar 下标
    symbols: dict[str, dict] = field(default_factory=dict)  # code -> {vals, dates}
    train_days: list[TrainDay] = field(default_factory=list)
    es_days: list[EvalDay] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _calc_stamps(dates: pd.DatetimeIndex) -> np.ndarray:
    """官方 finetune/dataset.py 同口径：[T,5] = minute/hour/weekday/day/month。"""
    df = pd.DataFrame(index=dates)
    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["day"] = df.index.day
    df["month"] = df.index.month
    return df.values.astype(np.float32)


def _window_zscore_clip(x: np.ndarray) -> np.ndarray:
    """窗口 z-score + clip5（KronosPredictor/build_daily_samples 同口径）。"""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    z = (x - mean) / (std + 1e-5)
    return np.clip(z, -CLIP, CLIP).astype(np.float32)


def _load_symbol_tables(pool: str) -> dict[str, dict]:
    """train+val 拼接 → {code: {vals float32 [n,6], dates DatetimeIndex}}。"""
    from h1_readout.corpus_loader import load_corpus_split

    tr = load_corpus_split(pool, "train")
    va = load_corpus_split(pool, "val")
    out: dict[str, dict] = {}
    for code in sorted(set(tr) | set(va)):
        parts = [d for d in (tr.get(code), va.get(code)) if d is not None and len(d)]
        if not parts:
            continue
        df = pd.concat(parts) if len(parts) > 1 else parts[0]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        out[code] = {"vals": df[_FEATURES].values.astype(np.float32),
                     "dates": df.index}
    return out


def _build_es_days() -> list[EvalDay]:
    """早停段 = csi300 PIT × 2025H1，R1 早停口径逐字（``build_daily_samples``：
    PIT 成员、停牌剔除、日历窗语义——与 R1 的 g5 缓存/打分路径同源）。

    **标签上界（计划 §1 合法性前提）**：早停决策日的 +10 交易日标签必须留在
    2025-06-30 内——否则溢出到 2025H2 评估窗开头，违反"早停段为 2025H1"的
    半污染声明。故决策日上界 = 日历上 2025-06-30 回退 10 个交易日。
    """
    from cross_section_kda.data import build_daily_samples
    from kronos_qlib import QlibProvider

    provider = QlibProvider("csi300", ES_START, ES_END)
    days = provider.trading_days(ES_START, ES_END)
    # 标签上界：最后决策日 = 06-30 前 10 个交易日（t+10 恰为 06-30）
    es_cal = provider.trading_days(ES_START, ES_END)
    max_decision = es_cal[-11] if len(es_cal) >= 11 else es_cal[0]
    logger.info(f"早停决策日上界 {max_decision.date()}（其 +10 标签 = {es_cal[-1].date()}，"
                f"留在 2025H1 内）")
    out: list[EvalDay] = []
    for d in days:
        if d > max_decision:
            continue
        b = build_daily_samples(provider, date=d.strftime("%Y-%m-%d"), pool="csi300")
        if b is None or len(b.codes) < MIN_CROSS_SECTION:
            continue
        out.append(EvalDay(b.date, b.codes, b.x_norm, b.stamp, b.y_z, b.fwd_ret_raw))
    return out


def build_corpus(pool: str) -> Corpus:
    """构造 H1 语料：训练日截面（在线索引，G1 同源 union pkl）+ 早停日（PIT csi300）。

    训练段 = union pkl 语义（计划 §1"G1 同语料"：PIT 并集 pkl 的全部符号-日样本，
    G1 predictor 吃同样的饭）；早停段恒 **csi300 PIT**（R1 早停口径，§1"其余一字
    不改"）——b 臂（ashares 训练）的早停与 a 臂完全一致，仅训练数据不同。
    """
    assert pool in ("csi300", "ashares"), f"未知语料池 {pool}"
    symbols = _load_symbol_tables(pool)

    cal = sorted(set().union(*(s["dates"] for s in symbols.values())))
    calendar = pd.DatetimeIndex(cal)
    cal_pos = {d: i for i, d in enumerate(calendar)}
    n_cal = len(calendar)
    logger.info(f"[{pool}] 符号 {len(symbols)} | 交易日历 {calendar[0].date()}~"
                f"{calendar[-1].date()} 共 {n_cal} 日")

    def _scan(syms: dict[str, dict], day_lo: str, day_end: str,
              label_end: pd.Timestamp):
        """按符号扫描决策日样本 → {date: [(code, row, y_raw)]}。"""
        lo, hi = pd.Timestamp(day_lo), pd.Timestamp(day_end)
        per_day: dict[pd.Timestamp, list[tuple[str, int, float]]] = {}
        for code, tab in syms.items():
            dates, vals = tab["dates"], tab["vals"]
            close = vals[:, 3]
            date_pos = {d: i for i, d in enumerate(dates)}
            # 决策日 = 该符号在 [lo, hi] 内的行日期
            i0 = dates.searchsorted(lo, side="left")
            i1 = dates.searchsorted(hi, side="right")
            for i in range(i0, i1):
                t = dates[i]
                c = cal_pos[t]
                if c + PREDICT_LEN >= n_cal:
                    continue
                t_label = calendar[c + PREDICT_LEN]
                if t_label > label_end:
                    continue
                j = date_pos.get(t_label)
                if j is None or j <= i:
                    continue
                if i < LOOKBACK - 1:
                    continue
                # 90 行须全部落在 90 交易日日历窗内（行不足=停牌/缺数 → 跳过）
                if dates[i - LOOKBACK + 1] < calendar[c - LOOKBACK + 1]:
                    continue
                y = float(close[j] / close[i] - 1.0)
                per_day.setdefault(t, []).append((code, i, y))
        return per_day

    # —— 训练段（在线索引）——
    per_day = _scan(symbols, TRAIN_START, TRAIN_LABEL_END, pd.Timestamp("2024-12-31"))
    train_days: list[TrainDay] = []
    n_train_samples = 0
    for d in sorted(per_day):
        items = per_day[d]
        if len(items) < MIN_CROSS_SECTION:
            continue
        codes = [it[0] for it in items]
        rows = np.array([it[1] for it in items], dtype=np.int64)
        fwd = np.array([it[2] for it in items], dtype=np.float64)
        mu, sd = fwd.mean(), fwd.std()
        y_z = ((fwd - mu) / (sd + 1e-8)).astype(np.float32)
        train_days.append(TrainDay(d, codes, rows, y_z, fwd))
        n_train_samples += len(items)

    # —— 早停段（PIT csi300，R1 口径；物化张量）——
    es_days = _build_es_days()
    n_es_samples = sum(len(d.codes) for d in es_days)

    corpus = Corpus(pool=pool, calendar=calendar, cal_pos=cal_pos, symbols=symbols,
                    train_days=train_days, es_days=es_days)
    corpus.stats = {
        "pool": pool, "n_symbols": len(symbols), "n_cal_days": n_cal,
        "n_train_days": len(train_days), "n_train_samples": n_train_samples,
        "n_es_days": len(es_days), "n_es_samples": n_es_samples,
        "train_day_range": [str(train_days[0].date.date()), str(train_days[-1].date.date())],
        "es_day_range": [str(es_days[0].date.date()), str(es_days[-1].date.date())],
        "es_pool": "csi300(PIT, build_daily_samples 同 R1 口径)",
    }
    logger.info(f"[{pool}] 训练 {len(train_days)} 日 / {n_train_samples:,} 样本 | "
                f"早停 {len(es_days)} 日 / {n_es_samples:,} 样本（csi300 PIT）")
    return corpus


def build_train_batch(corpus: Corpus, day: TrainDay,
                      idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按截面成员下标物化一个批的 (x_norm [B,90,6], stamp [B,90,5], y_z [B])。

    stamp 为决策日 90 交易日日历窗的 [90,5]（批内共享，expand 到 B）。
    """
    xs, ys = [], []
    for k in idx:
        code, row = day.codes[k], int(day.rows[k])
        tab = corpus.symbols[code]
        xs.append(_window_zscore_clip(tab["vals"][row - LOOKBACK + 1: row + 1]))
        ys.append(day.y_z[k])
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.array(ys, dtype=np.float32))
    c = corpus.cal_pos[day.date]
    win_cal = corpus.calendar[c - LOOKBACK + 1: c + 1]
    stamp = torch.from_numpy(_calc_stamps(win_cal))[None, :, :].expand(x.shape[0], -1, -1)
    return x, stamp.contiguous(), y


__all__ = [
    "TRAIN_START", "TRAIN_LABEL_END", "ES_START", "ES_END",
    "LOOKBACK", "PREDICT_LEN", "CLIP", "MIN_CROSS_SECTION",
    "Corpus", "TrainDay", "EvalDay", "build_corpus", "build_train_batch",
]
