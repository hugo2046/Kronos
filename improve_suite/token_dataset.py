"""冻结 tokenizer 词表的监督数据集（计划 §6 阶段 4）。

冻结 ``KronosTokenizer`` 把每 (date, code) 的 L=60 日 K 线窗口（预处理与
``KronosPredictor.predict`` 逐字一致：OHLCVA、窗口 z-score、clip=5）编码为
``(s1_token, s2_token)`` 序列；标签 ``y = mean(close[t+1..t+5])/close_t − 1``（后复权）。

切分（跑前冻结）：train 2019-01-01~2024-06-30，valid = paper 窗，judge = oos1 窗。
样本 = csi300 point-in-time 截面。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

OHLCVA = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK = 60
HORIZON = 5
CLIP = 5


def preprocess_window(window_df: pd.DataFrame, clip: float = CLIP):
    """窗口 z-score + clip（与 ``KronosPredictor.predict`` 逐字一致）。

    **只用窗口内数据**计算 mean/std（无泄漏）。

    :returns: ``(x_norm, mean, std)``——``x_norm`` ``(L, feat)`` 已归一化裁剪。
    """
    x = window_df[OHLCVA].values.astype(np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x_norm = (x - mean) / (std + 1e-5)
    x_norm = np.clip(x_norm, -clip, clip)
    return x_norm, mean, std


def build_sample_dates(t, lookback: int, horizon: int, calendar: pd.DatetimeIndex):
    """决策日 t 的特征窗口日期 + 标签日期（无重叠）。

    :returns: ``(window_dates, label_dates)``——窗口 = [t-lookback+1 .. t]，
        标签 = [t+1 .. t+horizon]（交易日）。
    """
    t = pd.Timestamp(t)
    t_pos = calendar.get_loc(t)
    win_start = max(0, t_pos - lookback + 1)
    window_dates = calendar[win_start : t_pos + 1]
    label_dates = calendar[t_pos + 1 : t_pos + 1 + horizon]
    return window_dates, label_dates


def extract_label(close: pd.Series, t, horizon: int = HORIZON) -> float:
    """y = mean(close[t+1..t+horizon]) / close_t - 1。"""
    t = pd.Timestamp(t)
    t_pos = close.index.get_loc(t)
    future = close.iloc[t_pos + 1 : t_pos + 1 + horizon]
    if len(future) < horizon:
        return float("nan")
    return float(np.mean(future.values) / close.loc[t] - 1.0)


def encode_window(tokenizer, window_df: pd.DataFrame, device: str = "cuda:0"):
    """冻结 tokenizer 把 L 日窗口编码为 (s1, s2) token 序列。

    :returns: ``(s1_tokens, s2_tokens)``——各 ``(L,)`` long tensor（已在 device 上）。
    """
    x_norm, _, _ = preprocess_window(window_df)
    x_t = torch.from_numpy(x_norm)[None].to(device)  # (1, L, feat)
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x_t, half=True)
    return s1[0], s2[0]  # (L,), (L,)


def build_samples(
    provider,
    tokenizer,
    *,
    dates: pd.DatetimeIndex,
    pool: str = "csi300",
    device: str = "cuda:0",
    progress_every: int = 20,
) -> dict:
    """逐决策日构造 (s1, s2, y) 样本（csi300 point-in-time 截面）。

    :returns: ``{"s1": Tensor(N,L), "s2": Tensor(N,L), "y": Tensor(N,), "dates": [...], "codes": [...]}``。
    """
    calendar = provider.trading_days()
    s1_list, s2_list, y_list, meta = [], [], [], []
    n_skipped = 0
    ENC_CHUNK = 128  # 批量编码块大小（显存安全）
    for i, d in enumerate(dates):
        ds = d.strftime("%Y-%m-%d")
        members = provider.list_pool_at(pool, ds)
        if len(members) == 0:
            continue
        win_dates, label_dates = build_sample_dates(d, LOOKBACK, HORIZON, calendar)
        if len(win_dates) < LOOKBACK or len(label_dates) < HORIZON:
            n_skipped += 1
            continue
        fetch_start = win_dates[0].strftime("%Y-%m-%d")
        fetch_end = label_dates[-1].strftime("%Y-%m-%d")
        orig_s, orig_e, orig_inst = provider._start_date, provider._end_date, provider.instruments_
        try:
            provider._start_date = fetch_start
            provider._end_date = fetch_end
            provider.instruments_ = members
            df = provider.fetch([f"${c}" for c in OHLCVA], freq="day")
        finally:
            provider._start_date, provider._end_date, provider.instruments_ = orig_s, orig_e, orig_inst

        if "instrument" not in df.index.names:
            continue
        # 先收集当日所有合法样本的 (code, x_norm, y)，再批量编码
        day_codes, day_x, day_y = [], [], []
        for code in members:
            try:
                sub = df.xs(code, level="instrument")
            except KeyError:
                continue
            sub = sub.sort_index()
            if len(sub) < LOOKBACK + HORIZON:
                continue
            win_df = sub.loc[win_dates]
            if win_df[OHLCVA].isnull().any().any() or len(win_df) < LOOKBACK:
                continue
            close_t = float(sub.loc[d, "close"])
            if close_t <= 0 or pd.isna(close_t):
                continue
            future_close = sub.loc[label_dates, "close"]
            if future_close.isnull().any() or len(future_close) < HORIZON:
                continue
            y = float(np.mean(future_close.values) / close_t - 1.0)
            x_norm, _, _ = preprocess_window(win_df)
            day_codes.append(code)
            day_x.append(x_norm)
            day_y.append(y)
        # 批量编码（块内一次 tokenizer.encode，省 GPU 调用数）
        for s in range(0, len(day_x), ENC_CHUNK):
            e = min(s + ENC_CHUNK, len(day_x))
            batch = np.stack(day_x[s:e]).astype(np.float32)
            batch_t = torch.from_numpy(batch).to(device)
            with torch.no_grad():
                s1_b, s2_b = tokenizer.encode(batch_t, half=True)  # 各 (chunk, L)
            for j in range(e - s):
                s1_list.append(s1_b[j].cpu())
                s2_list.append(s2_b[j].cpu())
                y_list.append(day_y[s + j])
                meta.append((ds, day_codes[s + j]))
        if (i + 1) % progress_every == 0 or i == 0:
            logger.info(f"token 样本 [{i + 1}/{len(dates)}] {ds}：累计 {len(y_list)} 样本")
    logger.info(f"token 样本构建完成：{len(y_list)} 样本，跳过 {n_skipped} 日（窗口不足）")
    return {
        "s1": torch.stack(s1_list) if s1_list else torch.empty(0, LOOKBACK, dtype=torch.long),
        "s2": torch.stack(s2_list) if s2_list else torch.empty(0, LOOKBACK, dtype=torch.long),
        "y": torch.tensor(y_list, dtype=torch.float32),
        "meta": meta,
    }
