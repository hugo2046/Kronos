"""信号批产（计划 §2.3）。

四组信号逐日逐**档**生成，2024-07-01~2026-07-24 窗口（预训练截止后）：

- **K** = Kronos-base zero-shot canonical mean（N=20, T=1.0, top_p=0.9, seed=42, L=90/H=10）；
- **M** = 10 日动量（``close_t/close_{t-10} - 1``）；
- **R** = 10 日反转（``-momentum``）；
- **P** = 随机占位（``default_rng(seed).standard_normal``，零点门禁用）。

效率设计：四档成员大量重叠（withst ⊃ exst，三档百分位带不交），故**每日先取
全档 union 宇宙跑一次 Kronos 推理**，再按 (bucket, st_track) 成员表切片——
避免跨档重复推理（计划 §2.3 算力 12-15h，本设计把它压到 union 量级）。

断点续跑：union 推理按计划 §2.3 "文件名带档位标签" 命名
``daily_signals_<SIGNAL>_union.parquet``，已跑日期跳过；切片表确定性派生。
推理窗口构造 :func:`build_windows_for_codes` **逐字镜像** ``kronos_qlib/windows.py``
的 7 条语义（仅把 ``pool`` 字符串换成显式 ``members`` 列表），不改 kronos_qlib。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from liquidity_strat.common import (
    DATA_DIR,
    NEW_SIGNALS,
    SIGNAL_KRONOS,
    SIGNAL_MOM,
    SIGNAL_PLACEHOLDER,
    SIGNAL_REV,
    LiquidityConfig,
)
from paper_replication.signal import predict_batch_chunked

# 与 kronos_qlib/windows.py 逐字一致（不改 kronos_qlib，故本地镜像）
_REQUIRED_COLS: list[str] = ["open", "high", "low", "close", "volume", "amount"]
_AUX_COLS: list[str] = ["preclose", "tradestatuscode"]


def daily_union_universe(
    strat_df: pd.DataFrame, provider, start: str, end: str
) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, list[str]]]:
    """把月末分档扩展到每个交易日，并取全档 union 宇宙。

    :returns: ``(trading_days, universe_map)``，每个交易日 → 当月（最近月末）
        全档（3 档 × 2 轨）union 成员列表。任意交易日的成员 = ``<= d`` 的最近月末。
    """
    cal = provider.trading_days(start, end)
    reb_ts = pd.DatetimeIndex(sorted(strat_df["date"].unique()))

    def nearest_reb(d: pd.Timestamp) -> pd.Timestamp:
        le = reb_ts[reb_ts <= d]
        return le[-1] if len(le) else reb_ts[0]

    # 预算每个月末的 union 成员
    month_union: dict[pd.Timestamp, list[str]] = {}
    for d in reb_ts:
        codes = sorted(strat_df.loc[strat_df["date"] == d, "code"].unique().tolist())
        month_union[d] = codes

    universe_map: dict[pd.Timestamp, list[str]] = {}
    for d in cal:
        universe_map[d] = month_union[nearest_reb(d)]

    sizes = [len(v) for v in universe_map.values()]
    logger.info(
        f"union 宇宙：{len(cal)} 交易日，日均 {np.mean(sizes):.0f} 只 "
        f"(min {min(sizes)} / max {max(sizes)})"
    )
    return cal, universe_map


def build_windows_for_codes(
    provider,
    rebalance_date: str,
    members: list[str],
    *,
    lookback: int = 90,
    predict_len: int = 10,
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.Series], list[str], dict]:
    """显式 ``members`` 版的 :func:`kronos_qlib.windows.build_inference_windows`。

    逐字镜像 windows.py 的 7 条语义（PIT 池、≥L 行、停牌跳过、列序、不归一化、
    y_ts 来自 D.calendar、ST 留给上层），唯一差别：``members`` 显式传入（流动性
    档宇宙每月变、且非 qlib 注册的市场字符串），而非 ``provider.list_pool_at``。
    """
    from kronos_qlib.windows import _fetch_via  # 复用同一 fetch 路径

    if len(members) == 0:
        return [], [], [], [], {"n_pool": 0, "n_kept": 0, "skipped_short": 0, "skipped_halt": 0}

    t = pd.Timestamp(rebalance_date)
    full_cal = provider.trading_days()
    cal_le_t = full_cal[full_cal <= t]
    if len(cal_le_t) == 0:
        raise ValueError(f"build_windows_for_codes：日历中无 <= {rebalance_date} 的交易日")
    t_effective = cal_le_t[-1]
    t_pos = len(cal_le_t) - 1

    x_start_pos = max(0, t_pos - lookback + 1)
    x_window_cal = full_cal[x_start_pos : t_pos + 1]
    y_window_cal = full_cal[t_pos + 1 : t_pos + 1 + predict_len]
    if len(y_window_cal) < predict_len:
        raise ValueError(
            f"build_windows_for_codes：{rebalance_date} 之后交易日不足 {predict_len} 个"
        )

    fetch_start = x_window_cal[0].strftime("%Y-%m-%d")
    fetch_end = y_window_cal[-1].strftime("%Y-%m-%d")
    fields = [f"${c}" for c in _REQUIRED_COLS + _AUX_COLS]
    # 无 filter_pipe → 传 list 精准取成分（windows.py 陷阱 5）
    raw = _fetch_via(provider, members, fetch_start, fetch_end, fields, None)

    if "instrument" not in raw.index.names:
        raise ValueError(
            f"fetch 缺 instrument 索引层，实际 {raw.index.names}"
        )
    available = raw.index.get_level_values("instrument").unique()

    df_list: list[pd.DataFrame] = []
    x_ts_list: list[pd.Series] = []
    y_ts_list: list[pd.Series] = []
    codes: list[str] = []
    stats = {"n_pool": len(members), "n_kept": 0, "skipped_short": 0, "skipped_halt": 0}
    y_window_idx = pd.DatetimeIndex(y_window_cal)

    for code in members:
        if code not in available:
            stats["skipped_short"] += 1
            continue
        sub = raw.xs(code, level="instrument")
        sub = sub.loc[:t_effective]
        if len(sub) < lookback:
            stats["skipped_short"] += 1
            continue
        window = sub.iloc[-lookback:]
        if "tradestatuscode" in window.columns and (
            (window["tradestatuscode"] == 0)
        ).any():
            stats["skipped_halt"] += 1
            continue
        window_df = window[_REQUIRED_COLS].copy()
        x_ts = pd.Series(window_df.index)
        y_ts = pd.Series(y_window_idx)
        df_list.append(window_df)
        x_ts_list.append(x_ts)
        y_ts_list.append(y_ts)
        codes.append(code)
        stats["n_kept"] += 1

    return df_list, x_ts_list, y_ts_list, codes, stats


def run_kronos_signals(
    predictor,
    provider,
    cfg: LiquidityConfig,
    trading_days: pd.DatetimeIndex,
    universe_map: dict[pd.Timestamp, list[str]],
    *,
    progress_every: int = 5,
    checkpoint_dir: Path | None = DATA_DIR,
) -> pd.DataFrame:
    """逐日对 union 宇宙跑 Kronos canonical mean，断点续跑。

    :returns: union 宽表 ``index=date, columns=code, values=Kronos mean 信号``。
        checkpoint：``<checkpoint_dir>/daily_signals_K_union.parquet``。
    """
    ckpt = Path(checkpoint_dir) / "daily_signals_K_union.parquet" if checkpoint_dir else None
    rows: list[dict] = []
    done: set[pd.Timestamp] = set()
    if ckpt and ckpt.exists():
        existing = pd.read_parquet(ckpt)
        done = set(pd.to_datetime(existing.index))
        rows = [existing.loc[d].dropna().to_dict() for d in existing.index]
        logger.info(f"K 断点续跑：已有 {len(done)} 日，跳过")

    pending = [d for d in trading_days if d not in done]
    logger.info(
        f"K 信号：{len(trading_days)} 日总计，{len(pending)} 日待跑 "
        f"(N={cfg.sample_count}, union 日均 ~{np.mean([len(v) for v in universe_map.values()]):.0f})"
    )

    for i, d in enumerate(pending):
        ds = d.strftime("%Y-%m-%d")
        members = universe_map[d]
        if len(members) == 0:
            rows.append({})
            continue
        df_list, x_ts_list, y_ts_list, codes, stats = build_windows_for_codes(
            provider, ds, members, lookback=cfg.lookback, predict_len=cfg.predict_len
        )
        if len(df_list) == 0:
            logger.warning(f"{ds}: union 无可用窗口（{stats}）")
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
        day_sig: dict[str, float] = {}
        for j, pred_df in enumerate(preds):
            vals = pred_df[cfg.signal_field].values
            day_sig[codes[j]] = float(np.mean(vals) / last_closes[j] - 1.0)
        rows.append(day_sig)

        if (i + 1) % progress_every == 0 or i == 0:
            logger.info(
                f"K [{i + 1}/{len(pending)}] {ds}: kept={stats['n_kept']}/{stats['n_pool']} "
                f"std={np.std(list(day_sig.values())):.4f}"
            )
            if ckpt:
                _dump_k(rows, trading_days, done, pending[: i + 1], ckpt)

    wide = pd.DataFrame(rows, index=trading_days)
    if ckpt:
        wide.to_parquet(ckpt)
        logger.info(f"K 信号落盘：{ckpt}（{wide.shape[0]} 日 × {wide.shape[1]} 列）")
    return wide


def _dump_k(rows, trading_days, done, pending, ckpt) -> None:
    done_sorted = sorted(done)
    n_done = len(done_sorted)
    done_rows = rows[:n_done] if n_done else []
    new_rows = rows[n_done:]
    all_idx = done_sorted + list(pending[: len(new_rows)])
    pd.DataFrame(done_rows + new_rows, index=all_idx).to_parquet(ckpt)


def run_mr_signals(
    provider,
    trading_days: pd.DatetimeIndex,
    universe_map: dict[pd.Timestamp, list[str]],
    backtest_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日算 union 宇宙的 10 日动量 / 反转（与 paper_replication 同口径）。

    一次性取 union 全部代码的 close（回看足够），再逐日切片。
    """
    all_codes = sorted({c for codes in universe_map.values() for c in codes})
    fetch_start = (trading_days[0] - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    orig_start, orig_end, orig_inst = (
        provider._start_date,
        provider._end_date,
        provider.instruments_,
    )
    try:
        provider._start_date = fetch_start
        provider._end_date = backtest_end
        provider.instruments_ = all_codes
        df = provider.fetch(["$close"], freq="day")
    finally:
        provider._start_date = orig_start
        provider._end_date = orig_end
        provider.instruments_ = orig_inst

    wide = df["close"].unstack("instrument").sort_index() if "instrument" in df.index.names else df["close"]
    mom_rows, rev_rows = [], []
    for d in trading_days:
        if d not in wide.index:
            mom_rows.append({})
            rev_rows.append({})
            continue
        loc = wide.index.get_loc(d)
        if loc < 10:
            mom_rows.append({})
            rev_rows.append({})
            continue
        mom = wide.iloc[loc] / wide.iloc[loc - 10] - 1.0
        mom_rows.append(mom.to_dict())
        rev_rows.append((-mom).to_dict())
    mom_wide = pd.DataFrame(mom_rows, index=trading_days)
    rev_wide = pd.DataFrame(rev_rows, index=trading_days)
    logger.info(f"M/R 信号：{mom_wide.shape[0]} 日 × 平均 {mom_wide.notna().sum(axis=1).mean():.0f} 只/日")
    return mom_wide, rev_wide


def run_placeholder_signals(
    trading_days: pd.DatetimeIndex,
    universe_map: dict[pd.Timestamp, list[str]],
    seed: int = 42,
) -> pd.DataFrame:
    """每日每股标准正态占位（零点门禁用），仅对 union 宇宙非空格生成。"""
    rng = np.random.default_rng(seed)
    rows = []
    for d in trading_days:
        codes = universe_map[d]
        row = {c: float(rng.standard_normal()) for c in codes}
        rows.append(row)
    wide = pd.DataFrame(rows, index=trading_days)
    logger.info(f"P 占位信号：{wide.shape[0]} 日 × 平均 {wide.notna().sum(axis=1).mean():.0f} 只/日")
    return wide


def slice_bucket_signals(
    union_signals: dict[str, pd.DataFrame],
    strat_df: pd.DataFrame,
    provider,
    start: str,
    end: str,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """把 union 信号表按 (bucket, st_track) 成员切片，得到档内宽表。

    :param union_signals: ``{signal_tag: union_wide_df}``（index=date, columns=code）。
    :returns: ``{(bucket, st_track, signal_tag): wide_df}``，每张表只含该档该轨成员列。
    """
    cal = provider.trading_days(start, end)
    reb_ts = pd.DatetimeIndex(sorted(strat_df["date"].unique()))

    def nearest_reb(d: pd.Timestamp) -> pd.Timestamp:
        le = reb_ts[reb_ts <= d]
        return le[-1] if len(le) else reb_ts[0]

    # 每个交易日每 (bucket, track) 的成员
    daily_members: dict[tuple[pd.Timestamp, str, str], set[str]] = {}
    buckets = strat_df["bucket"].unique()
    tracks = strat_df["st_track"].unique()
    for d in cal:
        src = nearest_reb(d)
        sub = strat_df[strat_df["date"] == src]
        for b in buckets:
            for tr in tracks:
                codes = set(sub[(sub.bucket == b) & (sub.st_track == tr)]["code"])
                daily_members[(d, b, tr)] = codes

    sliced: dict[tuple[str, str, str], pd.DataFrame] = {}
    for signal_tag, union_df in union_signals.items():
        for b in buckets:
            for tr in tracks:
                # 对每个交易日只保留该档该轨成员列
                out = pd.DataFrame(index=union_df.index, columns=[], dtype=float)
                parts = []
                for d in union_df.index:
                    members = sorted(daily_members.get((d, b, tr), set()))
                    row = union_df.loc[d].reindex(members)
                    parts.append(row)
                wide = pd.DataFrame(parts, index=union_df.index)
                sliced[(b, tr, signal_tag)] = wide
    return sliced
