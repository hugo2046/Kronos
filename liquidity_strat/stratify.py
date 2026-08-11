"""PIT 滚动分档（计划 §2.1：补硬伤①的前半——分档必须 point-in-time）。

每月末按**过去 60 交易日日均成交额**的截面百分位重新分档。这是与帖子（疑似静态
分档、含前视）的关键区别：本模块每一个分档决策只用决策日 t 时点可知的信息。

输出：``stratify(provider, cfg)`` 返回长表 ``DataFrame(date, bucket, st_track, code)``，
以及按月扩展到每个交易日的 ``expand_to_daily()``（每个交易日沿用最近月末分档）。

ST 双轨（计划 §2.1）：主口径 ``exst`` 在分档**前** PIT 剔除 ST（影响百分位排名与
成员）；附加口径 ``withst`` 不剔除。两组成员分开产出，不得混合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from liquidity_strat.common import (
    BUCKETS,
    DATA_END,
    ST_TRACKS,
    ST_TRACK_MAIN,
    LiquidityConfig,
)


def month_end_rebalance_dates(
    provider, start: str, end: str
) -> pd.DatetimeIndex:
    """窗口内每月最后一个交易日（分档 / 调仓时点）。

    :returns: 升序 ``DatetimeIndex``，含 start..end 间每个月的月末交易日。
    """
    cal = provider.trading_days(start, end)
    s = pd.Series(cal, index=cal)
    # 按年月分组取最后一行 → 月末交易日
    month_ends = s.groupby([s.index.year, s.index.month]).last()
    return pd.DatetimeIndex(sorted(month_ends.values))


def _st_codes_at(provider, t: str | pd.Timestamp) -> set[str]:
    """t 时点的 ST 股票代码集合（PIT，计划 §2.1）。

    qlib 的 ``D.instruments("st")`` 返回 ST 标记的时间段；
    ``list_instruments(..., start_time=t, end_time=t, as_list=True)`` 取 t 当日为 ST 的代码。
    本实例 ST 表可得性在阶段 0 探测；不可用时抛错，由上层降级（不静默）。
    """
    from qlib.data import D

    t = pd.Timestamp(t).strftime("%Y-%m-%d")
    inst = D.instruments("st")
    members = D.list_instruments(inst, start_time=t, end_time=t, as_list=True)
    return set(members or [])


def _avg_amount_pct(
    provider, t: pd.Timestamp, lookback: int
) -> pd.Series:
    """t 回看 lookback 个交易日的日均成交额（按股票），返回代码→均额 Series。

    只取 t 时点 ashares 成员；新上市不足 lookback 日的按实际可用日取均值（nanmean）。
    """
    cal = provider.trading_days(end=t.strftime("%Y-%m-%d"))
    if len(cal) < lookback:
        raise ValueError(
            f"日均成交额回看不足：t={t.date()} 日历仅 {len(cal)} 交易日 < lookback={lookback}"
        )
    win_cal = cal[-lookback:]
    start = win_cal[0].strftime("%Y-%m-%d")
    members = provider.list_pool_at("ashares", t.strftime("%Y-%m-%d"))
    # 临时把 provider 的区间 / instruments 指到本次窗口（沿用 kronos_qlib 约定）
    orig_start, orig_end, orig_inst = (
        provider._start_date,
        provider._end_date,
        provider.instruments_,
    )
    try:
        provider._start_date = start
        provider._end_date = t.strftime("%Y-%m-%d")
        provider.instruments_ = members
        raw = provider.fetch(["$amount"], freq="day")
    finally:
        provider._start_date = orig_start
        provider._end_date = orig_end
        provider.instruments_ = orig_inst

    if raw.empty:
        return pd.Series(dtype=float)
    # raw: MultiIndex(datetime, instrument)；按 instrument 取 mean
    amt = raw["amount"].groupby(level="instrument").mean()
    return amt


def stratify(provider, cfg: LiquidityConfig) -> pd.DataFrame:
    """逐月末 PIT 分档，返回长表。

    :returns: ``DataFrame`` 列 ``[date, bucket, st_track, code]``，每个 (月末,
        bucket, st_track) 组合下若干 code 行。``bucket`` 取 :data:`BUCKETS` 的标签；
        csi300 对照档不在本表（复用 baseline_suite）。
    """
    rebalances = month_end_rebalance_dates(provider, cfg.window_start, cfg.window_end)
    logger.info(f"分档月末时点 {len(rebalances)} 个：{rebalances[0].date()}..{rebalances[-1].date()}")

    records: list[dict] = []
    for t in rebalances:
        amt = _avg_amount_pct(provider, t, cfg.stratify_lookback)
        if amt.empty:
            logger.warning(f"{t.date()}: 日均成交额为空，跳过")
            continue
        try:
            st_set = _st_codes_at(provider, t)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"ST 表读取失败（{t.date()}）——阶段 0 须确认 ST 数据可得性；"
                f"不可用则按计划降级。原因：{type(exc).__name__}"
            ) from exc

        for track in ST_TRACKS:
            if track == ST_TRACK_MAIN:
                pool_amt = amt[~amt.index.isin(st_set)]
            else:  # withst：不剔除
                pool_amt = amt
            # 截面百分位排名（rank/pct，0~1）
            pct = pool_amt.rank(pct=True)
            for tag, lo, hi in cfg.buckets:
                mask = (pct >= lo) & (pct <= hi)
                codes = pct[mask].index.tolist()
                for code in codes:
                    records.append(
                        {
                            "date": t,
                            "bucket": tag,
                            "st_track": track,
                            "code": code,
                        }
                    )
        if t == rebalances[0] or t == rebalances[-1] or t == rebalances[len(rebalances) // 2]:
            # 抽样日志：每档每轨样本量（不泄露窗口内数字，仅结构性信息）
            sub = pd.DataFrame(records)
            sub_t = sub[sub["date"] == t]
            summary = (
                sub_t.groupby(["st_track", "bucket"])["code"].count().to_dict()
            )
            st_ratio = (len(st_set) / max(len(amt), 1))
            logger.info(
                f"{t.date()}: 母池={len(amt)} ST占比={st_ratio:.1%} "
                f"分档样本量(轨,档)=N:{summary}"
            )

    df = pd.DataFrame(records)
    logger.info(
        f"分档完成：{len(df)} 条 (date,bucket,st_track,code)，"
        f"月末数={df['date'].nunique()}"
    )
    return df


def expand_to_daily(
    strat_df: pd.DataFrame, provider, start: str, end: str
) -> pd.DataFrame:
    """把月末分档扩展到每个交易日（沿用最近月末成员）。

    :returns: 长表，列同 ``stratify`` 但 ``date`` 扩展为窗口内每个交易日。
        任意交易日 d 的成员 = ``<= d`` 的最近月末分档结果（forward-fill 语义）。
    """
    cal = provider.trading_days(start, end)
    rebalances = sorted(strat_df["date"].unique())
    # 每个交易日 → 最近月末（<= d）
    reb_ts = pd.DatetimeIndex(rebalances)
    def nearest_reb(d: pd.Timestamp) -> pd.Timestamp:
        le = reb_ts[reb_ts <= d]
        return le[-1] if len(le) else reb_ts[0]

    parts = []
    for d in cal:
        src_date = nearest_reb(d)
        chunk = strat_df[strat_df["date"] == src_date].copy()
        chunk["date"] = d
        parts.append(chunk)
    daily = pd.concat(parts, ignore_index=True)
    logger.info(
        f"分档扩展到日线：{daily['date'].nunique()} 交易日 × "
        f"{daily.groupby(['st_track','bucket'])['code'].nunique().count()} (轨,档) 组合"
    )
    return daily


def bucket_members_at(
    strat_df: pd.DataFrame, date: pd.Timestamp, bucket: str, st_track: str
) -> list[str]:
    """取某交易日某档某轨的成员列表（便捷查询）。"""
    sub = strat_df[
        (strat_df["date"] == date)
        & (strat_df["bucket"] == bucket)
        & (strat_df["st_track"] == st_track)
    ]
    return sub["code"].tolist()
