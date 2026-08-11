"""组合层回测（计划 §2.4.2）：每档 × 每信号调 ``qlib_bt.QlibBacktestEngine``。

强制口径（计划 §1.1）：``limit_threshold`` 默认（涨跌停拦截）、``only_tradable=True``
（停牌跳过）、成本用 qlib_bt 默认（``open_cost=0.0005`` 万五 / ``close_cost=0.0015``
万十五含印花税；计划 §2.4 "qlib 默认"，注：计划写 open 0.1% 实为 0.05%）；
策略 ``PeriodicTopkRebalanceStrategy``、``rebalance_freq="m"``、``rebalance_day=-1``
（月末调仓 = 分档时点，PIT 自洽）。

基准双轨（计划 §2.4.3，补硬伤①核心）：
- ``qlib_bt`` 内置 benchmark 填 ``000852.SH``（中证1000，仅展示）；
- **主判读基准 = 自算本档 PIT 等权日收益**：``engine.get_returns()``（扣费）减去
  本档等权 → 剥离流动性 / 规模 beta 后的**选股超额**。门禁用此口径（判读 1）。

csi300 对照档：复用 ``baseline_suite`` paper+oos mean 信号（计划 §2.1，不重算推理），
仅跑 qlib_bt。topk = 各档平均成员数 × 10%（lo5/lo5_10 ≈26、mid45_55 ≈53、csi300 ≈34）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import empyrical as ep

from liquidity_strat.common import (
    DATA_DIR,
    DATA_END,
    HIGH_LIQ_BUCKET,
    NEW_SIGNALS,
    SIGNAL_KRONOS,
    SIGNAL_MOM,
    SIGNAL_PLACEHOLDER,
    SIGNAL_REV,
    ST_TRACK_MAIN,
    ST_TRACKS,
    LiquidityConfig,
    init_analyzer_auth,
)

UNION_FILES = {
    SIGNAL_KRONOS: DATA_DIR / "daily_signals_K_union.parquet",
    SIGNAL_MOM: DATA_DIR / "daily_signals_M_union.parquet",
    SIGNAL_REV: DATA_DIR / "daily_signals_R_union.parquet",
    SIGNAL_PLACEHOLDER: DATA_DIR / "daily_signals_P_union.parquet",
}


def _dynamic_factor_wide(
    union_wide: pd.DataFrame, strat_df: pd.DataFrame, bucket: str, track: str
) -> pd.DataFrame:
    """档内动态成员因子宽表（非当月成员置 NaN），复用 analysis 同口径。"""
    from liquidity_strat.analysis import _build_dynamic_factor

    return _build_dynamic_factor(union_wide, strat_df, bucket, track)


def _to_long_signal(wide: pd.DataFrame) -> pd.Series:
    """宽表 → qlib_bt 契约的 MultiIndex(datetime, instrument) long Series。

    level 名必须精确为 ``datetime`` / ``instrument``（qlib_bt 校验）。
    """
    long = wide.stack(dropna=True)
    long.index = long.index.set_names(["datetime", "instrument"])
    return long


def bucket_equal_weight_returns(
    strat_df: pd.DataFrame, bucket: str, track: str, provider, start: str, end: str
) -> pd.Series:
    """本档 PIT 等权日收益（主判读基准）。

    逐日取当月分档成员的 close 收益（``pct_change``），等权平均（非空成员）。
    与 qlib_bt 同源（DolphinDB / qlib close），halt 日 close 为 NaN 自动剔除。
    """
    reb_ts = pd.DatetimeIndex(sorted(strat_df["date"].unique()))

    def nearest_reb(d: pd.Timestamp) -> pd.Timestamp:
        le = reb_ts[reb_ts <= d]
        return le[-1] if len(le) else reb_ts[0]

    sub = strat_df[(strat_df.bucket == bucket) & (strat_df.st_track == track)]
    all_codes = sorted(sub["code"].unique())
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    orig_s, orig_e, orig_i = provider._start_date, provider._end_date, provider.instruments_
    try:
        provider._start_date = fetch_start
        provider._end_date = end
        provider.instruments_ = all_codes
        df = provider.fetch(["$close"], freq="day")
    finally:
        provider._start_date = orig_s
        provider._end_date = orig_e
        provider.instruments_ = orig_i
    px = df["close"].unstack("instrument").sort_index()

    cal = provider.trading_days(start, end)
    month_members = {d: set(sub.loc[sub.date == d, "code"]) for d in reb_ts}
    daily_ret = px.pct_change()
    eq = []
    for d in cal:
        members = month_members[nearest_reb(d)]
        if d not in daily_ret.index or not members:
            eq.append(np.nan)
            continue
        r = daily_ret.loc[d].reindex(list(members)).dropna()
        eq.append(r.mean() if len(r) else np.nan)
    return pd.Series(eq, index=cal, name="bucket_eq")


def _run_qlib_bt(
    signal_long: pd.Series,
    start: str,
    end: str,
    topk: int,
    codes,
) -> dict:
    """跑一次 qlib_bt，返回指标 + 日收益序列。"""
    import sys

    sys.path.insert(0, "/home/user/workspace/AlphaFarmer")
    from qlib_bt import (
        BacktestConfig,
        QlibBacktestEngine,
        StrategyConfig,
    )
    from qlib_bt.strategies import PeriodicTopkRebalanceStrategy

    engine = QlibBacktestEngine(
        signal_data=signal_long,
        backtest_config=BacktestConfig(
            start_time=start,
            end_time=end,
            benchmark="000852.SH",  # 仅展示
            codes=codes,
        ),
        strategy_config=StrategyConfig(
            strategy_cls=PeriodicTopkRebalanceStrategy,
            topk=topk,
            rebalance_freq="m",
            rebalance_day=-1,
            only_tradable=True,
        ),
    )
    engine.run_backtest()
    metrics = engine.calculate_risk_metrics().to_dict()["指标"]
    returns = engine.get_returns()
    return {"metrics": {k: float(v) for k, v in metrics.items()}, "returns": returns}


def _excess_metrics(strategy_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """strategy − 本档等权 的超额指标（empyrical，与 qlib_bt 同口径）。"""
    common = strategy_ret.index.intersection(bench_ret.index)
    s = strategy_ret.reindex(common).fillna(0.0)
    b = bench_ret.reindex(common).fillna(0.0)
    excess = s - b
    if len(excess) < 10:
        return {"ok": False, "n": int(len(excess))}
    return {
        "ok": True,
        "n": int(len(excess)),
        "annual_return": float(ep.annual_return(excess)),
        "sharpe": float(ep.sharpe_ratio(excess)),
        "max_drawdown": float(ep.max_drawdown(excess)),
        "annual_volatility": float(ep.annual_volatility(excess)),
    }


def _topk_for_bucket(strat_df: pd.DataFrame, bucket: str, track: str) -> int:
    """档内约 10% 持仓数（月末成员平均 × 10%，至少 5）。"""
    sub = strat_df[(strat_df.bucket == bucket) & (strat_df.st_track == track)]
    mean_n = sub.groupby("date")["code"].count().mean()
    return max(5, int(round(mean_n * 0.10)))


def _load_csi300_signal(start: str, end: str) -> pd.DataFrame:
    """复用 baseline_suite paper+oos mean，拼成流动性窗口宽表。"""
    paper = pd.read_parquet(DATA_DIR.parent / "baseline_suite" / "data" / "daily_signals_paper_mean.parquet")
    oos = pd.read_parquet(DATA_DIR.parent / "baseline_suite" / "data" / "daily_signals_oos_mean.parquet")
    paper.index = pd.to_datetime(paper.index)
    oos.index = pd.to_datetime(oos.index)
    wide = pd.concat([paper, oos])
    wide = wide[~wide.index.duplicated(keep="last")]
    cal_mask = (wide.index >= start) & (wide.index <= end)
    return wide.loc[cal_mask]


def run_backtest_all(cfg: LiquidityConfig, *, tracks=(ST_TRACK_MAIN,)) -> dict:
    """跑全部（档 × 轨 × 信号）+ csi300 对照的组合层回测，落盘 JSON。"""
    init_analyzer_auth()  # qlib 复用同一 init
    from kronos_qlib import QlibProvider

    provider = QlibProvider(cfg.pool, cfg.window_start, DATA_END)
    strat = pd.read_parquet(DATA_DIR / "strat_membership.parquet")
    strat["date"] = pd.to_datetime(strat["date"])

    union = {}
    for sig in NEW_SIGNALS:
        df = pd.read_parquet(UNION_FILES[sig])
        df.index = pd.to_datetime(df.index)
        union[sig] = df

    results: dict = {}
    for bucket, _, _ in cfg.buckets:
        for track in tracks:
            topk = _topk_for_bucket(strat, bucket, track)
            codes = sorted(strat[(strat.bucket == bucket) & (strat.st_track == track)]["code"].unique())
            bench = bucket_equal_weight_returns(strat, bucket, track, provider, cfg.window_start, cfg.window_end)
            for sig in NEW_SIGNALS:
                tag = f"{bucket}/{track}/{sig}"
                logger.info(f"qlib_bt [{tag}] topk={topk}")
                wide = _dynamic_factor_wide(union[sig], strat, bucket, track)
                long_sig = _to_long_signal(wide)
                if long_sig.empty:
                    logger.warning(f"[{tag}] 信号为空，跳过")
                    results[f"{bucket}|{track}|{sig}"] = {"ok": False, "reason": "empty_signal"}
                    continue
                try:
                    bt = _run_qlib_bt(long_sig, cfg.window_start, cfg.window_end, topk, codes)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[{tag}] qlib_bt 失败：{type(exc).__name__}: {exc}")
                    results[f"{bucket}|{track}|{sig}"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
                    continue
                exc_m = _excess_metrics(bt["returns"], bench)
                rec = {
                    "ok": True,
                    "topk": topk,
                    "strategy": bt["metrics"],
                    "excess_vs_bucket_eq": exc_m,
                }
                results[f"{bucket}|{track}|{sig}"] = rec
                if exc_m.get("ok"):
                    logger.info(
                        f"[{tag}] AER={bt['metrics'].get('年化收益率', 0):+.4f} "
                        f"超额AER={exc_m['annual_return']:+.4f} "
                        f"超额Sharpe={exc_m['sharpe']:.2f} "
                        f"超额MaxDD={exc_m['max_drawdown']:.4f}"
                    )

    # csi300 对照档
    logger.info("qlib_bt [csi300 control] topk=34")
    csi_wide = _load_csi300_signal(cfg.window_start, cfg.window_end)
    csi_long = _to_long_signal(csi_wide)
    csi_codes = sorted(csi_wide.columns.tolist())
    # csi300 等权基准
    csi_strat = pd.DataFrame({"date": [], "code": []})
    # 复用 bucket_equal_weight_returns 需 strat 表形态：造一个 csi300 的伪 strat
    csi_strat_df = _csi300_pseudo_strat(csi_wide)
    csi_bench = bucket_equal_weight_returns(csi_strat_df, HIGH_LIQ_BUCKET, ST_TRACK_MAIN, provider, cfg.window_start, cfg.window_end)
    try:
        bt = _run_qlib_bt(csi_long, cfg.window_start, cfg.window_end, 34, csi_codes)
        exc_m = _excess_metrics(bt["returns"], csi_bench)
        results[f"{HIGH_LIQ_BUCKET}|{ST_TRACK_MAIN}|{SIGNAL_KRONOS}"] = {
            "ok": True,
            "topk": 34,
            "note": "复用 baseline_suite mean 信号",
            "strategy": bt["metrics"],
            "excess_vs_bucket_eq": exc_m,
        }
        logger.info(f"[csi300] AER={bt['metrics'].get('年化收益率',0):+.4f} 超额AER={exc_m.get('annual_return',float('nan')):+.4f}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[csi300] qlib_bt 失败：{type(exc).__name__}: {exc}")
        results[f"{HIGH_LIQ_BUCKET}|{ST_TRACK_MAIN}|{SIGNAL_KRONOS}"] = {"ok": False, "reason": str(exc)}

    out = DATA_DIR / "analysis_portfolio_layer.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"组合层回测落盘：{out}（{len(results)} 组）")
    return results


def _csi300_pseudo_strat(csi_wide: pd.DataFrame) -> pd.DataFrame:
    """把 csi300 宽表转成 strat 表形态（date/bucket/st_track/code）供等权基准复用。"""
    records = []
    for d in csi_wide.index:
        codes = csi_wide.loc[d].dropna().index.tolist()
        for c in codes:
            records.append({"date": d, "bucket": HIGH_LIQ_BUCKET, "st_track": ST_TRACK_MAIN, "code": c})
    return pd.DataFrame(records)
