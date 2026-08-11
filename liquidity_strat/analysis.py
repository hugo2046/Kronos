"""信号层分析（计划 §2.4.1）：每档 × 每信号调 ``analyzer.analyze_factor``。

强制口径（计划 §1.1）：``quantiles=10``、``periods=(1,5,10)``、``delay=1``（T+1）、
``skip_paused=True``、``skip_one_word_limit=True``（补硬伤③）、``industry='sw_l1'``。

行业降级（阶段 0 结论）：本实例 ``sw_l1`` 字段全常数（唯一值=1，无区分力）；
``analyzer`` 拒绝 ``industry=None``，故传 ``industry='sw_l1'`` —— 全同行业下中性化
等价于减截面整体均值，即不实质性中性化。已在结果文档记入局限。

动态成员处理：档成员每月末 PIT 重定，逐日把**非当月成员**置 NaN。
``analyze_factor`` 的分位/IC 计算是逐日截面，NaN 自动剔除，故宽表可含全档历史
代码、按日 mask 成员。这是与"静态分档"（含前视）的本质区别。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from liquidity_strat.common import (
    BUCKETS,
    DATA_DIR,
    DATA_END,
    HIGH_LIQ_BUCKET,
    NEW_SIGNALS,
    SIGNAL_KRONOS,
    SIGNAL_MOM,
    SIGNAL_PLACEHOLDER,
    SIGNAL_REV,
    ST_TRACKS,
    ST_TRACK_MAIN,
    LiquidityConfig,
    init_analyzer_auth,
)

# analyze_factor 的 period 10 期 → 年化乘子（244 交易日/年 ÷ 10）
TRADING_DAYS_PER_YEAR = 244
_ANN_PERIOD10 = TRADING_DAYS_PER_YEAR / 10

UNION_FILES = {
    SIGNAL_KRONOS: DATA_DIR / "daily_signals_K_union.parquet",
    SIGNAL_MOM: DATA_DIR / "daily_signals_M_union.parquet",
    SIGNAL_REV: DATA_DIR / "daily_signals_R_union.parquet",
    SIGNAL_PLACEHOLDER: DATA_DIR / "daily_signals_P_union.parquet",
}


def _build_dynamic_factor(
    union_wide: pd.DataFrame,
    strat_df: pd.DataFrame,
    bucket: str,
    track: str,
) -> pd.DataFrame:
    """构造档内动态成员因子宽表（非当月成员置 NaN）。

    :returns: 宽表 ``index=date, columns=code``，列 = 该档该轨历史曾出现的全部代码，
        每日只保留当月（最近月末）成员的信号值，其余 NaN。
    """
    reb_ts = pd.DatetimeIndex(sorted(strat_df["date"].unique()))

    def nearest_reb(d: pd.Timestamp) -> pd.Timestamp:
        le = reb_ts[reb_ts <= d]
        return le[-1] if len(le) else reb_ts[0]

    # 该档该轨所有历史代码
    sub = strat_df[(strat_df.bucket == bucket) & (strat_df.st_track == track)]
    all_codes = sorted(sub["code"].unique())
    factor = union_wide.reindex(columns=all_codes)
    # 按日 mask：只保留当日（最近月末分档）成员
    mask = pd.DataFrame(False, index=factor.index, columns=factor.columns)
    # 预算每个月末成员集
    month_members: dict[pd.Timestamp, set[str]] = {
        d: set(sub.loc[sub.date == d, "code"]) for d in reb_ts
    }
    for d in factor.index:
        members = month_members[nearest_reb(d)]
        mask.loc[d, [c for c in all_codes if c in members]] = True
    return factor.where(mask)


def analyze_one(
    factor: pd.DataFrame, *, need_plot: bool = False, tag: str = ""
) -> dict:
    """对一张因子宽表跑 analyze_factor，抽取判读所需指标（不画图，避免副作用）。

    :returns: dict 含 spread/IC/单调性指标（period_1/5/10）。
    """
    import sys

    sys.path.insert(0, "/home/user/workspace/AlphaFarmer")
    import analyzer

    factor = factor.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if factor.shape[1] < 20 or len(factor) < 30:
        logger.warning(f"[{tag}] 样本不足 ({factor.shape})，跳过")
        return {"ok": False, "shape": list(factor.shape)}

    res = analyzer.analyze_factor(
        factor,
        quantiles=10,
        periods=(1, 5, 10),
        delay=1,
        skip_paused=True,
        skip_one_word_limit=True,
        industry="sw_l1",  # 阶段0降级：全常数等价不中性化
        # 动态成员 mask 致 ~75% NaN（PIT 每月末重定档，代码仅在其所属月份非空）——
        # 这是设计意图（point-in-time 成员），非数据质量损失，故放宽 max_loss。
        max_loss=1.0,
    )
    spread = res.calc_spread_significance()
    info = res.calc_information_table()
    mono = res.calc_monotonicity_table()
    mrq = res.mean_return_by_quantile.copy()

    out: dict = {"ok": True, "n_days": int(len(factor)), "n_codes": int(factor.shape[1])}
    for p in ("period_1", "period_5", "period_10"):
        out[p] = {
            "spread_mean": float(spread.loc["mean", p]),
            "spread_t_nw": float(spread.loc["t_nw", p]),
            "spread_p_nw": float(spread.loc["p_nw", p]),
            "ic_mean": float(info.loc["IC Mean", p]),
            "ic_ir_raw": float(info.loc["IR", p]),  # 日频 IC_mean/IC_std
            "ic_t_nw": float(info.loc["t-stat NW(IC)", p]),
            "mono_rho": float(mono.loc["rho", p]),
            "mono_p": float(mono.loc["p_value", p]),
            "monotonic": bool(mono.loc["monotonic", p]),
            "q10_return": float(mrq.loc[10, p]),
            "q1_return": float(mrq.loc[1, p]),
        }
    # period_10 年化（判读主口径）
    s10 = out["period_10"]
    s10["spread_annualized"] = s10["spread_mean"] * _ANN_PERIOD10
    s10["ic_ir_annualized"] = s10["ic_ir_raw"] * (TRADING_DAYS_PER_YEAR ** 0.5)
    # 10 分位收益序列（单调性可视化用）
    out["period_10"]["quantile_returns"] = {
        int(q): float(v) for q, v in mrq["period_10"].items()
    }
    return out


def run_analysis(cfg: LiquidityConfig, *, tracks=(ST_TRACK_MAIN,)) -> dict:
    """跑全部（档 × 轨 × 信号）的信号层分析，落盘 JSON。

    :returns: ``{(bucket, track, signal): metrics}``。
    """
    init_analyzer_auth()
    strat = pd.read_parquet(DATA_DIR / "strat_membership.parquet")
    strat["date"] = pd.to_datetime(strat["date"])

    union = {}
    for sig in NEW_SIGNALS:
        p = UNION_FILES[sig]
        if not p.exists():
            raise FileNotFoundError(f"union 信号缺失：{p}")
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        union[sig] = df

    results: dict = {}
    for bucket, _, _ in cfg.buckets:
        for track in tracks:
            for sig in NEW_SIGNALS:
                tag = f"{bucket}/{track}/{sig}"
                logger.info(f"analyze_factor [{tag}]")
                factor = _build_dynamic_factor(union[sig], strat, bucket, track)
                m = analyze_one(factor, tag=tag)
                results[f"{bucket}|{track}|{sig}"] = m
                if m.get("ok"):
                    s = m["period_10"]
                    logger.info(
                        f"[{tag}] P10 spread={s['spread_mean']:.5f} "
                        f"(ann {s['spread_annualized']:+.4f}) t={s['spread_t_nw']:.2f} "
                        f"IC={s['ic_mean']:.4f} mono_rho={s['mono_rho']:.2f}"
                    )
    out = DATA_DIR / "analysis_signal_layer.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"信号层分析落盘：{out}（{len(results)} 组）")
    return results
