"""标准单因子检验（计划 §5）。

对 signal 做标准单因子检验，**同时对基线因子跑完全相同的评估代码**。

指标：
    1. 逐调仓日 RankIC → 均值、ICIR、t 值（111 期，t 值标注样本量）；
    2. 分 5 组等权，组合收益单调性 + 多空（Q5-Q1）每期收益、累计净值、年化与最大回撤；
    3. 多空收益扣除单边 15bp 近似成本后的净值（每 10 日双边换手上限 200%）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from cross_section.common import ExperimentConfig

# 每年交易日数（A股约 244）；用于年化
_TRADING_DAYS_PER_YEAR = 244
# 调仓间隔（H=10 个交易日）
_PERIODS_PER_YEAR = 244 / 10


@dataclass
class ICResult:
    """RankIC 检验结果。"""

    name: str
    n_periods: int
    rankic_mean: float
    rankic_std: float
    icir: float
    t_stat: float
    p_value: float
    rankic_positive_ratio: float
    rankic_series: pd.Series


@dataclass
class GroupResult:
    """分组 / 多空检验结果。"""

    name: str
    group_mean_returns: pd.Series  # 各组平均期收益（单调性）
    long_short_series: pd.Series  # 多空每期收益
    long_short_nav: pd.Series  # 多空累计净值（毛）
    long_short_nav_net: pd.Series  # 多空累计净值（扣 15bp 单边成本）
    annualized: float  # 年化（净）
    max_drawdown: float  # 最大回撤（净）


def compute_rankic(
    signals: pd.DataFrame, factor_col: str, fwd_col: str = "fwd_ret_10d"
) -> ICResult:
    """逐调仓日 RankIC（Spearman）+ ICIR / t 值。

    :param signals: 长表，含 ``date / code / factor_col / fwd_col``。
    :param factor_col: 因子列名。
    :param fwd_col: 前向收益列名。
    :returns: :class:`ICResult`。
    """
    series = []
    for date, sub in signals.groupby("date"):
        sub = sub.dropna(subset=[factor_col, fwd_col])
        if len(sub) < 5:
            series.append(np.nan)
            continue
        rho, _ = stats.spearmanr(sub[factor_col], sub[fwd_col])
        series.append(rho)
    rankic = pd.Series(series, index=sorted(signals["date"].unique()), name=factor_col)
    valid = rankic.dropna()
    n = len(valid)
    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    icir = mean / std if std > 0 else np.nan
    t_stat, p_value = stats.ttest_1samp(valid, 0.0)
    pos_ratio = float((valid > 0).mean())
    return ICResult(
        name=factor_col,
        n_periods=n,
        rankic_mean=mean,
        rankic_std=std,
        icir=icir,
        t_stat=float(t_stat),
        p_value=float(p_value),
        rankic_positive_ratio=pos_ratio,
        rankic_series=rankic,
    )


def compute_groups(
    signals: pd.DataFrame,
    factor_col: str,
    *,
    n_groups: int = 5,
    cost_bps: float = 15.0,
    fwd_col: str = "fwd_ret_10d",
) -> GroupResult:
    """分 N 组等权 → 多空（最高组 - 最低组）净值（含单边成本）。

    :param signals: 长表。
    :param factor_col: 因子列名。
    :param n_groups: 分组数。
    :param cost_bps: 单边成本（bp），多空双边换手上限 200% → 每期扣 2*cost。
    :param fwd_col: 前向收益列名。
    :returns: :class:`GroupResult`。
    """
    group_rows = []
    ls_rows = []
    for date, sub in signals.groupby("date"):
        sub = sub.dropna(subset=[factor_col, fwd_col]).copy()
        if len(sub) < n_groups:
            continue
        # 按 factor 分位分 n_groups 组（rank qcut，避免极值聚集）
        sub["grp"] = pd.qcut(sub[factor_col].rank(method="first"), n_groups, labels=False)
        grp_ret = sub.groupby("grp")[fwd_col].mean()
        # 组号 0=最低, n-1=最高
        group_rows.append(grp_ret.rename(date))
        # 多空 = 最高组 - 最低组
        ls_rows.append({"date": date, "ls": grp_ret.iloc[-1] - grp_ret.iloc[0]})

    group_df = pd.DataFrame(group_rows).sort_index()
    group_mean = group_df.mean()  # 各组全期平均收益（单调性）

    ls = pd.DataFrame(ls_rows).set_index("date")["ls"].sort_index()
    # 毛净值
    nav = (1 + ls).cumprod()
    # 扣成本：每期双边换手上限 200%，单边 cost_bps → 双边 2*cost
    cost_per_period = 2 * cost_bps / 1e4
    ls_net = ls - cost_per_period
    nav_net = (1 + ls_net).cumprod()

    # 年化（净）+ 最大回撤（净）
    n_years = len(ls_net) / _PERIODS_PER_YEAR
    ann = float(nav_net.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan
    dd = (nav_net / nav_net.cummax() - 1).min()

    return GroupResult(
        name=factor_col,
        group_mean_returns=group_mean,
        long_short_series=ls,
        long_short_nav=nav,
        long_short_nav_net=nav_net,
        annualized=ann,
        max_drawdown=float(dd),
    )


def evaluate_factor(
    signals: pd.DataFrame, factor_col: str, cfg: ExperimentConfig
) -> tuple[ICResult, GroupResult]:
    """对一个因子跑完整 IC + 分组评估。"""
    ic = compute_rankic(signals, factor_col)
    grp = compute_groups(
        signals, factor_col, n_groups=cfg.n_groups, cost_bps=cfg.cost_bps
    )
    logger.info(
        f"[{factor_col}] RankIC 均值={ic.rankic_mean:+.4f} ICIR={ic.icir:+.3f} "
        f"t={ic.t_stat:+.2f} (n={ic.n_periods}) 正向期占比={ic.rankic_positive_ratio:.2%} | "
        f"多空年化(净)={grp.annualized:+.2%} 最大回撤={grp.max_drawdown:.2%}"
    )
    return ic, grp
