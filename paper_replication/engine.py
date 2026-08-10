"""top-k / drop-n long-only 组合引擎（计划 §2.1）。

这是与既有 ``cross_section/`` 评估的本质区别——那边是分组多空，这边是论文的
long-only 规则。引擎规则严格按东证/论文描述（notebooklm 问答 Q1/Q6）：

    - 每个交易日 t 收盘：按当日信号对池内可用股票排序；
    - 目标持仓 k=50 等权；每日最大调入/调出 n=5 只；
    - 最少持有期 5 个交易日：持有不满 5 日的股票不得卖出（即便信号变差）；
    - 卖出候选 = 持有≥5日 且 信号排名最差者；买入候选 = 未持有 且 信号排名最好者；
      每日实际换手 = min(n, 可卖数, 可买数)；
    - 交易成本：单边 0.15%，按换手额从组合收益中扣除；
    - 停牌股：当日不可买卖（沿用 ``tradestatuscode==0`` 判定），持仓中停牌的顺延；
    - 初始建仓：首日按信号 top-k 一次建满（论文未述，属实现假设，记录于计划 §2.1）。

等权再平衡口径（计划 §2.2.3，选定并测试）：
    只在调入调出时再平衡**新腿**——新买入的股票按当日等权（1/k）配仓；
    存量腿随行就市（持有期内不再平衡，其权重随股价漂移）。这是最贴近"long-only
    等权 + 每日仅动 n 只"原意、且换手最可控的口径。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

# 每年交易日数（A 股约 244）；用于年化
TRADING_DAYS_PER_YEAR = 244


@dataclass
class EngineConfig:
    """引擎参数（计划 §2.1）。"""

    top_k: int = 50          # 目标持仓数
    drop_n: int = 5          # 每日最大换手（单边）
    min_hold: int = 5        # 最少持有期（交易日）
    cost_bps: float = 15.0   # 单边交易成本（bp）


@dataclass
class TradeLog:
    """换手日志（用于事后核验与日均换手统计）。"""

    rows: list[dict] = field(default_factory=list)

    def append(self, date, held_out: list[str], bought_in: list[str], turnover: float) -> None:
        self.rows.append(
            {
                "date": pd.Timestamp(date),
                "sold": held_out,
                "bought": bought_in,
                "turnover_ratio": turnover,  # 换手额 / 组合净值（单边）
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _rank_signals(signal: pd.Series, ascending: bool = True) -> pd.Series:
    """信号排名（默认大=好，rank 1..N）。NaN 信号排到最后（不可买）。"""
    # method="first" 保证无平局；NaN 给最大 rank（最差）
    return signal.rank(method="first", ascending=ascending, na_option="bottom")


def _daily_returns(
    px: pd.DataFrame,
) -> pd.DataFrame:
    """逐日个股后复权收益率（close-to-close）。

    :param px: 宽表 ``index=date, columns=code, values=后复权 close``。
    :returns: 同形状的日收益率 DataFrame。
    """
    return px.pct_change()


def run_portfolio(
    signal_wide: pd.DataFrame,
    px_wide: pd.DataFrame,
    tradeable: pd.DataFrame,
    *,
    cfg: EngineConfig,
) -> tuple[pd.Series, pd.Series, TradeLog]:
    """逐日推进 top-k/drop-n long-only 组合。

    :param signal_wide: 宽表 ``index=date, columns=code, values=signal``。
        每行的信号用于当日收盘后的调仓决策（信号越大约好）。
    :param px_wide: 宽表 ``index=date, columns=code, values=后复权 close``。
        用于算逐日收益与换手额。
    :param tradeable: 宽表 ``index=date, columns=code, values=bool``。
        ``True`` 表示当日可交易（未停牌）。与 ``px_wide`` 同形状。
    :param cfg: 引擎参数。
    :returns: ``(daily_ret, daily_excess_nav, trades)``：
        ``daily_ret`` = 组合逐日总收益（已扣单边成本）；
        ``daily_excess_nav`` = 相对**基准**的逐日超额收益序列的累计净值
        （基准在调用方合并；本函数只返毛组合收益，超额见 :func:`attach_benchmark`）；
        ``trades`` = :class:`TradeLog`。

    约定：
        - 决策时点：每个交易日 t 收盘后用 ``signal_wide.loc[t]`` 决策，
          次日 t+1 起按新持仓收取收益（避免同日买卖同日收益的 lookahead）。
        - 建仓：首日（信号序列第一个交易日）按 top-k 一次建满，建仓成本照扣。
        - 等权口径：新买入按 1/k 配仓；存量腿随行就市，持有期内不再平衡。
    """
    dates = signal_wide.index.intersection(px_wide.index).sort_values()
    if len(dates) == 0:
        raise ValueError("signal_wide 与 px_wide 无公共交易日")

    rets = _daily_returns(px_wide.loc[dates])
    # —— 状态 ——
    holding_since: dict[str, pd.Timestamp] = {}   # code -> 入场日
    weights: dict[str, float] = {}                # code -> 当前权重（随价格漂移）
    trades = TradeLog()
    daily_ret_list: list[tuple[pd.Timestamp, float]] = []

    target_w = 1.0 / cfg.top_k

    for i, d in enumerate(dates):
        sig = signal_wide.loc[d]
        can_trade = tradeable.loc[d] if d in tradeable.index else pd.Series(True, index=sig.index)

        # —— 1. 收当日组合收益（基于昨日的 weights 与今日个股收益）——
        # 首日无持仓，收益为 0
        if i == 0:
            port_ret = 0.0
        else:
            prev_w = pd.Series(weights, dtype=float)
            # 只对仍在 px 里的持仓算收益（退市/极端缺失记 0）
            day_r = rets.loc[d].reindex(prev_w.index).fillna(0.0)
            port_ret_gross = float((prev_w * day_r).sum())
            port_ret = port_ret_gross  # 成本在换手时扣（下方），这里先记毛

        # —— 2. 收盘后调仓决策（用今日 signal）——
        if i == 0:
            # 首日建仓：取 top-k
            ranked = _rank_signals(sig)
            # 只在可交易 & 信号非 NaN 的股票里选
            eligible = can_trade.reindex(sig.index).fillna(False) & sig.notna()
            buyable = sig[eligible].sort_values(ascending=False)
            buys = list(buyable.head(cfg.top_k).index)
            held_out: list[str] = []
            bought_in = buys
        else:
            # —— 卖出候选：持有≥min_hold 且 信号排名最差者 ——
            ranks = _rank_signals(sig)
            held_codes = list(weights.keys())
            # 持仓中今日不可交易的不能卖（停牌顺延）
            sellable = [
                c
                for c in held_codes
                if (d - holding_since[c]).days >= 0  # 占位，下面用交易日计数
                and _trading_days_between(dates, holding_since[c], d) >= cfg.min_hold
                and bool(can_trade.get(c, False))
            ]
            # 在可卖池里按信号排名升序（最差优先卖）
            sell_candidates = sorted(sellable, key=lambda c: ranks.get(c, np.inf))
            # —— 买入候选：未持有 且 可交易 且 信号排名最好者 ——
            not_held = [c for c in sig.index if c not in weights]
            buyable = [
                c
                for c in not_held
                if bool(can_trade.get(c, False)) and pd.notna(sig.get(c, np.nan))
            ]
            buy_candidates = sorted(buyable, key=lambda c: ranks.get(c, -np.inf), reverse=True)
            actual = min(cfg.drop_n, len(sell_candidates), len(buy_candidates))
            held_out = sell_candidates[:actual]
            bought_in = buy_candidates[:actual]

        # —— 3. 执行换手 & 扣成本（单边 cost_bps 按换手额）——
        turnover_ratio = 0.0  # 单边换手额 / 组合净值
        if held_out or bought_in:
            # 卖出：按卖出腿的当前权重变现（加入现金，但 long-only 等权近似无现金腿，
            # 直接把权重腾给买入腿）
            freed = 0.0
            for c in held_out:
                freed += weights.pop(c, 0.0)
                holding_since.pop(c, None)
            # 买入：每只新腿按 target_w 配仓（等权口径，§2.2.3）
            new_alloc_total = freed
            # 若有买入但无卖出（首日建仓后的补充极少见），用 target_w 占位
            if bought_in and not held_out and i == 0:
                new_alloc_total = target_w * len(bought_in)
            per_new = new_alloc_total / len(bought_in) if bought_in else 0.0
            for c in bought_in:
                weights[c] = per_new
                holding_since[c] = d
            # 单边换手额（买 + 卖 的市值变动），这里用变现额 + 配仓额近似，
            # 因等权再平衡买入额≈卖出额≈freed（除首日建仓）
            turnover_ratio = (freed + sum(weights[c] for c in bought_in)) / 2.0
            # 首日建仓特殊：全额配仓
            if i == 0:
                turnover_ratio = sum(weights.values())

        cost = turnover_ratio * cfg.cost_bps / 1e4
        port_ret_net = port_ret - cost

        trades.append(d, held_out, bought_in, turnover_ratio)
        daily_ret_list.append((d, port_ret_net))

        # —— 4. 存量腿随行就市：权重随昨日→今日收益漂移（已在 step 1 用收益体现，
        #         这里把权重本身向前推进，保持总权重≈1）——
        if i > 0 and weights:
            day_r = rets.loc[d]
            new_w = {}
            for c, w in weights.items():
                r = day_r.get(c, 0.0)
                if pd.isna(r):
                    r = 0.0
                new_w[c] = w * (1.0 + r)
            total = sum(new_w.values())
            if total > 0:
                weights = {c: w / total for c, w in new_w.items()}

    daily_ret = pd.Series(dict(daily_ret_list)).sort_index()
    daily_ret.name = "portfolio_ret_net"
    return daily_ret, daily_ret, trades


def _trading_days_between(
    dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    """``start`` 与 ``end`` 之间（含两端）的交易日数（在给定日历内）。"""
    mask = (dates >= start) & (dates <= end)
    return int(mask.sum()) - 1  # 不含入场当日


def attach_benchmark(
    daily_ret: pd.Series, benchmark_ret: pd.Series
) -> pd.Series:
    """组合日收益 − 基准日收益 = 超额日收益。

    :param daily_ret: 组合逐日净收益（已扣成本）。
    :param benchmark_ret: 基准逐日收益（沪深300 指数日收益，或池等权日收益）。
    :returns: 逐日超额收益（已对齐索引）。
    """
    common = daily_ret.index.intersection(benchmark_ret.index)
    excess = daily_ret.loc[common] - benchmark_ret.loc[common]
    excess.name = "excess_ret"
    return excess


@dataclass
class PerfStats:
    """组合绩效（计划 §4 指标）。"""

    name: str
    n_days: int
    aer: float               # 年化超额（扣成本），相对基准
    ir: float                # 信息比率 = 超额日收益均值/std × √252
    max_drawdown: float      # 超额净值最大回撤
    daily_turnover: float    # 日均单边换手率

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_days": self.n_days,
            "aer": self.aer,
            "ir": self.ir,
            "max_drawdown": self.max_drawdown,
            "daily_turnover": self.daily_turnover,
        }


def compute_perf(
    excess_ret: pd.Series, trades: TradeLog, *, name: str
) -> PerfStats:
    """从超额日收益与换手日志算 AER / IR / 最大回撤 / 日均换手。

    :param excess_ret: 逐日超额收益（组合 − 基准，已扣成本）。
    :param trades: :class:`TradeLog`。
    :param name: 结果标签。
    :returns: :class:`PerfStats`。
    """
    valid = excess_ret.dropna()
    n = len(valid)
    if n == 0:
        return PerfStats(name, 0, float("nan"), float("nan"), float("nan"), float("nan"))

    # AER：超额累计净值的年化（几何）
    nav = (1 + valid).cumprod()
    n_years = n / TRADING_DAYS_PER_YEAR
    aer = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else float("nan")
    # IR：超额日收益均值/std × √252
    mu = float(valid.mean())
    sd = float(valid.std(ddof=1))
    ir = mu / sd * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else float("nan")
    # 最大回撤（超额净值）
    dd = float((nav / nav.cummax() - 1).min())
    # 日均换手
    tdf = trades.to_frame()
    daily_to = float(tdf["turnover_ratio"].mean()) if len(tdf) else float("nan")
    return PerfStats(name=name, n_days=n, aer=aer, ir=ir, max_drawdown=dd, daily_turnover=daily_to)
