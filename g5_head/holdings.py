"""持仓重放（G5 计划 §1 阶段 0 归因用）。

canonical 引擎 :func:`paper_replication.engine.run_portfolio` 只外泄换手日志，
不外泄逐日持仓集合。本模块逐字镜像引擎的**选股分支**（首日建仓 / 卖出候选 /
买入候选 / min(drop_n, 可卖, 可买)），只跟踪持仓成员与时钟，不算收益——
选股逻辑只依赖信号 + 可交易掩码 + 持仓成员（不依赖权重），故成员集合与引擎
逐日一致（tests/test_g5_head.py::TestHoldingsReplay 与引擎 TradeLog 逐位对拍）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from paper_replication.engine import EngineConfig, _rank_signals, _trading_days_between


def replay_holdings(
    signal_wide: pd.DataFrame,
    px_wide: pd.DataFrame,
    tradeable: pd.DataFrame,
    *,
    cfg: EngineConfig,
) -> tuple[dict[pd.Timestamp, frozenset[str]], list[tuple[pd.Timestamp, list[str], list[str]]]]:
    """重放引擎的逐日持仓集合。

    :param signal_wide: 宽表 ``index=date, columns=code, values=signal``（越大越好）。
    :param px_wide: 后复权 close 宽表（只用于交易日历交集，与引擎一致）。
    :param tradeable: bool 宽表（True=当日可交易）。
    :param cfg: 与正式回测相同的 :class:`EngineConfig`。
    :returns: ``(holdings, events)``：
        ``holdings`` = ``{date: frozenset[code]}``（当日收盘调仓后的持仓集合）；
        ``events`` = ``[(date, sold, bought), ...]``（与引擎 TradeLog 逐位一致）。

    分支逐字对应 engine.run_portfolio：首日 ``eligible & sig.notna`` 按信号降序
    取 top-k；此后卖出候选 = 持有 ≥ min_hold 交易日且当日可交易、按信号排名升序；
    买入候选 = 未持有且可交易且信号非 NaN、按排名降序；actual = min(drop_n, ...)。
    """
    dates = signal_wide.index.intersection(px_wide.index).sort_values()
    if len(dates) == 0:
        raise ValueError("signal_wide 与 px_wide 无公共交易日")

    holding_since: dict[str, pd.Timestamp] = {}
    weights_members: set[str] = set()
    holdings: dict[pd.Timestamp, frozenset[str]] = {}
    events: list[tuple[pd.Timestamp, list[str], list[str]]] = []

    for i, d in enumerate(dates):
        sig = signal_wide.loc[d]
        can_trade = tradeable.loc[d] if d in tradeable.index else pd.Series(True, index=sig.index)

        if i == 0:
            ranked = _rank_signals(sig)
            eligible = can_trade.reindex(sig.index).fillna(False) & sig.notna()
            buyable = sig[eligible].sort_values(ascending=False)
            buys = list(buyable.head(cfg.top_k).index)
            held_out: list[str] = []
            bought_in = buys
        else:
            ranks = _rank_signals(sig)
            held_codes = list(weights_members)
            sellable = [
                c
                for c in held_codes
                if _trading_days_between(dates, holding_since[c], d) >= cfg.min_hold
                and bool(can_trade.get(c, False))
            ]
            sell_candidates = sorted(sellable, key=lambda c: ranks.get(c, np.inf))
            not_held = [c for c in sig.index if c not in weights_members]
            buyable = [
                c
                for c in not_held
                if bool(can_trade.get(c, False)) and pd.notna(sig.get(c, np.nan))
            ]
            buy_candidates = sorted(buyable, key=lambda c: ranks.get(c, -np.inf), reverse=True)
            actual = min(cfg.drop_n, len(sell_candidates), len(buy_candidates))
            held_out = sell_candidates[:actual]
            bought_in = buy_candidates[:actual]

        for c in held_out:
            weights_members.discard(c)
            holding_since.pop(c, None)
        for c in bought_in:
            weights_members.add(c)
            holding_since[c] = d

        holdings[d] = frozenset(weights_members)
        events.append((d, held_out, bought_in))

    return holdings, events


def daily_overlap(
    holdings_a: dict, holdings_b: dict, *, k: int
) -> float:
    """逐日持仓重合度均值：``mean(|A ∩ B| / k)``（两字典按公共日对齐）。"""
    common = sorted(set(holdings_a) & set(holdings_b))
    if not common:
        raise ValueError("两持仓序列无公共交易日")
    vals = [len(holdings_a[d] & holdings_b[d]) / k for d in common]
    return float(np.mean(vals))
