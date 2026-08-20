"""E1 缓冲带引擎契约测试（计划 §2，20260820 G8+E1 计划）。

冻结规则（§2）——双阈值滞回：
    持仓上限 k=50、min_hold=5、单边 15bp、每日净换手上限 n=5 只——全部沿用 canonical；
    卖出：持仓股当日信号排名跌出前 100（且已过 min_hold）才卖；
    买入：从未持仓股中按排名从高到低补足至 50，仅限排名 ≤ 30 者；
    候选不足时允许持仓 < 50（等权重分于现有持仓），不强行补足。

单测覆盖（计划 3.1 冻结清单）：min_hold 不变式、持仓数 ≤ top_k、无前视
（t 日决策仅用 t 日信号）、同输入重放逐位一致；另加：滞回带内不换手、
买入门槛 ≤ buy_rank 边界、日换手上限 n、首日建仓口径（≤ buy_rank 只、
等权满仓）、冻结默认值防漂移、成本口径与 canonical 同式。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from e1_buffer.engine import BufferEngineConfig, run_buffer_portfolio

# —— 合成市场小玩具参数（缩小阈值等比验证逻辑；冻结默认值另测）——
TOY = BufferEngineConfig(
    top_k=4, drop_n=2, min_hold=5, cost_bps=15.0, buy_rank=2, sell_rank=5
)


def make_market(n_days: int, n_stocks: int, start="2026-01-05"):
    """平价市场：价格恒 100（无漂移），全日可交易。"""
    dates = pd.bdate_range(start, periods=n_days)
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    px = pd.DataFrame(100.0, index=dates, columns=cols)
    trd = pd.DataFrame(True, index=dates, columns=cols)
    return dates, cols, px, trd


def signals_from_order(order_rows: list[list[int]], dates, cols) -> pd.DataFrame:
    """每日名次顺序表 → 信号宽表。

    ``order_rows[i]`` = 第 i 日从**最好到最差**排列的股票下标列表
    （如 ``[0,2,1]`` 表示 S000 第 1 名、S002 第 2 名、S001 第 3 名）。
    信号 = n_stocks − 名次（名次 1 信号最大）。
    """
    n = len(cols)
    sig = pd.DataFrame(0.0, index=dates, columns=cols)
    for i, order in enumerate(order_rows):
        assert sorted(order) == list(range(n)), f"order 行 {i} 非下标排列"
        for pos, j in enumerate(order):
            sig.iloc[i, j] = float(n - pos)
    return sig


def holdings_series(trades) -> pd.Series:
    """由换手日志重建逐日持仓数。"""
    hold, out = 0, {}
    for row in trades.rows:
        hold += len(row["bought"]) - len(row["sold"])
        out[row["date"]] = hold
    return pd.Series(out)


# ============================================================================
# 1. 冻结默认值（防漂移：§2 规则跑前冻结）
# ============================================================================
def test_frozen_defaults():
    cfg = BufferEngineConfig()
    assert (cfg.top_k, cfg.drop_n, cfg.min_hold, cfg.cost_bps) == (50, 5, 5, 15.0)
    assert cfg.buy_rank == 30 and cfg.sell_rank == 100


# ============================================================================
# 2. min_hold 不变式：跌出卖出门槛但持有不满 5 交易日 → 不卖
# ============================================================================
def test_min_hold_invariant():
    dates, cols, px, trd = make_market(10, 8)
    base = list(range(8))
    # 首日买 top-2（S000/S001）；自第 1 日起 S001 掉到第 8 名（> sell_rank=5）
    worst_s001 = [0, 2, 3, 4, 5, 6, 7, 1]
    orders = [list(base)] + [list(worst_s001)] * 9
    sig = signals_from_order(orders, dates, cols)

    _, _, trades = run_buffer_portfolio(sig, px, trd, cfg=TOY)
    sold_days = [row["date"] for row in trades.rows if "S001" in row["sold"]]
    # 首日建仓 = dates[0]，满 5 个交易日（不含入场日）最早 dates[5] 可卖
    assert sold_days and sold_days[0] == dates[5], (
        f"S001 应在 {dates[5].date()} 首次卖出，实际 {[str(d.date()) for d in sold_days]}"
    )


# ============================================================================
# 3. 滞回带：带内（buy_rank < 名次 ≤ sell_rank）持仓不卖 / 空仓不买
# ============================================================================
def test_hysteresis_band_no_churn():
    dates, cols, px, trd = make_market(10, 8)
    base = list(range(8))
    # 此后 S001 名次=4（带内持仓）、S005 名次=3（带内空仓）、S002 名次=2（可买）
    drifted = [0, 2, 5, 1, 3, 4, 6, 7]
    orders = [list(base)] + [list(drifted)] * 9
    sig = signals_from_order(orders, dates, cols)

    _, _, trades = run_buffer_portfolio(sig, px, trd, cfg=TOY)
    for row in trades.rows[1:]:
        assert "S001" not in row["sold"], "带内持仓（名次4≤sell_rank）不得卖出"
        assert "S005" not in row["bought"], "带内空仓股（名次3>buy_rank）不得买入"


# ============================================================================
# 4. 首日建仓：仅买排名 ≤ buy_rank、候选不足允许 < top_k、等权满仓
# ============================================================================
def test_day0_build_rank_cap_and_full_investment():
    dates, cols, px, trd = make_market(3, 8)
    orders = [list(range(8))] * 3  # 名次恒定
    sig = signals_from_order(orders, dates, cols)

    _, _, trades = run_buffer_portfolio(sig, px, trd, cfg=TOY)
    d0 = trades.rows[0]
    assert sorted(d0["bought"]) == ["S000", "S001"], "首日只买 top buy_rank"
    assert d0["sold"] == []
    assert d0["turnover_ratio"] == 1.0, "首日等权满仓（换手=1，同 canonical）"
    assert d0["n_holdings"] == 2, "候选不足（2 < top_k=4）允许持仓 < top_k"
    # 首日之后名次恒定 → 零换手
    assert all(r["turnover_ratio"] == 0.0 for r in trades.rows[1:])


# ============================================================================
# 5. 买入门槛边界：名次恰为 buy_rank 可买，buy_rank+1 不可买
# ============================================================================
def test_buy_rank_boundary():
    dates, cols, px, trd = make_market(8, 8)
    base = list(range(8))
    # 此后 S001 第 1、S002 第 2（=buy_rank，可买）、S003 第 3（>buy_rank，永不买）、
    # S000 掉到第 8（> sell_rank，满持有期后卖）
    drifted = [1, 2, 3, 4, 5, 6, 7, 0]
    orders = [list(base)] + [list(drifted)] * 7
    sig = signals_from_order(orders, dates, cols)

    _, _, trades = run_buffer_portfolio(sig, px, trd, cfg=TOY)
    ever_bought = {c for row in trades.rows for c in row["bought"]}
    assert "S003" not in ever_bought, "名次 > buy_rank 的空仓股不得买入"
    assert "S002" in ever_bought, "名次恰为 buy_rank 的空仓股在有空位时应买入"
    # S000 第 8 名 > sell_rank=5，dates[5] 满持有期被卖
    sold_days = [row["date"] for row in trades.rows if "S000" in row["sold"]]
    assert sold_days and sold_days[0] == dates[5]


# ============================================================================
# 6. 持仓数 ≤ top_k（任意信号路径）+ 首日 ≤ buy_rank
# ============================================================================
def test_holdings_never_exceed_top_k():
    rng = np.random.default_rng(7)
    dates, cols, px, trd = make_market(40, 60)
    sig = pd.DataFrame(rng.normal(size=(40, 60)), index=dates, columns=cols)
    cfg = BufferEngineConfig(
        top_k=6, drop_n=2, min_hold=3, cost_bps=15.0, buy_rank=3, sell_rank=8
    )
    _, _, trades = run_buffer_portfolio(sig, px, trd, cfg=cfg)
    hs = holdings_series(trades)
    assert (hs <= cfg.top_k).all(), f"持仓数越界：max={hs.max()}"
    assert hs.iloc[0] <= cfg.buy_rank


# ============================================================================
# 7. 日换手上限：首日之后每日卖出/买入各 ≤ drop_n
# ============================================================================
def test_daily_swap_cap():
    rng = np.random.default_rng(11)
    dates, cols, px, trd = make_market(30, 40)
    sig = pd.DataFrame(rng.normal(size=(30, 40)), index=dates, columns=cols)
    cfg = BufferEngineConfig(
        top_k=5, drop_n=2, min_hold=2, cost_bps=15.0, buy_rank=2, sell_rank=4
    )
    _, _, trades = run_buffer_portfolio(sig, px, trd, cfg=cfg)
    for row in trades.rows[1:]:
        assert len(row["sold"]) <= cfg.drop_n
        assert len(row["bought"]) <= cfg.drop_n


# ============================================================================
# 8. 无前视：t 日决策仅用 t 日信号（改 t+1 之后的信号不得影响 ≤t 的决策与收益）
# ============================================================================
def test_no_lookahead():
    rng = np.random.default_rng(3)
    dates, cols, px, trd = make_market(15, 12)
    sig = pd.DataFrame(rng.normal(size=(15, 12)), index=dates, columns=cols)
    cfg = BufferEngineConfig(
        top_k=4, drop_n=2, min_hold=2, cost_bps=15.0, buy_rank=2, sell_rank=5
    )
    ret_full, _, trades_full = run_buffer_portfolio(sig, px, trd, cfg=cfg)

    # 篡改第 4 日（index 4）之后的信号行
    sig_mut = sig.copy()
    sig_mut.iloc[4:] = rng.permutation(sig_mut.iloc[4:])
    ret_mut, _, trades_mut = run_buffer_portfolio(sig_mut, px, trd, cfg=cfg)

    t = dates[3]
    assert (ret_full.loc[:t] == ret_mut.loc[:t]).all(), "前 4 日收益被未来信号影响"
    for a, b in zip(trades_full.rows[:4], trades_mut.rows[:4]):
        assert a["date"] == b["date"]
        assert a["sold"] == b["sold"] and a["bought"] == b["bought"], (
            f"{a['date'].date()} 决策被未来信号影响"
        )


# ============================================================================
# 9. 同输入重放逐位一致
# ============================================================================
def test_replay_bitwise_deterministic():
    rng = np.random.default_rng(5)
    dates, cols, px, trd = make_market(20, 15)
    sig = pd.DataFrame(rng.normal(size=(20, 15)), index=dates, columns=cols)
    cfg = BufferEngineConfig(
        top_k=4, drop_n=2, min_hold=2, cost_bps=15.0, buy_rank=2, sell_rank=6
    )
    r1, _, t1 = run_buffer_portfolio(sig, px, trd, cfg=cfg)
    r2, _, t2 = run_buffer_portfolio(sig, px, trd, cfg=cfg)
    assert r1.equals(r2), "同输入重放收益序列不一致"
    for a, b in zip(t1.rows, t2.rows):
        assert a == b, "同输入重放换手日志不一致"


# ============================================================================
# 10. 成本口径：平价市场净收益 = −换手×15bp（与 canonical 同式）
# ============================================================================
def test_cost_accounting_flat_market():
    dates, cols, px, trd = make_market(6, 8)
    orders = [list(range(8))] * 6
    sig = signals_from_order(orders, dates, cols)
    _, ret, trades = run_buffer_portfolio(sig, px, trd, cfg=TOY)
    # 名次恒定 → 首日之后零换手；净收益 = 首日 −1×15bp，其余 0
    assert abs(ret.iloc[0] - (-TOY.cost_bps / 1e4)) < 1e-12
    assert (np.abs(ret.iloc[1:].values) < 1e-12).all()
