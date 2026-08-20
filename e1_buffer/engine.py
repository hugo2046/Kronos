"""E1 缓冲带引擎：双阈值滞回 top-k long-only（计划 §2，规则冻结）。

与 canonical（``paper_replication.engine.run_portfolio``）的唯一差异 = 换仓规则：

    - 卖出：持仓股当日信号排名跌出前 ``sell_rank``（且已过 min_hold）才卖；
    - 买入：从未持仓股中按排名从高到低补足至 ``top_k``，仅限排名 ≤ ``buy_rank`` 者；
    - 候选不足时允许持仓 < top_k（等权重分于现有持仓），不强行补足。

canonical 的规则是"每天持有中信号最差者优先轮换"——引擎每日强制轮换
贡献 ~10%/日机械换手底；E1 以滞回带 [buy_rank, sell_rank] 吸收排名噪声抖动
（进严 30 / 出宽 100），只有信号真正跨过带边界才动仓。

实现假设（计划 §2 未细化处的落点，比照 canonical §2.1 的先例如实记录）：

    1. 首日建仓：沿用 canonical "首日一次建满" 结构，但受买入门槛约束——
       只买当日排名 ≤ buy_rank 的可交易股（至多 buy_rank 只），等权满仓
       （每只 1/m，"等权重分于现有持仓"）；不设 drop_n 上限（同 canonical 首日）。
    2. 新腿配仓 = 1 / (换仓后持仓数)——补足语义下的等权满仓；资金来源优先
       卖出腾挪额（freed），不足部分按比例从存量腿刮取（pro-rata shave）。
       与 canonical 的 freed/len(buys) 在"卖买等量、存量腿约等权"时近似一致，
       在卖多买少（候选不足）时不会把全部腾挪额压进少数新腿。
    3. 卖出腾挪额若无人接（买入候选不足）则暂留，由日终 pro-rata 重归一
       分摊回存量腿（canonical 的随行就市重归一口径，零成本）。
    4. 换手额口径 = (变现额 + 配仓额) / 2（单边，同 canonical）；首日 = 1.0。

无前视：t 日收盘决策仅用 ``signal_wide.loc[t]``（同 canonical）；t+1 起按
新持仓收收益。同输入重放逐位一致（纯确定性路径，无随机源）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# canonical 引擎只读复用（纪律 §4：不改其一行代码）
from paper_replication.engine import (
    TradeLog,
    _daily_returns,
    _rank_signals,
    _trading_days_between,
)


@dataclass
class BufferEngineConfig:
    """缓冲带引擎参数（计划 §2 冻结；前四项与 canonical 逐字一致）。"""

    top_k: int = 50          # 持仓上限（等权分母仅在建仓时刻）
    drop_n: int = 5          # 每日净换手上限（单边，只）
    min_hold: int = 5        # 最少持有期（交易日）
    cost_bps: float = 15.0   # 单边交易成本（bp）
    buy_rank: int = 30       # 买入门槛：仅排名 ≤ 30 者可买
    sell_rank: int = 100     # 卖出门槛：跌出前 100 才卖


@dataclass
class BufferTradeLog(TradeLog):
    """换手日志（继承 canonical 字段，追加缓冲带描述量）。"""

    rows: list[dict] = field(default_factory=list)

    def append(  # type: ignore[override]
        self,
        date,
        held_out: list[str],
        bought_in: list[str],
        turnover: float,
        *,
        n_holdings: int,
        new_leg_w: float,
        freed: float,
        shaved: float,
    ) -> None:
        self.rows.append(
            {
                "date": pd.Timestamp(date),
                "sold": held_out,
                "bought": bought_in,
                "turnover_ratio": turnover,
                "n_holdings": n_holdings,   # 换仓后持仓数
                "new_leg_w": new_leg_w,     # 新腿配仓权重（1/n_after）
                "freed": freed,             # 卖出腾挪额
                "shaved": shaved,           # 存量腿 pro-rata 刮取额
            }
        )


def run_buffer_portfolio(
    signal_wide: pd.DataFrame,
    px_wide: pd.DataFrame,
    tradeable: pd.DataFrame,
    *,
    cfg: BufferEngineConfig,
) -> tuple[pd.Series, pd.Series, BufferTradeLog]:
    """逐日推进双阈值滞回 long-only 组合（接口与 canonical ``run_portfolio`` 同构）。

    :param signal_wide: 宽表 ``index=date, columns=code, values=signal``（越大越好）。
    :param px_wide: 后复权 close 宽表（收益与换手额）。
    :param tradeable: 可交易掩码宽表（True=未停牌）。
    :param cfg: :class:`BufferEngineConfig`。
    :returns: ``(daily_ret, daily_ret, trades)``——净收益（已扣单边成本）、
        同序列（与 canonical 返回形状对齐）、:class:`BufferTradeLog`。
    """
    dates = signal_wide.index.intersection(px_wide.index).sort_values()
    if len(dates) == 0:
        raise ValueError("signal_wide 与 px_wide 无公共交易日")

    rets = _daily_returns(px_wide.loc[dates])
    holding_since: dict[str, pd.Timestamp] = {}
    weights: dict[str, float] = {}
    trades = BufferTradeLog()
    daily_ret_list: list[tuple[pd.Timestamp, float]] = []

    for i, d in enumerate(dates):
        sig = signal_wide.loc[d]
        can_trade = (
            tradeable.loc[d]
            if d in tradeable.index
            else pd.Series(True, index=sig.index)
        )

        # —— 1. 收当日组合收益（基于昨日权重与今日个股收益；首日为 0）——
        if i == 0:
            port_ret = 0.0
        else:
            prev_w = pd.Series(weights, dtype=float)
            day_r = rets.loc[d].reindex(prev_w.index).fillna(0.0)
            port_ret = float((prev_w * day_r).sum())

        # —— 2. 收盘后调仓决策（仅用今日 signal 行，无前视）——
        # 质量名次（rank 1 = 最好信号；NaN 落底=最差）——计划 §2 的"排名 ≤ 30/
        # 跌出前 100"语义。注意 canonical 引擎内部用 ascending=True（rank 1=最差）
        # 配合其卖差买好的排序；本引擎的阈值比较需要质量名次方向。
        ranks = _rank_signals(sig, ascending=False)
        eligible = can_trade.reindex(sig.index).fillna(False) & sig.notna()

        if i == 0:
            # 首日建仓：全部排名 ≤ buy_rank 的可交易股，等权满仓
            buys = [
                c
                for c in sig.index
                if eligible.get(c, False) and ranks.get(c, float("-inf")) <= cfg.buy_rank
            ]
            held_out: list[str] = []
        else:
            # 卖出：持有≥min_hold 且 排名跌出前 sell_rank 且 当日可交易；
            # 信号行缺失的持仓码按最差名次处理（对齐 canonical 的 np.inf 先卖语义）
            held_codes = list(weights.keys())
            sellable = [
                c
                for c in held_codes
                if _trading_days_between(dates, holding_since[c], d) >= cfg.min_hold
                and ranks.get(c, float("inf")) > cfg.sell_rank
                and bool(can_trade.get(c, False))
            ]
            # 最差排名优先卖（质量名次最大者），单日上限 drop_n
            sellable.sort(key=lambda c: ranks.get(c, float("inf")), reverse=True)
            held_out = sellable[: cfg.drop_n]

            # 买入：未持有 且 可交易 且 排名 ≤ buy_rank，从高到低补足至 top_k
            not_held = [c for c in sig.index if c not in weights]
            buy_cands = [
                c
                for c in not_held
                if bool(can_trade.get(c, False))
                and pd.notna(sig.get(c, np.nan))
                and ranks.get(c, float("inf")) <= cfg.buy_rank
            ]
            buy_cands.sort(key=lambda c: ranks.get(c, float("inf")))
            slots = max(cfg.top_k - (len(held_codes) - len(held_out)), 0)
            buys = buy_cands[: min(cfg.drop_n, slots)]

        # —— 3. 执行换手、配仓与扣成本 ——
        turnover_ratio = 0.0
        new_leg_w = 0.0
        freed = 0.0
        shaved = 0.0
        if held_out or buys:
            for c in held_out:
                freed += weights.pop(c, 0.0)
                holding_since.pop(c, None)
            n_after = len(weights) + len(buys)
            if buys:
                # 新腿配仓 = 1/n_after（补足语义的等权满仓）
                new_leg_w = 1.0 / n_after
                need = new_leg_w * len(buys)
                if freed >= need:
                    # 卖出腾挪额足够：盈余暂留（日终重归一分摊回存量腿）
                    shaved = 0.0
                else:
                    # 不足部分按比例从存量腿刮取
                    total_existing = sum(weights.values())
                    shaved = min(need - freed, total_existing)
                    if total_existing > 0 and shaved > 0:
                        scale = (total_existing - shaved) / total_existing
                        weights = {c: w * scale for c, w in weights.items()}
                for c in buys:
                    weights[c] = new_leg_w
                    holding_since[c] = d
                buy_amount = new_leg_w * len(buys)
            else:
                buy_amount = 0.0
            turnover_ratio = (freed + shaved + buy_amount) / 2.0
            if i == 0:
                turnover_ratio = 1.0  # 首日全额建仓（canonical 同款）

        cost = turnover_ratio * cfg.cost_bps / 1e4
        port_ret_net = port_ret - cost
        trades.append(
            d, held_out, buys, turnover_ratio,
            n_holdings=len(weights), new_leg_w=new_leg_w,
            freed=freed, shaved=shaved,
        )
        daily_ret_list.append((d, port_ret_net))

        # —— 4. 存量腿随行就市：权重按收益漂移后重归一（同 canonical）——
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
    daily_ret.name = "buffer_portfolio_ret_net"
    return daily_ret, daily_ret, trades
