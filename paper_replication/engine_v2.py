"""top-k / drop-n long-only 组合引擎 v2 —— 旧引擎六项口径修正（可逐项开关）。

背景：Codex_review 复核报告（2026-09-05）§2-B/§2-C 指出旧引擎
（``paper_replication/engine.py``，不改）六处口径问题。本模块逐项修正，
每项独立开关、默认全开；**六开关全关时数值上复现旧引擎**（随机数据逐日
等价断言见 ``tests/test_engine_v2.py::TestLegacyEquivalence``），因此任何
"旧 vs v2"差值都只来自这些修正本身。

六项修正（编号与复核报告一致）：

    (1) ``fix_double_sided_cost`` —— 成本按每笔双边计：
        ``cost = (freed + bought) × cost_bps/1e4``。旧引擎按
        ``(freed + bought)/2 × bps``（单边近似）计，系统性少扣一半成本：
        一次换手 X（卖 X 买 X）旧引擎只扣 15bp·X，应扣 2×15bp·X。

    (2) ``fix_delay_1`` —— t 日信号在 t+1 收盘成交，首个收益日为 t+2。
        旧引擎 t 日收盘决策并同刻成交：t+1 的涨跌被记在 t 日才买入的腿上
        （同日 lookahead：买入价即 t 日收盘价，t+1 收益本应归属组合，但
        引擎同时把 t 日（决策日）的涨幅通过权重漂移虚记给新腿）。v2 决策
        与成交分离一日：d 收盘决策 → d+1 收盘成交（可交易性与涨跌停在
        **成交日**复核）→ d+2 起按新持仓收收益。初始建仓同样顺延一日。
        附带语义：持有期资格（≥ min_hold 交易日）在**决策日**评估（保守，
        不预支成交日），顺延成交后实际持有 ≥ min_hold+1 个交易日。

    (3) ``fix_limit_block`` —— 涨跌停/一字板不可成交：买入排除当日**收盘
        涨停**股（涨停价上的买单按未成交处理），卖出排除当日**收盘跌停**股。
        判定方案（按优先级，见 :func:`build_limit_masks`）：

          a. DDB ``up_down_limit_status`` 字段（``dfs://QlibFeaturesDay.Features``，
             qlib 表达式 ``$up_down_limit_status``）。实测（2025-06 全市场交叉
             验证）：**+1 = 收盘涨停**（1433/1433 例 close==high）、**-1 = 收盘
             跌停**（close==low）、0/NaN = 未封板；20cm 板（创业板/科创板）与
             ST 档位自动覆盖；除权除息日的极端 pctchange 不误报。
             注意 ``limit`` 列是**未复权**涨停价，与后复权 close 不同尺度，
             不可直接比较——故用状态字段而非价格字段。
          b. 字段缺失时回退**一字板**判定：``high == low``（全天一档价），
             方向按 close 相对 preclose 的符号：正 = 涨停一字（禁买）、
             负 = 跌停一字（禁卖）、零 = 罕见平价一字（不禁）。

        被禁委托当日不成交：买入顺延到候选名单下一位（决策日已给出完整
        排序候选），卖出顺延（持仓保留，由后续交易日决策重评）。

    (4) ``fix_new_leg_drift`` —— 新买入腿权重不参与当日收益漂移。新腿在
        成交日收盘才建仓，成交日（及之前）的价格变动不属于本组合；其权重
        从成交日的**下一交易日**起才随价格漂移。实现上把权重漂移移到当日
        交易执行**之前**、且只作用于当日开盘时已持有的腿。旧引擎在收盘
        交易后统一漂移全部权重（含刚成交的新腿），新腿被乘上 (1+r_成交日)
        ——把成交前已发生的价格变动虚记进新腿权重（幻觉加仓：组合从未
        赚到那笔钱，权重却按赚到了漂移）。

    (5) ``fix_ew_benchmark_mask``（见 :func:`build_pool_equal_weight_benchmark_v2`）
        —— 等权基准掩码 = ``tradeable ∧ signal.notna()``：基准选股域与该臂
        信号的覆盖域一致。旧掩码只看 tradeable：信号缺失股（新上市/长停复牌
        等）的涨跌会污染"同池等权"基准，而组合根本不可能持有它们。

    (6) ``fix_annualization_252``（见 :func:`compute_perf_v2`）—— 年化常量
        252：AER 复利年化 ``nav^(252/n)`` 与 IR ``mean/std×√252`` 一致用 252。
        旧引擎 244（且 IR 处注释写 √252、实算 √244，自相矛盾）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 修正 (6)：年化常量 252；244 仅作旧口径开关关闭时的回退值
TRADING_DAYS_PER_YEAR_V2 = 252
_LEGACY_TRADING_DAYS_PER_YEAR = 244


@dataclass
class EngineConfigV2:
    """引擎 v2 参数（六项修正各自独立开关，默认全开）。"""

    top_k: int = 50
    drop_n: int = 5
    min_hold: int = 5
    cost_bps: float = 15.0
    # —— 六项修正开关（编号对应复核报告 §2-B/§2-C）——
    fix_double_sided_cost: bool = True    # (1) 双边成本
    fix_delay_1: bool = True              # (2) t 信号 → t+1 收盘成交
    fix_limit_block: bool = True          # (3) 涨跌停/一字板不可成交
    fix_new_leg_drift: bool = True        # (4) 新腿权重不参与当日漂移
    fix_annualization_252: bool = True    # (6) 年化 252
    # 注：(5) 等权基准掩码在 build_pool_equal_weight_benchmark_v2 的参数上


@dataclass
class TradeLogV2:
    """换手日志（含决策日/成交日分离与双边换手额，用于事后核验）。"""

    rows: list[dict] = field(default_factory=list)

    def append(
        self,
        date,
        decision_date,
        held_out: list[str],
        bought_in: list[str],
        freed: float,
        bought_amt: float,
        cost: float,
    ) -> None:
        one_side = bought_amt if not held_out else (freed + bought_amt) / 2.0
        self.rows.append(
            {
                "date": pd.Timestamp(date),            # 成交日
                "decision_date": pd.Timestamp(decision_date),
                "sold": held_out,
                "bought": bought_in,
                "freed": freed,                        # 卖出腿权重变现额
                "bought_amt": bought_amt,              # 买入腿配仓额
                "turnover_one_side": one_side,         # 旧口径单边换手（可比）
                "turnover_double": freed + bought_amt, # 双边换手额
                "cost": cost,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _tds_between(dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """``start`` 与 ``end`` 之间的交易日数（不含入场当日；与旧引擎同口径）。"""
    mask = (dates >= start) & (dates <= end)
    return int(mask.sum()) - 1


def _drift(weights: dict[str, float], day_r: pd.Series) -> None:
    """权重随当日收益漂移并归一（Σ=1）。只作用于传入 dict 中的既有腿。"""
    if not weights:
        return
    new_w = {}
    for c, w in weights.items():
        r = day_r.get(c, 0.0)
        r = 0.0 if pd.isna(r) else float(r)
        new_w[c] = w * (1.0 + r)
    total = sum(new_w.values())
    if total > 0:
        weights.clear()
        weights.update({c: w / total for c, w in new_w.items()})


def run_portfolio_v2(
    signal_wide: pd.DataFrame,
    px_wide: pd.DataFrame,
    tradeable: pd.DataFrame,
    *,
    cfg: EngineConfigV2,
    buy_blocked: pd.DataFrame | None = None,
    sell_blocked: pd.DataFrame | None = None,
) -> tuple[pd.Series, TradeLogV2]:
    """逐日推进 top-k/drop-n long-only 组合（v2 口径）。

    :param signal_wide: 宽表 ``index=date, columns=code, values=signal``（越大越好）。
        ``fix_delay_1`` 开时，d 日信号用于 d 收盘的**决策**，成交在 d+1 收盘。
    :param px_wide: 后复权 close 宽表（逐日收益与换手额）。
    :param tradeable: bool 宽表（停牌=False），**成交日**复核。
    :param cfg: :class:`EngineConfigV2`。
    :param buy_blocked: bool 宽表，True = 当日收盘涨停（禁买）。None = 全 False。
        由 :func:`build_limit_masks` 构造；仅在 ``fix_limit_block`` 开时生效。
    :param sell_blocked: bool 宽表，True = 当日收盘跌停（禁卖）。同上。
    :returns: ``(daily_ret_net, trades)``：逐日净收益（成本计在**成交日**）。

    时序约定（``fix_delay_1`` 开，默认）：

        - d0：首日决策（建仓 top-k，含完整排序候选清单）→ pending；
        - d1：收盘执行 pending（成交日复核 tradeable/涨跌停）；d0/d1 无持仓
          收益（=0）；d1 收盘后再按 d1 信号决策次日订单；
        - d2 起：按昨日收盘持仓收当日收益——首个收益日 = 信号日 + 2。
        - 每日流程顺序：①开盘持仓收当日收益 → ②既有腿权重漂移（修正 4：
          先漂移，新腿成交后进入，不再吃当日漂移）→ ③执行 pending 订单
          （成交日复核）→ ④当日信号决策次日订单。
    """
    dates = signal_wide.index.intersection(px_wide.index).sort_values()
    if len(dates) == 0:
        raise ValueError("signal_wide 与 px_wide 无公共交易日")
    rets = px_wide.loc[dates].pct_change(fill_method=None)

    weights: dict[str, float] = {}
    holding_since: dict[str, pd.Timestamp] = {}
    trades = TradeLogV2()
    daily: list[tuple[pd.Timestamp, float]] = []
    pending: dict | None = None  # {"build": bool, "sells": [...], "buys": [...]}
    pending_from: pd.Timestamp | None = None
    target_w = 1.0 / cfg.top_k

    def _blocked(mask: pd.DataFrame | None, d, codes) -> pd.Series:
        if mask is None or d not in mask.index:
            return pd.Series(False, index=codes)
        return mask.loc[d].reindex(codes).fillna(False).astype(bool)

    def decide(d: pd.Timestamp, sig: pd.Series, can_trade: pd.Series) -> dict:
        """d 收盘决策（用 d 日信号 + 决策时点持仓）。

        ``fix_delay_1`` 开时不做可交易性预过滤（留给成交日复核），只要求
        信号非 NaN；关时按旧引擎在决策日过滤（legacy 等价性要求）。
        卖出资格（≥min_hold 交易日）在决策日评估（见模块 docstring (2)）。
        """
        rank = sig.rank(method="first", ascending=True, na_option="bottom")
        if not weights:
            if cfg.fix_delay_1:
                elig = sig.notna()
            else:
                elig = can_trade.reindex(sig.index).fillna(False) & sig.notna()
            # 建仓保留完整排序候选（不预截 top_k）：成交日涨停/停牌挡下
            # 前列股票时按排序顺延补齐，截断在 fill_order 成交时做
            buys = list(sig[elig].sort_values(ascending=False).index)
            return {"build": True, "sells": [], "buys": buys}
        sell_ok = [
            c
            for c in weights
            if _tds_between(dates, holding_since[c], d) >= cfg.min_hold
            and (cfg.fix_delay_1 or bool(can_trade.get(c, False)))
        ]
        sells = sorted(sell_ok, key=lambda c: rank.get(c, np.inf))  # 信号最差在前
        not_held = [c for c in sig.index if c not in weights]
        if cfg.fix_delay_1:
            buy_ok = [c for c in not_held if pd.notna(sig.get(c, np.nan))]
        else:
            buy_ok = [
                c
                for c in not_held
                if bool(can_trade.get(c, False)) and pd.notna(sig.get(c, np.nan))
            ]
        buys = sorted(buy_ok, key=lambda c: rank.get(c, -np.inf), reverse=True)
        return {"build": False, "sells": sells, "buys": buys}

    def fill_order(
        d: pd.Timestamp,
        order: dict,
        can_trade: pd.Series,
        buy_block: pd.Series | None,
        sell_block: pd.Series | None,
    ) -> tuple[list[str], list[str], float, float]:
        """d 收盘成交订单（成交日复核可交易性与涨跌停），更新持仓。"""
        use_limit = cfg.fix_limit_block and buy_block is not None
        if order["build"]:
            buys = [
                c
                for c in order["buys"]
                if bool(can_trade.get(c, False))
                and not (use_limit and bool(buy_block.get(c, False)))
            ][: cfg.top_k]
            bought_amt = target_w * len(buys)
            for c in buys:
                weights[c] = target_w
                holding_since[c] = d
            return [], buys, 0.0, bought_amt
        exec_sells = [
            c
            for c in order["sells"]
            if c in weights
            and bool(can_trade.get(c, False))
            and not (use_limit and bool(sell_block.get(c, False)))
        ][: cfg.drop_n]
        exec_buys = [
            c
            for c in order["buys"]
            if c not in weights
            and bool(can_trade.get(c, False))
            and not (use_limit and bool(buy_block.get(c, False)))
        ][: cfg.drop_n]
        actual = min(len(exec_sells), len(exec_buys))
        held_out, bought_in = exec_sells[:actual], exec_buys[:actual]
        freed = 0.0
        for c in held_out:
            freed += weights.pop(c, 0.0)
            holding_since.pop(c, None)
        per_new = freed / len(bought_in) if bought_in else 0.0
        for c in bought_in:
            weights[c] = per_new
            holding_since[c] = d
        return held_out, bought_in, freed, per_new * len(bought_in)

    for i, d in enumerate(dates):
        sig = signal_wide.loc[d]
        can_trade = (
            tradeable.loc[d]
            if d in tradeable.index
            else pd.Series(True, index=sig.index)
        )
        buy_block = _blocked(buy_blocked, d, sig.index)
        sell_block = _blocked(sell_blocked, d, sig.index)

        # ① 当日组合收益：基于开盘时点（昨日收盘形成）的权重
        if not weights:
            port_ret = 0.0
        else:
            prev_w = pd.Series(weights, dtype=float)
            day_r = rets.loc[d].reindex(prev_w.index).fillna(0.0)
            port_ret = float((prev_w * day_r).sum())

        # ② 权重漂移（修正 4 开：成交前漂移既有腿；新腿成交后进入，不吃当日）
        if cfg.fix_new_leg_drift:
            _drift(weights, rets.loc[d])

        # ③ 成交：delay 开 → 成交昨日 pending；delay 关 → 决策即成交（legacy）
        if cfg.fix_delay_1:
            if pending is not None:
                held_out, bought_in, freed, bought_amt = fill_order(
                    d, pending, can_trade, buy_block, sell_block
                )
                decision_date = pending_from
                pending = None
            else:
                held_out, bought_in, freed, bought_amt, decision_date = [], [], 0.0, 0.0, d
        else:
            order = decide(d, sig, can_trade)
            held_out, bought_in, freed, bought_amt = fill_order(
                d, order, can_trade, buy_block, sell_block
            )
            decision_date = d

        # 成本：修正 (1) 双边；关 → 旧口径 (freed+bought)/2（建仓日=全额，同旧引擎）
        if cfg.fix_double_sided_cost:
            cost = (freed + bought_amt) * cfg.cost_bps / 1e4
        else:
            base = bought_amt if not held_out else (freed + bought_amt) / 2.0
            cost = base * cfg.cost_bps / 1e4
        trades.append(d, decision_date, held_out, bought_in, freed, bought_amt, cost)
        daily.append((d, port_ret - cost))

        # ④ 决策次日订单（delay 开）
        if cfg.fix_delay_1:
            pending = decide(d, sig, can_trade)
            pending_from = d

        # 修正 (4) 关 → 旧行为：收盘交易后漂移**全部**权重（含当日新腿，i>0 起）
        if not cfg.fix_new_leg_drift and i > 0:
            _drift(weights, rets.loc[d])

    daily_ret = pd.Series(dict(daily)).sort_index()
    daily_ret.name = "portfolio_ret_net_v2"
    return daily_ret, trades


def build_limit_masks(
    uls_wide: pd.DataFrame | None = None,
    high_wide: pd.DataFrame | None = None,
    low_wide: pd.DataFrame | None = None,
    preclose_wide: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """构造 (buy_blocked, sell_blocked) 掩码（方案见模块 docstring 修正 (3)）。

    :param uls_wide: DDB ``up_down_limit_status`` 宽表（+1 收盘涨停 / -1 收盘
        跌停 / 0 或 NaN 未封板）。提供时优先使用，其余参数忽略。
    :param high_wide / low_wide / preclose_wide: 回退一字板判定所需
        （high==low 且 close 相对 preclose 定方向；preclose 缺失时一律
        high==low 即双向禁）。
    :returns: ``(buy_blocked, sell_blocked)``；输入全缺返回 ``(None, None)``。
    """
    if uls_wide is not None:
        buy_blocked = (uls_wide == 1).fillna(False)
        sell_blocked = (uls_wide == -1).fillna(False)
        return buy_blocked, sell_blocked
    if high_wide is None or low_wide is None:
        return None, None
    one_line = (high_wide == low_wide) & high_wide.notna() & low_wide.notna()
    if preclose_wide is not None:
        up = one_line & (high_wide > preclose_wide)
        down = one_line & (high_wide < preclose_wide)
    else:
        up, down = one_line, one_line
    return up.fillna(False), down.fillna(False)


def build_pool_equal_weight_benchmark_v2(
    px_wide: pd.DataFrame,
    tradeable: pd.DataFrame,
    signal_wide: pd.DataFrame | None = None,
    *,
    fix_mask: bool = True,
) -> pd.Series:
    """同池等权基准 v2（修正 (5)：掩码 = tradeable ∧ signal.notna()）。

    :param fix_mask: True（默认）= 修正口径，分母 = 掩码 ∧ 当日收益非缺失的
        股票数；False = 逐字复现旧 ``build_pool_equal_weight_benchmark``
        （分子 rets.where(tradeable).sum、分母 tradeable.sum，含旧口径在
        收益缺失股上的轻微下偏）。
    """
    rets = px_wide.pct_change(fill_method=None)
    if fix_mask and signal_wide is not None:
        mask = (
            tradeable.reindex_like(rets).fillna(False)
            & signal_wide.reindex_like(rets).notna().fillna(False)
        )
        denom = (mask & rets.notna()).sum(axis=1).replace(0, np.nan)
        bench = (rets.where(mask).sum(axis=1) / denom).dropna()
    else:
        masked = rets.where(tradeable)
        bench = (
            masked.sum(axis=1) / tradeable.sum(axis=1).replace(0, np.nan)
        ).dropna()
    bench.name = "bench_ew_v2"
    return bench


def attach_benchmark_v2(daily_ret: pd.Series, benchmark_ret: pd.Series) -> pd.Series:
    """组合日收益 − 基准日收益 = 超额日收益（索引对齐）。"""
    common = daily_ret.index.intersection(benchmark_ret.index)
    excess = daily_ret.loc[common] - benchmark_ret.loc[common]
    excess.name = "excess_ret_v2"
    return excess


@dataclass
class PerfStatsV2:
    """组合绩效（与旧 PerfStats 同字段，年化常量随修正 (6) 可切换）。"""

    name: str
    n_days: int
    aer: float
    ir: float
    max_drawdown: float
    daily_turnover: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_days": self.n_days,
            "aer": self.aer,
            "ir": self.ir,
            "max_drawdown": self.max_drawdown,
            "daily_turnover": self.daily_turnover,
        }


def compute_perf_v2(
    excess_ret: pd.Series,
    trades: TradeLogV2 | None,
    *,
    name: str,
    fix_annualization_252: bool = True,
) -> PerfStatsV2:
    """从超额日收益与换手日志算 AER / IR / 最大回撤 / 日均换手（修正 (6)）。

    ``fix_annualization_252`` 开：AER = ``nav^(252/n)−1``、IR = ``μ/σ×√252``；
    关：逐字复现旧口径 244。
    """
    days = TRADING_DAYS_PER_YEAR_V2 if fix_annualization_252 else _LEGACY_TRADING_DAYS_PER_YEAR
    valid = excess_ret.dropna()
    n = len(valid)
    if n == 0:
        return PerfStatsV2(name, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    nav = (1 + valid).cumprod()
    n_years = n / days
    aer = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else float("nan")
    mu = float(valid.mean())
    sd = float(valid.std(ddof=1))
    ir = mu / sd * np.sqrt(days) if sd > 0 else float("nan")
    dd = float((nav / nav.cummax() - 1).min())
    if trades is not None and len(trades.rows):
        daily_to = float(trades.to_frame()["turnover_one_side"].mean())
    else:
        daily_to = float("nan")
    return PerfStatsV2(name=name, n_days=n, aer=aer, ir=ir, max_drawdown=dd, daily_turnover=daily_to)
