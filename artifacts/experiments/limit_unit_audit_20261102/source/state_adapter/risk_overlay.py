"""冻结D信号上的因果风险特征与组合订单约束，不改变预测。"""
from __future__ import annotations

import copy
from typing import Any
import numpy as np
import pandas as pd
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest.decision import Order, TradeDecisionWO


def build_risk_panel(close: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """按决策日计算风险，只使用当日及以前的61个收盘价。

    :param close: 交易日×股票历史收盘价，不做缺失填补。
    :param signals: 当日有效CSI300 PIT冻结信号。
    :returns: 日期、股票双索引风险面板。
    """
    if close.index.max() > pd.Timestamp('2026-07-24'):
        raise ValueError('行情越过封存线')
    close = close.sort_index().where(np.isfinite(close) & (close > 0))
    returns = close.pct_change(fill_method=None)
    sigma = returns.rolling(60, min_periods=60).std(ddof=1)
    vol20 = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    beta = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    if '000300.SH' in returns:
        market = returns['000300.SH']
        beta = returns.rolling(60).cov(market).div(market.rolling(60).var(), axis=0)
    rows = []
    # 估值允许使用最后已知价格，但不将填充值用于收益/风险计算。
    known = close.where(np.isfinite(close) & (close > 0)).ffill()
    last = pd.DataFrame({c: pd.Series(close.index.where(close[c].notna()), index=close.index).ffill() for c in close})
    for date, scores in signals.iterrows():
        valid = scores.index[np.isfinite(scores)]
        if date not in close.index:
            raise ValueError(f'信号日无历史日历: {date}')
        s = sigma.loc[date].reindex(valid)
        median = float(s.dropna().median()) if s.notna().any() else np.nan
        # 保存所有历史股票，持仓已退出信号池时仍可取得历史估值；中位数只由有效池计算。
        all_s = sigma.loc[date]
        all_f = np.sqrt(median / all_s.clip(lower=1e-8)).clip(.5, 1.).fillna(.5)
        frame = pd.DataFrame(dict(sigma=all_s, median_sigma=median, factor=all_f,
            price=known.loc[date], last_history_date=last.loc[date], vol20=vol20.loc[date],
            vol60=all_s*np.sqrt(252), beta60=beta.loc[date]))
        frame['signal_valid'] = frame.index.isin(valid)
        # 信号资格仅限定横截面中位数；退出池的存量持仓仍使用其合法历史风险。
        frame['missing_risk'] = frame.sigma.isna()
        frame['date'] = date; frame['instrument'] = frame.index
        rows.append(frame.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True).set_index(['date','instrument']).sort_index()


def constrain_orders(orders: list[Order], position: Any, exchange: Any, panel: pd.DataFrame,
                     mode: str, start: pd.Timestamp, end: pd.Timestamp, freq: str,
                     hold_thresh: int, forbid_all_trade_at_limit: bool) -> tuple[list[Order], list[dict[str, Any]]]:
    """约束原订单数量，合并部分减仓并保留原交易所成交规则。

    :param orders: super策略原订单。
    :param position: 日初真实账户。
    :param exchange: 原交易所。
    :param panel: 截至前一交易日的风险面板。
    :param mode: neutral/cap_only/vol_only/cap_vol。
    :param start: 执行日起始时间。
    :param end: 执行日结束时间。
    :param freq: 持仓计数频率。
    :param hold_thresh: 最短持有计数。
    :param forbid_all_trade_at_limit: 原策略涨跌停规则。
    :returns: 新订单及逐笔约束/未执行原因审计。
    """
    if mode == 'neutral':
        return orders, []
    if mode not in {'cap_only','vol_only','cap_vol'}:
        raise ValueError(mode)
    use_cap = mode in {'cap_only','cap_vol'}
    nav = float(position.calculate_value())
    held = set(position.get_stock_list())
    sells = {o.stock_id: copy.deepcopy(o) for o in orders if o.direction == Order.SELL}
    audit = []
    def features(code: str) -> tuple[float,float]:
        if code not in panel.index: return np.nan, .5 if mode != 'cap_only' else 1.
        row = panel.loc[code]
        return float(row.price), 1. if mode == 'cap_only' else float(row.factor)
    def rounded(code: str, amount: float) -> float:
        factor = exchange.get_factor(stock_id=code, start_time=start, end_time=end)
        return float(exchange.round_amount_by_trade_unit(max(0.,amount), factor))
    for code in sorted(held) if use_cap else []:
        amount = float(position.get_stock_amount(code)); price, f = features(code)
        old = float(sells[code].amount) if code in sells else 0.
        count = position.get_stock_count(code, bar=freq)
        row = dict(date=start.normalize(), instrument=code, kind='risk_sell', before=old, after=old,
                   extra=0., price=price, factor=f, cap=.03*f, hold_count=count, rejected=False)
        if old >= amount: continue
        if not np.isfinite(price) or price <= 0 or not np.isfinite(nav) or nav <= 0:
            row['reason']='missing_valuation'; audit.append(row); continue
        desired = max(0., amount - .03*f*nav/price)
        if desired <= old: continue
        row['weight']=amount*price/nav
        if count < hold_thresh:
            row['reason']='hold_thresh'; audit.append(row); continue
        if not exchange.is_stock_tradable(stock_id=code,start_time=start,end_time=end,
                direction=None if forbid_all_trade_at_limit else Order.SELL):
            row.update(reason='not_tradable',rejected=True)
            # 只记录原交易所既有可卖标记，不用这些当日字段估计风险。
            if hasattr(exchange, 'check_stock_suspended'):
                row['exchange_suspended']=bool(exchange.check_stock_suspended(code,start,end))
                row['exchange_limit']=bool(exchange.check_stock_limit(code,start,end,None if forbid_all_trade_at_limit else Order.SELL))
            audit.append(row); continue
        target = min(amount, max(old, rounded(code, desired)))
        if target <= old:
            row['reason']='rounding'; audit.append(row); continue
        sell = Order(stock_id=code,amount=target,direction=Order.SELL,start_time=start,end_time=end)
        if not exchange.check_order(sell):
            row.update(reason='exchange_rejected',rejected=True); audit.append(row); continue
        sells[code] = sell
        row.update(after=target,extra=target-old,reason='cap_reduction')
        audit.append(row)
    # 排名卖出顺序不变，额外部分减仓置于原卖出之后；买单不使用减仓新增现金。
    result = [sells.pop(o.stock_id) for o in orders if o.direction==Order.SELL]
    result.extend(sells.values())
    sold = {o.stock_id:float(o.amount) for o in result}
    for original in orders:
        if original.direction != Order.BUY: continue
        code=original.stock_id; price,f=features(code)
        row=dict(date=start.normalize(),instrument=code,kind='buy',before=float(original.amount),price=price,
                 factor=f,cap=.03*f if use_cap else np.nan,extra=0.,rejected=False,hold_count=0)
        amount=float(original.amount)*f
        if not np.isfinite(price) or price<=0 or not np.isfinite(nav) or nav<=0:
            amount=0.; reason='missing_valuation'
        else:
            remaining=max(0.,float(position.get_stock_amount(code))-sold.get(code,0.)) if code in held else 0.
            if use_cap: amount=min(amount,max(0.,.03*f*nav/price-remaining))
            amount=rounded(code,amount); reason='discount_cap' if use_cap else 'discount'
        row.update(after=amount,reason=reason); audit.append(row)
        if amount>0:
            order=copy.deepcopy(original); order.amount=amount; result.append(order)
    for row in audit:
        row['before_value']=row['before']*row['price']; row['after_value']=row['after']*row['price']
    return result,audit


class RiskOverlayStrategy(TopkDropoutStrategy):
    """继承原排名决策，只在返回前约束数量。

    :param risk_panel: 日期、股票双索引的因果风险面板。
    :param mode: 预设约束模式。
    """
    def __init__(self, *args: Any, risk_panel: pd.DataFrame | None = None, mode: str = 'neutral', **kwargs: Any) -> None:
        """初始化原策略与风险审计。

        :param args: 原策略位置参数。
        :param risk_panel: 决策日风险面板。
        :param mode: 固定风险臂。
        :param kwargs: 原策略关键字参数。
        :returns: 无。
        """
        super().__init__(*args,**kwargs)
        self.risk_panel=risk_panel; self.mode=mode; self.order_audit=[]; self.raw_orders=[]; self.final_orders=[]
    def generate_trade_decision(self, execute_result: Any = None) -> TradeDecisionWO:
        """调用原策略后施加数量约束。

        :param execute_result: 引擎原执行反馈。
        :returns: 原生Qlib订单决策。
        """
        decision=super().generate_trade_decision(execute_result)
        step=self.trade_calendar.get_trade_step()
        start,end=self.trade_calendar.get_step_time(step)
        t,_=self.trade_calendar.get_step_time(step,shift=1)
        orders=decision.get_decision()
        def record(items: list[Order]) -> list[dict[str,Any]]:
            return [dict(date=start.normalize(),instrument=o.stock_id,direction=int(o.direction),amount=float(o.amount)) for o in items]
        self.raw_orders.extend(record(orders))
        if self.mode=='neutral':
            self.final_orders.extend(record(orders)); return decision
        # 首日可能尚无前日信号；原策略空决策且空仓时原样返回。
        if t.normalize() not in self.risk_panel.index.get_level_values('date'):
            if not orders and not self.trade_position.get_stock_list():
                return decision
            raise ValueError('非空账户/订单缺少决策日风险面板')
        panel=self.risk_panel.xs(t.normalize(),level='date')
        constrained,audit=constrain_orders(orders,self.trade_position,self.trade_exchange,panel,self.mode,start,end,
             self.trade_calendar.get_freq(),self.hold_thresh,self.forbid_all_trade_at_limit)
        for row in audit: row['decision_date']=t.normalize()
        self.order_audit.extend(audit); self.final_orders.extend(record(constrained))
        return TradeDecisionWO(constrained,self)


def acceptance(metrics: dict[str,dict[str,dict[str,float]]]) -> dict[str,Any]:
    """按预先固定的两窗联合门槛判定，不择优替换主候选。

    :param metrics: 各窗口各臂指标。
    :returns: 门槛明细与是否通过。
    """
    if set(metrics) != {'W3','W4'}:
        raise ValueError('联合门槛必须包含完整两窗')
    checks={}
    for window, arms in metrics.items():
        r,d=arms['cap_vol'],arms['D']
        checks[window]=dict(above_g1=r['net_sum']>arms['G1_original']['net_sum'],
            above_b=r['net_sum']>arms['B']['net_sum'],sharpe=r['sharpe']>=d['sharpe'],
            mdd=abs(r['mdd'])<=abs(d['mdd']))
    r,d=metrics['W3']['cap_vol'],metrics['W3']['D']
    strict=r['sharpe']>d['sharpe'] or abs(r['mdd'])<abs(d['mdd'])
    return dict(passed=bool(all(all(v.values()) for v in checks.values()) and strict),checks=checks,w3_strict=bool(strict))
