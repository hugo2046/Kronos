"""只观测Qlib真实账户和成交，执行W3损益及风险归因。"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from state_adapter.factorial_run import OUT, MC, DATE, load, save
from state_adapter.mc40 import _wide


def stock_pnl(start_amount: float, start_price: float, close: float,
              trades: list[tuple[float, float]]) -> float:
    """按旧持仓估值变动与成交相对收盘价计算毛损益。

    :param start_amount: 日初股数。
    :param start_price: 前日估值价格。
    :param close: 当日估值价格。
    :param trades: 带符号股数、执行价格；买正卖负。
    :returns: 未扣成本的货币损益。
    """
    return start_amount * (close - start_price) + sum(qty * (close - price) for qty, price in trades)


def drawdown_episode(net: pd.Series) -> dict[str, Any]:
    """定位包含初始净值1的最大回撤区间。

    :param net: 日期索引日净收益。
    :returns: 峰谷、恢复、持续期与MDD。
    """
    values = np.r_[1., (1 + net).cumprod().values]
    dd = values / np.maximum.accumulate(values) - 1
    trough = int(np.argmin(dd)); peak = int(np.argmax(values[:trough + 1]))
    hits = np.flatnonzero(values[trough + 1:] >= values[peak])
    recovery = trough + 1 + int(hits[0]) if len(hits) else None
    labels = ['初始净值1'] + [str(d.date()) for d in net.index]
    return dict(mdd=float(dd[trough]), peak=labels[peak], trough=labels[trough],
                recovery=labels[recovery] if recovery is not None else None,
                peak_to_trough_sessions=trough - peak,
                underwater_sessions=(recovery if recovery is not None else len(net)) - peak,
                interval_dates=[str(d.date()) for d in net.index[peak:trough]])


def daily_diagnostics(reports: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """保存日报层收益/成本恒等分解与后验风险诊断。

    :param reports: 原G1、B和D同轴日报。
    :returns: 可序列化诊断结果。
    :raises ValueError: 日期轴不同。
    """
    from state_adapter.date_balanced_run import metrics
    base = reports['D'].index
    if any(not r.index.equals(base) for r in reports.values()):
        raise ValueError('日报必须使用相同时间索引')
    summary: dict[str, Any] = {}
    for arm, report in reports.items():
        net = report['return'] - report.cost
        summary[arm] = dict(metrics(report), mean=float(net.mean()), sample_std=float(net.std(ddof=1)),
                            annual_vol=float(net.std(ddof=1) * np.sqrt(252)), drawdown=drawdown_episode(net))
    for ref in ['B', 'G1_original']:
        d, b = reports['D'], reports[ref]
        gross = d['return'] - b['return']; savings = b.cost - d.cost
        delta = (d['return'] - d.cost) - (b['return'] - b.cost)
        np.testing.assert_allclose(delta, gross + savings, atol=1e-15, rtol=0)
        largest = delta.abs().nlargest(5).index
        summary['D_minus_' + ref] = dict(gross_pp=float(gross.sum() * 100), cost_saving_pp=float(savings.sum() * 100),
            net_pp=float(delta.sum() * 100), positive_days=int((delta > 0).sum()), negative_days=int((delta < 0).sum()),
            positive_sum_pp=float(delta[delta > 0].sum() * 100), negative_sum_pp=float(delta[delta < 0].sum() * 100),
            best5={str(k.date()): float(v * 100) for k, v in delta.nlargest(5).items()},
            worst5={str(k.date()): float(v * 100) for k, v in delta.nsmallest(5).items()},
            excluded_abs5=[str(d.date()) for d in largest], without_abs5_pp=float(delta.drop(largest).sum() * 100),
            mean_ratio=summary['D']['mean'] / summary[ref]['mean'],
            volatility_ratio=summary['D']['sample_std'] / summary[ref]['sample_std'])
        pd.DataFrame(dict(gross_delta=gross, cost_saving=savings, net_delta=delta)).to_csv(OUT / f'risk/daily_difference_D_{ref}.csv')
    summary['same_drawdown_intervals'] = {}
    for owner, stats in list(summary.items()):
        if owner not in reports:
            continue
        dates = pd.to_datetime(stats['drawdown']['interval_dates'])
        summary['same_drawdown_intervals'][owner] = {arm: dict(net_sum=float((r['return'] - r.cost).loc[dates].sum()),
          compound=float((1 + (r['return'] - r.cost).loc[dates]).prod() - 1)) for arm, r in reports.items()}
    return summary


def replay_account(signal: pd.Series, start: str, end: str, directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只读观察账户日终回调，导出实际持仓、成交和估值损益。

    原方法完整调用并原样返回；只读取已执行的trade_info，排除策略模拟订单。

    :param signal: Qlib信号。
    :param start: 起日。
    :param end: 末日，必须不晚于封存线。
    :param directory: 本轮导出目录。
    :returns: 原Qlib日报及元数据。
    :raises ValueError: 日期越过封存线。
    """
    if pd.Timestamp(end) > pd.Timestamp('2026-07-24'):
        raise ValueError('禁止读取封存行情')
    from qlib.backtest.account import Account
    from qlib_mean_comparison.run import run_qlib_backtest
    original = Account.update_bar_end
    signature = inspect.signature(original)
    snapshots: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    balances: list[dict[str, Any]] = []
    prior: dict[str, Any] = {'cash': 100000000.}
    previous_nav = 100000000.

    def observe(account: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal prior, previous_nav
        bound = signature.bind(account, *args, **kwargs); bound.apply_defaults()
        params = bound.arguments
        result = original(account, *args, **kwargs)
        if not params['atomic'] or not account.is_port_metr_enabled():
            return result
        date = pd.Timestamp(params['trade_start_time']).normalize()
        if date > pd.Timestamp('2026-07-24'):
            raise ValueError('账户回调越过封存线')
        current = copy.deepcopy(account.current_position.position)
        nav = float(account.current_position.calculate_value())
        day_trades: dict[str, list[tuple[float, float]]] = {}
        costs: dict[str, float] = {}
        for order, value, cost, price in params['trade_info']:
            if float(value) <= 1e-5:
                continue
            signed = float(order.deal_amount) * (1 if int(order.direction) == 1 else -1)
            day_trades.setdefault(order.stock_id, []).append((signed, float(price)))
            costs[order.stock_id] = costs.get(order.stock_id, 0.) + float(cost)
            trades.append(dict(date=date, instrument=order.stock_id, quantity=signed, price=float(price),
                               value=float(value), cost=float(cost), previous_nav=previous_nav))
        codes = sorted({k for p in [prior, current] for k, v in p.items() if isinstance(v, dict)} | set(day_trades))
        exchange = params['trade_exchange']
        for code in codes:
            old = prior.get(code, {}); new = current.get(code, {})
            q0, q1 = float(old.get('amount', 0)), float(new.get('amount', 0))
            p0 = float(old.get('price', 0))
            close = new.get('price')
            if close is None:
                close = exchange.get_close(code, params['trade_start_time'], params['trade_end_time'])
            if close is None or not np.isfinite(close):
                close = p0 if q0 else day_trades[code][-1][1]
            close = float(close)
            pnl = stock_pnl(q0, p0, close, day_trades.get(code, []))
            cost = costs.get(code, 0.)
            contributions.append(dict(date=date, instrument=code, pnl=pnl, cost=cost,
                gross_contribution=pnl / previous_nav, cost_contribution=cost / previous_nav,
                net_contribution=(pnl - cost) / previous_nav, previous_nav=previous_nav))
            snapshots.append(dict(date=date, instrument=code, start_amount=q0, end_amount=q1,
                start_price=p0, end_price=close, start_weight=q0 * p0 / previous_nav,
                end_weight=q1 * close / nav, previous_nav=previous_nav, nav=nav))
        balances.append(dict(date=date, previous_nav=previous_nav, nav=nav, start_cash=float(prior['cash']),
                             end_cash=float(current['cash']), cash_delay=float(current.get('cash_delay', 0.))))
        prior = current; previous_nav = nav
        return result

    Account.update_bar_end = observe
    try:
        from state_adapter.market_guard import market_guard
        with market_guard(directory / 'market_requests.jsonl'):
            report, meta = run_qlib_backtest(signal, start, end)
    finally:
        Account.update_bar_end = original
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in [('positions', snapshots), ('trades', trades), ('stock_daily_contributions', contributions), ('balances', balances)]:
        pd.DataFrame(rows).to_parquet(directory / f'{name}.parquet', index=False)
    report.to_parquet(directory / 'report.parquet')
    c = pd.DataFrame(contributions).groupby('date')[['gross_contribution', 'cost_contribution']].sum().reindex(report.index, fill_value=0)
    residual = pd.DataFrame({'gross_residual': report['return'] - c.gross_contribution,
                             'cost_residual': report.cost - c.cost_contribution})
    residual.to_csv(directory / 'reconciliation_residuals.csv')
    audit = dict(gross_max_abs=float(residual.gross_residual.abs().max()), cost_max_abs=float(residual.cost_residual.abs().max()),
                 gross_residual_sum=float(residual.gross_residual.sum()), cost_residual_sum=float(residual.cost_residual.sum()),
                 exact=bool(residual.abs().max().max() <= 1e-10), observation='Account.update_bar_end原样执行及透传返回；只读实际trade_info和估值后position')
    (directory / 'reconciliation_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return report, dict(meta, reconciliation=audit)


def main() -> None:
    """独立CPU进程重放三模型W3并保存风险证据。

    :returns: 无。
    :raises RuntimeError: 重放与冻结日报不一致。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()
    p, sha = load()
    if args.analysis_only:
        analyze_holdings()
        analyze_variance()
        return
    (OUT / 'risk').mkdir(exist_ok=True)
    reports = {arm: pd.read_parquet(path) for arm, path in {
        'G1_original': MC / 'report_W3_G1_original.parquet', 'B': MC / 'report_W3_State_MC40.parquet',
        'D': DATE / 'report_W3_Date_balanced_MC40.parquet'}.items()}
    save('risk/daily_diagnostics.json', daily_diagnostics(reports))
    audits = {}
    for arm, path in {'G1_original': Path(p['baseline']['signal_paths']['W3']),
                      'B': MC / 'signal_W3_State_MC40.parquet', 'D': DATE / 'signal_W3_Date_balanced_MC40.parquet'}.items():
        logger.info('重放W3真实账户 {}', arm)
        wide = _wide(str(path)); signal = wide.stack(); signal.index.names = ['datetime', 'instrument']
        report, meta = replay_account(signal.swaplevel().sort_index(), *p['windows']['W3'], OUT / 'risk' / arm)
        expected = reports[arm]
        if not report.index.equals(expected.index):
            raise RuntimeError('风险重放日期不匹配')
        delta = float(np.abs(report - expected).to_numpy().max())
        if delta > 1e-10:
            raise RuntimeError(f'{arm} 风险重放误差{delta}')
        audits[arm] = dict(replay_max_abs=delta, **meta)
        save('risk/replay_audit.json', audits)
    save('risk/capture_complete.json', dict(complete=True, protocol_sha=sha, arms=list(reports)))


def historical_exposures(close: pd.DataFrame) -> pd.DataFrame:
    """计算滞后一日的20/60日波动、60日动量和市场beta。

    :param close: 日期×股票收盘价，含000300.SH，末日不晚于封存线。
    :returns: 日期/股票索引暴露；缺失窗口保持NaN。
    :raises ValueError: 行情越过封存线或缺少市场序列。
    """
    if close.index.max() > pd.Timestamp('2026-07-24') or '000300.SH' not in close:
        raise ValueError('暴露行情越界或缺少市场序列')
    returns = close.pct_change(fill_method=None)
    market = returns['000300.SH']
    rows = []
    for code in close.columns:
        if code == '000300.SH':
            continue
        r = returns[code]
        frame = pd.DataFrame(dict(vol20=r.rolling(20).std(ddof=1) * np.sqrt(252),
            vol60=r.rolling(60).std(ddof=1) * np.sqrt(252), momentum60=close[code] / close[code].shift(60) - 1,
            beta60=r.rolling(60).cov(market) / market.rolling(60).var(ddof=1))).shift(1)
        frame['instrument'] = code; frame.index.name = 'date'
        rows.append(frame.reset_index())
    return pd.concat(rows, ignore_index=True).set_index(['date', 'instrument']).sort_index()


def analyze_holdings() -> None:
    """汇总真实持仓贡献、集中度及因果可用价量暴露。

    :returns: 无，写入risk目录。
    :raises RuntimeError: 缺少已核验账户产物或暴露数据不合规。
    """
    from kronos_qlib import QlibProvider
    from state_adapter.run import CensoredProvider
    root = OUT / 'risk'
    audit = json.loads((root / 'replay_audit.json').read_text())
    if any(not v['reconciliation']['exact'] for v in audit.values()):
        raise RuntimeError('账户未对账，不能声明精确归因')
    arms = ['G1_original', 'B', 'D']
    positions = {a: pd.read_parquet(root / a / 'positions.parquet') for a in arms}
    contrib = {a: pd.read_parquet(root / a / 'stock_daily_contributions.parquet').set_index(['date', 'instrument']) for a in arms}
    diag = json.loads((root / 'daily_diagnostics.json').read_text())
    codes = sorted(set().union(*(set(x.instrument) for x in positions.values())))
    provider = CensoredProvider(QlibProvider(instruments=codes + ['000300.SH'], start_date='2025-03-01', end_date='2025-12-31'))
    from state_adapter.market_guard import market_guard
    with market_guard(root / 'exposure_market_requests.jsonl'):
        frame = provider.fetch(['$close'])
    close = frame['close'].unstack('instrument').sort_index()
    close.to_parquet(root / 'exposure_close_history.parquet')
    exposures = historical_exposures(close)
    exposures.to_parquet(root / 'stock_historical_exposures.parquet')
    availability = dict(industry='缺项：未发现经过核验的历史行业数据接口，不用最新行业替代',
        float_market_cap='缺项：未发现经过核验的时点流通市值接口',
        max_market_date=str(provider.max_market_date), max_market_request_end=str(provider.max_market_request_end),
        exposure_timing='日初权重×截至前一交易日收盘的历史滚动暴露；缺失不填造值，另报可用权重',
        beta_model='仅60日沪深300简单beta，不是完整风险模型')
    save('risk/exposure_availability.json', availability)
    concentration = []; weighted = []
    for arm, pos in positions.items():
        data = pos.set_index(['date', 'instrument']).join(exposures)
        for date, day in data.groupby(level='date'):
            weights = day.start_weight
            rec = dict(date=date, arm=arm, hhi=float((weights ** 2).sum()), max_weight=float(weights.max()),
                       invested_weight=float(weights.sum()), cash_weight=float(1 - weights.sum()))
            concentration.append(rec)
            rec_exp: dict[str, Any] = dict(date=date, arm=arm)
            for field in ['vol20', 'vol60', 'momentum60', 'beta60']:
                valid = day[field].notna()
                rec_exp[field] = float((weights[valid] * day.loc[valid, field]).sum())
                rec_exp[field + '_covered_weight'] = float(weights[valid].sum())
            weighted.append(rec_exp)
    conc = pd.DataFrame(concentration); exp = pd.DataFrame(weighted)
    conc.to_csv(root / 'concentration_daily.csv', index=False)
    exp.to_csv(root / 'weighted_exposures.csv', index=False)
    summary: dict[str, Any] = dict(concentration=conc.groupby('arm').mean(numeric_only=True).to_dict('index'), comparisons={})
    for ref in ['B', 'G1_original']:
        delta = contrib['D'].net_contribution.subtract(contrib[ref].net_contribution, fill_value=0)
        delta.rename('net_difference').to_csv(root / f'stock_daily_difference_D_{ref}.csv')
        periods = {'full_W3': delta.index.get_level_values('date').unique(),
                   'D_drawdown': pd.to_datetime(diag['D']['drawdown']['interval_dates'])}
        results = {}
        for period, dates in periods.items():
            selected = delta.loc[delta.index.get_level_values('date').isin(dates)]
            by_stock = selected.groupby('instrument').sum().sort_values(ascending=False)
            by_stock.rename('net_difference').to_csv(root / f'stock_difference_D_{ref}_{period}.csv')
            absolute = by_stock.abs().sort_values(ascending=False); denominator = float(absolute.sum())
            results[period] = dict(total_pp=float(by_stock.sum() * 100), top10_pp=(by_stock.head(10)*100).to_dict(),
                bottom10_pp=(by_stock.tail(10).sort_values()*100).to_dict(),
                absolute_contribution_sum_pp=denominator * 100,
                absolute_top_shares={str(n): float(absolute.head(n).sum() / denominator) if denominator else 0. for n in [1, 5, 10]},
                signed_top_shares={str(n): float(by_stock.head(n).sum() / by_stock.sum()) if by_stock.sum() else None for n in [1, 5, 10]})
        dp = positions['D'].pivot(index='date', columns='instrument', values='start_weight').fillna(0)
        bp = positions[ref].pivot(index='date', columns='instrument', values='start_weight').fillna(0)
        dp, bp = dp.align(bp, fill_value=0)
        overlap = pd.DataFrame(dict(weight_overlap=np.minimum(dp, bp).sum(axis=1),
                                    jaccard=((dp > 0) & (bp > 0)).sum(axis=1) / ((dp > 0) | (bp > 0)).sum(axis=1)))
        overlap.to_csv(root / f'overlap_D_{ref}.csv')
        diff_exp = exp[exp.arm == 'D'].set_index('date').drop(columns='arm') - exp[exp.arm == ref].set_index('date').drop(columns='arm')
        diff_exp.to_csv(root / f'exposure_difference_D_{ref}.csv')
        results['overlap_mean'] = overlap.mean().to_dict()
        results['exposure_difference_mean'] = diff_exp.mean().to_dict()
        summary['comparisons']['D_minus_' + ref] = results
    for arm in arms:
        dates = pd.to_datetime(diag['D']['drawdown']['interval_dates'])
        c = contrib[arm]
        selected = c.loc[c.index.get_level_values('date').isin(dates)]
        (selected.net_contribution.groupby('instrument').sum().sort_values()*100).rename('net_contribution_pp').to_csv(root / f'D_drawdown_stock_contribution_{arm}.csv')
    save('risk/holdings_analysis.json', summary)
    save('risk/analysis_complete.json', dict(complete=True, arms=arms, historical_industry_available=False,
                                           historical_float_market_cap_available=False))


def variance_contributions(stocks: pd.DataFrame, net: pd.Series) -> pd.Series:
    """通过Cov(个股净贡献,组合净收益)分解样本方差，显式纳入残差。

    :param stocks: 同轴日个股净收益贡献，已含各股交易成本。
    :param net: Qlib日净收益。
    :returns: 每股方差贡献和残差项，加总等于组合方差。
    :raises ValueError: 日期或数值不合规。
    """
    if not stocks.index.equals(net.index) or not np.isfinite(stocks).all().all() or not np.isfinite(net).all():
        raise ValueError('方差归因必须同轴且有限')
    parts = stocks.copy()
    parts['__residual__'] = net - parts.sum(axis=1)
    return parts.apply(lambda column: column.cov(net))


def analyze_variance() -> None:
    """逐股方差及D-B方差恒等式核验并导出风险主导股票。

    :returns: 无。
    :raises RuntimeError: 方差贡献未对账。
    """
    root = OUT / 'risk'; results: dict[str, Any] = {}; variances = {}; nets = {}
    for arm in ['G1_original', 'B', 'D']:
        r = pd.read_parquet(root / arm / 'report.parquet'); net = r['return'] - r.cost; nets[arm] = net
        c = pd.read_parquet(root / arm / 'stock_daily_contributions.parquet')
        stocks = c.pivot(index='date', columns='instrument', values='net_contribution').fillna(0).reindex(net.index, fill_value=0)
        parts = variance_contributions(stocks, net).sort_values(ascending=False); variances[arm] = parts
        residual = float(parts.sum() - net.var(ddof=1))
        if abs(residual) > 1e-14:
            raise RuntimeError('方差贡献未对账')
        parts.rename('sample_variance_contribution').to_csv(root / f'variance_contributions_{arm}.csv')
        results[arm] = dict(sample_variance=float(net.var(ddof=1)), reconciliation_residual=residual,
                           top10=parts.head(10).to_dict(), top1_5_10_shares={str(n): float(parts.head(n).sum()/net.var(ddof=1)) for n in [1, 5, 10]})
    for ref in ['B', 'G1_original']:
        diff = nets['D'] - nets[ref]
        base_var = float(nets[ref].var(ddof=1)); diff_var = float(diff.var(ddof=1)); twice_cov = float(2*nets[ref].cov(diff))
        attribution = variances['D'].subtract(variances[ref], fill_value=0).sort_values(ascending=False)
        attribution.rename('sample_variance_difference_contribution').to_csv(root / f'variance_difference_D_{ref}.csv')
        results['D_minus_' + ref] = dict(base_variance=base_var, difference_variance=diff_var,
            twice_covariance=twice_cov, reconstructed_D_variance=base_var+diff_var+twice_cov,
            variance_identity_residual=float(nets['D'].var(ddof=1) - base_var - diff_var - twice_cov),
            variance_increase=float(nets['D'].var(ddof=1)-base_var), top10_increase=attribution.head(10).to_dict(),
            bottom10_increase=attribution.tail(10).to_dict())
    save('risk/variance_analysis.json', results)


def variance_evidence_lines() -> list[str]:
    """将逐股方差对账结果转换为报告证据段落。

    :returns: Markdown段落。
    """
    root = OUT / 'risk'
    variance = json.loads((root / 'variance_analysis.json').read_text())
    lines = ['', '### 方差来源的精确恒等式', '',
             '以已扣交易成本的逐股日贡献计算Cov(个股贡献,组合日净收益)，另将账户对账残差作为独立项；加总应等于组合日净收益样本方差。以下是方差的会计分解，不是干预因果。', '',
             '|模型|日净收益样本方差|逐股方差对账残差|Top1/5/10方差贡献占比|', '|---|---:|---:|---|']
    for arm in ['G1_original', 'B', 'D']:
        r = variance[arm]
        lines.append(f'|{arm}|{r["sample_variance"]:.10f}|{r["reconciliation_residual"]:.3e}|{r["top1_5_10_shares"]}|')
    for ref in ['B', 'G1_original']:
        r = variance['D_minus_' + ref]
        lines += ['', f'相对{ref}：Var(D)=Var({ref})+Var(D−{ref})+2Cov({ref},D−{ref}) = {r["base_variance"]:.10f}+{r["difference_variance"]:.10f}+{r["twice_covariance"]:.10f}={r["reconstructed_D_variance"]:.10f}，恒等式残差{r["variance_identity_residual"]:.3e}。',
                  f'方差增量主要股票（样本方差单位）：{dict(list(r["top10_increase"].items())[:5])}。']
    return lines


if __name__ == '__main__':
    main()
