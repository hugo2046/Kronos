"""从落盘账户与历史价格独立复算风险证据，不调用行情服务或CUDA。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def main() -> None:
    """独立核对逐股方差、统一回撤区间和历史暴露。

    :returns: 无，写入同目录round6_risk_audit.json。
    :raises AssertionError: 任一原始产物与报告证据不符。
    """
    out = Path(__file__).resolve().parent.parent
    root = out / 'risk'
    result: dict[str, Any] = {'at': datetime.now(timezone.utc).isoformat()}
    diagnostics = json.loads((root / 'daily_diagnostics.json').read_text())
    holdings = json.loads((root / 'holdings_analysis.json').read_text())
    nets: dict[str, pd.Series] = {}
    contributions: dict[str, pd.Series] = {}
    result['arms'] = {}
    for arm in ['G1_original', 'B', 'D']:
        report = pd.read_parquet(root / arm / 'report.parquet')
        net = report['return'] - report['cost']
        nets[arm] = net
        stocks = pd.read_parquet(root / arm / 'stock_daily_contributions.parquet')
        contributions[arm] = stocks.set_index(['date', 'instrument']).net_contribution
        matrix = stocks.pivot(index='date', columns='instrument', values='net_contribution').fillna(0).reindex(net.index, fill_value=0)
        matrix['__residual__'] = net - matrix.sum(axis=1)
        x = matrix.to_numpy()
        y = net.to_numpy()
        covariance = (x - x.mean(axis=0)).T @ (y - y.mean()) / (len(y) - 1)
        stored = pd.read_csv(root / f'variance_contributions_{arm}.csv', index_col=0).iloc[:, 0].reindex(matrix.columns)
        error = float(np.max(np.abs(covariance - stored.to_numpy())))
        total_error = float(abs(covariance.sum() - np.var(y, ddof=1)))
        assert error < 1e-14 and total_error < 1e-14
        nav = np.r_[1., np.cumprod(1. + y)]
        drawdown = nav / np.maximum.accumulate(nav) - 1.
        trough = int(np.argmin(drawdown))
        peak = int(np.argmax(nav[:trough + 1]))
        dates = [None, *net.index.strftime('%Y-%m-%d')]
        assert dates[peak] == diagnostics[arm]['drawdown']['peak']
        assert dates[trough] == diagnostics[arm]['drawdown']['trough']
        result['arms'][arm] = dict(covariance_max_abs=error, variance_sum_error=total_error,
                                  peak=dates[peak], trough=dates[trough], mdd=float(drawdown[trough]))
    result['comparisons'] = {}
    for ref in ['B', 'G1_original']:
        delta = contributions['D'].subtract(contributions[ref], fill_value=0)
        checks = {}
        for period in ['full_W3', 'D_drawdown']:
            dates = nets['D'].index if period == 'full_W3' else pd.to_datetime(diagnostics['D']['drawdown']['interval_dates'])
            selected = delta[delta.index.get_level_values('date').isin(dates)]
            actual = selected.groupby('instrument').sum()
            stored = pd.read_csv(root / f'stock_difference_D_{ref}_{period}.csv', index_col=0).iloc[:, 0]
            actual, stored = actual.align(stored, fill_value=0)
            error = float((actual - stored).abs().max())
            net_difference = float((nets['D'] - nets[ref]).loc[dates].sum())
            account_error = abs(float(actual.sum()) - net_difference)
            assert error < 1e-12 and account_error < 1e-12
            evidence = holdings['comparisons']['D_minus_' + ref][period]
            shares = {str(n): float(actual.abs().nlargest(n).sum() / actual.abs().sum()) for n in [1, 5, 10]}
            assert max(abs(shares[n] - evidence['absolute_top_shares'][n]) for n in shares) < 1e-12
            checks[period] = dict(stock_max_abs=error, account_error=account_error, net_difference_pp=net_difference * 100)
        result['comparisons'][ref] = checks
    close = pd.read_parquet(root / 'exposure_close_history.parquet')
    assert close.index.max() <= pd.Timestamp('2025-12-31')
    exposures = pd.read_parquet(root / 'stock_historical_exposures.parquet')
    # 原价格为float32；收益先保留原精度，再与pandas滚动统计一样升为float64。
    returns = (close.to_numpy()[1:] / close.to_numpy()[:-1] - 1.).astype(np.float64)
    market_index = close.columns.get_loc('000300.SH')
    errors: list[float] = []
    count = 0
    for (date, code), row in exposures.iterrows():
        if date < pd.Timestamp('2025-07-01'):
            continue
        i = close.index.get_loc(date)
        j = close.columns.get_loc(code)
        # 仅取date之前收盘所形成的收益，date当天价格不参与估计。
        r20, r60 = returns[i-21:i-1, j], returns[i-61:i-1, j]
        market = returns[i-61:i-1, market_index]
        values = [np.std(r20, ddof=1) * np.sqrt(252), np.std(r60, ddof=1) * np.sqrt(252),
                  close.iloc[i-1, j] / close.iloc[i-61, j] - 1.,
                  np.cov(r60, market, ddof=1)[0, 1] / np.var(market, ddof=1)]
        recorded = row[['vol20', 'vol60', 'momentum60', 'beta60']].to_numpy(dtype=float)
        assert np.array_equal(np.isfinite(values), np.isfinite(recorded))
        finite = np.isfinite(recorded)
        if finite.any():
            errors.append(float(np.max(np.abs(np.asarray(values)[finite] - recorded[finite]))))
        count += 1
    assert max(errors) < 1e-10
    result['exposures'] = dict(rows=count, max_abs=max(errors), max_price_date=str(close.index.max()),
                               timing='独立数组切片：全部取估值日前一交易日及以前')
    concentration = pd.read_csv(root / 'concentration_daily.csv', parse_dates=['date']).set_index(['arm', 'date'])
    weighted = pd.read_csv(root / 'weighted_exposures.csv', parse_dates=['date']).set_index(['arm', 'date'])
    position_errors: list[float] = []
    for arm in nets:
        positions = pd.read_parquet(root / arm / 'positions.parquet')
        for date, day in positions.groupby('date'):
            weights = day.set_index('instrument').start_weight
            expected = dict(hhi=float(weights @ weights), max_weight=float(weights.max()),
                            invested_weight=float(weights.sum()), cash_weight=float(1 - weights.sum()))
            for field, value in expected.items():
                position_errors.append(abs(value - concentration.loc[(arm, date), field]))
            observed = exposures.xs(date, level='date').reindex(weights.index)
            for field in ['vol20', 'vol60', 'momentum60', 'beta60']:
                valid = observed[field].notna()
                value = float(weights[valid] @ observed.loc[valid, field])
                position_errors.append(abs(value - weighted.loc[(arm, date), field]))
                covered = float(weights[valid].sum())
                position_errors.append(abs(covered - weighted.loc[(arm, date), field + '_covered_weight']))
    assert max(position_errors) < 1e-12
    result['holding_weighted_evidence'] = dict(max_abs=max(position_errors), checked_values=len(position_errors))
    result['status'] = 'passed'
    (out / 'verification/round6_risk_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info('W3独立风险复核通过：{}行历史暴露，最大误差{}', count, max(errors))


if __name__ == '__main__':
    main()
