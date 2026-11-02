"""成交限制单位回归：直接调用真实Exchange判定。"""
import numpy as np
import pandas as pd
from qlib.backtest.exchange import Exchange


def flags(threshold):
    ex = Exchange.__new__(Exchange)
    pct = np.array([-1.0531, -1.0531, -10., 10., -1., 1., 0.])
    ex.quote_df = pd.DataFrame({'$close': [51.196499, 2., 90., 110., 99., 101., np.nan],
                              '$change': [-.46, -.02, -10., 10., -1., 1., 0.]})
    if isinstance(threshold, tuple):
        ex.quote_df[threshold[0]] = pct / 100 >= .095
        ex.quote_df[threshold[1]] = pct / 100 <= -.095
    ex._update_limit(threshold)
    return ex.quote_df[['limit_buy','limit_sell']]


def test_old_amount_bug():
    old = flags(.095)
    assert old.limit_sell.iloc[0] and not old.limit_sell.iloc[1]


def test_correct_units():
    from state_adapter.limit_unit_audit import LIMIT
    actual = flags(LIMIT)
    assert actual.values.tolist() == [[False,False],[False,False],[False,True],
                                     [True,False],[False,False],[False,False],[True,True]]


def test_contribution_empty_first_day():
    from state_adapter.limit_unit_report import contribution_delta
    dates=pd.date_range('2025-07-01',periods=2)
    old=pd.DataFrame({'X':[.01]},index=dates[1:])
    new=pd.DataFrame({'X':[.02]},index=dates[1:])
    result=contribution_delta(new,old,dates)
    assert result.index.equals(dates)
    np.testing.assert_allclose(result.sum(axis=1),[0.,.01])


def test_real_samples():
    from pathlib import Path
    from state_adapter.limit_unit_audit import LIMIT
    frame=pd.read_csv(Path('artifacts/experiments/limit_unit_audit_20261102/fixed_sample.csv'))
    rows=frame[frame.datetime.eq('2025-08-14')]
    assert len(rows)==2
    old=Exchange.__new__(Exchange); old.quote_df=rows.copy(); old._update_limit(.095)
    new=Exchange.__new__(Exchange); new.quote_df=rows.copy(); new._update_limit(LIMIT)
    assert old.quote_df.limit_sell.all()
    assert not new.quote_df[['limit_buy','limit_sell']].any().any()


def test_direction_and_missing_quote():
    from qlib.backtest.decision import Order
    class Quote:
        def get_all_stock(self): return ['up','down','suspended']
        def get_data(self,code,start,end,field,method=None):
            return {'$close':np.nan if code=='suspended' else 100.,'limit_buy':code=='up','limit_sell':code=='down'}[field]
    ex=Exchange.__new__(Exchange); ex.quote=Quote()
    date=pd.Timestamp('2025-08-14')
    assert not ex.is_stock_tradable('up',date,date,Order.BUY)
    assert ex.is_stock_tradable('up',date,date,Order.SELL)
    assert not ex.is_stock_tradable('up',date,date,None)
    assert not ex.is_stock_tradable('down',date,date,Order.SELL)
    assert ex.is_stock_tradable('down',date,date,Order.BUY)
    for code in ['missing','suspended']:
        for direction in [None,Order.BUY,Order.SELL]: assert not ex.is_stock_tradable(code,date,date,direction)


def test_plot_rejects_mixed_execution(tmp_path, monkeypatch):
    import json
    import pytest
    import state_adapter.limit_unit_report as report
    from state_adapter.limit_unit_audit import sha
    monkeypatch.setattr(report,'OUT',tmp_path)
    monkeypatch.setattr(report,'SRC',tmp_path)
    directory=tmp_path/'accounts/new/W3/D'; directory.mkdir(parents=True)
    frame=pd.DataFrame({'return':[0.], 'cost':[0.]})
    frame.to_parquet(directory/'report.parquet')
    frame.to_parquet(tmp_path/'signal_W3_D.parquet')
    metadata=dict(execution_id='old',signal_sha256=sha(tmp_path/'signal_W3_D.parquet'),report_sha256=sha(directory/'report.parquet'))
    (directory/'complete.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='口径或信号SHA'): report.load('new','W3','D')
    metadata['execution_id']='new'; metadata['signal_sha256']='wrong'
    (directory/'complete.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='口径或信号SHA'): report.load('new','W3','D')
