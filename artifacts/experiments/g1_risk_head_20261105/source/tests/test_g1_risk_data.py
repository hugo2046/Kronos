from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest


def module():
    p=Path(__file__).resolve().parents[1]/'g1_risk_head/data.py'
    assert p.exists(), 'risk data implementation missing'
    spec=importlib.util.spec_from_file_location('risk_data',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_labels_use_market_calendar_next_open_and_keep_missing():
    m=module();cal=pd.bdate_range('2024-01-02',periods=15)
    f=pd.DataFrame({'open':np.arange(15)+100.,'close':np.arange(15)+110.},index=cal)
    f=f.drop(cal[3])
    result=m.make_labels(f,cal[:2],cal,cal[10])
    assert result.index.equals(cal[:2])
    assert result.iloc[0]['y']==pytest.approx(120/101-1)
    assert pd.isna(result.iloc[1]['y'])
    assert result.iloc[0]['target_date']==cal[10]


def test_future_bad_open_only_masks_label():
    m=module();cal=pd.bdate_range('2024-01-02',periods=15)
    f=pd.DataFrame({'open':100.,'close':110.},index=cal);f.loc[cal[1],'open']=np.nan
    result=m.make_labels(f,cal[:2],cal,cal[-1])
    assert len(result)==2 and not result.iloc[0].valid and result.iloc[1].valid


def test_scale_uses_20_returns_and_rejects_bad_history():
    m=module();closes=np.ones((2,21))*10
    assert np.array_equal(m.history_scale(closes),[1e-4,1e-4])
    closes[0,-1]=0
    with pytest.raises(ValueError):m.history_scale(closes)
    with pytest.raises(ValueError):m.history_scale(np.ones((1,20)))


def test_asof_rejects_before_fetch_and_membership_future():
    from state_adapter.tta import AsOfProvider
    seen=[]
    p=AsOfProvider(lambda *a:seen.append(a),lambda:pd.bdate_range('2026-07-01',periods=30),lambda *a:['000001.SZ'])
    with p.at('2026-07-24'):
        with pytest.raises(ValueError):p.read(['000001.SZ'],['$close'],'2026-07-24','2026-07-25')
        with pytest.raises(ValueError):p.list_pool_at('csi300','2026-07-25')
    assert not seen
