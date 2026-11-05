import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import pytest


def module():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/evaluate.py';assert p.exists(),'risk score missing'
 spec=importlib.util.spec_from_file_location('risk_eval',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_zero_penalty_is_exact_original_and_risk_direction():
 m=module();g=pd.Series([20.,4.,10.],index=['b','a','c']);q=pd.Series([-.4,.1,-.1],index=g.index)
 pd.testing.assert_series_equal(m.risk_adjusted_score(g,q,0),g)
 out=m.risk_adjusted_score(g,q)
 np.testing.assert_allclose(out.to_numpy(),[.75,1/3-.25/3,2/3-.25*2/3])
 with pytest.raises(ValueError):m.risk_adjusted_score(g,q.drop('b'))


def test_constant_risk_preserves_order_and_annualization():
 m=module();g=pd.Series([1.,3.,2.]);q=pd.Series([-1.,-1.,-1.])
 assert m.risk_adjusted_score(g,q).sort_values().index.equals(g.sort_values().index)
 assert m.annualized_return(np.full(252,.001))==pytest.approx(1.001**252-1)


def test_execution_new_ties_use_code_not_sort_backend():
 m=module();s=pd.Series([1.,1.,2.],index=['000002.SZ','000001.SZ','000003.SZ'])
 assert hasattr(m,'execution_score'),'explicit stable tie-break missing'
 x=m.execution_score(s)
 assert x.is_unique
 assert x.sort_values(ascending=False).index.tolist()==['000003.SZ','000001.SZ','000002.SZ']


def test_report_distinguishes_strategy_and_learned_increment():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/report.py';assert p.exists(),'report gates missing'
 spec=importlib.util.spec_from_file_location('risk_report',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 a={'annualized':.2,'sharpe':2.,'mdd':-.1};b={'annualized':.1,'sharpe':1.,'mdd':-.2};c={'annualized':.3,'sharpe':3.,'mdd':-.05}
 out=m.economic_gate({w:{'learned':a,'G1_original':b,'MC40':b,'volatility':c} for w in ['W3','W4']})
 assert out['strategy_joint_passed'] and not out['learned_increment_passed'] and not out['all_passed']


def test_bootstrap_recomputes_risk_statistics():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/report.py';assert p.exists(),'risk interval implementation missing'
 spec=importlib.util.spec_from_file_location('risk_report2',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 x=np.array([.01,-.02,.03,.01,-.01]*6)
 result=m.paired_intervals(x,x)
 for key in ['net_sum','annualized','sharpe','mdd']:
  assert result[key]['observed']==0 and result[key]['ci95']==[0.,0.]
