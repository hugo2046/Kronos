import importlib.util
from pathlib import Path
import numpy as np


def module():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/diagnostics.py';assert p.exists(),'risk diagnostics missing'
 spec=importlib.util.spec_from_file_location('risk_diag',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_auc_ties_and_single_class():
 m=module()
 assert m.auc(np.array([0,1,0,1]),np.array([0.,1.,0.,1.]))==1
 assert m.auc(np.array([0,1]),np.array([.5,.5]))==.5
 assert np.isnan(m.auc(np.array([1,1]),np.array([.3,.7])))


def test_y_scale_pinball_date_equal_weight():
 m=module();y=np.array([1.,1.,1.]);pred=np.zeros((3,3));dates=np.array(['a','a','b'])
 result=m.summarize(y,pred,dates)
 assert result['pinball']==.5 and result['p10']==.1 and result['coverage10']==0


def test_selection_includes_initial_and_ties_choose_earliest():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/train.py';assert p.exists(),'training implementation missing'
 spec=importlib.util.spec_from_file_location('risk_train',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.best_epoch([.1,.2,.1])==0
 assert m.best_epoch([.3,.1,.1,.2])==1


def test_protocol_and_budget_cannot_reset(tmp_path):
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/runtime.py';assert p.exists(),'runtime missing'
 spec=importlib.util.spec_from_file_location('risk_runtime',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.budget_exhausted({'first_model_at':100.,'gpu_seconds':21601.,'active_since':None},101)
 assert m.budget_exhausted({'first_model_at':100.,'gpu_seconds':0.,'active_since':None},36101)
 assert not m.budget_exhausted({'first_model_at':None,'gpu_seconds':0.,'active_since':None},100000)


def test_cached_transaction_rejects_changed_bytes(tmp_path):
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/pipeline.py';assert p.exists(),'pipeline missing'
 spec=importlib.util.spec_from_file_location('risk_pipeline',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 import json,hashlib
 data=tmp_path/'day.npz';data.write_bytes(b'original')
 meta=tmp_path/'day.json';meta.write_text(json.dumps({'identity':{'protocol':'test'},'sha':hashlib.sha256(b'original').hexdigest()}))
 assert m.valid_transaction(data,{'protocol':'test'})
 data.write_bytes(b'changed')
 import pytest
 with pytest.raises(RuntimeError):m.valid_transaction(data,{'protocol':'test'})


def test_result_identity_rejects_stale_stage():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/train.py'
 spec=importlib.util.spec_from_file_location('risk_train_identity',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert hasattr(m,'validate_result_identity'),'B-to-C identity check missing'
 import pytest
 expected={'protocol':'new','sources':{'run.py':'current'},'feature_sha':{'day':'actual'}}
 with pytest.raises(RuntimeError):m.validate_result_identity({'passed':True,'identity':{'protocol':'old'}},{'identity':expected},expected)
 with pytest.raises(RuntimeError):m.validate_result_identity({'passed':True,'identity':expected},{'identity':{'protocol':'old'}},expected)
 m.validate_result_identity({'passed':True,'identity':expected},{'identity':expected},expected)


def test_calibration_top50_date_blocks():
 m=module();assert hasattr(m,'calibration_by_day'),'conditional calibration missing'
 import pandas as pd
 f=pd.DataFrame({'date':['a','a','b'],'y':[-2.,2.,0.],'q10':[-1.,-1.,-1.],'q50':[0.,0.,0.],'q90':[1.,1.,1.]})
 daily,stats=m.calibration_by_day(f,['q10','q50','q90'])
 assert len(daily)==2 and stats['coverage10']['mean']==.25
 assert len(stats['coverage10']['ci95'])==2


def test_long_call_guard_watchdog_exits(tmp_path):
 import subprocess,sys
 root=Path(__file__).resolve().parents[1]
 code='''from pathlib import Path
import time
from g1_risk_head import runtime as r
import state_adapter.run as old
r.parent.OUT=Path(__import__('sys').argv[1]);r.OUT=r.parent.OUT
old.in_registry_guard=lambda:True
b=r.Budget();b.begin();b.watchdog();time.sleep(4)
'''
 result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=root,timeout=10)
 assert result.returncode==75,'long calls must yield at reserved window'
