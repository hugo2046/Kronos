import importlib.util
from pathlib import Path
import torch
import pytest


def module():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/model.py';assert p.exists(),'risk head missing'
 spec=importlib.util.spec_from_file_location('risk_model',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_head_initialization_order_and_backprop():
 m=module();q=torch.tensor([-.2,.01,.3]);head=m.QuantileHead(4,q)
 h=torch.ones(2,4);out=head(h)
 torch.testing.assert_close(out,q.expand(2,3),rtol=0,atol=0)
 m.pinball(out,torch.tensor([.5,-.4])).mean().backward()
 assert head.linear.weight.grad.abs().sum()>0
 with torch.no_grad():head.linear.bias.copy_(torch.tensor([3.,-2.,1.]))
 assert torch.all(head(h)[:,1:]>=head(h)[:,:-1])


def test_pinball_hand_values_and_shape():
 m=module();loss=m.pinball(torch.zeros(2,3),torch.tensor([1.,-1.]))
 torch.testing.assert_close(loss,torch.tensor([[.1,.5,.9],[.9,.5,.1]]))
 with pytest.raises(ValueError):m.pinball(torch.zeros(2,3),torch.ones(2,1))


def test_feature_preprocess_has_only_history_and_matches_hand_values():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/features.py';assert p.exists(),'feature extraction missing'
 spec=importlib.util.spec_from_file_location('risk_features',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 import numpy as np
 import pandas as pd
 raw=np.ones((1,90,6));raw[0,:,0]=np.arange(90)
 x,ts=m.preprocess(raw,pd.bdate_range('2024-01-02',periods=90))
 assert x.shape==(1,90,6) and ts.shape==(1,90,5)
 assert np.array_equal(x[0,:,1:],np.zeros((90,5)))
 assert float(x[0,:,0].mean())==pytest.approx(0,abs=1e-7)
 with pytest.raises(ValueError):m.preprocess(raw[:,:89],pd.bdate_range('2024-01-02',periods=90))


def test_preprocess_uses_original_g1_float32_normalization():
 p=Path(__file__).resolve().parents[1]/'g1_risk_head/features.py'
 spec=importlib.util.spec_from_file_location('risk_features32',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 import numpy as np
 import pandas as pd
 raw=np.random.default_rng(5).normal(1e7,2,size=(1,90,6))
 original=raw[0].astype(np.float32)
 expected=np.clip((original-original.mean(0))/(original.std(0)+1e-5),-5,5)
 x,_=m.preprocess(raw,pd.bdate_range('2024-01-02',periods=90))
 np.testing.assert_array_equal(x[0],expected)
