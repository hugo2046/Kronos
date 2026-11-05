from pathlib import Path
import sys,json,torch,numpy as np
root=Path('/home/user/workspace/Kronos-g1-risk-head');sys.path.insert(0,str(root))
from g1_risk_head import train as module
from g1_risk_head.runtime import OUT,freeze,Budget
from state_adapter.tta import atomic_json
p=freeze();budget=Budget();original=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False)
work=OUT/'verification/resume_production';work.mkdir(parents=True,exist_ok=True)
link=work/'features'
if not link.exists():link.symlink_to(OUT/'features',target_is_directory=True)
assert not (work/'latest.pt').exists(), 'verification already ran; inspect evidence rather than overwrite'
module.OUT=work
class StopAtCheckpoint(Exception):pass
def stop(stage,**kwargs):
 if stage=='training' and kwargs.get('epoch')==1:raise StopAtCheckpoint()
module.status=stop
try:module.train(p,budget)
except StopAtCheckpoint:pass
partial=torch.load(work/'latest.pt',map_location='cpu',weights_only=False);assert partial['epoch']==1
module.status=lambda *a,**k:None
module.train(p,budget)
resumed=torch.load(work/'latest.pt',map_location='cpu',weights_only=False)
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
checks={key:equal(original[key],resumed[key]) for key in ['epoch','head','optimizer','rng','history','best','bad']}
assert all(checks.values()), checks
atomic_json(OUT/'training_resume_audit.json',dict(passed=True,scope='production train interrupted after epoch1 atomic checkpoint, resumed to epoch3; research outputs untouched',checks=checks))
print('PRODUCTION TRAIN RESUME PASS',checks)
