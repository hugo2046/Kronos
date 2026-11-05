from pathlib import Path
import sys,json,numpy as np
root=Path('/home/user/workspace/Kronos-g1-risk-head');sys.path.insert(0,str(root))
from g1_risk_head.runtime import OUT,freeze,identity,sources
from state_adapter import tta_runtime as parent
from state_adapter.tta import atomic_json
from state_adapter.mc40 import _wide
from kronos_qlib.windows import build_inference_windows,REQUIRED_COLS
from g1_risk_head.data import history_scale
p=freeze();parent.OUT=OUT/'coverage';summary={}
with parent.live_provider() as provider:
 for w in ['W3','W4']:
  base=_wide(p['reference_paths'][w]['g142']);rows=[]
  for date in base.index:
   with provider.at(date):frames,xt,yt,codes,_=build_inference_windows(provider,str(date.date()))
   assert set(codes)==set(base.loc[date].dropna().index)
   raw=np.stack([f[REQUIRED_COLS].to_numpy() for f in frames]);assert np.isfinite(raw).all()
   scale=history_scale(raw[:,:,3]);assert np.isfinite(scale).all()
   rows.append({'date':str(date.date()),'stocks':len(codes),'scale_min':float(scale.min())})
  assert len(rows)==p['expected'][w][0] and sum(x['stocks'] for x in rows)==p['expected'][w][1]
  summary[w]=rows
atomic_json(OUT/'A_full_input_grid.json',dict(passed=True,protocol_sha=identity(p),source_hashes=sources(),windows=summary,max_read=str(provider.max_read),model_loaded=False))
print('FULL INPUT GRID PASS',flush=True)
