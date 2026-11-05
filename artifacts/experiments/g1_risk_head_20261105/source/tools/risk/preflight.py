"""A阶段只读真实入口核验；不加载模型、不启动训练。"""
from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from state_adapter import tta_runtime as runtime
from state_adapter.tta import atomic_json,sha
from kronos_qlib.windows import build_inference_windows
from g1_risk_head.data import history_scale,make_labels
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/experiments/g1_risk_head_20261105'
runtime.OUT=OUT
prior=Path('/home/user/workspace/Kronos-mc40-tta/artifacts/experiments/mc40_tta_research_20261104/protocol.json')
protocol=json.loads(prior.read_text())
identities={path:sha(path) for path in protocol['reference_files']}
assert identities==protocol['reference_files']
# 不读取旧共享行情缓存或旧pkl；在新进程中经受审逐时点原始SQL读取。
records=[]
with runtime.live_provider() as provider:
    calendar=provider.trading_days()
    dates=['2014-01-02','2024-12-17','2025-02-05','2025-04-01','2025-07-01','2025-12-31','2026-01-05','2026-07-24']
    for date in dates:
        cutoff=('2024-12-31' if date<'2025-01-01' else '2025-03-31' if date<'2025-04-01' else '2025-06-30' if date<'2025-07-01' else '2025-12-31' if date<'2026-01-01' else '2026-07-24')
        with provider.at(date):
            frames,xt,yt,codes,stats=build_inference_windows(provider,date)
        arr=np.stack([f[['open','high','low','close','volume','amount']].to_numpy() for f in frames])
        scale=history_scale(arr[:,:,3])
        target=pd.Timestamp(yt[0].iloc[-1])
        count=None
        if target<=pd.Timestamp(cutoff):
            with provider.at(cutoff):
                raw=provider.read(codes,['$open','$close'],str(pd.Timestamp(yt[0].iloc[0]).date()),str(target.date()))
            count=0
            for code in codes:
                if code not in raw.index.get_level_values('instrument'):continue
                label=make_labels(raw.xs(code,level='instrument'),pd.DatetimeIndex([date]),calendar,pd.Timestamp(cutoff))
                count+=int(label.valid.sum())
        record=dict(date=date,stocks=len(codes),input_end=str(max(pd.to_datetime(x).max() for x in xt)),scale_min=float(scale.min()),
                    target_end=str(target.date()),label_boundary=cutoff,mature_labels=count,stats=stats,
                    code_sha=hashlib.sha256(json.dumps(codes).encode()).hexdigest())
        if date in ['2025-07-01','2025-12-31','2026-01-05','2026-07-24']:
            w='W3' if date<'2026-01-01' else 'W4'
            from state_adapter.mc40 import _wide
            base=_wide(protocol['reference_paths'][w]['g142'])
            expected=base.loc[pd.Timestamp(date)].dropna().index
            assert set(codes)==set(expected)
            record['original_grid_exact']=True
        records.append(record);print(date,len(codes),'labels',count,flush=True)
atomic_json(OUT/'A_readonly_preflight.json',dict(at=time.time(),source_commit='047ac95',frozen_references=identities,
    sampled_inputs=records,max_read=str(provider.max_read),shared_market_cache_used=False,model_loaded=False,gpu_used=False,
    availability='current-final assumed; historical vintage not verified',
    pending=['Q2 complete original G1 score source unresolved; request permission for bounded generation',
             'full grid/feature identity/lambda0 account parity and independent code review not yet passed']))
