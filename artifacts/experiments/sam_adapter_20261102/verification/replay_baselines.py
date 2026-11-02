"""本轮冻结参照信号实际CPU重放；不调用旧OUT的run。"""
import json
import time
from pathlib import Path
import pandas as pd
from state_adapter.sam_run import load,OUT,save
from state_adapter import run as legacy
from state_adapter.mc40 import _wide
from state_adapter.limit_unit_audit import execution
from state_adapter.risk_attribution import replay_account
p,sha=load();gates={}
for w,bounds in p['windows'].items():
    for tag,path in p['reference_signals'][w].items():
        if time.time()>=p['budget']['deadline']:raise RuntimeError('固定墙钟耗尽')
        directory=OUT/'accounts'/w/tag;directory.mkdir(parents=True,exist_ok=True)
        identity=dict(protocol_sha=sha,execution_id=p['execution_id'],signal_sha256=legacy._sha256_file(path))
        if (directory/'complete.json').exists():raise RuntimeError('已有产物，拒绝重复启动')
        wide=_wide(path);signal=wide.stack();signal.index.names=['datetime','instrument'];signal=signal.swaplevel().sort_index()
        with execution('new'):report,meta=replay_account(signal,*bounds,directory)
        assert meta['reconciliation']['exact']
        expected=pd.read_parquet(p['reference_reports'][w][tag]);pd.testing.assert_frame_equal(report,expected,atol=1e-10,rtol=0)
        meta.update(identity,report_sha256=legacy._sha256_file(directory/'report.parquet'))
        save(str((directory/'complete.json').relative_to(OUT)),meta)
        gates[f'{w}/{tag}']=float((report-expected).abs().max().max());save('reference_replay_gate.json',gates)
        print(w,tag,gates[f'{w}/{tag}'],flush=True)
