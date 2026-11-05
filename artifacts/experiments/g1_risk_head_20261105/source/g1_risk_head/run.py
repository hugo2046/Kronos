"""本轮唯一执行入口；A审查与保真、B训练门禁、C经济评价顺序固定。"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from g1_risk_head.runtime import ROOT,OUT,freeze,require_gate,identity,sources,status,Budget,provider
from g1_risk_head.pipeline import load_backbone,collect,scheduled_dates
from g1_risk_head.features import preprocess
from g1_risk_head.train import train,require_mechanism
from state_adapter import tta_runtime as parent
from state_adapter.tta import atomic_json,frozen_hash
from kronos_qlib.windows import build_inference_windows,REQUIRED_COLS


def pilot(p: dict,budget: Budget,reader,backbone,predictor) -> None:
    """真实原G1输入与表示保真，测量预算可行性。

    :param p: 固定协议。
    :param budget: 累计预算。
    :param reader: 有界入口。
    :param backbone: 原G1。
    :param predictor: 原生生成器。
    :return: 无。
    """
    with reader.at('2025-04-01'):frames,xt,yt,codes,_=build_inference_windows(reader,'2025-04-01')
    raw=np.stack([f[REQUIRED_COLS].to_numpy() for f in frames[:64]])
    x,stamp=preprocess(raw,xt[0]);captured={};original=predictor.generate
    def capture(a,b,c,*args,**kwargs):
        captured.update(x=a.copy(),stamp=b.copy())
        return np.zeros((len(a),10,6),dtype=np.float32)
    predictor.generate=capture
    try:predictor.predict_batch(frames[:64],xt[:64],yt[:64],pred_len=10,sample_count=20,verbose=False)
    finally:predictor.generate=original
    np.testing.assert_array_equal(x,captured['x']);np.testing.assert_array_equal(stamp,captured['stamp'])
    before=(frozen_hash(backbone.kronos),frozen_hash(backbone.tokenizer));times=[]
    with budget.compute(),torch.no_grad():
        a=torch.tensor(x,device='cuda:0');b=torch.tensor(stamp,device='cuda:0')
        for _ in range(3):
            torch.cuda.synchronize();t=time.time();h=backbone.extract(a,b);torch.cuda.synchronize();times.append(time.time()-t)
        h2=backbone.extract(a,b)
        torch.testing.assert_close(h,h2,rtol=0,atol=0)
        s1,s2=backbone.tokenizer.encode(a,half=True);observed=[]
        hook=backbone.kronos.norm.register_forward_hook(lambda module,inputs,output:observed.append(output.detach().clone()))
        try:backbone.kronos(s1,s2,stamp=b,use_teacher_forcing=True,s1_targets=s1)
        finally:hook.remove()
        torch.testing.assert_close(h,observed[0],rtol=0,atol=0)
    with budget.compute():
        torch.manual_seed(42);t=time.time()
        predictor.predict_batch(frames[:32],xt[:32],yt[:32],pred_len=10,T=1,top_k=0,top_p=.9,sample_count=20,verbose=False)
        torch.cuda.synchronize();generation_seconds=time.time()-t
    assert before==(frozen_hash(backbone.kronos),frozen_hash(backbone.tokenizer))
    assert all(not v.requires_grad for v in backbone.parameters())
    counts={stage:len(scheduled_dates(reader,p,stage)) for stage in ['train','Q1','Q2','W3','W4']}
    estimate=sum(counts.values())*5*max(times)+counts['Q2']*10*generation_seconds
    conservative=estimate*1.5+600
    feasible=conservative+budget.state['gpu_seconds']<21600
    atomic_json(OUT/'A_model_pilot.json',dict(identity=dict(protocol=identity(p),sources=sources()),
        normalization_exact=True,native_hidden_exact=True,repeated_hidden_exact=True,frozen_hashes=before,
        history_only=True,feature_seconds64=times,generation_seconds32=generation_seconds,dates=counts,
        gpu_estimate_seconds=estimate,conservative_gpu_seconds=conservative,budget_feasible=feasible))
    if not feasible:raise RuntimeError('原GPU预算不可行，不缩样本或增预算')


def main() -> None:
    """执行已批准阶段，门禁失败不会默默进入后续。

    :return: 无。
    """
    args=argparse.ArgumentParser();args.add_argument('stage',choices=['pilot','features','train','eval_features','all']);stage=args.parse_args().stage
    # GPU操作前原子独占，预算时钟跨进程累计。
    with parent.worker_lock('g1-risk-'+stage):
        p=freeze();require_gate(p);budget=Budget();watch=budget.watchdog()
        try:
            budget.check()
            if stage in ['pilot','features','eval_features','all']:
                if stage=='eval_features':require_mechanism(p)
                with provider() as reader:
                    backbone,predictor=load_backbone(budget)
                    if stage in ['pilot','all']:pilot(p,budget,reader,backbone,predictor)
                    if stage in ['features','all']:
                        proof=json.loads((OUT/'A_model_pilot.json').read_text())
                        if proof['identity']!=dict(protocol=identity(p),sources=sources()) or not proof['budget_feasible']:raise RuntimeError('A模型门禁身份不符')
                        parity=json.loads((OUT/'A_account_parity.json').read_text())
                        if parity['identity']!=proof['identity'] or not parity['passed']:raise RuntimeError('A账户门禁不通过')
                        collect(reader,backbone,predictor,p,budget,['train','Q1','Q2'])
                    if stage=='eval_features':collect(reader,backbone,predictor,p,budget,['W3','W4'])
                    del backbone,predictor;budget.resident=None;torch.cuda.empty_cache()
            if stage in ['train','all']:
                require_gate(p)
                gate=train(p,budget)
                status('mechanism_passed' if gate['passed'] else 'mechanism_failed',selected_epoch=gate['selected_epoch'])
            elif stage=='pilot':status('A_model_passed')
            else:status(stage+'_complete')
        finally:watch.set()


if __name__=='__main__':main()
