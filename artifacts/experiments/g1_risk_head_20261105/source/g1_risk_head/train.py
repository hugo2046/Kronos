"""冻结特征上的唯一分位数头训练，Q1选点，Q2只做机制判断。"""
from __future__ import annotations
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import torch
from loguru import logger
from g1_risk_head.model import QuantileHead,pinball
from g1_risk_head.diagnostics import summarize,auc,calibration_by_day
from g1_risk_head.runtime import OUT,identity,sources,Budget,status
from g1_risk_head.pipeline import valid_transaction
from state_adapter.tta import atomic_json,sha


def best_epoch(losses: list[float]) -> int:
    """包含初始化点，损失并列选择最早。

    :param losses: 按epoch顺序的Q1损失。
    :return: 最早最小损失下标。
    """
    if not losses or not np.isfinite(losses).all():raise ValueError('选点损失非法')
    return int(np.argmin(losses))


def read_segment(stage: str,p: dict) -> dict[str,np.ndarray]:
    """验证原始事务与日历后装配监督有效数组。

    :param stage: 分段。
    :param p: 固定协议。
    :return: 隐状态、尺度、Y、日期、证券和可选G1信号。
    """
    folder=OUT/'features'/stage;cal=json.loads((folder/'calendar.json').read_text())
    expected=dict(protocol=identity(p),sources=sources());parts=[]
    assert cal['identity']==expected
    for date in cal['dates']:
        path=folder/f'{date}.npz'
        if not valid_transaction(path,expected):raise RuntimeError('特征日缺失')
        with np.load(path,allow_pickle=False) as d:
            ok=np.isfinite(d['y'])
            # 先用原点完整股票格选Top50，再应用监督mask。
            top=np.zeros(len(ok),dtype=bool)
            if stage=='Q2':
                order=np.lexsort((d['codes'],-d['g1']));top[order[:50]]=True
            parts.append({k:d[k][ok] for k in ('hidden','scale','y','codes','g1')})
            parts[-1].update(date=np.full(ok.sum(),date),top50=top[ok])
    return {k:np.concatenate([d[k] for d in parts]) for k in parts[0]}


def predict(head: QuantileHead,data: dict) -> np.ndarray:
    """输出真实Y尺度分位数。

    :param head: 头。
    :param data: 完整监督数组。
    :return: 样本乘3分位数。
    """
    with torch.no_grad():
        pieces=[head(torch.from_numpy(data['hidden'][i:i+2048])).numpy() for i in range(0,len(data['y']),2048)]
    return np.concatenate(pieces)*data['scale'][:,None]


def train(p: dict,budget: Budget) -> dict:
    """按批准配置训练并判定机制门禁。

    :param p: 固定协议。
    :param budget: 原累计预算。
    :return: 机制结果。
    """
    torch.set_num_threads(4);torch.manual_seed(100)
    tr=read_segment('train',p);q1=read_segment('Q1',p);q2=read_segment('Q2',p)
    z=tr['y']/tr['scale'];constants=np.quantile(z,[.1,.5,.9]).astype(np.float32)
    head=QuantileHead(tr['hidden'].shape[1],torch.from_numpy(constants))
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,betas=(.9,.999),weight_decay=.01)
    x=torch.from_numpy(tr['hidden']);target=torch.tensor(z,dtype=torch.float32)
    initial=summarize(q1['y'],predict(head,q1),q1['date'])['pinball']
    history=[dict(epoch=0,Q1=initial)];best={k:v.clone() for k,v in head.state_dict().items()};bad=0;start=1
    fingerprint=training_identity(p)
    latest=OUT/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['identity']!=fingerprint:raise RuntimeError('训练恢复身份漂移')
        head.load_state_dict(ck['head']);optimizer.load_state_dict(ck['optimizer']);history=ck['history'];best=ck['best'];bad=ck['bad'];start=ck['epoch']+1
        torch.set_rng_state(ck['rng'])
    for epoch in range(start,21):
        if bad>=3:break
        budget.check();order=torch.randperm(len(target),generator=torch.Generator().manual_seed(100+epoch))
        losses=[]
        for idx in order.split(2048):
            budget.check();optimizer.zero_grad(set_to_none=True)
            loss=pinball(head(x[idx]),target[idx]).mean()
            if not torch.isfinite(loss):raise RuntimeError('非有限训练损失')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val=summarize(q1['y'],predict(head,q1),q1['date'])['pinball']
        old_min=min(h['Q1'] for h in history);history.append(dict(epoch=epoch,Q1=val,train_Z=float(np.mean(losses))))
        if val<old_min:best={k:v.clone() for k,v in head.state_dict().items()};bad=0
        else:bad+=1
        ck=dict(identity=fingerprint,epoch=epoch,head=head.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),history=history,best=best,bad=bad)
        tmp=latest.with_suffix('.tmp');torch.save(ck,tmp);os.replace(tmp,latest)
        status('training',epoch=epoch,Q1_pinball=val);logger.info('epoch {} Q1 pinball {}',epoch,val)
    selected=best_epoch([h['Q1'] for h in history]);head.load_state_dict(best)
    torch.save(dict(identity=fingerprint,head=best,constants=constants,epoch=selected),OUT/'head.pt')
    pred=predict(head,q2);base=q2['scale'][:,None]*constants
    metrics={name:summarize(q2['y'],v,q2['date']) for name,v in [('learned',pred),('volatility',base)]}
    threshold=float(np.quantile(tr['y'],.1));df=pd.DataFrame(dict(date=q2['date'],code=q2['codes'],y=q2['y'],scale=q2['scale'],top50=q2['top50']))
    for j,key in enumerate(['q10','q50','q90']):df[key]=pred[:,j];df['base_'+key]=base[:,j]
    df['bad']=df.y<threshold;df['risk']=np.maximum(0,-pred[:,0]);df['base_risk']=np.maximum(0,-base[:,0])
    daily=[]
    for date,g in df[df.top50].groupby('date'):
        daily.append(dict(date=date,stocks=len(g),learned_auc=auc(g.bad.to_numpy(),g.risk.to_numpy()),
                          baseline_auc=auc(g.bad.to_numpy(),g.base_risk.to_numpy())))
    au=pd.DataFrame(daily);delta=(au.learned_auc-au.baseline_auc).mean()
    passed=bool(selected>0 and metrics['learned']['p10']<metrics['volatility']['p10'] and np.isfinite(delta) and delta>0)
    df.to_parquet(OUT/'Q2_diagnostics.parquet');au.to_csv(OUT/'Q2_top50_auc.csv',index=False)
    bins=np.quantile(tr['scale'],[.2,.4,.6,.8]);df['vol_group']=np.searchsorted(bins,df.scale)+1
    groups={str(k):{name:summarize(g.y.to_numpy(),g[cols].to_numpy(),g.date.to_numpy()) for name,cols in [('learned',['q10','q50','q90']),('volatility',['base_q10','base_q50','base_q90'])]} for k,g in df.groupby('vol_group')}
    calibrations={}
    for group,g in [('all',df),('G1_top50',df[df.top50]),*[(f'vol_{k}',g) for k,g in df.groupby('vol_group')]]:
        calibrations[group]={}
        for name,cols in [('learned',['q10','q50','q90']),('volatility',['base_q10','base_q50','base_q90'])]:
            daily_cal,summary=calibration_by_day(g,cols)
            daily_cal.to_csv(OUT/f'Q2_calibration_{group}_{name}.csv',index=False)
            calibrations[group][name]=summary
    out=dict(calibration=calibrations,passed=passed,selected_epoch=selected,history=history,train_samples=len(z),Q1_samples=len(q1['y']),Q2_samples=len(q2['y']),
        metrics=metrics,highscore_auc_delta=float(delta) if np.isfinite(delta) else None,auc_valid_dates=int(au.learned_auc.notna().sum()),
        train_bad_threshold=threshold,train_vol_bins=bins.tolist(),conditional_groups=groups,head_sha=sha(OUT/'head.pt'),identity=fingerprint)
    atomic_json(OUT/'mechanism_gate.json',out)
    return out


def training_identity(p: dict) -> dict:
    """绑定实际当前特征清单和协议源码。:param p: 固定协议。:return: 身份。"""
    return dict(protocol=identity(p),sources=sources(),feature_sha={str(f.relative_to(OUT)):sha(f) for stage in ['train','Q1','Q2'] for f in (OUT/'features'/stage).glob('*.json')})


def validate_result_identity(gate: dict,checkpoint: dict,expected: dict) -> None:
    """禁止陈旧B结果跨版本进入C。

    :param gate: 机制结果。
    :param checkpoint: 权重容器。
    :param expected: 当前真实输入/代码身份。
    :return: 无。
    """
    if gate.get('passed') is not True or gate.get('identity')!=expected or checkpoint.get('identity')!=expected:
        raise RuntimeError('B-to-C训练产物身份不符')


def require_mechanism(p: dict) -> dict:
    """重验完整B特征、头及机制身份。:param p: 协议。:return: 头容器。"""
    gate=json.loads((OUT/'mechanism_gate.json').read_text())
    checkpoint=torch.load(OUT/'head.pt',map_location='cpu',weights_only=False)
    validate_result_identity(gate,checkpoint,training_identity(p))
    if gate.get('head_sha')!=sha(OUT/'head.pt'):raise RuntimeError('头内容漂移')
    # 日历与每个数据SHA再次验证，不能只验证可被保留的manifest字节。
    for stage in ['train','Q1','Q2']:read_segment(stage,p)
    return checkpoint
