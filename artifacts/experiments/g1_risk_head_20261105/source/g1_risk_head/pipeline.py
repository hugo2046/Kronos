"""逐日有界特征事务、Q2原G1补生成和小样本保真。"""
from __future__ import annotations
from pathlib import Path
import json
import time
import os
import numpy as np
import pandas as pd
import torch
from loguru import logger
from g1_risk_head.runtime import OUT,identity,sources,status,Budget
from g1_risk_head.features import preprocess
from g1_risk_head.data import history_scale
from state_adapter.tta import atomic_json,sha,frozen_hash
from kronos_qlib.windows import build_inference_windows,REQUIRED_COLS


def valid_transaction(path: Path, expected: dict) -> bool:
    """验证完整日事务，拒绝身份/内容漂移。

    :param path: npz文件。
    :param expected: 精确协议和源码身份。
    :return: 完整事务是否存在。
    """
    meta=path.with_suffix('.json')
    if not meta.exists():return False
    m=json.loads(meta.read_text())
    if m['identity']!=expected or not path.exists() or sha(path)!=m['sha']:
        raise RuntimeError('日事务身份或内容漂移:'+str(path))
    return True


def save_day(path: Path,arrays: dict,expected: dict) -> None:
    """先原子保存数据，再提交SHA标志。

    :param path: 目标文件。
    :param arrays: 无pickle数组。
    :param expected: 固定身份。
    :return: 无。
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending.npz')
    np.savez_compressed(tmp,**arrays)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)
    atomic_json(path.with_suffix('.json'),dict(identity=expected,sha=sha(path),stocks=len(arrays['codes'])))


def load_backbone(budget: Budget):
    """只加载冻结原G1，计入真实GPU预算。

    :param budget: 累计预算。
    :return: 冻结表示模型及原生生成预测器。
    """
    from lora_mean_pilot.evaluate import G1_MODEL
    from model import Kronos,KronosTokenizer,KronosPredictor
    from cross_section_kda.backbone import KronosFrozenBackbone
    with budget.compute():
        tok=KronosTokenizer.from_pretrained(G1_MODEL['tokenizer']).to('cuda:0').eval()
        model=Kronos.from_pretrained(G1_MODEL['predictor']).to('cuda:0').eval()
        backbone=KronosFrozenBackbone(tok,model,device='cuda:0')
    from types import SimpleNamespace
    budget.resident=SimpleNamespace(model=model,tokenizer=tok,device='cuda:0')
    return backbone,KronosPredictor(model,tok,device='cuda:0',max_context=512,clip=5)


def extract_day(provider,backbone,predictor,date: str,label_end: str,budget: Budget,
                with_g1: bool=False) -> dict:
    """输入与标签分开读，前者永远截止原点。

    :param provider: 有界数据入口。
    :param backbone: 冻结原G1。
    :param predictor: 原生G1采样器。
    :param date: 发行原点。
    :param label_end: 分段标签截止。
    :param budget: 持久化预算。
    :param with_g1: 只为Q2补生成原信号。
    :return: 完整输入资格上的数组，标签可缺失。
    """
    with provider.at(date):frames,xt,yt,codes,stats=build_inference_windows(provider,date)
    if not codes:raise RuntimeError('无有效股票:'+date)
    raw=np.stack([x[REQUIRED_COLS].to_numpy() for x in frames])
    hist=pd.DatetimeIndex(xt[0]);future=pd.DatetimeIndex(yt[0])
    assert hist.max()<=pd.Timestamp(date)
    x,stamp=preprocess(raw,hist);scale=history_scale(raw[:,:,3])
    hidden=[]
    for start in range(0,len(codes),64):
        with budget.compute(),torch.no_grad():
            h=backbone.extract(torch.tensor(x[start:start+64],device='cuda:0'),torch.tensor(stamp[start:start+64],device='cuda:0'))[:,-1]
            hidden.append(h.cpu().numpy())
    h=np.concatenate(hidden)
    if not np.isfinite(h).all():raise RuntimeError('隐藏状态非有限')
    g1=np.full(len(codes),np.nan)
    if with_g1:
        torch.manual_seed(42)
        # 每个股票块保持原32股×20路径顺序；不让特征抽取推进采样RNG。
        for start in range(0,len(codes),32):
            with budget.compute():
                pred=predictor.predict_batch(frames[start:start+32],xt[start:start+32],yt[start:start+32],
                    pred_len=10,T=1,top_k=0,top_p=.9,sample_count=20,verbose=False)
            for j,out in enumerate(pred):g1[start+j]=out.close.mean()/frames[start+j].close.iloc[-1]-1
        if not np.isfinite(g1).all():raise RuntimeError('Q2原G1信号非有限')
    y=np.full(len(codes),np.nan);label_raw=np.full((len(codes),10,2),np.nan)
    if future[-1]<=pd.Timestamp(label_end):
        with provider.at(label_end):
            labels=provider.read(codes,['$open','$close'],str(future[0].date()),str(future[-1].date()))
        full=pd.MultiIndex.from_product([future,codes],names=['datetime','instrument'])
        label_raw=labels.reindex(full)[['open','close']].to_numpy().reshape(10,len(codes),2).transpose(1,0,2)
        op=label_raw[:,0,0];cl=label_raw[:,-1,1]
        ok=np.isfinite(op)&np.isfinite(cl)&(op>0)&(cl>0)
        y[ok]=cl[ok]/op[ok]-1
    return dict(hidden=h,raw=raw,codes=np.array(codes),history_dates=hist.to_numpy(),
        future_dates=future.to_numpy(),label_raw=label_raw,scale=scale,y=y,g1=g1)


def scheduled_dates(provider,p: dict,stage: str) -> pd.DatetimeIndex:
    """生成预登记原点，不根据效果选择。

    :param provider: 仅日历接口。
    :param p: 固定协议。
    :param stage: train/Q1/Q2/W3/W4。
    :return: 日期清单。
    """
    bounds=p[stage] if stage in ('train','Q1','Q2') else p['windows'][stage]
    dates=provider.trading_days(*bounds)
    if stage=='train':dates=dates[::5]
    if stage in ('train','Q1'):
        cal=provider.trading_days();positions=cal.get_indexer(dates)
        dates=dates[(positions+10<len(cal)) & (cal[np.minimum(positions+10,len(cal)-1)]<=pd.Timestamp(bounds[1]))]
    return dates


def collect(provider,backbone,predictor,p: dict,budget: Budget,stages: list[str]) -> None:
    """提取预登记日期，已完成事务仅核验后复用。

    :param provider: 数据入口。
    :param backbone: 原G1。
    :param predictor: 原G1采样器。
    :param p: 协议。
    :param budget: 预算。
    :param stages: 有序分段。
    :return: 无。
    """
    expected=dict(protocol=identity(p),sources=sources())
    before=(frozen_hash(backbone.kronos),frozen_hash(backbone.tokenizer))
    from state_adapter.mc40 import _wide
    for stage in stages:
        dates=scheduled_dates(provider,p,stage)
        bounds=p[stage] if stage in ('train','Q1','Q2') else p['windows'][stage]
        base=_wide(p['reference_paths'][stage]['g142']) if stage in p['windows'] else None
        if base is not None and not dates.equals(base.index):raise RuntimeError('日期格不匹配')
        atomic_json(OUT/'features'/stage/'calendar.json',dict(identity=expected,dates=dates.strftime('%Y-%m-%d').tolist()))
        for i,date in enumerate(dates):
            budget.guard(budget.resident);ds=str(date.date());path=OUT/'features'/stage/f'{ds}.npz'
            if valid_transaction(path,expected):continue
            status('features',segment=stage,completed=i,total=len(dates),date=ds)
            values=extract_day(provider,backbone,predictor,ds,bounds[1],budget,stage=='Q2')
            if base is not None and set(values['codes'])!=set(base.loc[date].dropna().index):
                raise RuntimeError('完整原G1股票格不匹配，禁止删格:'+ds)
            save_day(path,values,expected)
            logger.info('{} {}/{} {} 股票{}',stage,i+1,len(dates),ds,len(values['codes']))
        if before!=(frozen_hash(backbone.kronos),frozen_hash(backbone.tokenizer)):
            raise RuntimeError('原权重被改变')
