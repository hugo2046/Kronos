"""SAM单变量可恢复执行：诊断、控制复现、训练、逐日推理与纠偏回测。"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
import numpy as np
import pandas as pd
import torch
from loguru import logger
from state_adapter import adapter_state,base_state_hash,load_adapter_state
from state_adapter import run as legacy
from state_adapter.data import WindowCorpus
from state_adapter.gpu_budget import GPUBudget,GPUTimeBudgetExceeded
from state_adapter.mc40 import _wide,pool_signals
from state_adapter.sam_train import sam_train_batch
from state_adapter.sam_diagnostics import perturbation_diagnostic,rng_equal
from state_adapter.train import train_arm,validation_ce,_capture_rng,_restore_rng

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/experiments/sam_adapter_20261102'


def save(name: str, value: Any) -> None:
    """原子保存审计。:param name: 相对路径。 :param value: JSON数据。 :returns: 无。"""
    legacy._atomic_text(OUT/name,json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')


def bind() -> None:
    """将复用入口输出全部绑定本轮目录。:returns: 无。"""
    legacy.OUT=OUT;legacy.PROTOCOL_PATH=OUT/'protocol.json';legacy.PROTOCOL_SHA_PATH=OUT/'protocol.json.sha256'
    legacy.STATUS_PATH=OUT/'bot_status.json';legacy.GPU_BUDGET_PATH=OUT/'gpu_budget.json'


def load() -> tuple[dict[str,Any],str]:
    """重查冻结协议及全部输入SHA。:returns: 协议及canonical SHA。"""
    bind();p,sha=legacy.load_protocol()
    for path,digest in p['reference_files'].items():
        if legacy._sha256_file(path)!=digest:raise RuntimeError('冻结源SHA漂移: '+path)
    for key in ['reference_adapter','validation_cache']:
        if legacy._sha256_file(p[key]['path'])!=p[key]['sha256']:raise RuntimeError(key+' SHA漂移')
    return p,sha


def cfg_sha(p: dict[str,Any], arm: str) -> str:
    """配置绑定算法/rho及seed。:param p: 协议。 :param arm: control/sam。 :returns: SHA。"""
    config=dict(p,training=p['control_training'] if arm=='control' else p['training'])
    return legacy.config_sha(config,100)


def status(stage: str, **extra: Any) -> None:
    """记录真实PID和阶段。:param stage: 阶段。 :param extra: 进度。 :returns: 无。"""
    save('bot_status.json',dict(status='running',pid=os.getpid(),stage=stage,at=time.time(),**extra))


def require_control() -> None:
    """严格控制复现门禁。:raises RuntimeError: 任一差异。 :returns: 无。"""
    comparison=json.loads((OUT/'control_comparison.json').read_text())
    if not all(comparison[k] for k in ['step_equal','ce_equal','tensors_equal']):raise RuntimeError('AdamW未精确复现，禁止解释SAM效应')


def signal_identity(path: Path, identity: dict[str,Any]) -> None:
    """恢复前检查算法和全部信号身份。:param path: 信号。 :param identity: 身份。 :returns: 无。"""
    meta_path=path.with_suffix('.meta.json')
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if any(meta.get(k)!=v for k,v in identity.items()):raise RuntimeError('逐日信号身份不一致')
        if meta.get('signal_sha256') and (not path.is_file() or meta['signal_sha256']!=legacy._sha256_file(path)):
            raise RuntimeError('已完成信号内容SHA漂移，拒绝重绑定')
    else:
        if path.exists():raise RuntimeError('孤立信号缺少身份')
        legacy._atomic_text(meta_path,json.dumps(dict(identity,completed=[],forward_cutoff='2026-07-24',max_market_request_end=None),indent=2)+'\n')


def verify_candidate_inputs(p: dict[str,Any], sha: str) -> None:
    """在独立回测/报告入口只读验证训练权重与四流身份及完整覆盖。

    :param p: 冻结协议。
    :param sha: canonical协议SHA。
    :returns: 无，不创建或修复缺失manifest。
    :raises RuntimeError: 权重、manifest、信号内容或封存边界不一致。
    """
    require_control()
    checkpoint=OUT/'arms/sam/best_adapter.pt'
    record=json.loads((OUT/'training_sam.json').read_text())
    digest=legacy._sha256_file(checkpoint)
    expected=dict(algorithm='SAM',rho=.05,protocol_sha=sha,config_sha=cfg_sha(p,'sam'),adapter_sha256=digest)
    if any(record.get(k)!=v for k,v in expected.items()):raise RuntimeError('训练权重身份不一致')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    checkpoint_identity=dict(protocol_sha=sha,config_sha=cfg_sha(p,'sam'),seed=100,family='dynamic',base_hash=p['reference_adapter']['base_hash'],step=record['best_step'])
    if any(payload.get(k)!=v for k,v in checkpoint_identity.items()):raise RuntimeError('选点checkpoint身份不一致')
    for window in p['windows']:
        baseline=_wide(p['baseline']['signal_paths'][window])
        for seed in [42,100]:
            tag=f'SAM_train100_infer{seed}'
            path=OUT/'signals'/tag/f'{window}.parquet';meta_path=path.with_suffix('.meta.json')
            if not path.is_file() or not meta_path.is_file():raise RuntimeError('候选信号或manifest缺失')
            meta=json.loads(meta_path.read_text())
            identity=dict(expected,tag=tag,window=window,seed=seed,train_seed=100,execution_id=p['execution_id'])
            if any(meta.get(k)!=v for k,v in identity.items()):raise RuntimeError('回测输入流身份不一致')
            if not meta.get('complete') or meta.get('signal_sha256')!=legacy._sha256_file(path):raise RuntimeError('流未完成或信号内容SHA漂移')
            wide=_wide(str(path))
            dates=[str(d.date()) for d in baseline.index]
            if sorted(meta.get('completed',[]))!=dates:raise RuntimeError('manifest日期不完整')
            for field in ['max_market_date','max_market_request_end']:
                if meta.get(field) and pd.Timestamp(meta[field])>legacy.FORWARD_CUTOFF:raise RuntimeError('候选流封存边界不一致')
            legacy.check_signal_coverage(baseline,wide,baseline.index)
            expected_coverage=p['expected_coverage'][window]
            if len(wide)!=expected_coverage['dates'] or int(np.isfinite(wide).sum().sum())!=expected_coverage['finite']:
                raise RuntimeError('候选流覆盖计数不一致')


def verdict(records: dict[str,Any]) -> dict[str,bool]:
    """分别判定跨窗收益和风险兼容。:param records: 两窗指标。 :returns: 两个门槛。"""
    returns=all(records[w]['SAM_MC40']['net_sum']>records[w][tag]['net_sum'] for w in ['W3','W4'] for tag in ['G1_original','MC40'])
    risk=all(records[w]['SAM_MC40']['sharpe'] is not None and records[w]['MC40']['sharpe'] is not None and records[w]['SAM_MC40']['sharpe']>=records[w]['MC40']['sharpe'] and records[w]['SAM_MC40']['mdd']>=records[w]['MC40']['mdd'] for w in ['W3','W4'])
    return dict(historical_return=returns,risk_compatible=risk)


def check_budget(budget: GPUBudget) -> None:
    """阶段内预算检查。:param budget: 累计预算。 :returns: 无。"""
    if budget.should_stop():raise GPUTimeBudgetExceeded('固定GPU/墙钟预算耗尽')


def diagnose(p: dict[str,Any], sha: str, budget: GPUBudget, arm: str) -> None:
    """在冻结2025H1 cache上诊断原MC40或SAM。:param p: 协议。 :param sha: SHA。 :param budget: 预算。 :param arm: reference/sam。 :returns: 无。"""
    name=f'diagnostic_{arm}.json'
    checkpoint=Path(p['reference_adapter']['path']) if arm=='reference' else OUT/'arms/sam/best_adapter.pt'
    digest=legacy._sha256_file(checkpoint)
    if (OUT/name).exists():
        result=json.loads((OUT/name).read_text())
        if result['adapter_sha256']!=digest or result['protocol_sha']!=sha:raise RuntimeError('诊断身份漂移')
        return
    status('diagnostic_'+arm)
    with budget.stage('diagnostic_'+arm):
        identity=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model,tokenizer,_=legacy._load_g1(torch.device('cuda:0'),'dynamic',checkpoint,
            protocol_sha=identity['protocol_sha'] if arm=='reference' else sha,
            cfg_sha=identity['config_sha'] if arm=='reference' else cfg_sha(p,'sam'),seed=100)
        batches=torch.load(p['validation_cache']['path'],map_location='cpu',weights_only=False)['batches']
        def evaluate() -> float:
            check_budget(budget)
            return validation_ce(model,batches,'cuda:0',eval_seed=p['diagnostic']['eval_seed'])
        result=perturbation_diagnostic(model,evaluate,seed=p['diagnostic']['seed'])
        result.update(adapter_sha256=digest,protocol_sha=sha)
        if result['base_hash_after']!=p['reference_adapter']['base_hash']:raise RuntimeError('底座漂移')
        save(name,result)
    del model,tokenizer;torch.cuda.empty_cache()


def smoke(p: dict[str,Any], sha: str, budget: GPUBudget) -> None:
    """真实GPU小批集成，保存随机/恢复/梯度及CE位置证据。:param p: 协议。 :param sha: SHA。 :param budget: 预算。 :returns: 无。"""
    if (OUT/'smoke.json').exists():return
    status('real_gpu_smoke')
    corpus=WindowCorpus(p['corpus']['train_path'],target_end='2024-12-31')
    batch=corpus.sample(2,np.random.default_rng(100))
    with budget.stage('smoke'):
        check_budget(budget)
        model,tokenizer,_=legacy._load_g1(torch.device('cuda:0'),'dynamic')
        before=adapter_state(model);base=base_state_hash(model)
        optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=3e-4,betas=(.9,.95),weight_decay=.01)
        observed=[]
        def pre(*args: Any) -> None:observed.append(_capture_rng())
        handle=model.register_forward_pre_hook(pre)
        original=optimizer.step;restored=[]
        def step(*args: Any,**kwargs: Any) -> Any:
            restored.append(all(torch.equal(v,adapter_state(model)[k]) for k,v in before.items()))
            return original(*args,**kwargs)
        optimizer.step=step
        values=sam_train_batch(model,tokenizer,batch,optimizer,'cuda:0',rho=.05)
        handle.remove()
        cache=torch.load(p['validation_cache']['path'],map_location='cpu',weights_only=False)['batches']
        rng=_capture_rng();current=adapter_state(model)
        ce=validation_ce(model,cache[:1],'cuda:0',eval_seed=103100)
        from state_adapter.train import future_token_loss
        # logits只有89..98影响CE：真实smoke同一损失函数的边界证据。
        z=torch.zeros(1,99,4,device='cuda:0');y=torch.zeros(1,99,dtype=torch.long,device='cuda:0')
        ce0=future_token_loss(z,z,y,y)[0];z[:,:89,0]=10
        positions_ok=bool(torch.equal(ce0,future_token_loss(z,z,y,y)[0]))
        record=dict(values,unperturbed_validation_ce=ce,restored_before_step=all(restored),
                    common_rng=rng_equal(observed[0],observed[1]),validation_rng_equal=rng_equal(rng,_capture_rng()),
                    validation_parameters_equal=all(torch.equal(v,adapter_state(model)[k]) for k,v in current.items()),
                    base_hash_before=base,base_hash_after=base_state_hash(model),positions_89_98=positions_ok,
                    protocol_sha=sha,device=torch.cuda.get_device_name(),batch=2)
        assert all(record[k] for k in ['restored_before_step','common_rng','validation_rng_equal','validation_parameters_equal','positions_89_98'])
        assert record['grad_norm']>0 and base==record['base_hash_after']==p['reference_adapter']['base_hash']
        save('smoke.json',record)
    del model,tokenizer,optimizer,corpus;torch.cuda.empty_cache()


def train(p: dict[str,Any], sha: str, budget: GPUBudget) -> None:
    """原AdamW精确复现通过后才训练固定SAM。:param p: 协议。 :param sha: SHA。 :param budget: 预算。 :returns: 无。"""
    validation=WindowCorpus(p['corpus']['val_path'],target_start='2025-01-01',target_end='2025-06-30',history_path=p['corpus']['train_path'])
    corpus=WindowCorpus(p['corpus']['train_path'],target_end='2024-12-31')
    for arm in ['control','sam']:
        if arm=='sam':require_control()
        directory=OUT/'arms'/arm;name=f'training_{arm}.json'
        if (OUT/name).exists():
            done=json.loads((OUT/name).read_text())
            if done['adapter_sha256']!=legacy._sha256_file(directory/'best_adapter.pt') or done['config_sha']!=cfg_sha(p,arm):raise RuntimeError('训练完成身份漂移')
            continue
        if legacy._sha256_file(directory/'validation_batches.pt')!=p['validation_cache']['sha256']:raise RuntimeError('cache漂移')
        status('train_'+arm,step=0)
        def progress(record: dict[str,Any]) -> None:
            with (OUT/f'training_{arm}.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            status('train_'+arm,**record)
        def sam_step(*args: Any) -> dict[str,float]:
            values=sam_train_batch(*args,rho=.05)
            with (OUT/'sam_batch_metrics.jsonl').open('a') as stream:stream.write(json.dumps(values)+'\n')
            return values
        with budget.stage('train_'+arm):
            torch.cuda.reset_peak_memory_stats()
            result=train_arm('dynamic',model_path=p['baseline']['model_paths']['predictor'],tokenizer_path=p['baseline']['model_paths']['tokenizer'],
                train_corpus=corpus,validation_corpus=validation,output_dir=directory,protocol_sha=sha,config_sha=cfg_sha(p,arm),
                seed=100,steps=3000,batch_size=64,learning_rate=3e-4,weight_decay=.01,validation_windows=2048,validation_every=500,device='cuda:0',
                guard_fn=legacy.in_registry_guard,budget_guard_fn=budget.should_stop,progress_fn=progress,batch_step_fn=sam_step if arm=='sam' else None)
            result.update(peak_memory_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=3000,training_forward_count=6000 if arm=='sam' else 3000,
                          training_backward_count=6000 if arm=='sam' else 3000,algorithm='SAM' if arm=='sam' else 'AdamW',rho=.05 if arm=='sam' else 0.,
                          adapter_sha256=legacy._sha256_file(directory/'best_adapter.pt'),config_sha=cfg_sha(p,arm),protocol_sha=sha)
        if result['base_hash']!=p['reference_adapter']['base_hash']:raise RuntimeError('训练底座漂移')
        if arm=='control':
            a=torch.load(directory/'best_adapter.pt',map_location='cpu',weights_only=False)
            b=torch.load(p['reference_adapter']['path'],map_location='cpu',weights_only=False)
            diffs={k:float((a['adapter_state'][k]-v).abs().max()) for k,v in b['adapter_state'].items()}
            save('control_comparison.json',dict(step_equal=a['step']==b['step'],ce_equal=a['val_ce']==b['val_ce'],tensors_equal=all(v==0 for v in diffs.values()),
                new_step=a['step'],old_step=b['step'],new_ce=a['val_ce'],old_ce=b['val_ce'],tensor_max_abs=diffs,max_abs=max(diffs.values())))
            require_control()
        save(name,result);torch.cuda.empty_cache()


def infer(p: dict[str,Any], sha: str, budget: GPUBudget) -> None:
    """冻结权重后逐日双流推理。:param p: 协议。 :param sha: SHA。 :param budget: 预算。 :returns: 无。"""
    require_control()
    record=json.loads((OUT/'training_sam.json').read_text());checkpoint=OUT/'arms/sam/best_adapter.pt'
    if legacy._sha256_file(checkpoint)!=record['adapter_sha256']:raise RuntimeError('SAM权重身份漂移')
    from state_adapter.market_guard import market_guard
    coverage={}
    with budget.stage('infer'),market_guard(OUT/'inference_market_requests.jsonl'):
        model,tokenizer,_=legacy._load_g1(torch.device('cuda:0'),'dynamic',checkpoint,protocol_sha=sha,cfg_sha=cfg_sha(p,'sam'),seed=100)
        predictor=legacy.StateKronosPredictor(model,tokenizer,device='cuda:0',max_context=512,clip=5.)
        provider=legacy._new_provider()
        for seed in [42,100]:
            tag=f'SAM_train100_infer{seed}'
            for window in p['windows']:
                path=OUT/'signals'/tag/f'{window}.parquet'
                signal_identity(path,dict(tag=tag,window=window,seed=seed,train_seed=100,algorithm='SAM',rho=.05,protocol_sha=sha,config_sha=cfg_sha(p,'sam'),adapter_sha256=record['adapter_sha256'],execution_id=p['execution_id']))
                legacy.score_window(predictor,provider,tag,window,path,seed=seed,budget=budget)
                wide=_wide(str(path));baseline=_wide(p['baseline']['signal_paths'][window])
                coverage[f'{seed}_{window}']=legacy.check_signal_coverage(baseline,wide,baseline.index)
                assert len(wide)==p['expected_coverage'][window]['dates'] and int(np.isfinite(wide).sum().sum())==p['expected_coverage'][window]['finite']
                meta_path=path.with_suffix('.meta.json');meta=json.loads(meta_path.read_text())
                meta['signal_sha256']=legacy._sha256_file(path)
                legacy._atomic_text(meta_path,json.dumps(meta,indent=2)+'\n')
    save('coverage.json',coverage)
    del model,tokenizer,predictor;torch.cuda.empty_cache()


def backtest(p: dict[str,Any], sha: str) -> None:
    """真实重放纠偏对照和候选，保留持仓交易对账。:param p: 协议。 :param sha: SHA。 :returns: 无。"""
    verify_candidate_inputs(p,sha)
    from state_adapter.limit_unit_audit import execution
    from state_adapter.risk_attribution import replay_account
    from state_adapter.date_balanced_run import metrics
    records={};comparisons={};gates={};ranks={}
    for window,bounds in p['windows'].items():
        candidate=pool_signals(*[_wide(str(OUT/'signals'/f'SAM_train100_infer{s}'/f'{window}.parquet')) for s in [42,100]])
        baseline=_wide(p['baseline']['signal_paths'][window]);legacy.check_signal_coverage(baseline,candidate,baseline.index)
        candidate_path=OUT/f'signal_{window}_SAM_MC40.parquet';legacy.atomic_write_signal(candidate_path,candidate)
        signals={**p['reference_signals'][window],'SAM_MC40':str(candidate_path)};reports={};records[window]={}
        for tag,path in signals.items():
            directory=OUT/'accounts'/window/tag;directory.mkdir(parents=True,exist_ok=True)
            identity=dict(protocol_sha=sha,execution_id=p['execution_id'],signal_sha256=legacy._sha256_file(path))
            if (directory/'complete.json').exists():
                meta=json.loads((directory/'complete.json').read_text())
                if any(meta.get(k)!=v for k,v in identity.items()) or meta['report_sha256']!=legacy._sha256_file(directory/'report.parquet'):raise RuntimeError('回测身份漂移')
                report=pd.read_parquet(directory/'report.parquet')
            else:
                status('backtest',window=window,arm=tag)
                wide=_wide(path);legacy.check_signal_coverage(baseline,wide,baseline.index)
                signal=wide.stack();signal.index.names=['datetime','instrument'];signal=signal.swaplevel().sort_index()
                with execution('new'):report,meta=replay_account(signal,*bounds,directory)
                if not meta['reconciliation']['exact']:raise RuntimeError('持仓收益费用对账失败')
                meta.update(identity,report_sha256=legacy._sha256_file(directory/'report.parquet'))
                save(str((directory/'complete.json').relative_to(OUT)),meta)
            if tag!='SAM_MC40':
                expected=pd.read_parquet(p['reference_reports'][window][tag])
                pd.testing.assert_frame_equal(report,expected,atol=1e-10,rtol=0)
                gates[f'{window}/{tag}']=float((report-expected).abs().max().max())
            reports[tag]=report;record=metrics(report);record['volatility']=float((report['return']-report.cost).std(ddof=1)*np.sqrt(252));records[window][tag]=record
            report.to_parquet(OUT/f'report_{window}_{tag}.parquet')
            net=report['return']-report.cost
            pd.DataFrame(dict(net_daily=net,cumulative=net.cumsum(),nav=(1+net).cumprod())).to_parquet(OUT/f'daily_{window}_{tag}.parquet')
        for tag in ['G1_original','MC40']:
            a=reports['SAM_MC40']['return']-reports['SAM_MC40'].cost;b=reports[tag]['return']-reports[tag].cost
            assert a.index.equals(b.index)
            comparisons[f'{window}/{tag}']=dict(delta_pp=float((a-b).sum()*100),bootstrap=legacy.block_bootstrap_sum_difference(a,b,block_length=20,resamples=5000,seed=100))
            reference=_wide(signals[tag]);rank=candidate.corrwith(reference,axis=1,method='spearman')
            ranks[f'{window}/{tag}']=dict(mean_spearman=float(rank.mean()),min_spearman=float(rank.min()),mean_top50_overlap=float(np.mean([len(set(candidate.loc[d].nlargest(50).index)&set(reference.loc[d].nlargest(50).index))/50 for d in candidate.index])))
        save('reference_replay_gate.json',gates)
    save('metrics.json',dict(metrics=records,comparisons=comparisons,verdict=verdict(records),ranking=ranks))
    pd.DataFrame({(w,t):m for w,rows in records.items() for t,m in rows.items()}).T.to_csv(OUT/'metrics.csv')


def budget_exhausted(state: dict[str,Any], *, now: float, deadline: float) -> bool:
    """按原子预算快照检查硬截止。:param state: 预算。 :param now: 当前秒。 :param deadline: 固定截止。 :returns: 耗尽。"""
    active=state.get('active')
    live=max(0.,now-datetime.fromisoformat(active['started_at']).timestamp()) if active else 0.
    return now>=deadline or float(state.get('used_seconds',0))+live>=21600


def main() -> None:
    """独占runner，恢复不重置预算；all一直执行到报告。:returns: 无。"""
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['train','infer','backtest','report','all'],default='all');args=parser.parse_args()
    bind();lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p,sha=load();deadline=p['budget']['deadline']
    if time.time()>=deadline:raise RuntimeError('原8小时墙钟耗尽')
    budget=GPUBudget(OUT/'gpu_budget.json',limit_seconds=21600,cumulative=True,heartbeat_interval=2)
    stop=budget.should_stop;budget.should_stop=lambda:stop() or time.time()>=deadline
    # 独立监督线确保长算子/失败重启亦受硬截止约束，检查点只来自完整batch。
    done=threading.Event()
    def watchdog() -> None:
        while not done.wait(1):
            state=json.loads(budget.path.read_text()) if budget.path.exists() else {}
            if budget_exhausted(state,now=time.time(),deadline=deadline):
                save('bot_status.json',dict(status='blocked',pid=os.getpid(),stage='hard_budget_limit',at=time.time()))
                os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start()
    save('execution_version.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),pid=os.getpid(),stage=args.stage,at=time.time(),protocol_sha=sha,
        source_sha256={str(path):legacy._sha256_file(path) for path in [Path(__file__),ROOT/'state_adapter/sam_train.py',ROOT/'state_adapter/train.py',ROOT/'state_adapter/sam_diagnostics.py']}))
    try:
        if args.stage in ['train','all']:
            diagnose(p,sha,budget,'reference');smoke(p,sha,budget);train(p,sha,budget);diagnose(p,sha,budget,'sam')
        if args.stage in ['infer','all']:infer(p,sha,budget)
        if args.stage in ['backtest','all']:backtest(p,sha)
        if args.stage in ['report','all']:
            from state_adapter.sam_report import report
            report()
        else:save('bot_status.json',dict(status='continue',pid=os.getpid(),stage=args.stage+'_finished',next_step='resume --stage all; do not reset clock'))
    except BaseException as exc:
        save('bot_status.json',dict(status='continue',pid=os.getpid(),stage=args.stage,error=f'{type(exc).__name__}: {exc}',next_step='inspect logs and resume frozen protocol'))
        raise
    finally:done.set()


if __name__=='__main__':main()
