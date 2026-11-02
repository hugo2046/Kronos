"""等待唯一runner完成后独立复算已有产物并本地提交；不运行GPU或读取新行情。"""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd

ROOT=Path('/home/user/workspace/Kronos-sam-adapter')
OUT=ROOT/'artifacts/experiments/sam_adapter_20261102'


def read(name: str):
    return json.loads((OUT/name).read_text())


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(name: str,value) -> None:
    path=OUT/name;temp=path.with_suffix(path.suffix+'.finalizer.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def wide(path: Path) -> pd.DataFrame:
    frame=pd.read_parquet(path)
    if not isinstance(frame.index,pd.DatetimeIndex):frame=frame.T
    frame.index=pd.to_datetime(frame.index)
    return frame.sort_index().sort_index(axis=1)


def audit() -> dict:
    """从逐日日报、原始归因表与请求JSONL独立重算，不调用报告指标函数。"""
    p=read('protocol.json');summary=read('metrics.json');endpoints=read('figure_endpoints.json')
    result={'passed':False,'finalizer_pid':os.getpid(),'at':datetime.now().isoformat(),'metrics':{},'requests':0,'max_effective_market_end':None}
    for w in ['W3','W4']:
        baseline=wide(Path(p['baseline']['signal_paths'][w]));streams=[]
        for seed in [42,100]:
            path=OUT/'signals'/f'SAM_train100_infer{seed}'/f'{w}.parquet';frame=wide(path);meta=json.loads(path.with_suffix('.meta.json').read_text())
            assert frame.index.equals(baseline.index) and frame.columns.equals(baseline.columns)
            np.testing.assert_array_equal(np.isfinite(frame),np.isfinite(baseline))
            assert meta['complete'] and meta['seed']==seed and meta['rho']==.05 and meta['algorithm']=='SAM'
            assert meta['signal_sha256']==digest(path)
            assert meta['adapter_sha256']==read('training_sam.json')['adapter_sha256']
            streams.append(frame)
        candidate=wide(OUT/f'signal_{w}_SAM_MC40.parquet')
        np.testing.assert_allclose(candidate.values,(streams[0].values+streams[1].values)*.5,atol=0,rtol=0,equal_nan=True)
        nets={}
        for tag in ['G1_original','MC40','SAM_MC40']:
            directory=OUT/'accounts'/w/tag;report=pd.read_parquet(directory/'report.parquet');meta=json.loads((directory/'complete.json').read_text())
            assert meta['report_sha256']==digest(directory/'report.parquet') and meta['execution_id']==p['execution_id']
            pd.testing.assert_frame_equal(report,pd.read_parquet(OUT/f'report_{w}_{tag}.parquet'),check_exact=True)
            assert len(report)==p['expected_coverage'][w]['dates'] and report.index.max()<=pd.Timestamp('2026-07-24')
            net=report['return'].to_numpy()-report['cost'].to_numpy();nets[tag]=net
            nav=np.concatenate(([1.],np.cumprod(1.+net)));sd=np.std(net,ddof=1)
            actual=dict(net_sum=float(np.sum(net)),compound=float(nav[-1]-1),sharpe=float(np.sqrt(252)*np.mean(net)/sd) if sd else None,
                        mdd=float(np.min(nav/np.maximum.accumulate(nav)-1)),cost=float(report.cost.sum()),turnover=float(report.turnover.mean()),volatility=float(sd*np.sqrt(252)))
            for key,value in actual.items():
                assert value is not None,'零波动需处理报告格式，不能跳过'
                assert abs(value-summary['metrics'][w][tag][key])<1e-12,(w,tag,key,value)
            assert abs(endpoints[f'{w}/{tag}']-actual['net_sum'])<1e-12
            contributions=pd.read_parquet(directory/'stock_daily_contributions.parquet')
            grouped=contributions.groupby('date')[['gross_contribution','cost_contribution']].sum().reindex(report.index,fill_value=0)
            assert np.max(np.abs(grouped.gross_contribution-report['return']))<=1e-10
            assert np.max(np.abs(grouped.cost_contribution-report.cost))<=1e-10
            if tag!='SAM_MC40':
                expected=pd.read_parquet(p['reference_reports'][w][tag]);pd.testing.assert_frame_equal(report,expected,atol=1e-10,rtol=0)
            result['metrics'][f'{w}/{tag}']=actual
        for tag in ['G1_original','MC40']:
            d=nets['SAM_MC40']-nets[tag];rng=np.random.default_rng(100);n=len(d);samples=[]
            for _ in range(5000):
                starts=rng.integers(0,n,size=int(np.ceil(n/20)))
                positions=np.concatenate([np.arange(start,start+20)%n for start in starts])[:n]
                samples.append(d[positions].sum())
            limits=np.quantile(samples,[.025,.975]);comp=summary['comparisons'][f'{w}/{tag}']
            assert abs(comp['delta_pp']-sum(d)*100)<1e-10
            np.testing.assert_allclose(limits,[comp['bootstrap']['lower_95'],comp['bootstrap']['upper_95']],atol=1e-12,rtol=0)
    for request_file in [OUT/'inference_market_requests.jsonl',*sorted((OUT/'accounts').glob('*/*/market_requests.jsonl'))]:
        assert request_file.is_file()
        for line in request_file.read_text().splitlines():
            row=json.loads(line);end=pd.Timestamp(row['effective_end'])
            assert end<=pd.Timestamp('2026-07-24')
            result['requests']+=1
            old=result['max_effective_market_end'];result['max_effective_market_end']=str(end) if old is None or end>pd.Timestamp(old) else old
    m=summary['metrics']
    returns=all(m[w]['SAM_MC40']['net_sum']>m[w][tag]['net_sum'] for w in m for tag in ['G1_original','MC40'])
    risk=all(m[w]['SAM_MC40']['sharpe']>=m[w]['MC40']['sharpe'] and m[w]['SAM_MC40']['mdd']>=m[w]['MC40']['mdd'] for w in m)
    assert summary['verdict']==dict(historical_return=returns,risk_compatible=risk)
    budget=read('gpu_budget.json');assert budget['active'] is None and budget['used_seconds']<=21600
    assert time.time()<p['budget']['deadline']
    for name in ['cumulative_return','drawdown','cumulative_difference']:assert (OUT/'figures'/f'{name}.png').stat().st_size>1000
    result.update(passed=True,verdict=summary['verdict'],gpu_seconds=budget['used_seconds'])
    return result


def main() -> None:
    p=read('protocol.json');deadline=p['budget']['deadline']
    write('finalizer_status.json',dict(status='waiting',pid=os.getpid(),deadline=deadline,worker_pid=603798))
    while time.time()<deadline:
        if (OUT/'figure_endpoints.json').exists() and read('bot_status.json').get('stage')=='report_complete':break
        try:os.kill(603798,0)
        except ProcessLookupError:raise RuntimeError('唯一runner已退出但报告未完成，需接力恢复')
        time.sleep(10)
    else:raise RuntimeError('固定墙钟已到期，未重置')
    checked=audit();write('independent_recalculation.json',checked)
    with (OUT/'execution_ledger.md').open('a') as stream:
        stream.write('\nTask 3: complete，四流520日期全部完成，finite mask与原G1逐格一致。\nTask 4: complete，六账户真实回测/持仓/成交对账、配对bootstrap、三张图与中文报告完成；独立finalizer从原始日报和请求记录重新计算指标/区间/端点/预算/封存全部通过，见independent_recalculation.json。\n')
    # 只加入本实验小产物和报告，不碰其他工作树或分支索引。
    paths=[ROOT/'docs/SAM状态适配器实验报告_20261102.md']
    for path in OUT.rglob('*'):
        if not path.is_file():continue
        if path.suffix in {'.json','.csv','.png','.svg','.md','.txt','.py'} and path.name not in {'bot_status.json','supervisor_status.json','bot_supervisor.py','finalizer_status.json'} and not path.name.startswith('bot_final_'):
            paths.append(path)
    subprocess.run(['git','add','--',*[str(x.relative_to(ROOT)) for x in paths]],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','docs: 交付固定SAM两窗完整收益诊断与独立复算', '-m','保留冻结权重、双流完整覆盖、纠偏账户对账、累计收益/回撤/累计差图和配对区间。单seed先导结果不自动采纳；本地实验分支不合并不推送。','-m','Co-Authored-By: Hugo <shen.lan123@gmail.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    state=read('bot_status.json');state.update(pid=os.getpid(),worker_pid=603798,completed_dates=520,total_dates=520,independent_recalculation=True,commit=commit)
    write('bot_status.json',state);write('finalizer_status.json',dict(status='complete',pid=os.getpid(),commit=commit))
    print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('finalizer_status.json',dict(status='continue',pid=os.getpid(),error=f'{type(exc).__name__}: {exc}'))
        raise
