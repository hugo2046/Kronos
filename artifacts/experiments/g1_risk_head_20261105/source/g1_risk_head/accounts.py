"""独立进程使用已受审隔离评价器，基准保真与风险候选同口径。"""
from __future__ import annotations
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from g1_risk_head.runtime import OUT,freeze,require_gate,Budget,identity,sources,status
from g1_risk_head.pipeline import valid_transaction
from g1_risk_head.evaluate import risk_adjusted_score,annualized_return,execution_score
from g1_risk_head.model import QuantileHead
from state_adapter.tta import atomic_json,sha


def account_metrics(report: pd.DataFrame) -> dict[str,float]:
    """同一日净收益计算全部经济指标。

    :param report: 引擎日报。
    :return: 收益与风险指标。
    """
    x=(report['return']-report.cost).to_numpy();nav=np.r_[1.,np.cumprod(1+x)]
    return dict(annualized=annualized_return(x),net_sum=float(x.sum()),compound=float(nav[-1]-1),
        sharpe=float(np.sqrt(252)*x.mean()/x.std(ddof=1)),mdd=float(np.min(nav/np.maximum.accumulate(nav)-1)),
        cost=float(report.cost.sum()),turnover=float(report.turnover.mean()))


def run(parity: bool=False) -> None:
    """执行原G1零惩罚保真或已冻结风险头评价。

    :param parity: 为真时只运行原G1精确控制。
    :return: 无。
    """
    from state_adapter.market_guard import isolated_evaluation,market_guard
    from state_adapter.limit_unit_audit import execution
    from state_adapter.risk_attribution import replay_account
    from state_adapter.tta_report import observe_execution_timing,validate_account
    from state_adapter.mc40 import _wide
    from state_adapter import tta_runtime as parent
    with parent.worker_lock('risk-parity' if parity else 'risk-evaluate'):
        p=freeze();require_gate(p);budget=Budget();watch=budget.watchdog();budget.check()
        if not parity:
            from g1_risk_head.train import require_mechanism
            ck=require_mechanism(p)
            head=QuantileHead(ck['head']['linear.weight'].shape[1],torch.tensor(ck['constants']));head.load_state_dict(ck['head']);head.eval()
        expected=dict(protocol=identity(p),sources=sources());all_results={}
        prefix='parity' if parity else 'evaluation'
        try:
            with isolated_evaluation(OUT/f'{prefix}_requests.jsonl',OUT/'evaluation_namespaces'/uuid.uuid4().hex):
                for w,bounds in p['windows'].items():
                    original=_wide(p['reference_paths'][w]['g142'])
                    zero=pd.DataFrame([risk_adjusted_score(row.dropna(),row.dropna()*0,0) for _,row in original.iterrows()],index=original.index).reindex(columns=original.columns)
                    pd.testing.assert_frame_equal(zero,original)
                    variants={'G1_zero':zero}
                    if not parity:
                        panels={'volatility':[],'learned':[]}
                        for date in original.index:
                            path=OUT/'features'/w/f'{date:%Y-%m-%d}.npz'
                            if not valid_transaction(path,expected):raise RuntimeError('完整风险特征缺失')
                            with np.load(path) as d:
                                native=original.loc[date].dropna()
                                if set(d['codes'])!=set(native.index):raise RuntimeError('原始股票格不符')
                                h=torch.from_numpy(d['hidden']);s=d['scale']
                                with torch.no_grad():q=head(h).numpy()[:,0]*s
                                for tag,riskq in [('learned',q),('volatility',ck['constants'][0]*s)]:
                                    qs=pd.Series(riskq,index=d['codes']).reindex(native.index)
                                    panels[tag].append(execution_score(risk_adjusted_score(native,qs)))
                        variants={tag:pd.DataFrame(rows,index=original.index).reindex(columns=original.columns) for tag,rows in panels.items()}
                    results={}
                    for tag,wide in variants.items():
                        budget.check();status('accounts',window=w,arm=tag)
                        np.testing.assert_array_equal(np.isfinite(wide),np.isfinite(original))
                        signal=wide.stack();signal.index.names=['datetime','instrument']
                        with market_guard(OUT/f'{prefix}_requests.jsonl'),execution('new'),observe_execution_timing() as timing:
                            report,meta=replay_account(signal.swaplevel().sort_index(),*bounds,OUT/'accounts'/w/tag)
                        meta['execution_timing']=timing;validate_account(meta)
                        if parity:
                            refdir=Path('/home/user/workspace/Kronos-limit-audit/artifacts/experiments/limit_unit_audit_20261102/accounts/new')/w/'G1_original'
                            ref=pd.read_parquet(refdir/'report.parquet')
                            pd.testing.assert_frame_equal(report,ref,atol=1e-10,rtol=0)
                            errors={}
                            for filename in ['positions','trades','balances','stock_daily_contributions']:
                                a=pd.read_parquet(OUT/'accounts'/w/tag/f'{filename}.parquet');b=pd.read_parquet(refdir/f'{filename}.parquet')
                                keys=[k for k in ['date','instrument','quantity','price'] if k in a]
                                a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
                                pd.testing.assert_frame_equal(a,b,atol=1e-8,rtol=1e-12)
                                errors[filename]=True
                            results[tag]=dict(max_report_error=float((report-ref).abs().max().max()),business_keys_exact=errors)
                        else:results[tag]=account_metrics(report)
                        folder=OUT/'signals';folder.mkdir(exist_ok=True)
                        wide.to_parquet(folder/f'{w}_{tag}.parquet')
                        atomic_json(OUT/'accounts'/w/tag/'identity.json',dict(identity=expected,meta=meta,signal_sha=sha(folder/f'{w}_{tag}.parquet'),
                            files={f.name:sha(f) for f in (OUT/'accounts'/w/tag).glob('*.parquet')}))
                    if not parity:
                        for label,tag in [('G1_original','G1_original'),('MC40','B')]:
                            ref=Path('/home/user/workspace/Kronos-limit-audit/artifacts/experiments/limit_unit_audit_20261102/accounts/new')/w/tag/'report.parquet'
                            f=pd.read_parquet(ref);f.to_parquet(OUT/'accounts'/w/f'{label}.parquet');results[label]=account_metrics(f)
                    all_results[w]=results
                budget.check()
            atomic_json(OUT/('A_account_parity.json' if parity else 'metrics.json'),dict(identity=expected,**({'passed':True} if parity else {'computation_complete':True}),results=all_results))
        finally:watch.set()


if __name__=='__main__':
    import sys
    run('--parity' in sys.argv)
