"""从落盘信号、账户和日报独立核验续训结果，不调用训练或回测引擎。"""
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parents[1]
prior = out.parent / 'sam_continuation_20261103'
hashfile = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((out / name).read_text())
audit = {'metrics': {}, 'accounts': {}, 'streams': {}, 'intervals': {}, 'requests': {}}
metrics = read('metrics.json')
protocol = read('protocol.json')
sha = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert sha == '11b9544188a1e5bf254fc38598da0278f393bc5f56ae83194c4d7b22a0b9cdca' == metrics['protocol_sha']
sources=[]
for step in [5000,5500,6000]:
    name=f'step_{step}.pt'; path=out/'sources'/name
    assert hashfile(path)==next(h for f,h in protocol['inputs'].items() if Path(f).name==name)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    assert ck['step']==step and ck['family']=='dynamic' and ck['seed']==100 and ck['config_sha']==protocol['trajectory_config'] and ck['protocol_sha']==protocol['parent_protocol'] and ck['base_hash']==protocol['reference_adapter']['base_hash']
    sources.append(ck['adapter_state'])
candidate=torch.load(out/'swa_adapter.pt',map_location='cpu',weights_only=True)
assert candidate['protocol_sha']==sha and list(candidate['source_steps'])==protocol['steps']==[5000,5500,6000]
state=candidate['adapter_state']
assert len(state)==12 and all(set(s)==set(state) for s in sources) and sum(x.numel() for x in state.values())==107712
for key,tensor in state.items():
    t=[src[key].numpy() for src in sources]
    expected=((t[0].astype(np.float64)+t[1].astype(np.float64)+t[2].astype(np.float64))/3).astype(t[0].dtype)
    np.testing.assert_array_equal(expected,tensor.numpy())
audit['weight_average']={'steps':protocol['steps'],'tensors':12,'parameters':107712,'exact':True}
for w,n,k in [('W3',126,37223),('W4',134,39563)]:
    base = pd.read_parquet(out.parent/'state_factorial_20261102'/f'signal_{w}_G1_original.parquet').sort_index().sort_index(axis=1)
    assert len(base)==n and np.isfinite(base).to_numpy().sum()==k
    nets = {}
    for tag in ['G1_original','MC40','AdamW6000','SWA']:
        f = pd.read_parquet(out/f'report_{w}_{tag}.parquet')
        assert f.index.equals(base.index) and f.index.is_unique
        x = (f['return']-f.cost).to_numpy(); nets[tag]=x
        nav = np.r_[1.,np.cumprod(1+x)]
        v = dict(net_sum=x.sum(),compound=nav[-1]-1,sharpe=np.sqrt(252)*x.mean()/x.std(ddof=1),mdd=np.min(nav/np.maximum.accumulate(nav)-1),cost=f.cost.sum(),turnover=f.turnover.mean(),days=n)
        np.testing.assert_allclose(list(v.values()),[metrics['metrics'][w][tag][key] for key in v],rtol=0,atol=1e-12)
        assert abs(np.cumsum(x)[-1]-read('figure_endpoints.json')[w+'/'+tag])<1e-12
        audit['metrics'][w+'/'+tag]=v
        if tag!='SWA':
            parent_report=prior/f'report_{w}_{tag}.parquet'
            assert hashfile(parent_report)==next(h for f,h in protocol['inputs'].items() if Path(f).name==parent_report.name)
            pd.testing.assert_frame_equal(f,pd.read_parquet(parent_report),check_exact=True)
            continue
        weight=hashfile(out/'swa_adapter.pt')
        assert weight==read('diagnostic.json')['weight_sha']
        streams=[]
        for seed in [42,100]:
            path=out/'signals'/f'{tag}_infer{seed}'/f'{w}.parquet'; raw=pd.read_parquet(path).T
            raw.index=pd.to_datetime(raw.index);raw=raw.sort_index().sort_index(axis=1)
            assert raw.index.equals(base.index) and raw.columns.equals(base.columns)
            np.testing.assert_array_equal(np.isfinite(raw),np.isfinite(base))
            meta=json.loads(path.with_suffix('.meta.json').read_text()); assert meta['complete'] and meta['signal_sha256']==hashfile(path) and meta['adapter_sha256']==weight and meta['protocol_sha']==sha
            assert pd.Timestamp(meta['max_market_date'])<=pd.Timestamp('2026-07-24')
            streams.append(raw);audit['streams'][f'{w}/{tag}/{seed}']=[n,k]
        pooled=out/f'signal_{w}_{tag}.parquet';pd.testing.assert_frame_equal(pd.read_parquet(pooled).sort_index().sort_index(axis=1),(streams[0]+streams[1])*.5,check_exact=True,check_names=False)
        d=out/'accounts'/w/tag; meta=json.loads((d/'complete.json').read_text()); assert meta['signal_sha256']==hashfile(pooled) and meta['report_sha256']==hashfile(d/'report.parquet') and meta['execution_id']==protocol['execution_id']
        pd.testing.assert_frame_equal(f,pd.read_parquet(d/'report.parquet'))
        pos=pd.read_parquet(d/'positions.parquet').set_index(['date','instrument']);tx=pd.read_parquet(d/'trades.parquet').set_index(['date','instrument']);bal=pd.read_parquet(d/'balances.parquet').set_index('date')
        t=tx.join(pos[['end_price']]);t['trade_pnl']=t.quantity*(t.end_price-t.price);t['cash_used']=t.quantity*t.price+t.cost
        sums=t.groupby(level=[0,1])[['quantity','trade_pnl','cost']].sum().reindex(pos.index).fillna(0)
        np.testing.assert_allclose(pos.end_amount-pos.start_amount,sums.quantity,atol=1e-8,rtol=0)
        pnl=pos.start_amount*(pos.end_price-pos.start_price)+sums.trade_pnl
        stock=pd.read_parquet(d/'stock_daily_contributions.parquet').set_index(['date','instrument']).reindex(pos.index)
        np.testing.assert_allclose(pnl/pos.previous_nav,stock.gross_contribution,atol=1e-12,rtol=0)
        np.testing.assert_allclose((pnl-sums.cost)/pos.previous_nav,stock.net_contribution,atol=1e-12,rtol=0)
        daily=(pnl/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0);cost=(sums.cost/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0)
        np.testing.assert_allclose(daily,f['return'],atol=1e-12,rtol=0);np.testing.assert_allclose(cost,f.cost,atol=1e-12,rtol=0)
        cash=bal.start_cash-t.groupby('date').cash_used.sum().reindex(bal.index,fill_value=0);equity=(pos.end_amount*pos.end_price).groupby('date').sum().reindex(bal.index,fill_value=0)
        np.testing.assert_allclose(cash,bal.end_cash,atol=1e-6,rtol=0);np.testing.assert_allclose(equity+bal.end_cash,bal.nav,atol=1e-6,rtol=0)
        audit['accounts'][w+'/'+tag]=float(abs(daily-f['return']).max())
    starts=np.random.default_rng(100).integers(0,n,(5000,(n+19)//20));indices=((starts[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
    for key,item in metrics['comparisons'].items():
        window,b=key.split('/');a='SWA'
        if window!=w:continue
        diff=nets[a]-nets[b];q=np.quantile(diff[indices].sum(axis=1),[.025,.975]);record=item['ci']
        np.testing.assert_allclose([diff.sum(),*q],[item['delta'],record['lower_95'],record['upper_95']],atol=1e-12,rtol=0);audit['intervals'][key]=(q*100).tolist()
for path in [*out.glob('market_*.jsonl'),*(out/'accounts').rglob('market_requests.jsonl')]:
    rows=list(map(json.loads,path.read_text().splitlines())); ends=[pd.Timestamp(row['effective_end']) for row in rows]
    assert ends and max(ends)<=pd.Timestamp('2026-07-24');audit['requests'][str(path.relative_to(out))]=len(rows)
assert len(audit['accounts'])==2 and len(audit['metrics'])==8 and len(audit['intervals'])==6 and sum(v[0] for v in audit['streams'].values())==520
assert len(audit['requests'])==5
m=metrics['metrics']
audit['return_gate']=all(m[w]['SWA']['net_sum']>max(m[w]['G1_original']['net_sum'],m[w]['MC40']['net_sum']) for w in m)
audit['risk_gate']=all(m[w]['SWA']['sharpe']>=m[w]['MC40']['sharpe'] and m[w]['SWA']['mdd']>=m[w]['MC40']['mdd'] for w in m)
audit['passed']=True
(out/'local_final_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print('PASS',audit['return_gate'],audit['risk_gate'],sum(audit['requests'].values()),audit['accounts'])
