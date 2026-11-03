"""从落盘信号、账户和日报独立核验续训结果，不调用训练或回测引擎。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(__file__).resolve().parents[1]
prior = out.parent / 'sam_adapter_20261102'
hashfile = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((out / name).read_text())
audit = {'metrics': {}, 'accounts': {}, 'streams': {}, 'intervals': {}, 'requests': {}, 'training_batches': {}}
metrics = read('metrics.json')
protocol = read('protocol.json')
sha = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert sha == '543a2d3133152ab196f0f3ef84ee0d20bc5358aee92d40de123ba3fe65fd9d79' == metrics['protocol_sha']
for arm in ('sam','control'):
    b = pd.read_json(out/f'batches_{arm}.jsonl',lines=True)
    np.testing.assert_array_equal(b.step, np.arange(3001,6001))
    audit['training_batches'][arm] = {str(start): float(b.loc[b.step.between(start,start+499),'loss'].mean()) for start in [5001,5501]}
for w,n,k in [('W3',126,37223),('W4',134,39563)]:
    base = pd.read_parquet(out.parent/'state_factorial_20261102'/f'signal_{w}_G1_original.parquet').sort_index().sort_index(axis=1)
    assert len(base)==n and np.isfinite(base).to_numpy().sum()==k
    nets = {}
    for tag in ['G1_original','MC40','SAM3000','AdamW6000','SAM6000']:
        f = pd.read_parquet(out/f'report_{w}_{tag}.parquet')
        assert f.index.equals(base.index) and f.index.is_unique
        x = (f['return']-f.cost).to_numpy(); nets[tag]=x
        nav = np.r_[1.,np.cumprod(1+x)]
        v = dict(net_sum=x.sum(),compound=nav[-1]-1,sharpe=np.sqrt(252)*x.mean()/x.std(ddof=1),mdd=np.min(nav/np.maximum.accumulate(nav)-1),cost=f.cost.sum(),turnover=f.turnover.mean(),days=n)
        np.testing.assert_allclose(list(v.values()),[metrics['metrics'][w][tag][key] for key in v],rtol=0,atol=1e-12)
        assert abs(np.cumsum(x)[-1]-read('figure_endpoints.json')[w+'/'+tag])<1e-12
        audit['metrics'][w+'/'+tag]=v
        if tag not in ['AdamW6000','SAM6000']:
            parent = 'SAM_MC40' if tag=='SAM3000' else tag
            pd.testing.assert_frame_equal(f,pd.read_parquet(prior/f'report_{w}_{parent}.parquet'),atol=1e-12,rtol=0)
            continue
        arm = 'sam' if tag=='SAM6000' else 'control'
        weight = hashfile(out/'arms'/arm/'best_adapter.pt'); assert weight==read(f'training_{arm}.json')['adapter_sha256']
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
        window,pair=key.split('/');a,b=pair.split('-')
        if window!=w:continue
        diff=nets[a]-nets[b];q=np.quantile(diff[indices].sum(axis=1),[.025,.975]);record=item['bootstrap']
        np.testing.assert_allclose([diff.sum(),*q],[item['delta'],record['lower_95'],record['upper_95']],atol=1e-12,rtol=0);audit['intervals'][key]=(q*100).tolist()
for path in [*out.glob('market_*.jsonl'),*(out/'accounts').rglob('market_requests.jsonl')]:
    rows=list(map(json.loads,path.read_text().splitlines())); ends=[pd.Timestamp(row['effective_end']) for row in rows]
    assert ends and max(ends)<=pd.Timestamp('2026-07-24');audit['requests'][str(path.relative_to(out))]=len(rows)
assert len(audit['accounts'])==4 and len(audit['metrics'])==10 and len(audit['intervals'])==12 and sum(v[0] for v in audit['streams'].values())==1040
assert sum(audit['requests'].values())==read('report_recalculation.json')['market_request_count']
audit['passed']=True
(out/'local_final_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print('PASS',audit['training_batches'],sum(audit['requests'].values()))
