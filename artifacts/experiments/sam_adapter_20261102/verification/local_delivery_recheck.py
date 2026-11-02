from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parents[1];data=json.loads((p/'metrics.json').read_text());audit={'metrics':{},'accounts':{},'comparisons':{},'signals':{}}
for w,n in [('W3',126),('W4',134)]:
 nets={};base=pd.read_parquet(p/f'report_{w}_G1_original.parquet')
 for tag in ['G1_original','MC40','SAM_MC40']:
  f=pd.read_parquet(p/f'report_{w}_{tag}.parquet');assert len(f)==n and f.index.equals(base.index) and f.index.is_unique
  np.testing.assert_allclose(f.bench,base.bench,atol=1e-12,rtol=0)
  x=(f['return']-f.cost).to_numpy();nets[tag]=x;nav=np.r_[1.,np.cumprod(1+x)]
  v={'net_sum':x.sum(),'compound':nav[-1]-1,'sharpe':np.sqrt(252)*x.mean()/x.std(ddof=1),'mdd':(nav/np.maximum.accumulate(nav)-1).min(),'cost':f.cost.sum(),'turnover':f.turnover.mean(),'volatility':np.sqrt(252)*x.std(ddof=1),'days':n}
  for k,z in v.items():assert abs(z-data['metrics'][w][tag][k])<1e-12,(w,tag,k)
  daily=pd.read_parquet(p/f'daily_{w}_{tag}.parquet');np.testing.assert_allclose(daily.cumulative,np.cumsum(x),atol=1e-12,rtol=0)
  ep=json.loads((p/'figure_endpoints.json').read_text());assert abs(ep[f'{w}/{tag}']-v['net_sum'])<1e-12
  audit['metrics'][f'{w}/{tag}']={k:float(z) for k,z in v.items()}
  d=p/'accounts'/w/tag;pos=pd.read_parquet(d/'positions.parquet').set_index(['date','instrument']);tx=pd.read_parquet(d/'trades.parquet').set_index(['date','instrument']);bal=pd.read_parquet(d/'balances.parquet').set_index('date')
  t=tx.join(pos[['end_price']]);t['trade_pnl']=t.quantity*(t.end_price-t.price);t['cash_used']=t.quantity*t.price+t.cost
  sums=t.groupby(level=[0,1])[['quantity','trade_pnl','cost']].sum().reindex(pos.index).fillna(0)
  np.testing.assert_allclose(pos.end_amount-pos.start_amount,sums.quantity,atol=1e-8,rtol=0)
  pnl=pos.start_amount*(pos.end_price-pos.start_price)+sums.trade_pnl
  gross=(pnl/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0);cost=(sums.cost/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0)
  np.testing.assert_allclose(gross,f['return'],atol=1e-12,rtol=0);np.testing.assert_allclose(cost,f.cost,atol=1e-12,rtol=0)
  cash=bal.start_cash-t.groupby('date').cash_used.sum().reindex(bal.index,fill_value=0);stock=(pos.end_amount*pos.end_price).groupby('date').sum().reindex(bal.index,fill_value=0)
  np.testing.assert_allclose(cash,bal.end_cash,atol=1e-6,rtol=0);np.testing.assert_allclose(stock+bal.end_cash,bal.nav,atol=1e-6,rtol=0)
  audit['accounts'][f'{w}/{tag}']={'gross_error':float(abs(gross-f['return']).max()),'cost_error':float(abs(cost-f.cost).max())}
 streams=[];base_signal=pd.read_parquet(p.parent/'state_factorial_20261102'/f'signal_{w}_G1_original.parquet').sort_index().sort_index(axis=1)
 for seed in [42,100]:
  q=p/'signals'/f'SAM_train100_infer{seed}'/f'{w}.parquet';f=pd.read_parquet(q).T;f.index=pd.to_datetime(f.index);f=f.sort_index().sort_index(axis=1)
  assert f.index.equals(base_signal.index) and f.columns.equals(base_signal.columns);np.testing.assert_array_equal(np.isfinite(f),np.isfinite(base_signal))
  meta=json.loads(q.with_suffix('.meta.json').read_text());assert meta['complete'] and meta['signal_sha256']==hashlib.sha256(q.read_bytes()).hexdigest()
  assert meta['adapter_sha256']=='9ed6b8d808276dc1b4d0de20646ed61f41c91638cac07700e36f23085a60ef28'
  streams.append(f)
 pooled=pd.read_parquet(p/f'signal_{w}_SAM_MC40.parquet').sort_index().sort_index(axis=1)
 pd.testing.assert_frame_equal(pooled,(streams[0]+streams[1])*.5,check_names=False,check_exact=True)
 audit['signals'][w]={'dates':len(pooled),'finite':int(np.isfinite(pooled).sum().sum()),'exact_pooling':True}
 starts=np.random.default_rng(100).integers(0,n,size=(5000,(n+19)//20));idx=((starts[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
 for tag in ['G1_original','MC40']:
  diff=nets['SAM_MC40']-nets[tag];ci=np.quantile(diff[idx].sum(axis=1),[.025,.975]);expected=data['comparisons'][f'{w}/{tag}']['bootstrap']
  np.testing.assert_allclose(ci,[expected['lower_95'],expected['upper_95']],atol=1e-12,rtol=0)
  audit['comparisons'][f'{w}/{tag}']={'delta_pp':float(diff.sum()*100),'ci95_pp':(ci*100).tolist()}
count=0;latest=pd.Timestamp.min
for q in p.rglob('*requests*.jsonl'):
 for line in q.read_text().splitlines():
  if not line.strip():continue
  rec=json.loads(line)
  if 'effective_end' in rec:
   t=pd.Timestamp(rec['effective_end']);assert t<=pd.Timestamp('2026-07-24');count+=1;latest=max(latest,t)
assert hashlib.sha256((p/'arms/sam/best_adapter.pt').read_bytes()).hexdigest()=='9ed6b8d808276dc1b4d0de20646ed61f41c91638cac07700e36f23085a60ef28'
audit.update(passed=True,market_requests=count,max_market_date=str(latest))
(p/'local_final_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print('PASS: six metrics/curves/accounts, four stream masks, exact pooling, four paired intervals, frozen candidate and cutoff',count)
