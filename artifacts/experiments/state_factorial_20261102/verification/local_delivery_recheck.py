from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[4];p=root/'artifacts/experiments/state_factorial_20261102'
m=json.loads((p/'metrics.json').read_text()); audit={'metrics':{},'mechanisms':{},'accounts':{}}
for w,n in [('W3',126),('W4',134)]:
 nets={}; base=pd.read_parquet(p/f'report_{w}_G1_original.parquet')
 for arm in ['G1_original','A','B','C','D']:
  f=pd.read_parquet(p/f'report_{w}_{arm}.parquet');assert f.index.equals(base.index) and len(f)==n
  np.testing.assert_allclose(f.bench,base.bench,rtol=0,atol=1e-13)
  x=(f['return']-f.cost).to_numpy();nets[arm]=x;nav=np.r_[1.,np.cumprod(1+x)]
  v={'net_sum':x.sum(),'compound':nav[-1]-1,'sharpe':np.sqrt(252)*x.mean()/x.std(ddof=1),'mdd':(nav/np.maximum.accumulate(nav)-1).min(),'cost':f.cost.sum(),'turnover':f.turnover.mean(),'days':n}
  for k,z in v.items():assert np.isclose(z,m['metrics'][f'{w}|{arm}'][k],rtol=0,atol=1e-12),(w,arm,k)
  daily=pd.read_parquet(p/f'daily_{w}_{arm}.parquet')
  curve_col='cumulative' if 'cumulative' in daily else 'cum_net'
  np.testing.assert_allclose(daily[curve_col],np.cumsum(x),atol=1e-12,rtol=0)
  audit['metrics'][f'{w}|{arm}']={k:float(z) for k,z in v.items()}
 contrasts={'state_window':nets['B']-nets['A'],'state_date':nets['D']-nets['C'],'sampling_static':nets['C']-nets['A'],'sampling_dynamic':nets['D']-nets['B'],'interaction':nets['D']-nets['C']-nets['B']+nets['A']}
 starts=np.random.default_rng(100).integers(0,n,size=(5000,(n+19)//20));idx=((starts[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
 for name,x in contrasts.items():
  lo,hi=np.quantile(x[idx].sum(axis=1)*100,[.025,.975]);expected=m['comparisons'][w][name]
  vals=[x.sum()*100,lo,hi];np.testing.assert_allclose(vals,[expected[k] for k in ['observed_sum_pp','lower_95_pp','upper_95_pp']],rtol=0,atol=1e-10)
  audit['mechanisms'][f'{w}|{name}']=[float(v) for v in vals]
for arm in ['G1_original','B','D']:
 d=p/'risk'/arm; pos=pd.read_parquet(d/'positions.parquet').set_index(['date','instrument']); trades=pd.read_parquet(d/'trades.parquet'); bal=pd.read_parquet(d/'balances.parquet').set_index('date');con=pd.read_parquet(d/'stock_daily_contributions.parquet').set_index(['date','instrument']);report=pd.read_parquet(d/'report.parquet')
 tx=trades.set_index(['date','instrument']).join(pos[['end_price']]);tx['pnl_trade']=tx.quantity*(tx.end_price-tx.price);tx['cash_used']=tx.quantity*tx.price+tx.cost
 sums=tx.groupby(level=[0,1])[['quantity','pnl_trade','cost']].sum().reindex(pos.index).fillna(0)
 np.testing.assert_allclose(pos.end_amount-pos.start_amount,sums.quantity,rtol=0,atol=1e-8)
 pnl=pos.start_amount*(pos.end_price-pos.start_price)+sums.pnl_trade
 np.testing.assert_allclose(pnl,con.pnl,rtol=0,atol=1e-6)
 gross=(pnl/pos.previous_nav).groupby('date').sum().reindex(report.index,fill_value=0)
 cost=(sums.cost/pos.previous_nav).groupby('date').sum().reindex(report.index,fill_value=0)
 np.testing.assert_allclose(gross,report['return'],rtol=0,atol=1e-12);np.testing.assert_allclose(cost,report.cost,rtol=0,atol=1e-12)
 stocks=(pos.end_amount*pos.end_price).groupby('date').sum().reindex(bal.index,fill_value=0)
 cash=bal.start_cash-tx.groupby('date').cash_used.sum().reindex(bal.index,fill_value=0)
 np.testing.assert_allclose(cash,bal.end_cash,atol=1e-6,rtol=0)
 np.testing.assert_allclose(stocks+bal.end_cash,bal.nav,atol=1e-6,rtol=0)
 net=report['return']-report.cost; c=con.net_contribution.unstack('instrument').reindex(report.index).fillna(0);res=net-c.sum(axis=1)
 mat=np.column_stack([c.to_numpy(),res.to_numpy()]);v=((mat-mat.mean(axis=0))*(net.to_numpy()-net.mean())[:,None]).sum(axis=0)/(len(net)-1)
 assert abs(v.sum()-net.var(ddof=1))<1e-16
 audit['accounts'][arm]={'gross_residual':float(abs(gross-report['return']).max()),'cost_residual':float(abs(cost-report.cost).max()),'variance_residual':float(v.sum()-net.var(ddof=1))}
ends=[];request_count=0
for f in p.rglob('*.jsonl'):
 if 'requests' not in f.name:continue
 for line in f.read_text().splitlines():
  if not line.strip():continue
  rec=json.loads(line)
  if 'effective_end' in rec:
   dt=pd.Timestamp(rec['effective_end']);assert dt<=pd.Timestamp('2026-07-24 23:59:59'),(f,dt)
   ends.append(dt);request_count+=1
assert request_count>0
assert hashlib.sha256((p/'arms/C/best_adapter.pt').read_bytes()).hexdigest()=='b09ee8f15bfdf23968f08ef7d62d7383803e5b30ad60fedece06ca196cb7f55f'
audit.update(passed=True,market_request_count=request_count,max_market_request_end=str(max(ends)))
(p/'local_final_audit_20261102.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('PASS: 10 reports and curves, 10 paired mechanism CIs, 3 trade/holding ledgers and variance reconciliation, frozen C weight, market request cutoff',request_count,max(ends))
