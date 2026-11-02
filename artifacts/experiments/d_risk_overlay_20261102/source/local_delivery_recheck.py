from pathlib import Path
import json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[4];p=root/'artifacts/experiments/d_risk_overlay_20261102'
m=json.loads((p/'metrics.json').read_text());comparisons=json.loads((p/'comparisons.json').read_text());audit={'metrics':{},'accounts':{},'risk_panels':{},'orders':{}}
for w,n in [('W3',126),('W4',134)]:
 base=pd.read_parquet(p/f'report_{w}_G1_original.parquet');nets={}
 for arm in ['G1_original','B','D','neutral','cap_only','vol_only','cap_vol']:
  f=pd.read_parquet(p/f'report_{w}_{arm}.parquet');assert f.index.equals(base.index) and len(f)==n
  np.testing.assert_allclose(f.bench,base.bench,rtol=0,atol=1e-12)
  x=(f['return']-f.cost).to_numpy();nets[arm]=x;nav=np.r_[1.,np.cumprod(1+x)]
  vals={'net_sum':x.sum(),'compound':nav[-1]-1,'sharpe':np.sqrt(252)*x.mean()/x.std(ddof=1),'mdd':(nav/np.maximum.accumulate(nav)-1).min(),'annual_vol':np.sqrt(252)*x.std(ddof=1),'cost':f.cost.sum(),'turnover':f.turnover.mean()}
  if arm!='neutral':
   for k,v in vals.items():assert abs(v-m[w][arm][k])<1e-12,(w,arm,k)
  daily=pd.read_parquet(p/f'daily_{w}_{arm}.parquet');np.testing.assert_allclose(daily.cumulative,np.cumsum(x),rtol=0,atol=1e-12)
  d=p/'accounts'/w/arm;pos=pd.read_parquet(d/'positions.parquet').set_index(['date','instrument']);tx=pd.read_parquet(d/'trades.parquet').set_index(['date','instrument']);bal=pd.read_parquet(d/'balances.parquet').set_index('date')
  t=tx.join(pos[['end_price']]);t['pnl_trade']=t.quantity*(t.end_price-t.price);t['cash_used']=t.quantity*t.price+t.cost
  sums=t.groupby(level=[0,1])[['quantity','pnl_trade','cost']].sum().reindex(pos.index).fillna(0)
  np.testing.assert_allclose(pos.end_amount-pos.start_amount,sums.quantity,atol=1e-8,rtol=0)
  pnl=pos.start_amount*(pos.end_price-pos.start_price)+sums.pnl_trade
  gross=(pnl/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0);cost=(sums.cost/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0)
  np.testing.assert_allclose(gross,f['return'],atol=1e-12,rtol=0);np.testing.assert_allclose(cost,f.cost,atol=1e-12,rtol=0)
  cash=bal.start_cash-t.groupby('date').cash_used.sum().reindex(bal.index,fill_value=0)
  stock=(pos.end_amount*pos.end_price).groupby('date').sum().reindex(bal.index,fill_value=0)
  np.testing.assert_allclose(cash,bal.end_cash,rtol=0,atol=1e-6);np.testing.assert_allclose(stock+bal.end_cash,bal.nav,rtol=0,atol=1e-6)
  audit['accounts'][w+'_'+arm]={'gross_max_error':float(abs(gross-f['return']).max()),'cost_max_error':float(abs(cost-f.cost).max())}
  audit['metrics'][w+'_'+arm]={k:float(v) for k,v in vals.items()}
  order_file=d/'order_audit.parquet'
  if order_file.exists():
   orders=pd.read_parquet(order_file)
   if orders.empty:continue
   buys=orders[orders.kind=='buy']
   assert (buys.after<=buys.before*buys.factor+1e-8).all()
   assert (pd.to_datetime(orders.decision_date)<pd.to_datetime(orders.date)).all()
   audit['orders'][w+'_'+arm]={'audited_rows':len(orders),'buy_discount_cap_passed':True,'decision_before_execution':True}
 np.testing.assert_allclose(nets['neutral'],nets['D'],rtol=0,atol=1e-12)
 starts=np.random.default_rng(100).integers(0,n-20+1,size=(5000,int(np.ceil(n/20))))
 idx=(starts[:,:,None]+np.arange(20)).reshape(5000,-1)[:,:n]
 for ref in ['G1_original','B','D']:
  dif=nets['cap_vol']-nets[ref]; draws=dif[idx].sum(axis=1)*100
  np.testing.assert_allclose(np.quantile(draws,[.025,.975]),comparisons[w][ref]['bootstrap']['ci95_pp'],rtol=0,atol=1e-10)
 panel=pd.read_parquet(p/f'risk_panel_{w}.parquet')
 sigma=panel.sigma; med=panel.median_sigma
 expected=np.sqrt(med/sigma.clip(lower=1e-8)).clip(.5,1).fillna(.5)
 np.testing.assert_allclose(expected,panel.factor,rtol=0,atol=1e-12)
 assert (pd.to_datetime(panel.last_history_date.dropna()).to_numpy()<=panel.index.get_level_values('date')[panel.last_history_date.notna()].to_numpy()).all()
 outside=(~panel.signal_valid)&panel.sigma.notna()
 audit['risk_panels'][w]={'rows':len(panel),'valid_sigma_outside_signal_pool':int(outside.sum()),'factor_and_history_date_passed':True}
ends=[]
for f in p.rglob('*requests*.jsonl'):
 for line in f.read_text().splitlines():
  if not line.strip():continue
  row=json.loads(line)
  if 'effective_end' in row:
   date=pd.Timestamp(row['effective_end']);assert date<=pd.Timestamp('2026-07-24 23:59:59');ends.append(date)
assert ends
audit.update(passed=True,market_requests=len(ends),max_effective_end=str(max(ends)),gpu_seconds=0)
(p/'local_independent_audit_20261102.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('PASS: 12 metrics, 14 curves/accounts, neutral identity, 6 registered non-circular block intervals, risk factor boundary/order causality, request cutoff',len(ends))
