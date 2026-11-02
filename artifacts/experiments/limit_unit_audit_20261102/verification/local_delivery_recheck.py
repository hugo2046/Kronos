from pathlib import Path
import json
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parents[1];metrics=pd.read_csv(p/'metrics.csv');out={'metrics':{},'accounts':{},'mechanisms':{}};nets={}
for row in metrics.itertuples():
 d=p/'accounts'/row.mode/row.window/row.arm;report_path=d/'report.parquet'
 if not report_path.exists():
  assert row.mode=='old' and row.arm in ['A','C']
  report_path=p.parent/'state_factorial_20261102'/f'report_{row.window}_{row.arm}.parquet'
 f=pd.read_parquet(report_path);n=126 if row.window=='W3' else 134
 assert len(f)==n and f.index.is_unique and f.index.is_monotonic_increasing
 x=(f['return']-f.cost).to_numpy();nets[(row.mode,row.window,row.arm)]=x;nav=np.r_[1.,np.cumprod(1+x)]
 vals={'net_sum':x.sum(),'compound':nav[-1]-1,'sharpe':np.sqrt(252)*x.mean()/x.std(ddof=1),'mdd':(nav/np.maximum.accumulate(nav)-1).min(),'annual_vol':np.sqrt(252)*x.std(ddof=1),'cost':f.cost.sum(),'turnover':f.turnover.mean()}
 for k,v in vals.items():assert abs(v-getattr(row,k))<1e-12,(row.mode,row.window,row.arm,k)
 out['metrics']['/'.join([row.mode,row.window,row.arm])]={k:float(v) for k,v in vals.items()}
for bpath in (p/'accounts').rglob('balances.parquet'):
 d=bpath.parent;bal=pd.read_parquet(bpath).set_index('date');pos=pd.read_parquet(d/'positions.parquet').set_index(['date','instrument']);tx=pd.read_parquet(d/'trades.parquet').set_index(['date','instrument']);f=pd.read_parquet(d/'report.parquet')
 t=tx.join(pos[['end_price']]);t['trade_pnl']=t.quantity*(t.end_price-t.price);t['cash_used']=t.quantity*t.price+t.cost
 z=t.groupby(level=[0,1])[['quantity','trade_pnl','cost']].sum().reindex(pos.index).fillna(0)
 np.testing.assert_allclose(pos.end_amount-pos.start_amount,z.quantity,atol=1e-8,rtol=0)
 pnl=pos.start_amount*(pos.end_price-pos.start_price)+z.trade_pnl
 gross=(pnl/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0);cost=(z.cost/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0)
 np.testing.assert_allclose(gross,f['return'],atol=1e-12,rtol=0);np.testing.assert_allclose(cost,f.cost,atol=1e-12,rtol=0)
 cash=bal.start_cash-t.groupby('date').cash_used.sum().reindex(bal.index,fill_value=0);stock=(pos.end_amount*pos.end_price).groupby('date').sum().reindex(bal.index,fill_value=0)
 np.testing.assert_allclose(cash,bal.end_cash,atol=1e-6,rtol=0);np.testing.assert_allclose(stock+bal.end_cash,bal.nav,atol=1e-6,rtol=0)
 out['accounts'][str(d.relative_to(p))]={'gross_error':float(abs(gross-f['return']).max()),'cost_error':float(abs(cost-f.cost).max())}
 if d.name=='neutral':
  base=pd.read_parquet(d.parent/'D/report.parquet');np.testing.assert_allclose(f,base,rtol=0,atol=1e-12)
fb=json.loads((p/'factorial_bootstrap.json').read_text())['windows']
for w,n in [('W3',126),('W4',134)]:
 a,b,c,d=[nets[('new',w,x)] for x in ['A','B','C','D']]
 contrasts={'B-A':b-a,'D-C':d-c,'C-A':c-a,'D-B':d-b,'D-C-B+A':d-c-b+a}
 starts=np.random.default_rng(100).integers(0,n,size=(5000,(n+19)//20));ix=((starts[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
 for k,x in contrasts.items():
  ci=np.quantile(x[ix].sum(axis=1)*100,[.025,.975]);np.testing.assert_allclose(ci,fb[w][k]['ci95_pp'],rtol=0,atol=1e-10);assert abs(x.sum()*100-fb[w][k]['delta_pp'])<1e-10
  out['mechanisms'][w+'/'+k]={'delta_pp':float(x.sum()*100),'ci95_pp':ci.tolist()}
f=pd.read_parquet(p/'fields_flags.parquet');v=f[~f.missing_fields]
for key,expect in [('old_buy',v['$change']>=.095),('old_sell',v['$change']<=-.095),('new_buy',v['$pctchange']/100>=.095),('new_sell',v['$pctchange']/100<=-.095)]:assert np.array_equal(v[key],expect),key
assert np.array_equal(v.new_buy,v['$pctchange / 100 >= 0.095'].astype(bool));assert np.array_equal(v.new_sell,v['$pctchange / 100 <= -0.095'].astype(bool))
ratio=v['$close']/v['$preclose']-1;dis=int(((ratio>=.095)!=v.new_buy).sum()+((ratio<=-.095)!=v.new_sell).sum());assert dis==0
assert (((v['$close']-v['$preclose'])/v['$factor']-v['$change']).abs()<=.011).all()
count=0;max_end=pd.Timestamp.min
for f in p.rglob('*requests*.jsonl'):
 for line in f.read_text().splitlines():
  if not line.strip():continue
  row=json.loads(line)
  if 'effective_end' in row:
   dt=pd.Timestamp(row['effective_end']);assert dt<=pd.Timestamp('2026-07-24 23:59:59');count+=1;max_end=max(max_end,dt)
out.update(passed=True,field_valid_rows=len(v),flags_using_price_ratio_disagree=dis,market_requests=count,max_effective_end=str(max_end))
(p/'local_independent_audit_20261102.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('PASS:',len(out['metrics']),'metric rows;',len(out['accounts']),'actual accounts; 10 joint-date bootstrap contrasts; field units/flags; cutoff requests',count)
