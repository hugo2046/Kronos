"""协调器独立复算：直接读取事务、头参数与账户，不调用实验实现。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
R=Path('/home/user/workspace/Kronos-g1-risk-head/artifacts/experiments/g1_risk_head_20261105')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((R/'protocol.json').read_text());gate=json.loads((R/'review_gate.json').read_text())
psha=hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False).encode()).hexdigest();assert psha==gate['protocol_sha']
audit={'transactions':{},'accounts':{},'metrics':{},'requests':{},'prediction':{}}
frames={}
for stage in ['train','Q1','Q2','W3','W4']:
 folder=R/'features'/stage
 if not folder.exists():continue
 cal=json.loads((folder/'calendar.json').read_text());dates=cal['dates'];records=[]
 expected={'protocol':psha,'sources':gate['source_hashes']}
 assert cal['identity']==expected
 for date in dates:
  f=folder/f'{date}.npz';m=json.loads(f.with_suffix('.json').read_text());assert m['identity']==expected and sha(f)==m['sha']
  with np.load(f,allow_pickle=False) as d:
   assert len(d['history_dates'])==90 and pd.Timestamp(d['history_dates'][-1])==pd.Timestamp(date)
   assert len(d['future_dates'])==10 and pd.Timestamp(d['future_dates'][0])>pd.Timestamp(date)
   raw=d['raw'];assert raw.shape==(len(d['codes']),90,6) and np.isfinite(raw).all()
   close=raw[:,-21:,3];scale=np.maximum((close[:,1:]/close[:,:-1]-1).std(1,ddof=1)*np.sqrt(10),1e-4)
   np.testing.assert_array_equal(scale,d['scale'])
   cutoff=p[stage][1] if stage in ['train','Q1','Q2'] else p['windows'][stage][1]
   labels=d['label_raw'];y=np.full(len(raw),np.nan)
   if pd.Timestamp(d['future_dates'][-1])<=pd.Timestamp(cutoff):
    op=labels[:,0,0];cl=labels[:,-1,1];ok=np.isfinite(op)&np.isfinite(cl)&(op>0)&(cl>0);y[ok]=cl[ok]/op[ok]-1
   else:assert np.isnan(labels).all()
   np.testing.assert_allclose(y,d['y'],rtol=0,atol=0,equal_nan=True)
   records.append(dict(date=date,stocks=len(raw),valid=int(np.isfinite(y).sum()),sha=m['sha']))
 audit['transactions'][stage]=dict(days=len(records),cells=sum(x['stocks'] for x in records),labels=sum(x['valid'] for x in records))
 frames[stage]=records
# 校验原始读取记录，覆盖独立全格审计。
for f in [R/'market_requests.jsonl',R/'coverage/market_requests.jsonl',R/'parity_requests.jsonl',R/'evaluation_requests.jsonl']:
 if not f.exists():continue
 rows=[json.loads(line) for line in f.read_text().splitlines()];ends=[]
 for row in rows:
  end=row.get('end',row.get('effective_end'))
  if end is None:continue
  end=pd.Timestamp(end);assert end<=pd.Timestamp('2026-07-24');ends.append(end)
  if 'as_of' in row:assert end<=pd.Timestamp(row['as_of'])
 audit['requests'][str(f.relative_to(R))]=dict(rows=len(rows),max_end=str(max(ends)))
mech=json.loads((R/'mechanism_gate.json').read_text());assert mech['identity']['protocol']==psha and mech['identity']['sources']==gate['source_hashes']
assert mech['head_sha']==sha(R/'head.pt')
ck=torch.load(R/'head.pt',map_location='cpu',weights_only=False);assert ck['identity']==mech['identity']
actual_features={str(f.relative_to(R)):sha(f) for stage in ['train','Q1','Q2'] for f in (R/'features'/stage).glob('*.json')}
assert ck['identity']['feature_sha']==actual_features
q2=pd.read_parquet(R/'Q2_diagnostics.parquet');recomputed=[]
for row in frames['Q2']:
 date=row['date']
 with np.load(R/'features/Q2'/f'{date}.npz') as d:
  ok=np.isfinite(d['y']);h=torch.from_numpy(d['hidden'])
  with torch.no_grad():q=(h@ck['head']['linear.weight'].T+ck['head']['linear.bias']).sort(-1).values.numpy()*d['scale'][:,None]
  expected=pd.DataFrame({'date':date,'code':d['codes'][ok],'y':d['y'][ok], 'scale':d['scale'][ok]})
  top=np.zeros(len(ok),bool);top[np.lexsort((d['codes'],-d['g1']))[:50]]=True;expected['top50']=top[ok]
  for j,key in enumerate(['q10','q50','q90']):expected[key]=q[ok,j]
  recomputed.append(expected)
expected=pd.concat(recomputed,ignore_index=True);actual=q2[['date','code','y','scale','top50','q10','q50','q90']]
pd.testing.assert_frame_equal(expected,actual,check_dtype=False,atol=2e-7,rtol=1e-6)
for name,cols in [('learned',['q10','q50','q90']),('volatility',['base_q10','base_q50','base_q90'])]:
 err=q2.y.to_numpy()[:,None]-q2[cols].to_numpy();q=np.array([.1,.5,.9]);loss=np.maximum(err*q,err*(q-1))
 daily=pd.DataFrame(loss,index=q2.date).groupby(level=0).mean()
 np.testing.assert_allclose(daily.to_numpy().mean(),mech['metrics'][name]['pinball'],atol=1e-12,rtol=0)
 np.testing.assert_allclose(daily.iloc[:,0].mean(),mech['metrics'][name]['p10'],atol=1e-12,rtol=0)
 audit['prediction'][name]=dict(pinball=float(daily.to_numpy().mean()),p10=float(daily.iloc[:,0].mean()))
result=json.loads((R/'result.json').read_text());endpoints=json.loads((R/'figure_endpoints.json').read_text())
for w in ['W3','W4']:
 for tag,m in result['metrics'][w].items():
  if tag in ['learned','volatility']:d=R/'accounts'/w/tag;f=pd.read_parquet(d/'report.parquet')
  else:
   name='B' if tag=='MC40' else 'G1_original';d=Path('/home/user/workspace/Kronos-limit-audit/artifacts/experiments/limit_unit_audit_20261102/accounts/new')/w/name;f=pd.read_parquet(d/'report.parquet')
  x=(f['return']-f.cost).to_numpy();nav=np.r_[1.,np.cumprod(1+x)]
  v=dict(annualized=float(np.prod(1+x)**(252/len(x))-1),net_sum=x.sum(),compound=nav[-1]-1,sharpe=np.sqrt(252)*x.mean()/x.std(ddof=1),mdd=np.min(nav/np.maximum.accumulate(nav)-1),cost=f.cost.sum(),turnover=f.turnover.mean())
  np.testing.assert_allclose(list(v.values()),[m[k] for k in v],atol=1e-12,rtol=0)
  assert abs(x.sum()-endpoints[f'cumulative/{w}/{tag}'])<1e-12
  audit['metrics'][f'{w}/{tag}']=v
  if tag not in ['learned','volatility']:continue
  pos=pd.read_parquet(d/'positions.parquet').set_index(['date','instrument']);tx=pd.read_parquet(d/'trades.parquet').set_index(['date','instrument']);bal=pd.read_parquet(d/'balances.parquet').set_index('date')
  t=tx.join(pos[['end_price']]);t['trade_pnl']=t.quantity*(t.end_price-t.price);t['cash_used']=t.quantity*t.price+t.cost
  sums=t.groupby(level=[0,1])[['quantity','trade_pnl','cost']].sum().reindex(pos.index).fillna(0)
  np.testing.assert_allclose(pos.end_amount-pos.start_amount,sums.quantity,atol=1e-8,rtol=0)
  pnl=pos.start_amount*(pos.end_price-pos.start_price)+sums.trade_pnl
  daily=(pnl/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0);cost=(sums.cost/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0)
  np.testing.assert_allclose(daily,f['return'],atol=1e-12,rtol=0);np.testing.assert_allclose(cost,f.cost,atol=1e-12,rtol=0)
  cash=bal.start_cash-t.groupby('date').cash_used.sum().reindex(bal.index,fill_value=0);equity=(pos.end_amount*pos.end_price).groupby('date').sum().reindex(bal.index,fill_value=0)
  np.testing.assert_allclose(cash,bal.end_cash,atol=1e-6,rtol=0);np.testing.assert_allclose(equity+bal.end_cash,bal.nav,atol=1e-6,rtol=0)
  audit['accounts'][w+'/'+tag]=float(abs(daily-f['return']).max())

if result['economic_complete']:
 audit['intervals']={};audit['signal_formula']={}
 for w in ['W3','W4']:
  native=pd.read_parquet(p['reference_paths'][w]['g142'])
  if not isinstance(native.index,pd.DatetimeIndex):native=native.T
  native.index=pd.to_datetime(native.index);native=native.sort_index().sort_index(axis=1)
  for tag in ['learned','volatility']:
   observed=pd.read_parquet(R/'signals'/f'{w}_{tag}.parquet')
   assert observed.index.equals(native.index) and set(observed.columns)==set(native.columns)
   np.testing.assert_array_equal(np.isfinite(observed.reindex(columns=native.columns)),np.isfinite(native))
   for date in native.index:
    with np.load(R/'features'/w/f'{date:%Y-%m-%d}.npz') as d:
     h=torch.from_numpy(d['hidden'])
     with torch.no_grad():q=(h@ck['head']['linear.weight'].T+ck['head']['linear.bias']).sort(-1).values.numpy()[:,0]
     riskq=(q if tag=='learned' else ck['constants'][0])*d['scale']
     g=native.loc[date].dropna();risk=pd.Series(np.maximum(0,-riskq),index=d['codes']).reindex(g.index)
     score=g.rank(method='average',pct=True)-.25*risk.rank(method='average',pct=True)
     order=np.lexsort((score.index.to_numpy(),-score.to_numpy()))
     expected_score=pd.Series(np.arange(len(score),0,-1,dtype=float),index=score.index[order]).reindex(g.index)
     pd.testing.assert_series_equal(expected_score,observed.loc[date].dropna().reindex(g.index),check_names=False,check_exact=True)
   audit['signal_formula'][w+'/'+tag]=True
  f=pd.read_parquet(R/'accounts'/w/'learned/report.parquet');a=(f['return']-f.cost).to_numpy();n=len(a)
  start=np.random.default_rng(100).integers(0,n,(5000,(n+19)//20));idx=((start[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
  def stat(x):
   x=np.atleast_2d(x);nav=np.concatenate([np.ones((len(x),1)),np.cumprod(1+x,axis=1)],axis=1)
   return {'net_sum':x.sum(1),'annualized':np.expm1(np.log1p(x).sum(1)*252/n),'sharpe':np.sqrt(252)*x.mean(1)/x.std(1,ddof=1),'mdd':np.min(nav/np.maximum.accumulate(nav,axis=1)-1,axis=1)}
  for tag in ['G1_original','MC40','volatility']:
   if tag=='volatility':f=pd.read_parquet(R/'accounts'/w/'volatility/report.parquet')
   else:
    name='B' if tag=='MC40' else 'G1_original';f=pd.read_parquet(Path('/home/user/workspace/Kronos-limit-audit/artifacts/experiments/limit_unit_audit_20261102/accounts/new')/w/name/'report.parquet')
   b=(f['return']-f.cost).to_numpy();sa=stat(a[idx]);sb=stat(b[idx]);oa=stat(a);ob=stat(b)
   for key in sa:
    d=sa[key]-sb[key];valid=np.isfinite(d);ci=np.quantile(d[valid],[.025,.975]);record=result['comparisons'][w][tag][key]
    np.testing.assert_allclose([float(oa[key][0]-ob[key][0]),*ci],[record['observed'],*record['ci95']],atol=1e-12,rtol=0)
   audit['intervals'][w+'/'+tag]=True

# 独立复核机制的选点和Top50 AUROC，不使用实验诊断函数。
history=mech['history'];assert mech['selected_epoch']==int(np.argmin([h['Q1'] for h in history]))
auc_rows=[]
for date,g in q2[q2.top50].groupby('date'):
 bad=(g.y.to_numpy()<mech['train_bad_threshold']);n1=int(bad.sum());n0=len(bad)-n1
 if not n1 or not n0:continue
 vals=[]
 for col in ['q10','base_q10']:
  risk=np.maximum(0,-g[col].to_numpy());ranks=pd.Series(risk).rank(method='average').to_numpy()
  vals.append(float((ranks[bad].sum()-n1*(n1+1)/2)/(n1*n0)))
 auc_rows.append(vals[0]-vals[1])
if auc_rows:
 delta=float(np.mean(auc_rows));assert abs(delta-mech['highscore_auc_delta'])<1e-12
else:
 delta=float('nan');assert mech['highscore_auc_delta'] is None
expected_pass=bool(mech['selected_epoch']>0 and audit['prediction']['learned']['p10']<audit['prediction']['volatility']['p10'] and np.isfinite(delta) and delta>0)
assert expected_pass==mech['passed'];audit['mechanism_passed']=expected_pass
if result['economic_complete']:
 strategy=[];increment=[]
 for w in ['W3','W4']:
  a=audit['metrics'][w+'/learned']
  for name in ['G1_original','MC40','volatility']:
   b=audit['metrics'][w+'/'+name]
   passed=a['annualized']>b['annualized'] and a['sharpe']>b['sharpe'] and (abs(a['mdd'])<=abs(b['mdd']) if name=='volatility' else abs(a['mdd'])<abs(b['mdd']))
   (increment if name=='volatility' else strategy).append(passed)
 eg=result['economic_gate'];assert eg['strategy_joint_passed']==all(strategy) and eg['learned_increment_passed']==all(increment) and eg['all_passed']==(all(strategy) and all(increment))
 audit['economic_gate']=eg
# 所有分组的逐日汇总均有固定20日循环块区间。
for group,models in mech.get('calibration',{}).items():
 for name,stats in models.items():
  daily=pd.read_csv(R/f'Q2_calibration_{group}_{name}.csv')
  if daily.empty:continue
  n=len(daily);starts=np.random.default_rng(100).integers(0,n,(5000,(n+19)//20));idx=((starts[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
  for col in ['pinball','p10','coverage10','coverage50','coverage90']:
   x=daily[col].to_numpy();ci=np.quantile(x[idx].mean(1),[.025,.975]);np.testing.assert_allclose([x.mean(),*ci],[stats[col]['mean'],*stats[col]['ci95']],atol=1e-12,rtol=0)
audit['calibration_intervals_passed']=True
audit['passed']=True
(R/'independent_final_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print('INDEPENDENT AUDIT PASS',audit['transactions'],audit['prediction'])
