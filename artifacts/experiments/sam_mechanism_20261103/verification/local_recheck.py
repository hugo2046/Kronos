"""独立于远端实现复算账户、几何记录和图端点。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter

root=Path(__file__).resolve().parents[4]
out=root/'artifacts/experiments/sam_mechanism_20261103'
old=root/'artifacts/experiments/sam_adapter_20261102'
s=json.loads((out/'summary.json').read_text())
g=json.loads((out/'geometry.json').read_text())
e=json.loads((out/'figure_endpoints.json').read_text())
checks={}; data={}
for w in ('W3','W4'):
 for a in ('G1_original','MC40','SAM_MC40'):
  r=pd.read_parquet(out/f'report_{w}_{a}.parquet')
  original=pd.read_parquet(old/'accounts'/w/a/'report.parquet')
  pd.testing.assert_frame_equal(r,original)
  x=(r['return']-r['cost']).to_numpy()
  nav=np.concatenate(([1.],np.cumprod(1+x)))
  value={'net_sum':float(np.sum(x)),'compound':float(nav[-1]-1),'sharpe':float(x.mean()/np.std(x,ddof=1)*np.sqrt(252)),
         'mdd':float(np.min(nav/np.maximum.accumulate(nav)-1)),'cost':float(r.cost.sum()),'turnover':float(r.turnover.mean()),
         'annual_vol':float(np.std(x,ddof=1)*np.sqrt(252))}
  assert max(abs(value[k]-s['stats'][w][a]['metrics'][k]) for k in value)<1e-12
  assert abs(x.cumsum()[-1]-e[w+'_'+a])<1e-12
  contrib=pd.read_parquet(old/'accounts'/w/a/'stock_daily_contributions.parquet')
  contrib['date']=pd.to_datetime(contrib.date)
  daily=contrib.groupby('date')[['gross_contribution','cost_contribution','net_contribution']].sum().reindex(r.index,fill_value=0)
  assert np.max(np.abs(daily.to_numpy()-np.c_[r['return'],r.cost,x]))<1e-12
  positions=pd.read_parquet(old/'accounts'/w/a/'positions.parquet')
  hhi=positions.assign(squared=positions.end_weight**2).groupby('date').squared.sum().mean()
  assert abs(hhi-s['stats'][w][a]['average_hhi'])<1e-14
  data[w,a]=(r,x,contrib)
  checks[w+'_'+a]=value
 for a in ('G1_original','MC40'):
  r,x,c=data[w,a];rr,y,cc=data[w,'SAM_MC40']
  rows=cc.groupby('instrument').net_contribution.sum().sub(c.groupby('instrument').net_contribution.sum(),fill_value=0)
  recorded=pd.read_csv(out/f'stock_difference_{w}_{a}.csv',index_col=0)
  assert np.max(np.abs(recorded.net_difference-rows.reindex(recorded.index)))<1e-12
  assert abs(recorded.net_difference.sum()-np.sum(y-x))<1e-12
  dec=s['attribution'][w+'_'+a]['sharpe']
  assert abs(dec['mean_effect']+dec['vol_effect']-np.sqrt(252)*(y.mean()/y.std(ddof=1)-x.mean()/x.std(ddof=1)))<1e-12
  assert abs(recorded.variance_difference_contribution.sum()-(y.var(ddof=1)-x.var(ddof=1)))<1e-12
for a in ('MC40','SAM'):
 clean=g[a]['clean'];assert clean['counts']==[1024,1024]
 assert clean['full_ce']==(clean['fit']['ce']+clean['crosscheck']['ce'])/2
 assert abs(clean['full_ce']-json.loads((old/('diagnostic_reference.json' if a=='MC40' else 'diagnostic_sam.json')).read_text())['zero_ce'])<1e-12
 for attack in g[a]['attacks']:
  assert attack['parameters_restored'] and g[a]['base_restored']
  assert abs(attack['best_fit_ce']-max(p['loss'] for p in attack['path']))<1e-12
  assert abs(attack['best_fit_ce']-attack['perturbed']['fit']['ce'])<1e-12
  assert max(p['norm'] for p in attack['path']) <= attack['rho']+1e-7
  assert abs(sum(l['delta_norm']**2 for l in attack['layers'])**.5-attack['rho'])<1e-7
font=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
for ax,w in zip(axes,['W3','W4']):
 for a,label,color in [('G1_original','SAM 减 原G1','#52657b'),('MC40','SAM 减 状态适配MC40','#00857d')]:
  r,x,_=data[w,a];_,y,_=data[w,'SAM_MC40']
  ax.plot(r.index,np.cumsum(y-x)*100,label=label,color=color)
 ax.axhline(0,color='#aaa',lw=.8);ax.set_title(w);ax.set_ylabel('净累计差额（百分点）');ax.grid(alpha=.2);ax.legend()
fig.suptitle('冻结SAM适配器的累计收益差额｜已有账户独立复算')
fig.tight_layout();fig.savefig(out/'figures/cumulative_difference.png',dpi=170)
result={'passed':True,'metrics':checks,'account_reconciliation':True,'stock_and_variance_differences':True,
        'geometry_records_and_original_ce':True,'figure_endpoints':True,'source_reports_equal':True,
        'note':'几何值内部一致性和原CE复核，不等于独立重跑GPU梯度；数值正确性另由CPU解析测试与独立代码审查覆盖。'}
(out/'local_independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS: 6组日报与指标、贡献差、方差差、Sharpe分解、图端点、几何记录/原CE。')
