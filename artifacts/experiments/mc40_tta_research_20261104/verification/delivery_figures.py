"""从独立复算账户生成交付图，不访问数据库。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter
OUT=Path(__file__).resolve().parents[1]
font=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc')
plt.rcParams['font.family']=font.get_name()
plt.rcParams['axes.unicode_minus']=False
colors={'原G1':'#536780','状态适配MC40':'#00857D','两标量适配 TTA2':'#DD753B'}
series={}
for w in ['W3','W4']:
 series[w]={}
 for name,folder,tag in [('原G1','root_reference_accounts','G1_original'),('状态适配MC40','root_reference_accounts','B'),('两标量适配 TTA2','accounts','MC40-TTA2')]:
  f=pd.read_parquet(OUT/folder/w/tag/'report.parquet');series[w][name]=f['return']-f.cost
endpoints={}
for kind,title,ylabel in [('cumulative','两标量测试期适配 · 扣费累计收益','每日净收益累加'),('drawdown','两标量测试期适配 · 复利净值回撤','相对历史峰值回撤'),('difference','两标量测试期适配 · 累计收益差额','候选减参照（百分点）')]:
 fig,axes=plt.subplots(1,2,figsize=(14,5.4),sharey=True)
 fig.suptitle(title,fontsize=20,x=.08,ha='left',y=.98,color='#283C4A')
 for ax,w in zip(axes,['W3','W4']):
  for name,s in series[w].items():
   if kind=='difference':
    if name=='两标量适配 TTA2':continue
    value=(series[w]['两标量适配 TTA2']-s).cumsum();label='TTA2 - '+name
   elif kind=='drawdown':
    nav=(1+s).cumprod();value=nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1;label=name
   else:value=s.cumsum();label=name
   ax.plot(value.index,value,label=label,color=colors[name],lw=2)
   endpoints[f'{kind}/{w}/{name}']=float(value.iloc[-1])
   if kind=='cumulative':ax.plot(value.index[-1],value.iloc[-1],'o',color=colors[name],ms=4)
  s=series[w]['原G1'];ax.set_title(f'{w}  {s.index[0]:%Y.%m.%d} — {s.index[-1]:%Y.%m.%d}',loc='left',fontsize=12)
  ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.18);ax.axhline(0,color='#8999A5',lw=.7)
  ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',rotation=25);ax.set_ylabel(ylabel)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.075,.90),ncol=3,frameon=False)
 fig.text(.08,.025,'研究性回放：当前历史最终值假设；保留旧测试越界事故。修复后隔离读取，所有曲线同一纠偏成交口径。',fontsize=10,color='#536780')
 fig.subplots_adjust(top=.73,bottom=.19,left=.08,right=.97,wspace=.14)
 fig.savefig(OUT/'figures'/f'{kind}.png',dpi=160);plt.close(fig)
(OUT/'verification/figure_endpoints.json').write_text(json.dumps(endpoints,ensure_ascii=False,indent=2)+'\n')
