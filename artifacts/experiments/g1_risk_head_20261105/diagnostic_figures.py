"""仅用已完成产物绘制训练与校准诊断。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parent
m=json.loads((R/'mechanism_gate.json').read_text());h=m['history'];font=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,3,figsize=(15,4.6))
x=[v['epoch'] for v in h];y=[v['Q1'] for v in h]
axes[0].plot(x,y,'o-',color='#DD753B');axes[0].scatter([0],[y[0]],color='#00857D',s=90,zorder=3)
axes[0].set_title('Q1选模：初始化最好');axes[0].set_xlabel('epoch（0为波动基线初始化）');axes[0].set_ylabel('日期等权pinball，越低越好');axes[0].set_xticks(x)
q=np.array([.1,.5,.9]);stats=m['calibration']['G1_top50']['learned'];observed=[stats[f'coverage{n}']['mean'] for n in [10,50,90]]
axes[1].plot(q,q,'--',color='#8999A5',label='名义覆盖率');axes[1].plot(q,observed,'o-',color='#DD753B',label='学习头＝波动基线');axes[1].set_title('Q2原G1 Top50：实际覆盖率');axes[1].set_xlabel('名义分位数');axes[1].set_ylabel('真实收益低于预测分位数的比例');axes[1].legend(fontsize=9)
a=pd.read_csv(R/'Q2_top50_auc.csv');good=a.learned_auc.notna()&a.baseline_auc.notna()
axes[2].plot(pd.to_datetime(a.loc[good,'date']),a.loc[good,'learned_auc']-a.loc[good,'baseline_auc'],color='#DD753B',lw=2)
axes[2].set_title(f'Q2风险识别增量：{good.sum()}个有效日均为0');axes[2].set_ylabel('学习头与波动基线的AUROC差');axes[2].set_ylim(-.05,.05);axes[2].tick_params(axis='x',rotation=30)
for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('G1并联风险头：未保留隐藏表示带来的增量',fontsize=16)
fig.tight_layout();fig.savefig(R/'figures/mechanism.png',dpi=160);plt.close(fig)
