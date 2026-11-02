"""从已核验日报生成D风险约束实验的中文交付图。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts/experiments/d_risk_overlay_20261102'
STYLES = [('G1_original','原 G1 · 主基准','#53677e','-'),
          ('B','MC40 · 已采纳参照','#008680','-'),
          ('D','原始 D · 日期等权动态','#956bb0','-'),
          ('cap_vol','主候选 R · 仓位上限＋波动折扣','#dc7935','-'),
          ('cap_only','对照 · 仅仓位上限','#a38531','--'),
          ('vol_only','对照 · 仅波动折扣','#279ba7','--')]


def main() -> None:
    """读取完整两窗日报，生成累计收益及回撤图。

    :returns: 无；图片写入本轮产物目录。
    """
    font = Path('/System/Library/Fonts/Hiragino Sans GB.ttc')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': FontProperties(fname=str(font)).get_name(), 'axes.unicode_minus': False, 'font.size': 12})
    for mode in ['main','controls','drawdown']:
        fig, axes = plt.subplots(1,2,figsize=(15,7),sharey=True)
        fig.subplots_adjust(left=.07,right=.975,bottom=.16,top=.65,wspace=.12)
        title = 'D 组合风险约束 · '+('回撤' if mode=='drawdown' else '扣费累计收益')
        fig.text(.07,.93,title,fontsize=23,color='#263e50')
        fig.text(.07,.87,'主候选：W3 收益代价较大；W4 保留收益优势并改善风险。两窗联合门槛未通过。',color='#596d80')
        for axis,window in zip(axes,['W3','W4']):
            for arm,label,color,line in STYLES:
                if mode!='controls' and arm in ['cap_only','vol_only']:
                    continue
                report=pd.read_parquet(OUT/f'report_{window}_{arm}.parquet')
                net=report['return']-report.cost
                # 主图使用累计和，回撤单独使用含初始1的复利净值。
                if mode=='drawdown':
                    nav=np.r_[1.,(1+net).cumprod().to_numpy()]
                    values=(nav/np.maximum.accumulate(nav)-1)[1:]
                else:
                    values=net.cumsum()
                axis.plot(net.index,values,color=color,ls=line,lw=2,label=label)
            axis.set_title(f'{window}   {net.index[0]:%Y.%m.%d} — {net.index[-1]:%Y.%m.%d}',loc='left',pad=14)
            axis.set_ylim((-.14,.005) if mode=='drawdown' else (-.05,.21))
            axis.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
            axis.grid(axis='y',color='#dce5eb',lw=.8)
            axis.axhline(0,color='#8ea2b3',lw=1)
            axis.tick_params(length=0,colors='#53677e',pad=8)
            for spine in axis.spines.values():spine.set_visible(False)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.065,.81),ncol=2 if mode!='controls' else 3,frameon=False,fontsize=10.5,handlelength=3)
        caption='复利净值回撤，历史峰值包含初始净值1；每窗独立计算。' if mode=='drawdown' else '每日净收益累加，已扣交易成本；每窗独立累计。风险约束仅改变组合订单，模型与预测信号冻结。'
        fig.text(.07,.055,caption,fontsize=10,color='#596d80')
        name='drawdown_zh.png' if mode=='drawdown' else f'cumulative_{mode}_zh.png'
        fig.savefig(OUT/name,dpi=150,facecolor='white');plt.close(fig)


if __name__=='__main__':
    main()
