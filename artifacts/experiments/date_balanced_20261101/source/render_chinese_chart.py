"""从已核验的逐日报表生成中文版累计收益交付图。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts/experiments/date_balanced_20261101'
DEST = ROOT / 'docs/assets/date_balanced_20261101'


def main() -> None:
    """读取三模型两窗净收益，输出共享纵轴的 PNG 和 SVG 图。

    :returns: 无；图片保存在报告资源目录。
    """
    font = Path('/System/Library/Fonts/Hiragino Sans GB.ttc')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus': False, 'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    fig.subplots_adjust(left=.075, right=.975, bottom=.18, top=.69, wspace=.13)
    fig.text(.075, .92, '日期等权适配 · 扣费累计收益', fontsize=23, color='#263e50')
    fig.text(.075, .855, '完整历史回测：两窗收益领先；W3 夏普下降、回撤加深。候选尚未采纳。',
             fontsize=12, color='#596d80')
    styles = [('G1_original', '原 G1 · 主基准', '#53677e'),
              ('State_MC40', '状态适配 MC40 · 已采纳', '#008680'),
              ('Date_balanced_MC40', '日期等权适配 MC40 · 新候选', '#dc7935')]
    for axis, window in zip(axes, ['W3', 'W4']):
        for tag, label, color in styles:
            report = pd.read_parquet(OUT / f'report_{window}_{tag}.parquet')
            # 与统一汇报规范一致：每日扣费净收益累加，不用复利替换主图。
            curve = (report['return'] - report['cost']).cumsum()
            axis.plot(curve.index, curve, color=color, lw=2.1, label=label)
            axis.scatter(curve.index[-1], curve.iloc[-1], color=color, s=24, zorder=4)
            if tag == 'Date_balanced_MC40':
                axis.annotate(f'{curve.iloc[-1]:+.2%}', (curve.index[-1], curve.iloc[-1]),
                              xytext=(-6, 13) if window == 'W3' else (-8, -35),
                              textcoords='offset points', ha='right',
                              color=color, fontsize=12)
        axis.set_title(f'{window}   {curve.index[0]:%Y.%m.%d} — {curve.index[-1]:%Y.%m.%d}',
                       loc='left', pad=17, fontsize=13)
        axis.set_ylim(-.05, .21)
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
        axis.grid(axis='y', color='#dce5eb', lw=.8)
        axis.axhline(0, color='#8ea2b3', lw=1)
        axis.tick_params(length=0, colors='#53677e', pad=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.07, .81),
               frameon=False, ncol=3, fontsize=11, handlelength=3)
    fig.text(.075, .065, '口径：每日净收益累加，已扣交易成本；每窗独立累计，三模型使用相同股票覆盖、交易规则和费用。',
             fontsize=10, color='#596d80')
    DEST.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST / 'cumulative_zh.png', dpi=160, facecolor='white')
    fig.savefig(DEST / 'cumulative_zh.svg', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
