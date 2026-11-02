"""从完整日报生成中文2×2累计收益和W3风险交付图。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts/experiments/state_factorial_20261102'
DEST = ROOT / 'docs/assets/state_factorial_20261102'
STYLES = [('G1_original', '原 G1 · 主基准', '#53677e', '-'),
          ('A', 'A 原抽样 · 固定状态', '#956bb0', '--'),
          ('B', 'B 原抽样 · 真实状态（MC40）', '#008680', '-'),
          ('C', 'C 日期等权 · 固定状态', '#259eaf', '--'),
          ('D', 'D 日期等权 · 真实状态', '#dc7935', '-')]


def style_axis(axis: plt.Axes) -> None:
    """设置百分比刻度、日期及浅色网格。

    :param axis: 待设置坐标轴。
    :returns: 无。
    """
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
    axis.grid(axis='y', color='#dce5eb', lw=.8)
    axis.axhline(0, color='#8ea2b3', lw=1)
    axis.tick_params(length=0, colors='#53677e', pad=8)
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    """从已核验日报绘图，保留原始回测图。

    :returns: 无；输出两张中文PNG。
    """
    font = Path('/System/Library/Fonts/Hiragino Sans GB.ttc')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': FontProperties(fname=str(font)).get_name(), 'axes.unicode_minus': False, 'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    fig.subplots_adjust(left=.07, right=.975, bottom=.16, top=.65, wspace=.12)
    fig.text(.07, .93, '2×2 机制对照 · 扣费累计收益', fontsize=23, color='#263e50')
    fig.text(.07, .87, 'C 在 W3 收益最高、W4 亏损；真实状态与日期等权的作用随窗口变化。', color='#596d80')
    for axis, window in zip(axes, ['W3', 'W4']):
        for arm, label, color, line in STYLES:
            report = pd.read_parquet(OUT / f'report_{window}_{arm}.parquet')
            # 主图严格使用净收益累计和，复利口径在报告中另列。
            curve = (report['return'] - report.cost).cumsum()
            axis.plot(curve.index, curve, color=color, linestyle=line, lw=2, label=label)
        axis.set_title(f'{window}   {curve.index[0]:%Y.%m.%d} — {curve.index[-1]:%Y.%m.%d}', loc='left', pad=14)
        axis.set_ylim(-.05, .21)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.065, .81), ncol=3, frameon=False, fontsize=10.5, handlelength=3)
    fig.text(.07, .055, '每日净收益累加，已扣交易成本；每窗独立累计。四个适配臂均使用两流各20条路径，原 G1 保持主基准配置。', fontsize=10, color='#596d80')
    fig.savefig(DEST / 'factorial_zh.png', dpi=160, facecolor='white')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(left=.07, right=.975, bottom=.17, top=.71, wspace=.16)
    fig.text(.07, .93, 'W3 风险诊断 · 更多收益与更深回撤', fontsize=22, color='#263e50')
    nets = {}
    for arm, label, color, _ in STYLES:
        if arm not in ['G1_original', 'B', 'D']:
            continue
        report = pd.read_parquet(OUT / f'report_W3_{arm}.parquet')
        net = report['return'] - report.cost
        nets[arm] = net
        nav = np.r_[1., (1 + net).cumprod().to_numpy()]
        dd = (nav / np.maximum.accumulate(nav) - 1)[1:]
        axes[0].plot(net.index, dd, color=color, lw=2, label=label)
    for arm, label, color in [('G1_original', '日期等权 D 减原 G1', '#53677e'), ('B', '日期等权 D 减 MC40', '#dc7935')]:
        delta = (nets['D'] - nets[arm]).cumsum()
        axes[1].plot(delta.index, delta, color=color, lw=2, label=label)
    axes[0].set_title('复利净值回撤（峰值含初始净值1）', loc='left', pad=15)
    axes[1].set_title('累计净收益差', loc='left', pad=15)
    for axis in axes:
        style_axis(axis)
        axis.legend(frameon=False, fontsize=10, loc='lower left')
    fig.text(.07, .81, 'D 最大回撤：7.84%；MC40：5.31%。D 的最大回撤峰谷为 2025.10.29—11.24。', color='#596d80')
    fig.text(.07, .055, '回撤使用复利净值；右图使用每日净收益差累加。两者口径分别标注。', fontsize=10, color='#596d80')
    fig.savefig(DEST / 'w3_drawdown_zh.png', dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
