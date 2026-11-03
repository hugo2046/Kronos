"""用既有真实账户与冻结预测进行失败归因，不执行回测或行情查询。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter
from state_adapter.sam_mechanism import OUT, SOURCE, ROOT, digest, save


def sharpe_decomposition(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """对称拆分Sharpe差，避免归因顺序影响。

    :param reference: 参照日净收益。
    :param candidate: 候选日净收益。
    :returns: 总差、均值项及波动项。
    """
    ma, mb = reference.mean(), candidate.mean()
    sa, sb = reference.std(ddof=1), candidate.std(ddof=1)
    if min(sa, sb) <= 0:
        raise ValueError('零波动不能分解Sharpe')
    return dict(difference=float(np.sqrt(252)*(mb/sb-ma/sa)),
                mean_effect=float(np.sqrt(252)*(mb-ma)*.5*(1/sa+1/sb)),
                vol_effect=float(np.sqrt(252)*.5*(ma+mb)*(1/sb-1/sa)))


def reconcile(report: pd.DataFrame, contributions: pd.DataFrame) -> float:
    """检验逐股毛/净/费用与日报一致。

    :param report: 完整日期日报。
    :param contributions: 逐股日贡献。
    :returns: 最大绝对误差。
    :raises ValueError: 日期越界、重复或残差超容限。
    """
    if contributions.duplicated(['date', 'instrument']).any() or report.index.has_duplicates:
        raise ValueError('对账有重复键')
    if not pd.Index(contributions.date.unique()).isin(report.index).all():
        raise ValueError('对账日期不完整')
    sums = contributions.groupby('date')[['gross_contribution', 'cost_contribution', 'net_contribution']].sum().reindex(report.index, fill_value=0.)
    expected = np.c_[report['return'], report.cost, report['return']-report.cost]
    err = float(np.max(np.abs(sums.to_numpy()-expected)))
    if not np.isfinite(err) or err > 1e-10:
        raise ValueError(f'逐股对账失败 {err}')
    return err


def metrics(report: pd.DataFrame) -> dict[str, float]:
    """计算统一规范指标。

    :param report: 扣费前收益、费用及换手日报。
    :returns: 净累计、复利、夏普、回撤、成本、换手及年化波动。
    """
    net = report['return']-report.cost
    nav = np.r_[1., np.cumprod(1+net)]
    return dict(net_sum=float(net.sum()), compound=float(nav[-1]-1),
                sharpe=float(np.sqrt(252)*net.mean()/net.std(ddof=1)),
                mdd=float(np.min(nav/np.maximum.accumulate(nav)-1)),
                cost=float(report.cost.sum()), turnover=float(report.turnover.mean()),
                annual_vol=float(net.std(ddof=1)*np.sqrt(252)))


def main() -> None:
    """生成完整对账、收敛、收益与机制图。\n\n:returns: 无。"""
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    manifest = {}
    def read(path: Path) -> pd.DataFrame:
        manifest[str(path)] = digest(path)
        return pd.read_parquet(path)
    labels = {'G1_original': '原G1 · 主基准', 'MC40': '状态适配MC40 · 参照', 'SAM_MC40': 'SAM适配器 · 已冻结负结果'}
    colors = {'G1_original': '#52657b', 'MC40': '#00857d', 'SAM_MC40': '#d47737'}
    reports, panels, stats, attribution = {}, {}, {}, {}
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    signal_stats = {}
    for window in ['W3', 'W4']:
        panels[window] = {}
        stats[window] = {}
        for arm in labels:
            directory = SOURCE / 'accounts' / window / arm
            report = read(directory / 'report.parquet')
            contributions = read(directory / 'stock_daily_contributions.parquet')
            contributions['date'] = pd.to_datetime(contributions.date)
            positions = read(directory / 'positions.parquet')
            balances = read(directory / 'balances.parquet')
            assert report.index.max() <= pd.Timestamp('2026-07-24')
            assert len(report) == (126 if window == 'W3' else 134)
            errors = reconcile(report, contributions)
            reports[window, arm] = report
            panels[window][arm] = contributions.pivot(index='date', columns='instrument', values='net_contribution').reindex(report.index).fillna(0.)
            daily_concentration = positions.groupby('date').agg(hhi=('end_weight', lambda x: np.square(x).sum()),
                                                               max_weight=('end_weight', 'max'), stock_weight=('end_weight', 'sum'))
            cash = balances.set_index('date').end_cash / balances.set_index('date').nav
            stats[window][arm] = dict(metrics=metrics(report), reconciliation_max_abs=errors,
                average_hhi=float(daily_concentration.hhi.mean()), average_max_weight=float(daily_concentration.max_weight.mean()),
                average_cash=float(cash.mean()), holding_days=len(daily_concentration))
            report.to_parquet(OUT / f'report_{window}_{arm}.parquet')
            daily_concentration.to_csv(OUT / f'concentration_{window}_{arm}.csv')
        sam_signal = read(SOURCE / f'signal_{window}_SAM_MC40.parquet')
        signal_stats[window] = {}
        for reference in ['G1_original', 'MC40']:
            ref_signal = read(Path(protocol['reference_signals'][window][reference])).reindex(index=sam_signal.index, columns=sam_signal.columns)
            if not np.array_equal(np.isfinite(ref_signal), np.isfinite(sam_signal)):
                raise ValueError('原G1/参照预测有限格不一致')
            assert int(np.isfinite(sam_signal).to_numpy().sum()) == (37223 if window == 'W3' else 39563)
            correlations = sam_signal.rank(axis=1).corrwith(ref_signal.rank(axis=1), axis=1)
            selected_sam = sam_signal.rank(axis=1, ascending=False, method='first') <= 50
            selected_ref = ref_signal.rank(axis=1, ascending=False, method='first') <= 50
            overlap = (selected_sam & selected_ref).sum(axis=1) / 50
            signal_stats[window][reference] = dict(rank_correlation=float(correlations.mean()), top50_overlap=float(overlap.mean()))
            pd.DataFrame({'rank_correlation': correlations, 'top50_overlap': overlap}).to_csv(OUT / f'signal_comparison_{window}_{reference}.csv')
            ref_report = reports[window, reference]
            sam_report = reports[window, 'SAM_MC40']
            assert sam_report.index.equals(ref_report.index)
            a = ref_report['return'] - ref_report.cost
            b = sam_report['return'] - sam_report.cost
            ref, sam = panels[window][reference].align(panels[window]['SAM_MC40'], join='outer', axis=1, fill_value=0.)
            difference = sam-ref
            assert np.max(np.abs(difference.sum(axis=1)-(b-a))) < 1e-10
            # 各股cov(贡献差, 两组合净收益之和)之和精确等于总净收益方差之差。
            centered = difference - difference.mean()
            risk = centered.mul((a+b)-(a+b).mean(), axis=0).sum()/(len(a)-1)
            summary = pd.DataFrame({'net_difference':difference.sum(), 'variance_difference_contribution':risk}).sort_values('net_difference')
            summary.index.name = 'instrument'
            summary.to_csv(OUT / f'stock_difference_{window}_{reference}.csv')
            daily = pd.DataFrame({'net_difference': b-a, 'gross_difference': sam_report['return']-ref_report['return'],
                                  'cost_difference':sam_report.cost-ref_report.cost})
            daily.to_csv(OUT / f'daily_difference_{window}_{reference}.csv')
            variance_error = float(abs(risk.sum()-(b.var(ddof=1)-a.var(ddof=1))))
            assert variance_error < 1e-12
            attribution[window+'_'+reference] = dict(sharpe=sharpe_decomposition(a.to_numpy(),b.to_numpy()),
                net_difference=float((b-a).sum()), gross_difference=float(daily.gross_difference.sum()),
                cost_difference=float(daily.cost_difference.sum()), variance_reconciliation=variance_error,
                worst_stocks=summary.head(8).reset_index().to_dict('records'),
                best_stocks=summary.tail(5).reset_index().to_dict('records'),
                worst_days=daily.nsmallest(5,'net_difference').reset_index().astype({'datetime':str}).to_dict('records'))
    trajectory = []
    for arm in ['control', 'sam']:
        path = SOURCE / f'training_{arm}.jsonl'
        manifest[str(path)] = digest(path)
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        trajectory.extend(dict(arm=arm, step=r['step'], validation_ce=r['validation_ce']) for r in rows)
    trajectory = pd.DataFrame(trajectory)
    trajectory.to_csv(OUT / 'training_trajectory.csv', index=False)
    path = SOURCE / 'sam_batch_metrics.jsonl'
    manifest[str(path)] = digest(path)
    batches = pd.DataFrame(json.loads(s) for s in path.read_text().splitlines())
    batches['step'] = np.arange(1, len(batches)+1)
    batches['ce_rise'] = batches.second_loss-batches.first_loss
    batch_groups = batches.groupby((batches.step-1)//500).agg(start_step=('step','min'),end_step=('step','max'),
        first_ce=('first_loss','mean'), perturbed_ce=('second_loss','mean'),ce_rise=('ce_rise','mean'),
        first_grad_norm=('first_grad_norm','mean'), second_grad_norm=('grad_norm','mean'))
    batch_groups.to_csv(OUT / 'sam_training_blocks.csv', index=False)
    geometry_path = OUT / 'geometry.json'
    geometry = json.loads(geometry_path.read_text())
    manifest[str(geometry_path)] = digest(geometry_path)
    save(OUT / 'summary.json', dict(stats=stats, attribution=attribution, signals=signal_stats, trained_updates=0, new_market_queries=0))
    save(OUT / 'input_manifest.json', manifest)
    for path, expected in manifest.items():
        assert digest(Path(path)) == expected
    fig, axes = plt.subplots(1,2,figsize=(14,5.2),sharey=True)
    endpoints = {}
    for ax, window in zip(axes,['W3','W4']):
        for arm in labels:
            r = reports[window,arm]
            y = (r['return']-r.cost).cumsum()
            ax.plot(y.index,y,label=labels[arm],color=colors[arm],linewidth=1.8)
            ax.scatter(y.index[-1],y.iloc[-1],color=colors[arm],s=18)
            endpoints[window+'_'+arm] = float(y.iloc[-1])
        ax.set_title(f'{window}  {r.index.min():%Y-%m-%d} — {r.index.max():%Y-%m-%d}')
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.axhline(0,color='#aab2bc',lw=.7);ax.grid(alpha=.2)
    fig.suptitle('SAM失败机制诊断：扣费累计收益（每日净收益累加）',fontsize=17)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
    fig.text(.5,.02,'复用上一轮冻结权重与纠偏账户，未训练新模型；各窗口独立累计。',ha='center')
    fig.tight_layout(rect=[0,.05,1,.82]);fig.savefig(figures/'cumulative_return.png',dpi=170);plt.close(fig)
    save(OUT / 'figure_endpoints.json',endpoints)
    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    for arm, label, color in [('control','AdamW适配器',colors['MC40']),('sam','SAM适配器',colors['SAM_MC40'])]:
        t = trajectory[trajectory.arm==arm]
        axes[0].plot(t.step,t.validation_ce,'o-',label=label,color=color)
    axes[0].set(title='固定缓存验证CE轨迹',xlabel='更新次数',ylabel='CE');axes[0].legend()
    axes[1].plot(batch_groups.end_step,batch_groups.first_ce,'o-',label='SAM未扰动训练CE')
    axes[1].plot(batch_groups.end_step,batch_groups.perturbed_ce,'o-',label='SAM扰动后训练CE')
    axes[1].set(title='每500批均值；非固定训练集损失',xlabel='更新次数',ylabel='CE');axes[1].legend()
    fig.tight_layout();fig.savefig(figures/'training_diagnostic.png',dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax, half in zip(axes,['fit','crosscheck']):
        for arm,color in [('MC40',colors['MC40']),('SAM',colors['SAM_MC40'])]:
            item=geometry[arm]
            xs=[r['rho'] for r in item['attacks']]
            ys=[r['perturbed'][half]['ce']-item['clean'][half]['ce'] for r in item['attacks']]
            ax.plot(xs,ys,'o-',color=color,label=arm)
        ax.set(title='扰动拟合半集' if half=='fit' else '交叉检查半集（非独立时期）',xlabel='全局L2半径',ylabel='CE增量');ax.legend()
    fig.suptitle('单起点5步上升：已探测高损失点，不代表全局最坏情况')
    fig.tight_layout();fig.savefig(figures/'adversarial_diagnostic.png',dpi=170);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(13,5))
    for ax,window in zip(axes,['W3','W4']):
        d=pd.read_csv(OUT/f'stock_difference_{window}_MC40.csv').head(8)
        ax.barh(d.instrument,d.net_difference*100,color=colors['SAM_MC40'])
        ax.set(title=f'{window} SAM相对MC40主要负贡献',xlabel='净累计差额（百分点）')
        ax.invert_yaxis()
    fig.tight_layout();fig.savefig(figures/'stock_attribution.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(stats=stats,attribution=attribution),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
