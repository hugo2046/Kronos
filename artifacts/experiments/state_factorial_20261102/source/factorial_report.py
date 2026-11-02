"""固定四格机制配对统计与统一研究报告。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def audit_signal_pair(baseline: pd.DataFrame, first: pd.DataFrame,
                      second: pd.DataFrame, pooled: pd.DataFrame,
                      dates: int, finite: int) -> dict[str, int | float]:
    """从落盘双流独立复算合并值并核验原G1有限格。

    :param baseline: 原G1完整窗口信号。
    :param first: seed42信号。
    :param second: seed100信号。
    :param pooled: 实际送入回测的双流合并信号。
    :param dates: 冻结交易日数。
    :param finite: 冻结股票日期有限格数。
    :returns: 覆盖计数及合并最大绝对误差。
    :raises RuntimeError: 覆盖、身份轴或合并值不符。
    """
    mask = np.isfinite(baseline.to_numpy())
    if len(baseline) != dates or int(mask.sum()) != finite or finite == 0:
        raise RuntimeError('信号覆盖计数不符')
    for frame in [baseline, first, second, pooled]:
        if (not frame.index.is_unique or not frame.columns.is_unique
                or not frame.index.equals(baseline.index)
                or not frame.columns.equals(baseline.columns)):
            raise RuntimeError('信号日期或股票身份不符')
        if np.isinf(frame.to_numpy()).any() or not np.array_equal(np.isfinite(frame), mask):
            raise RuntimeError('信号finite mask不符')
    expected = (first.to_numpy() + second.to_numpy()) * .5
    error = float(np.max(np.abs(pooled.to_numpy()[mask] - expected[mask])))
    if error > 1e-10:
        raise RuntimeError('双流合并值漂移')
    return dict(dates=dates, finite=finite, pool_max_abs=error)


def mechanism(net: pd.DataFrame) -> pd.DataFrame:
    """在共同日期上计算五个配对效应；先逐日交互再整块抽样。

    :param net: A/B/C/D四列同轴日净收益。
    :returns: 两个state效应、两个sampling效应及交互。
    :raises ValueError: 缺失臂、重复日期或非有限值。
    """
    if not set('ABCD').issubset(net.columns) or net.index.has_duplicates or not np.isfinite(net[list('ABCD')]).all().all():
        raise ValueError('四格必须具有完整共同有限日收益')
    return pd.DataFrame({'state_window': net.B - net.A, 'state_date': net.D - net.C,
                         'sampling_static': net.C - net.A, 'sampling_dynamic': net.D - net.B,
                         'interaction': (net.D - net.C) - (net.B - net.A)})


def bootstrap_effects(effects: pd.DataFrame) -> dict[str, dict[str, float]]:
    """对同轴逐日效应执行20日循环块bootstrap。

    :param effects: mechanism产生的日差额。
    :returns: 累计效应与95%区间，单位百分点。
    """
    from state_adapter.run import block_bootstrap_sum_difference
    result = {}
    for name in effects:
        stats = block_bootstrap_sum_difference(effects[name], effects[name] * 0, block_length=20, resamples=5000, seed=100)
        result[name] = {k + '_pp': stats[k] * 100 for k in ['observed_sum', 'lower_95', 'upper_95']}
    return result


NAMES = {'G1_original': 'Original G1', 'A': 'A: uniform / static MC40', 'B': 'B: adopted state MC40',
         'C': 'C: date-balanced / static MC40', 'D': 'D: date-balanced / dynamic MC40'}
CN = {'G1_original': '原G1', 'A': 'A 原窗口×静态', 'B': 'B 已采纳状态适配MC40', 'C': 'C 日期等权×静态', 'D': 'D 日期等权×动态'}
COLORS = {'G1_original': '#354b63', 'B': '#29965d', 'D': '#e78632', 'A': '#986bb1', 'C': '#269da8'}
EFFECT_NAMES = {'state_window': 'state 原抽样 B−A', 'state_date': 'state 日期等权 D−C',
                'sampling_static': 'sampling 静态 C−A', 'sampling_dynamic': 'sampling 动态 D−B',
                'interaction': '交互 (D−C)−(B−A)'}


def plot_results(reports: dict[tuple[str, str], pd.DataFrame], directory: 'Path') -> dict[str, float]:
    """绘制完整四格主曲线、机制差额、W3回撤及成本贡献。

    :param reports: 窗口和臂键的完整日报。
    :param directory: 图目录。
    :returns: 十条累计收益曲线端点。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    directory.mkdir(parents=True, exist_ok=True)
    ends = {}
    for filename, arms in [('cumulative', ['G1_original', 'B', 'D']), ('factorial', list(NAMES))]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, window in zip(axes, ['W3', 'W4']):
            for arm in arms:
                r = reports[window, arm]; net = r['return'] - r.cost
                curve = net.cumsum(); ends[f'{window}|{arm}'] = float(curve.iloc[-1])
                ax.plot(curve.index, curve, color=COLORS[arm], ls='--' if arm in ['A', 'C'] else '-', label=NAMES[arm])
            ax.set_title(f'{window}: {net.index[0]:%Y-%m-%d} to {net.index[-1]:%Y-%m-%d}')
            ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2); ax.tick_params(axis='x', rotation=25)
        axes[0].legend(fontsize=8); axes[0].set_ylabel('Cumulative net return: sum(return - cost)')
        fig.tight_layout(); fig.savefig(directory / f'{filename}.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, window in zip(axes, ['W3', 'W4']):
        daily = pd.DataFrame({a: reports[window, a]['return'] - reports[window, a].cost for a in 'ABCD'})
        effects = mechanism(daily)
        for name in effects:
            ax.plot(effects.index, effects[name].cumsum(), label=name)
        ax.axhline(0, color='#777', lw=.6); ax.grid(alpha=.2); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(window); ax.tick_params(axis='x', rotation=25)
    axes[0].legend(fontsize=8); fig.suptitle('Paired mechanism differences (daily net return sums)')
    fig.tight_layout(); fig.savefig(directory / 'mechanism.png', dpi=160); plt.close(fig)
    plot_risk({a: reports['W3', a] for a in ['G1_original', 'B', 'D']}, directory)
    return ends


def plot_risk(reports: dict[str, pd.DataFrame], directory: Path) -> None:
    """绘制已核验三模型W3回撤、净差及成本贡献，不伪造未完成控制。

    :param reports: 原G1/B/D同轴日报。
    :param directory: 图目录。
    :returns: 无。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for arm in ['G1_original', 'B', 'D']:
        r = reports[arm]; net = r['return'] - r.cost
        nav = (1 + net).cumprod(); peak = nav.cummax().clip(lower=1)
        axes[0].plot(net.index, nav / peak - 1, color=COLORS[arm], label=NAMES[arm])
    for ref in ['G1_original', 'B']:
        d = reports['D']; b = reports[ref]
        axes[1].plot(d.index, ((d['return'] - d.cost) - (b['return'] - b.cost)).cumsum(), label='D minus '+ref)
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.tick_params(axis='x', rotation=25)
    axes[0].set_title('W3 drawdown (initial NAV = 1)'); axes[1].set_title('W3 cumulative net differences')
    fig.tight_layout(); fig.savefig(directory / 'w3_drawdown.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ref in zip(axes, ['G1_original', 'B']):
        d, b = reports['D'], reports[ref]
        gross = float((d['return'] - b['return']).sum()); savings = float((b.cost - d.cost).sum())
        ax.bar(['Gross difference', 'Cost saving', 'Net difference'], [gross, savings, gross + savings], color=['#738eb0', '#29965d', '#e78632'])
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title('W3 D minus '+ref); ax.axhline(0, color='#777', lw=.6)
    fig.tight_layout(); fig.savefig(directory / 'w3_cost_contribution.png', dpi=160); plt.close(fig)


def report() -> None:
    """完整产物通过门禁后生成报告；缺失四格时拒绝完成标记。

    :returns: 无。
    :raises RuntimeError: 完整性、重放、图表或身份门禁失败。
    """
    import json
    from pathlib import Path
    import subprocess
    from state_adapter.factorial_run import OUT, ROOT, OLD, MC, DATE, load, save, require_control
    from state_adapter.mc40 import _wide
    from state_adapter.date_balanced_run import metrics
    p, sha = load(); require_control(p)
    from state_adapter.factorial_run import require_sampling, checkpoint_identity
    require_sampling()
    for arm in 'ABCD':
        checkpoint_identity(p, arm, sha)
    if not json.loads((OUT / 'inference_complete.json').read_text())['complete']:
        raise RuntimeError('推理未完成')
    if not json.loads((OUT / 'risk/analysis_complete.json').read_text())['complete']:
        raise RuntimeError('持仓归因未完成')
    gates = json.loads((OUT / 'backtest_replay_audit.json').read_text())
    if len(gates) != 6 or max(gates.values()) > 1e-10:
        raise RuntimeError('回测重放门禁未通过')
    manifest = json.loads((MC / 'manifest.json').read_text())
    signal_audit = {}
    for window in ['W3', 'W4']:
        baseline = _wide(p['baseline']['signal_paths'][window])
        pairs = {
            'A': [OUT / f'signals/A_infer42/{window}.parquet',
                  OLD / f'signals/State_static_100/{window}.parquet'],
            'B': [Path(manifest['paths'][window][f'state{s}']) for s in [42, 100]],
            'C': [OUT / f'signals/C_infer{s}/{window}.parquet' for s in [42, 100]],
            'D': [DATE / f'signals/Date_balanced_train100_infer{s}/{window}.parquet' for s in [42, 100]],
        }
        for arm, paths in pairs.items():
            first, second = [_wide(str(path)) for path in paths]
            signal_audit[f'{window}|{arm}'] = audit_signal_pair(
                baseline, first, second, _wide(str(OUT / f'signal_{window}_{arm}.parquet')),
                **p['expected_coverage'][window])
    save('final_signal_audit.json', signal_audit)
    reports = {(w, a): pd.read_parquet(OUT / f'report_{w}_{a}.parquet') for w in ['W3', 'W4'] for a in NAMES}
    stats = {}; comparisons = {}
    for window in ['W3', 'W4']:
        reference = reports[window, 'G1_original']
        for arm in NAMES:
            r = reports[window, arm]
            if not r.index.equals(reference.index) or len(r) != p['expected_coverage'][window]['dates']:
                raise RuntimeError('报告日历不一致')
            stats[f'{window}|{arm}'] = metrics(r)
            net = r['return'] - r.cost
            pd.DataFrame(dict(net_daily=net, cumulative=net.cumsum(), nav=(1+net).cumprod())).to_parquet(OUT / f'daily_{window}_{arm}.parquet')
        effects = mechanism(pd.DataFrame({a: reports[window, a]['return'] - reports[window, a].cost for a in 'ABCD'}))
        effects.to_csv(OUT / f'mechanism_daily_{window}.csv'); comparisons[window] = bootstrap_effects(effects)
    save('metrics.json', dict(metrics=stats, comparisons=comparisons))
    pd.DataFrame.from_dict(stats, orient='index').to_csv(OUT / 'metrics.csv')
    figures = ROOT / 'docs/assets/state_factorial_20261102'
    ends = plot_results(reports, figures)
    for key, value in ends.items():
        if abs(value - stats[key]['net_sum']) > 1e-12:
            raise RuntimeError('图表端点不同')
    risk = json.loads((OUT / 'risk/daily_diagnostics.json').read_text())
    holdings = json.loads((OUT / 'risk/holdings_analysis.json').read_text())
    audit = json.loads((OUT / 'risk/replay_audit.json').read_text())
    def verdict(effect: str) -> str:
        labels = []
        for window in ['W3', 'W4']:
            c = comparisons[window][effect]
            support = '正向区间不含零' if c['lower_95_pp'] > 0 else '负向区间不含零' if c['upper_95_pp'] < 0 else '区间含零'
            labels.append(f'{window} {c["observed_sum_pp"]:+.3f}pp（{support}）')
        return '，'.join(labels)
    lines = ['# 状态适配二乘二机制与W3风险归因报告', '',
        f'本固定训练seed100的完整历史消融中，原抽样state效应：{verdict("state_window")}；日期等权state效应：{verdict("state_date")}；静态sampling效应：{verdict("sampling_static")}；动态sampling效应：{verdict("sampling_dynamic")}；交互：{verdict("interaction")}。不自动采纳C或D，原G1仍是主模型基准。', '',
        '本轮只改变训练抽样与状态输入的两个机制。A为原窗口均匀×固定零状态static，B为原窗口均匀×真实状态dynamic（已采纳MC40），C为日期等权×static（唯一新增训练臂），D为日期等权×dynamic（冻结候选）。G1/tokenizer冻结；四臂均train seed100、last2/rank32和107712名义参数，static保留gate且仅将state置零，部分参数没有数据梯度，因此不宣称有效自由度严格相同。四格均为infer seed42/100各20路径等权；原G1为canonical seed42单流20。', '',
        '![扣费累计收益（每日净收益累加）](assets/state_factorial_20261102/cumulative.png)', '',
        '主图原G1深灰蓝、B绿色、D橙色；两窗独立面板共享纵轴，完整2×2见附图。', '',
        '|窗口|模型|净累计|复利|年化Sharpe|最大回撤|总成本|日均换手|相对原G1 pp|相对B pp|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for window in ['W3', 'W4']:
        for arm in NAMES:
            m = stats[f'{window}|{arm}']; g = stats[f'{window}|G1_original']; b = stats[f'{window}|B']
            lines.append(f'|{window}|{CN[arm]}|{m["net_sum"]:.4%}|{m["compound"]:.4%}|{m["sharpe"]:.6f}|{m["mdd"]:.4%}|{m["cost"]:.4%}|{m["turnover"]:.4%}|{100*(m["net_sum"]-g["net_sum"]):+.4f}|{100*(m["net_sum"]-b["net_sum"]):+.4f}|')
    lines += ['', '净累计为sum(return−cost)，复利另列；Sharpe为sqrt(252)×均值/样本标准差，无风险利率0；MDD包含初始NAV1。W3 126日/37223有限格，W4 134日/39563有限格，W4首交易日2026-01-05。Qlib原生Top50/drop5/hold5、t信号次日open，买0.001卖0.0015最低5，涨跌停阈值0.095，初始资金1亿，CSI300时点股票池；无额外信号shift、无样本裁剪。换手为Qlib实际成交金额/前日账户价值。', '',
        '![四格完整累计收益](assets/state_factorial_20261102/factorial.png)', '',
        '## 配对机制与区间', '', '|窗口|机制|累计差 pp|95%下界 pp|95%上界 pp|', '|---|---|---:|---:|---:|']
    for window in ['W3', 'W4']:
        for effect, c in comparisons[window].items():
            lines.append(f'|{window}|{EFFECT_NAMES[effect]}|{c["observed_sum_pp"]:+.4f}|{c["lower_95_pp"]:+.4f}|{c["upper_95_pp"]:+.4f}|')
    lines += ['', '同一时间索引的日净收益差直接执行20日circular block bootstrap，5000次、seed100；先算每日四序列交互再共同抽日期块，不是四个独立区间相减。以上CI仅对应净收益累计差，不能用于Sharpe或MDD差。历史窗口反复查看，不能宣称多训练seed、独立验证或现实严格因果。', '',
        '![机制累计差额](assets/state_factorial_20261102/mechanism.png)', '', '## W3收益提高但风险恶化', '',
        '|模型|日均净收益|日样本波动|年化波动|峰值日|谷值日|恢复日|峰谷交易日数|水下交易日数|', '|---|---:|---:|---:|---|---|---|---:|---:|']
    for arm in ['G1_original', 'B', 'D']:
        r = risk[arm]; dd = r['drawdown']
        lines.append(f'|{CN[arm]}|{r["mean"]:.6%}|{r["sample_std"]:.6%}|{r["annual_vol"]:.4%}|{dd["peak"]}|{dd["trough"]}|{dd["recovery"] or "窗内未恢复"}|{dd["peak_to_trough_sessions"]}|{dd["underwater_sessions"]}|')
    for ref in ['B', 'G1_original']:
        r = risk['D_minus_' + ref]
        lines += ['', f'D相对{CN[ref]}净收益差{r["net_pp"]:+.4f}pp = 毛收益差{r["gross_pp"]:+.4f}pp + 成本节省{r["cost_saving_pp"]:+.4f}pp。成本节省解释净优势的{r["cost_saving_pp"]/r["net_pp"]:.2%}。均值比为{r["mean_ratio"]:.4f}，波动比为{r["volatility_ratio"]:.4f}：波动增幅超过均值增幅，因此Sharpe下降。优势日{r["positive_days"]}天、劣势日{r["negative_days"]}天，正差额合计{r["positive_sum_pp"]:+.3f}pp、负差额合计{r["negative_sum_pp"]:+.3f}pp。去掉绝对差额最大的5天后差额为{r["without_abs5_pp"]:+.4f}pp，属于后验敏感性，不改变主指标或选策略。',
                  f'最高5日（pp）：{r["best5"]}；最低5日（pp）：{r["worst5"]}。']
    lines += ['', '![W3回撤及累计差额](assets/state_factorial_20261102/w3_drawdown.png)', '',
              '![W3毛收益与成本贡献](assets/state_factorial_20261102/w3_cost_contribution.png)', '',
              '各模型最大回撤区间内全部模型的净累计/复利收益：', '', '```json', json.dumps(risk['same_drawdown_intervals'], ensure_ascii=False, indent=2), '```', '',
              '## 真实账户与持仓证据', '',
              '通过只读Account.update_bar_end回调在原方法执行后观察实际trade_info与估值后的position，原方法返回值原样透传；没有用预测Top50冒充持仓。个股毛损益=前日股数×估值变化+各笔带符号成交股数×(当日估值−成交价格)，统一除以前日账户价值；费用单列。完全卖出的股票仍读取当日同Exchange收盘估值。', '',
              '|模型|冻结日报重放最大误差|个股毛收益对账最大残差|费用对账最大残差|', '|---|---:|---:|---:|']
    for arm, a in audit.items():
        lines.append(f'|{CN[arm]}|{a["replay_max_abs"]:.3e}|{a["reconciliation"]["gross_max_abs"]:.3e}|{a["reconciliation"]["cost_max_abs"]:.3e}|')
    for ref in ['B', 'G1_original']:
        a = holdings['comparisons']['D_minus_' + ref]
        lines += ['', f'### D减{CN[ref]}', '']
        for period in ['full_W3', 'D_drawdown']:
            r = a[period]
            lines += [f'{period}个股净贡献差合计{r["total_pp"]:+.4f}pp；按个股汇总后绝对贡献Top1/5/10占绝对贡献总和：{r["absolute_top_shares"]}。正负抵消使带符号贡献占净差额比例可能超过100%，不以该比例声称收益稳定。',
                      f'主要正贡献股票（pp）：{r["top10_pp"]}。主要负贡献股票（pp）：{r["bottom10_pp"]}。', '']
        lines += [f'平均持仓重叠：{a["overlap_mean"]}；日初权重加权暴露差均值：{a["exposure_difference_mean"]}。', '']
    lines += ['', '从上述真实持仓贡献看，D相对B的额外收益主要贡献股300308.SZ也位于额外方差贡献前列；D回撤区间的负差集中于002463.SZ、000338.SZ、300308.SZ等，D相对原G1的回撤差中601138.SH较突出。历史波动、动量和beta暴露差与风险增加方向一致，但不构成训练抽样改变真实市场风险的因果证明。行业及市值资料缺项，无法判断这些个股差额是否主要来自相应暴露。', '']
    from state_adapter.risk_attribution import variance_evidence_lines
    lines += variance_evidence_lines()
    lines += ['各模型日初集中度均值（HHI按股票权重平方和，现金另列）：', '', '```json', json.dumps(holdings['concentration'], ensure_ascii=False, indent=2), '```', '',
              '暴露仅使用当时前一交易日及更早的20/60日波动、60日动量与60日市场beta，采用日初真实持仓权重；逐日暴露差和可用权重保存在risk目录。行业和流通市值缺少已核验时点接口，明确缺项，未用最新行业替代。价量暴露和个股贡献提供描述性证据，不能证明因果，也不是完整风险模型。', '',
              '## 训练、复用及复现', '',
              'A旧权重冻结，另从零重训控制3000步与旧A逐tensor完全一致(max_abs=0)；旧A100每窗首中末三日核验后复用，新增A42。B旧MC40双流/信号/日报冻结复用，D上轮日期等权动态冻结复用，C为本轮从零训练static并新增42/100两流；没有将dynamic临时置零冒充static控制。旧A与共享验证cache逐tensor一致，容器SHA是否相同另列。', '',
              '所有训练均使用2024-12-31及以前标签、2025H1固定2048窗口验证，每500步最低CE选点，同分较早，未用W3/W4选checkpoint。C与D日期等权采样逐批重放审计见sampling_audit.json。', '',
              f'分支codex/state-factorial-20261102；报告生成提交{subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}；基点3aa1786efe989364a8e5351c6725ffd357b0ccb6；协议canonical SHA `{sha}`。', '',
              '|臂|权重SHA256|原训练协议|配置SHA|', '|---|---|---|---|']
    import torch
    from state_adapter.run import _sha256_file
    for arm, identity in p['arms'].items():
        ck = torch.load(identity['path'], map_location='cpu', weights_only=False)
        lines.append(f'|{arm}|{_sha256_file(identity["path"])}|{ck["protocol_sha"]}|{ck["config_sha"]}|')
    budget = json.loads((OUT / 'gpu_budget.json').read_text())
    lines += ['', f'本轮累计新GPU阶段耗时{budget["used_seconds"]:.1f}秒，上限36000秒；旧A/B/D训练推理为历史计算，未计成本轮重做。冻结输入、数据/底座/cache/SHA见protocol.json，旧实验预算分别位于各自源目录gpu_budget.json。', '',
              '原始小产物位于artifacts/experiments/state_factorial_20261102：report/daily/metrics、mechanism_daily、final_signal_audit、reuse_checks、control_comparison、sampling_audit及risk/{模型}/positions、trades、stock_daily_contributions、balances、reconciliation_audit；大权重在arms目录按SHA绑定。', '',
              '复算命令（quant Python）：', '', '```bash',
              '/home/user/miniconda3/envs/quant/bin/python -m state_adapter.factorial_run --stage backtest',
              'CUDA_VISIBLE_DEVICES= /home/user/miniconda3/envs/quant/bin/python -m state_adapter.risk_attribution',
              'CUDA_VISIBLE_DEVICES= /home/user/miniconda3/envs/quant/bin/python -m state_adapter.risk_attribution --analysis-only',
              '/home/user/miniconda3/envs/quant/bin/python -m state_adapter.factorial_report', '```', '',
              '主协调器负责独立复算、本地回收与master索引发布；本分支不push、不切master。历史收益优胜与机制区间证据分别报告，不自动采纳候选。']
    (ROOT / 'docs/状态适配二乘二机制与W3风险归因报告_20261102.md').write_text('\n'.join(lines) + '\n')
    save('report_complete.json', dict(complete=True, protocol_sha=sha, chart_endpoints=ends))


def progress_report() -> None:
    """推理期间汇报已完成W3归因，明确四格机制尚无结论。

    :returns: 无，写入同一报告路径，完整交付时由report覆盖。
    """
    import json
    from state_adapter.factorial_run import ROOT, OUT, MC, DATE
    from state_adapter.date_balanced_report import plot
    from state_adapter.risk_attribution import variance_evidence_lines
    root = OUT / 'risk'
    risk = json.loads((root/'daily_diagnostics.json').read_text())
    holdings = json.loads((root/'holdings_analysis.json').read_text())
    audit = json.loads((root/'replay_audit.json').read_text())
    reports = {(w, a): pd.read_parquet((DATE if a == 'Date_balanced_MC40' else MC)/f'report_{w}_{a}.parquet')
               for w in ['W3','W4'] for a in ['G1_original','State_MC40','Date_balanced_MC40']}
    plot(reports, ROOT/'docs/assets/state_factorial_20261102/progress')
    plot_risk({'G1_original': reports['W3','G1_original'], 'B': reports['W3','State_MC40'], 'D': reports['W3','Date_balanced_MC40']}, ROOT/'docs/assets/state_factorial_20261102/progress')
    lines = ['# 状态适配二乘二机制与W3风险归因报告（执行中）', '',
        '**阶段结论：state效应、sampling静态效应和四格交互尚待A/C完整双流推理回测，不能判定支持与否；W3真实账户归因已完成，D相对B收益提高但波动增幅更大，优势和新增风险均明显集中于若干股票。**', '',
        '本轮唯一新增训练臂C（日期等权×static）已从零完成3000步，最佳2025H1验证CE=2.4342959076@3000；A重训控制与旧A逐tensor完全相同max_abs=0。A/B/C/D均固定train seed100、last2/rank32、107712名义参数；static仅将state置零，保留gate，部分参数无数据梯度，有效自由度不能声称相同。A/B/D旧身份冻结，C新身份独立。', '',
        '以下累计收益图和指标仅展示已核验的原G1、B与D作为W3归因背景；**A/C曲线尚未完成，不能把该图当作完成2×2实验**。最终版将补齐四格附图、机制差额和共同日期块bootstrap。', '',
        '![已核验背景曲线，四格尚未完成](assets/state_factorial_20261102/progress/cumulative.png)', '',
        '|W3模型|净累计|复利|Sharpe|MDD含初始1|成本|日均换手|日均净收益|样本日波动|年化波动|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['G1_original','B','D']:
        r=risk[arm]
        lines.append(f'|{CN[arm]}|{r["net_sum"]:.4%}|{r["compound"]:.4%}|{r["sharpe"]:.6f}|{r["mdd"]:.4%}|{r["cost"]:.4%}|{r["turnover"]:.4%}|{r["mean"]:.6%}|{r["sample_std"]:.6%}|{r["annual_vol"]:.4%}|')
    lines += ['', '口径为同一126日完整Qlib日报，W3信号37223有限格；主指标sum(return−cost)，复利另列；Sharpe=sqrt(252)×均值/样本标准差，无风险利率0；Top50/drop5/hold5，t信号次日open，买0.001/卖0.0015/最低5，资金1亿，limit0.095，CSI300时点池，原引擎不额外shift。', '', '## 收益、成本与尾部日期', '']
    for ref in ['B','G1_original']:
        d=risk['D_minus_'+ref]
        lines += [f'D相对{CN[ref]}：净差{d["net_pp"]:+.4f}pp = 毛差{d["gross_pp"]:+.4f}pp + 成本节省{d["cost_saving_pp"]:+.4f}pp。成本节省解释净优势的{d["cost_saving_pp"]/d["net_pp"]:.2%}；均值提高{d["mean_ratio"]-1:.2%}，波动提高{d["volatility_ratio"]-1:.2%}，因此Sharpe下降。',
             f'正差日{d["positive_days"]}天、负差日{d["negative_days"]}天；正差合计{d["positive_sum_pp"]:+.3f}pp、负差合计{d["negative_sum_pp"]:+.3f}pp。去掉绝对差最大的5天后净差{d["without_abs5_pp"]:+.4f}pp，仅为后验敏感性，不改主指标。',
             f'最高5日（pp）：{d["best5"]}；最低5日（pp）：{d["worst5"]}。', '']
    lines += ['## 回撤区间与个股', '', 'D最大回撤峰值2025-10-29、谷值2025-11-24，窗末未恢复；以下个股差额统一取该峰值之后到谷值的相同交易日，不比较不同模型各自的回撤区间来推断个股因果。', '',
        '|模型|峰值日|谷值日|恢复日|峰谷交易日|水下交易日|D回撤区间净累计|D回撤区间复利|', '|---|---|---|---|---:|---:|---:|---:|']
    for arm in ['G1_original','B','D']:
        dd=risk[arm]['drawdown']; interval=risk['same_drawdown_intervals']['D'][arm]
        lines.append(f'|{CN[arm]}|{dd["peak"]}|{dd["trough"]}|{dd["recovery"] or "未恢复"}|{dd["peak_to_trough_sessions"]}|{dd["underwater_sessions"]}|{interval["net_sum"]:.4%}|{interval["compound"]:.4%}|')
    for ref in ['B','G1_original']:
        comp=holdings['comparisons']['D_minus_'+ref]
        lines += ['', f'### D减{CN[ref]}持仓贡献', '']
        for key in ['full_W3','D_drawdown']:
            item=comp[key]
            lines += [f'{key}净差合计{item["total_pp"]:+.4f}pp；正贡献前5股票：{dict(list(item["top10_pp"].items())[:5])}；负贡献前5股票：{dict(list(item["bottom10_pp"].items())[:5])}。',
                      f'按个股净差汇总后的绝对贡献Top1/5/10占绝对贡献总和：{item["absolute_top_shares"]}。带符号贡献存在抵消，单股贡献可超过总净优势。']
        exp=comp['exposure_difference_mean']
        lines += [f'平均日初权重加权暴露差：20日年化波动{exp["vol20"]:+.4f}、60日年化波动{exp["vol60"]:+.4f}、60日动量{exp["momentum60"]:+.4f}、60日市场beta{exp["beta60"]:+.4f}。使用前一交易日及以前行情；平均持仓重叠{comp["overlap_mean"]}。', '']
    lines += ['', 'D相对B的额外收益主要贡献股300308.SZ也位于额外方差贡献前列；在D回撤区间，002463.SZ、000338.SZ、300308.SZ等贡献了较大负差。D相对原G1的回撤差中601138.SH贡献尤为明显。更高的历史波动、动量和beta暴露与风险增加方向一致，但不足以证明训练抽样对真实市场风险的因果作用。', '']
    lines += ['', '![W3回撤与净差](assets/state_factorial_20261102/progress/w3_drawdown.png)', '', '![W3收益与成本贡献](assets/state_factorial_20261102/progress/w3_cost_contribution.png)', '']
    lines += variance_evidence_lines()
    lines += ['', '## 对账、可用性与未完成项', '', '|模型|日报重放最大误差|个股毛收益最大残差|费用最大残差|', '|---|---:|---:|---:|']
    for arm, a in audit.items():
        lines.append(f'|{CN[arm]}|{a["replay_max_abs"]:.3e}|{a["reconciliation"]["gross_max_abs"]:.3e}|{a["reconciliation"]["cost_max_abs"]:.3e}|')
    lines += ['', '实际持仓/成交来自只读Account.update_bar_end回调，完整调用原方法并透传其返回；不改交易规则、不记录模拟订单。毛损益为旧持仓估值变动加当日买卖相对收盘估值损益，除以前日账户价值；成本另列，方差贡献采用净贡献并单列残差。原始positions/trades/stock_daily_contributions/balances/reconciliation_audit在risk各模型目录。', '',
        '历史行业、时点流通市值缺项，未用最新行业替代。价量查询显式止于2025-12-31，D.features读取前守卫覆盖回测预载和暴露查询；守卫修订后已完整重放并重新对账。未来日历可以生成，不读取2026-07-25及以后真实行情。', '',
        '已完成A100两窗首中末六日逐值验收，最大差全部为0。待完成：A42与C42/C100完整两窗推理、四格回测、机制20日circular block bootstrap（5000次seed100）、完整图表与最终分支索引。单训练seed且历史窗口已反复查看，不是独立验证，不自动采纳C/D。', '',
        '分支codex/state-factorial-20261102，冻结协议提交ebae3ab；protocol.json及sha256、revision_01.json、gpu_budget.json和execution_clock.json保存身份、修订、累计预算和原deadline。本轮GPU硬上限36000秒、墙钟14小时，不重置。', '',
        '复算：`python -m state_adapter.risk_attribution`重放账户；`python -m state_adapter.risk_attribution --analysis-only`复算持仓/暴露/方差；`python -m state_adapter.factorial_report`仅在四格及全部门禁齐备后生成完整报告。', '']
    (ROOT/'docs/状态适配二乘二机制与W3风险归因报告_20261102.md').write_text('\n'.join(lines)+'\n')


def wait_and_report() -> None:
    """在原deadline内等待GPU进程完成后生成完整报告，不提前标记成功。

    :returns: 无。
    :raises RuntimeError: 原墙钟预算耗尽。
    """
    import fcntl
    import json
    import os
    import time
    from state_adapter.factorial_run import OUT, save
    deadline = json.loads((OUT / 'execution_clock.json').read_text())['deadline']
    lock = (OUT / 'report_watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while time.time() < deadline:
        ready = all((OUT / name).exists() for name in ['inference_complete.json', 'backtest_replay_audit.json', 'risk/analysis_complete.json'])
        save('report_watcher_status.json', dict(pid=os.getpid(), ready=ready, at=time.time(), deadline=deadline))
        if ready:
            runner_lock = (OUT / 'runner.lock').open('a')
            try:
                fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                runner_lock.close()
                time.sleep(30)
                continue
            with runner_lock:
                if json.loads((OUT / 'gpu_budget.json').read_text()).get('active') is not None:
                    raise RuntimeError('GPU阶段未结算，拒绝最终报告')
                report()
                save('report_watcher_status.json', dict(pid=os.getpid(), complete=True, at=time.time()))
                from state_adapter.run import _write_status
                _write_status('continue', '四格及W3归因完整报告已自动生成，待最终审查提交',
                              '核验图表/原始产物、更新本分支索引并提交；不要重复推理', pid=os.getpid())
                return
        time.sleep(30)
    raise RuntimeError('原14小时deadline已到，报告未齐备')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    wait_and_report() if args.wait else report()
