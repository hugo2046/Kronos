"""固定风险目标、分别重算的配对区间和中文研究报告。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def economic_gate(results: dict) -> dict:
    """区分三目标策略门槛与学习头超越波动基线。

    :param results: 两窗四臂指标。
    :return: 两种门槛与联合判定，不自动采纳。
    """
    strategy=True;increment=True;details={}
    for w,arms in results.items():
        a=arms['learned'];details[w]={}
        for ref in ['G1_original','MC40','volatility']:
            b=arms[ref]
            checks=dict(annualized=a['annualized']>b['annualized'],sharpe=a['sharpe']>b['sharpe'],
                mdd=(abs(a['mdd'])<=abs(b['mdd']) if ref=='volatility' else abs(a['mdd'])<abs(b['mdd'])))
            details[w][ref]=checks
            if ref=='volatility':increment &= all(checks.values())
            else:strategy &= all(checks.values())
    return dict(strategy_joint_passed=bool(strategy),learned_increment_passed=bool(increment),
        all_passed=bool(strategy and increment),details=details,auto_adopt=False)


def _statistics(values: np.ndarray) -> dict[str,np.ndarray]:
    values=np.atleast_2d(values);n=values.shape[1]
    nav=np.concatenate([np.ones((len(values),1)),np.cumprod(1+values,axis=1)],axis=1)
    sd=values.std(axis=1,ddof=1)
    return dict(net_sum=values.sum(1),annualized=np.expm1(np.log1p(values).sum(1)*252/n),
        sharpe=np.divide(np.sqrt(252)*values.mean(1),sd,out=np.full(len(values),np.nan),where=sd>0),
        mdd=(nav/np.maximum.accumulate(nav,axis=1)-1).min(1))


def paired_intervals(a: np.ndarray,b: np.ndarray) -> dict:
    """按相同日期块配对重算各指标，而非套用收益区间。

    :param a: 候选净收益。
    :param b: 同日参照净收益。
    :return: 每个指标的点差、95%区间及有效重采样数。
    """
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or a.ndim!=1 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('配对收益不符')
    n=len(a);start=np.random.default_rng(100).integers(0,n,(5000,(n+19)//20))
    idx=((start[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
    observed_a=_statistics(a);observed_b=_statistics(b);sa=_statistics(a[idx]);sb=_statistics(b[idx])
    out={}
    for name in sa:
        diff=sa[name]-sb[name];good=np.isfinite(diff)
        out[name]=dict(observed=float(observed_a[name][0]-observed_b[name][0]),
            ci95=np.quantile(diff[good],[.025,.975]).tolist() if good.any() else None,valid_resamples=int(good.sum()))
    return out


def main() -> None:
    """生成统一口径报告；机制未过时明确候选未回测。

    :return: 无。
    """
    from g1_risk_head.runtime import OUT,ROOT,freeze,require_gate,identity,sources
    from g1_risk_head.accounts import account_metrics
    from state_adapter.tta import atomic_json,sha
    from g1_risk_head.train import training_identity
    p=freeze();require_gate(p);mechanism=json.loads((OUT/'mechanism_gate.json').read_text())
    if mechanism['identity']!=training_identity(p) or mechanism['head_sha']!=sha(OUT/'head.pt'):raise RuntimeError('报告训练身份不符')
    complete=(OUT/'metrics.json').exists() and mechanism['passed']
    metrics={};reports={};comparisons={}
    if complete:
        execution=json.loads((OUT/'metrics.json').read_text())
        if not execution.get('computation_complete') or execution['identity']!=dict(protocol=identity(p),sources=sources()):raise RuntimeError('经济结果身份不符')
    for w in ['W3','W4']:
        reports[w]={}
        for name,tag in [('G1_original','G1_original'),('MC40','B')]:
            path=Path('/home/user/workspace/Kronos-limit-audit/artifacts/experiments/limit_unit_audit_20261102/accounts/new')/w/tag/'report.parquet'
            reports[w][name]=pd.read_parquet(path)
        if complete:
            for tag in ['volatility','learned']:reports[w][tag]=pd.read_parquet(OUT/'accounts'/w/tag/'report.parquet')
        metrics[w]={name:account_metrics(f) for name,f in reports[w].items()}
        if complete:
            comparisons[w]={name:paired_intervals((reports[w]['learned']['return']-reports[w]['learned'].cost).to_numpy(),(f['return']-f.cost).to_numpy()) for name,f in reports[w].items() if name!='learned'}
    gate=economic_gate(metrics) if complete else dict(all_passed=False,reason='Q2机制门禁未通过，未执行C',auto_adopt=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    labels={'G1_original':'原G1','MC40':'状态适配MC40','volatility':'G1＋历史波动风险','learned':'G1＋学习风险头'}
    colors={'G1_original':'#536780','MC40':'#00857D','volatility':'#8B77A9','learned':'#DD753B'}
    figures=OUT/'figures';figures.mkdir(exist_ok=True);endpoints={}
    for kind in ['cumulative','drawdown']:
        fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
        for ax,w in zip(axes,['W3','W4']):
            for name,f in reports[w].items():
                net=f['return']-f.cost
                if kind=='cumulative':curve=net.cumsum()
                else:
                    nav=(1+net).cumprod();curve=nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1
                ax.plot(curve.index,curve,color=colors[name],label=labels[name],lw=2)
                endpoints[f'{kind}/{w}/{name}']=float(curve.iloc[-1])
            ax.set_title(f'{w} {net.index[0]:%Y-%m-%d} 至 {net.index[-1]:%Y-%m-%d}')
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.suptitle(('扣费累计收益（每日净收益累加）' if kind=='cumulative' else '扣费复利最大回撤路径')+('' if complete else '；仅基准，风险候选未回测'))
        fig.autofmt_xdate();fig.tight_layout();fig.savefig(figures/f'{kind}.png',dpi=160);plt.close(fig)
    summary=dict(mechanism_passed=mechanism['passed'],economic_complete=complete,metrics=metrics,economic_gate=gate,comparisons=comparisons,
        protocol_sha=identity(p),head_sha=mechanism['head_sha'],bootstrap=dict(block=20,resamples=5000,seed=100))
    atomic_json(OUT/'result.json',summary);atomic_json(OUT/'figure_endpoints.json',endpoints)
    conclusion=('研究性回测完成，三目标与学习增量联合门槛'+('通过，但不自动采纳。' if gate['all_passed'] else '未通过，不替换MC40。')) if complete else 'Q2机制先导未通过，依固定计划停止，风险候选经济回测未运行。'
    lines=['# G1并联下行风险头研究报告','',conclusion,'','原G1收益信号保留；冻结其表示上学习分位数风险，使用固定0.25风险排名扣分。MC40是竞争参照，没有参与新头训练。',
        '', '![累计收益](../artifacts/experiments/g1_risk_head_20261105/figures/cumulative.png)',
        '', '|窗口|方案|扣费年化|净累计|复利|夏普|最大回撤|成本|日均换手|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for w,arms in metrics.items():
        for name,m in arms.items():lines.append(f"|{w}|{labels[name]}|{m['annualized']:.4%}|{m['net_sum']:.4%}|{m['compound']:.4%}|{m['sharpe']:.4f}|{m['mdd']:.4%}|{m['cost']:.4%}|{m['turnover']:.4%}|")
    lines+=['','年化按252交易日复利折算，夏普ddof=1，MDD包含初始净值1；W3/W4独立账户、不拼接成未验证实盘。原固定Top50/drop5/hold5、费用、pctchange/100简化9.5%和次日执行保留，原引擎无效open回退close的限制保留。',
        '', '![回撤](../artifacts/experiments/g1_risk_head_20261105/figures/drawdown.png)',
        '', '## 机制与限制','',f"Q1选定epoch {mechanism['selected_epoch']}；Q2 q10 pinball：学习 {mechanism['metrics']['learned']['p10']:.8f}，波动基线 {mechanism['metrics']['volatility']['p10']:.8f}；原G1 Top50有效日期风险AUROC增量 {mechanism['highscore_auc_delta']}。",
        '', 'q10是单股收益下分位，不是ES或组合MDD。标签是次日open到第十日close的毛收益，不等同动态换仓策略的逐笔净收益。单股校准改善不自动证明组合价值；历史波动基线采用相同尺度。',
        '', '当前库历史最终值为研究假设，available_at和历史复权vintage不能证明；2025H1及W3/W4反复查看，仅开发证据。上一轮TTA旧测试曾请求至2026-08-09、实际返回范围未知，本轮不撤销该事故、不声称历史全程未读。新读取从独立有界入口开始，审计和旧事故分别披露。',
        '', '分位数全体/原G1 Top50/历史波动分组逐日校准、20日循环块区间见mechanism_gate.json及Q2_calibration_*.csv。经济指标差的区间逐指标重算，收益区间不套用于夏普或MDD。',
        '', '完整机器结果：','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```',
        '', '复现入口：g1_risk_head.run、g1_risk_head.accounts、g1_risk_head.report。研究分支codex/g1-risk-head-20261105；完整协议、读取审计、日事务与冻结身份保存在artifacts/experiments/g1_risk_head_20261105。未合并实验代码。']
    (ROOT/'docs/G1并联下行风险头研究报告_20261105.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
