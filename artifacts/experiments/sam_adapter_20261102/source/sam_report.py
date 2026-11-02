"""SAM研究报告：统一净累计主图、风险与配对区间，保留负结果。"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from state_adapter import run as legacy

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/experiments/sam_adapter_20261102'
NAMES={'G1_original':'原G1','MC40':'状态适配MC40','SAM_MC40':'SAM状态适配MC40'}
COLORS={'G1_original':'#405568','MC40':'#309568','SAM_MC40':'#E88B32'}


def read(name: str) -> Any:
    """读取审计产物。:param name: 相对路径。 :returns: JSON。"""
    return json.loads((OUT/name).read_text())


def report() -> Path:
    """仅在完整经济产物与独立代码审查后生成最终报告。

    :returns: 中文报告路径。
    :raises RuntimeError: 缺失审查、对账、完整覆盖或图表端点不符。
    """
    review=read('review_gate.json')
    if not review.get('passed'):raise RuntimeError('等待主协调器独立代码审查完成')
    for path,digest in review['source_sha256'].items():
        if legacy._sha256_file(path)!=digest:raise RuntimeError('审查后源码漂移')
    p=read('protocol.json');result=read('metrics.json');m=result['metrics'];v=result['verdict']
    from state_adapter.sam_run import verify_candidate_inputs
    verify_candidate_inputs(p,legacy._sha256_object(p))
    control=read('control_comparison.json');sam=read('training_sam.json');coverage=read('coverage.json');gate=read('reference_replay_gate.json')
    assert all(control[k] for k in ['step_equal','ce_equal','tensors_equal']) and len(coverage)==4 and len(gate)==4 and max(gate.values())<=1e-10
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    ddfig,ddaxes=plt.subplots(1,2,figsize=(14,4),sharey=True)
    deltafig,deltaaxes=plt.subplots(1,2,figsize=(14,4),sharey=True)
    endpoints={}
    for index,w in enumerate(['W3','W4']):
        nets={}
        for tag in NAMES:
            r=pd.read_parquet(OUT/f'report_{w}_{tag}.parquet');net=r['return']-r.cost;nets[tag]=net
            assert np.isfinite(net).all() and len(net)==p['expected_coverage'][w]['dates']
            end=float(net.cumsum().iloc[-1]);assert abs(end-m[w][tag]['net_sum'])<1e-12;endpoints[f'{w}/{tag}']=end
            axes[index].plot(net.index,net.cumsum(),label=NAMES[tag],color=COLORS[tag],lw=2)
            nav=np.r_[1.,(1+net).cumprod().values];dd=nav/np.maximum.accumulate(nav)-1
            assert abs(dd.min()-m[w][tag]['mdd'])<1e-12
            ddaxes[index].plot(net.index,dd[1:],label=NAMES[tag],color=COLORS[tag],lw=1.6)
        for tag,style in [('G1_original','-'),('MC40','--')]:
            deltaaxes[index].plot(nets[tag].index,(nets['SAM_MC40']-nets[tag]).cumsum(),label='SAM − '+NAMES[tag],color=COLORS[tag],ls=style,lw=1.8)
        for group in [axes,ddaxes,deltaaxes]:
            group[index].set_title(f'{w}  {net.index[0]:%Y-%m-%d} — {net.index[-1]:%Y-%m-%d}')
            group[index].axhline(0,color='#b2b8bd',lw=.7);group[index].grid(alpha=.16);group[index].yaxis.set_major_formatter(PercentFormatter(1));group[index].tick_params(axis='x',rotation=25);group[index].legend(fontsize=9)
    for f,title,name in [(fig,'扣费累计收益（每日净收益累加）','cumulative_return'),(ddfig,'复利净值回撤（包含初始NAV=1）','drawdown'),(deltafig,'SAM相对参照的累计净收益差','cumulative_difference')]:
        f.suptitle(title);f.tight_layout();f.savefig(figures/f'{name}.png',dpi=180);f.savefig(figures/f'{name}.svg');plt.close(f)
    legacy._atomic_text(OUT/'figure_endpoints.json',json.dumps(endpoints,indent=2)+'\n')
    conclusion='两窗历史收益门槛通过' if v['historical_return'] else '未通过两窗历史收益门槛（negative）'
    risk='通过' if v['risk_compatible'] else '未通过'
    lines=[f'# SAM状态适配器实验报告', '',f'2026-11-02。冻结G1与状态适配器，仅将训练更新替换为SAM；已完成两窗MC40推理及纠偏口径真实账户回测，{conclusion}，风险兼容门槛{risk}。本轮候选未自动采纳。',
        '', '改动仅在107712个adapter权重的训练方法。原G1为主基准，状态适配MC40为既有参照；没有新增预测头、日期等权、SWA或测试期适配。',
        '', '![扣费累计收益](../artifacts/experiments/sam_adapter_20261102/figures/cumulative_return.png)', '',
        '| 窗口 | 模型 | 净累计 | 复利 | Sharpe | 最大回撤 | 总成本 | 日均换手 | 年化波动 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for w in ['W3','W4']:
        for tag in NAMES:
            a=m[w][tag];lines.append(f"| {w} | {NAMES[tag]} | {a['net_sum']:.4%} | {a['compound']:.4%} | {a['sharpe']:.4f} | {a['mdd']:.4%} | {a['cost']:.4%} | {a['turnover']:.4%} | {a['volatility']:.4%} |")
    lines+=['','W3 126日/37223有限格；W4 134日/39563有限格，实际2026-01-05起。两窗独立面板共享y轴。主指标为sum(return−cost)，并非年化收益；复利为prod(1+r)−1；Sharpe为sqrt(252)×mean/std(ddof=1)；MDD含初始NAV=1。日均换手为Qlib日报turnover均值（当日成交额/前日账户总值）。所有图表从同一日报复算。',
        '', '## 相对收益、风险与不确定性', '', '| 窗口 | 参照 | SAM净累计差（百分点） | 95%配对区间（百分点） |', '|---|---|---:|---:|']
    for key,a in result['comparisons'].items():
        w,tag=key.split('/');b=a['bootstrap'];lines.append(f"| {w} | {NAMES[tag]} | {a['delta_pp']:+.4f} | [{b['lower_95']*100:+.4f}, {b['upper_95']*100:+.4f}] |")
    lines+=['',f"历史收益条件为两窗均严格超过原G1及MC40：**{v['historical_return']}**。风险兼容为两窗Sharpe不低于MC40且MDD幅度不高于MC40：**{v['risk_compatible']}**。两项分别判断，不能用W3单窗领先代替跨窗成功。",
        '', '区间使用日净收益配对差、20日循环块bootstrap、5000次、seed100。它刻画已观察窗口内的时序不确定性，不修正反复查看历史窗口和候选探索造成的选择偏差。即便两项均通过，也只构成单次先导历史证据，不证明状态因果贡献。',
        '', '![回撤](../artifacts/experiments/sam_adapter_20261102/figures/drawdown.png)', '', '![累计差](../artifacts/experiments/sam_adapter_20261102/figures/cumulative_difference.png)',
        '', '## 训练机制与敏感性', '',
        f"原AdamW从零重训3000步，所有adapter tensors最大绝对差{control['max_abs']:.3g}，beststep={control['new_step']}、CE={control['new_ce']:.15f}与旧MC40逐值一致。SAM最低未扰动验证CE为{sam['best_val_ce']:.15f}，选点step={sam['best_step']}；每500步的全部记录见training_sam.jsonl。选点没有使用回测收益。",
        '', '两臂同为3000次AdamW更新；SAM为6000次训练前向/反向，控制为3000次，**同更新次数，不是等计算预算**。初始化/窗口均匀采样seed100；固定lr3e−4、betas(.9,.95)、weight_decay=.01、batch64，第二梯度clip1；冻结cache2048个2025H1窗口，目标位置89..98。rho=.05在看SAM W3/W4收益前锁定，非adaptive L2，epsilon=1e−12，不称最优。',
        '', '每批tokenizer只encode一次，两前向恢复相同CPU/CUDA/Python/NumPy起始随机状态；外部随机流只推进一次。这不保证扰动后的离散token相同。参数在finally精确copy还原，然后一次AdamW.step；weight decay仅一次。',
        '', '| 权重 | 零扰动CE | rho | CE增量均值 | 最小/最大增量 |', '|---|---:|---:|---:|---:|']
    for arm,label in [('reference','原MC40'),('sam','SAM选点')]:
        diagnostic=read(f'diagnostic_{arm}.json')
        for rho in [.01,.05]:
            values=[a['delta_ce'] for a in diagnostic['directions'] if a['rho']==rho]
            lines.append(f"| {label} | {diagnostic['zero_ce']:.9f} | {rho} | {np.mean(values):+.9f} | {min(values):+.9f} / {max(values):+.9f} |")
    lines+=['', '诊断每个非零rho共8个方向，即4个独立全局L2单位方向及其负方向，seed20261102；两幅度使用相同方向。固定eval_seed103100，两次零扰动完全一致。完整CE增量和实际半径见diagnostic_reference.json与diagnostic_sam.json。诊断只读2025H1验证cache，原权重/底座hash/RNG精确恢复。小幅CE变化或局部敏感性降低均不能外推交易收益，也不作为完成经济诊断的门槛。',
        '', '| 窗口/参照 | 日横截面Spearman均值 | 最小值 | Top50平均重合率 |', '|---|---:|---:|---:|']
    for key,a in result['ranking'].items():lines.append(f"| {key} | {a['mean_spearman']:.4f} | {a['min_spearman']:.4f} | {a['mean_top50_overlap']:.2%} |")
    lines+=['', '这些排序差异反映冻结SAM权重经同采样预算生成出口后的变化；两窗结果分别报告，不能将训练CE变化解释为收益变化的因果证明。',
        '', '## 执行口径、身份与复现', '',
        '推理训练seed100，infer42/100各20条路径，信号均值0.5/0.5；L90/H10/T1/top_p.9/top_k0/chunk32/clip5/maxcontext512，每日重置seed。日期/股票finite mask与原G1完全一致。',
        '', 'Top50/drop5/hold5，次日open，买费.001、卖费.0015、最低5，资金1亿元，CSI300 PIT。限价判断仅为pctchange/100对比±.095；统一9.5%与日频close判限仍是研究简化。原G1和MC40冻结信号已实际重放，四份纠偏源日报最大误差≤1e−10；未混用旧错误金额change口径。',
        '', '所有真实行情请求均由market_guard在读取前截断/拒绝，审计保存在inference_market_requests.jsonl和accounts下market_requests.jsonl；2026-07-25以后封存行情未用于本轮。持仓、实际成交、个股日损益及费用残差保存于accounts/{W3,W4}/{G1_original,MC40,SAM_MC40}。',
        '', f"实验分支：codex/sam-adapter-20261102；报告代码版本：`{subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}`。协议canonical SHA：`{legacy._sha256_object(p)}`。SAM选点SHA：`{sam['adapter_sha256']}`。",
        '', f"固定底座hash：`{p['reference_adapter']['base_hash']}`。验证cache SHA：`{p['validation_cache']['sha256']}`。来源路径及逐文件SHA详见protocol.json，执行版本见execution_version.json，回测complete.json绑定信号与日报SHA。",
        '', f"GPU累计已结算秒：{read('gpu_budget.json')['used_seconds']:.1f}/21600；墙钟起点：{p['budget']['started_at']}，固定8h恢复不重置。SAM训练峰值显存{sam['peak_memory_bytes']/2**30:.3f}GiB；单次训练记录耗时{sam['elapsed_seconds']:.1f}s，失败重启全部计入gpu_budget.json。",
        '', '复现入口：`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/user/miniconda3/envs/quant/bin/python -m state_adapter.sam_run --stage all`。超过冻结预算不可重置后重跑；本入口按身份恢复已有产物。测试命令及真实输出位于verification/，独立代码审查记录见review_gate.json及review.md；任务与修订见execution_ledger.md。',
        '', '## 限制与待确认事项', '', '单训练seed100、单固定rho，一次有界探索；既有W3/W4已被多次查看，本轮不是未见数据检验。未来确认仍需要跨训练seed、等计算量控制（如6000步AdamW）与独立时期；本轮不自动扩展。没有自动采纳、合并或push，主协调器另行决定研究索引发布。']
    path=ROOT/'docs/SAM状态适配器实验报告_20261102.md';path.write_text('\n'.join(lines)+'\n')
    legacy._atomic_text(OUT/'bot_status.json',json.dumps(dict(status='candidate' if all(v.values()) else 'negative',stage='report_complete',verdict=v,report=str(path),figures=[str(figures/f'{n}.png') for n in ['cumulative_return','drawdown','cumulative_difference']]),ensure_ascii=False,indent=2)+'\n')
    return path


if __name__=='__main__':report()
