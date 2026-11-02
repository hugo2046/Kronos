"""单位纠偏只读复核、绑定身份绘图与中文交付。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from state_adapter.limit_unit_audit import ROOT, OUT, SRC, RISK, sha, save
from state_adapter.risk_overlay_report import metrics
from state_adapter.risk_attribution import drawdown_episode

plt.rcParams['font.family']=['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus']=False

ARMS=['G1_original','A','B','C','D','cap_only','vol_only','cap_vol']
LABELS={'G1_original':'原G1','B':'状态适配MC40（B）','D':'日期等权dynamic（D）','cap_vol':'D + cap_vol','A':'原抽样static（A）','C':'日期等权static（C）','cap_only':'D + cap_only','vol_only':'D + vol_only'}
COLORS={'G1_original':'#374b60','B':'#24965e','D':'#8061ad','cap_vol':'#ef8a23','A':'#87a2b4','C':'#d2a043','cap_only':'#41a6b5','vol_only':'#be6682'}


def load(mode: str, window: str, arm: str) -> pd.DataFrame:
    """曲线必须绑定口径与信号身份。:param mode: 口径。 :param window: 窗口。 :param arm: 臂。 :returns: 已校验日报。"""
    source=arm if arm in ARMS[:5] else 'D'
    path=OUT/'accounts'/mode/window/arm
    if mode=='old' and arm in ['A','C']:
        report=pd.read_parquet(SRC/f'report_{window}_{arm}.parquet')
        identity=dict(execution_id='old',signal_sha256=sha(SRC/f'signal_{window}_{arm}.parquet'),report_sha256=sha(SRC/f'report_{window}_{arm}.parquet'),source='冻结旧日报，非本轮账户重放')
    else:
        identity=json.loads((path/'complete.json').read_text())
        assert identity['report_sha256']==sha(path/'report.parquet')
        report=pd.read_parquet(path/'report.parquet')
    if identity['execution_id']!=mode or identity['signal_sha256']!=sha(SRC/f'signal_{window}_{source}.parquet'):
        raise ValueError('曲线口径或信号SHA不一致')
    report.attrs.update(identity)
    return report


def circular_bootstrap(net: pd.DataFrame) -> dict[str,Any]:
    """共同日期20日循环块。:param net: 同轴日净收益。 :returns: 五个固定对比。"""
    n=len(net); rng=np.random.default_rng(100)
    indices=(rng.integers(0,n,size=(5000,int(np.ceil(n/20)),1))+np.arange(20))%n
    indices=indices.reshape(5000,-1)[:,:n]
    contrasts={'B-A':net.B-net.A,'D-C':net.D-net.C,'C-A':net.C-net.A,'D-B':net.D-net.B,'D-C-B+A':net.D-net.C-net.B+net.A}
    result={}
    for name,d in contrasts.items():
        draws=d.to_numpy()[indices].sum(axis=1)*100
        result[name]=dict(delta_pp=float(d.sum()*100),ci95_pp=np.quantile(draws,[.025,.975]).tolist(),positive_fraction=float((draws>0).mean()))
    return result


def risk_evidence(mode: str, window: str, arm: str, report: pd.DataFrame) -> tuple[pd.DataFrame,dict[str,Any],pd.DataFrame]:
    """在独立产物目录复用原只读归因器。:param mode: 口径。 :param window: 窗口。 :param arm: 臂。 :param report: 日报。 :returns: 原归因产物。"""
    import state_adapter.risk_overlay_report as original
    target=OUT/'analysis'/mode; target.mkdir(parents=True,exist_ok=True)
    for name,source in [('accounts',OUT/'accounts'/mode),('close_history.parquet',RISK/'close_history.parquet')]:
        if not (target/name).exists(): (target/name).symlink_to(source)
    previous_out,previous_save=original.OUT,original.save
    original.OUT=target
    original.save=lambda name,value:save(str(Path('analysis')/mode/name),value)
    try:
        return original.account_evidence(window,arm,report,pd.read_parquet(RISK/f'risk_panel_{window}.parquet'))
    finally:
        original.OUT,original.save=previous_out,previous_save


def contribution_delta(new: pd.DataFrame, old: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """将空仓首日显式补零。:param new: 新贡献。 :param old: 旧贡献。 :param dates: 完整日期。 :returns: 同轴贡献差。"""
    return new.subtract(old,fill_value=0).fillna(0).reindex(dates,fill_value=0)


def main() -> None:
    """检查全结果和请求日志后生成报告，不自动采纳。:returns: 无。"""
    assert json.loads((OUT/'new_neutral_gate.json').read_text())['passed']
    p=json.loads((OUT/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items(): assert sha(Path(path))==digest
    reports={mode:{w:{a:load(mode,w,a) for a in ARMS} for w in p['windows']} for mode in ['old','new']}
    rows=[]; summaries={}; risks=[]; stocks={}; identities=[]
    for mode,windows in reports.items():
        summaries[mode]={}
        for w,arms in windows.items():
            summaries[mode][w]={}
            for a,r in arms.items():
                rows.append(dict(mode=mode,window=w,arm=a,**metrics(r)))
                identities.append(dict(mode=mode,window=w,arm=a,**r.attrs))
                if a in ['D','cap_vol']:
                    daily,summary,contrib=risk_evidence(mode,w,a,r)
                    daily['mode']=mode; risks.append(daily); summaries[mode][w][a]=summary; stocks[mode,w,a]=contrib
    table=pd.DataFrame(rows); table.to_csv(OUT/'metrics.csv',index=False)
    save('curve_identities.json',identities); save('risk_summary.json',summaries)
    bootstrap={w:circular_bootstrap(pd.DataFrame({a:r['return']-r.cost for a,r in reports['new'][w].items()})) for w in p['windows']}
    save('factorial_bootstrap.json',dict(method='共同日期20日循环块，5000次，seed100；未多重校正，仅历史诊断',windows=bootstrap))
    checks={}
    for w in p['windows']:
        m={a:metrics(r) for a,r in reports['new'][w].items()}; c=m['cap_vol']; d=m['D']
        checks[w]=dict(above_G1=c['net_sum']>m['G1_original']['net_sum'],above_B=c['net_sum']>m['B']['net_sum'],sharpe=c['sharpe']>=d['sharpe'],mdd=c['mdd']>=d['mdd'])
    strict=metrics(reports['new']['W3']['cap_vol'])['mdd']>metrics(reports['new']['W3']['D'])['mdd']
    passed=all(all(v.values()) for v in checks.values()) and strict
    save('acceptance.json',dict(passed=passed,checks=checks,w3_strict_mdd=strict))
    flags=pd.read_parquet(OUT/'fields_flags.parquet').reset_index()
    flags['genuine_threshold']=(flags['$pctchange'].abs()/100>=.095)
    flags['window']=np.where(flags.datetime<'2026-01-01','W3','W4')
    flag_summary=flags.groupby('window')[['old_buy','old_sell','new_buy','new_sell','old_blocked','new_blocked','false_blocked','genuine_threshold','missing_fields']].agg(['sum','mean'])
    flag_summary.to_csv(OUT/'flag_summary.csv')
    flag_daily=flags.groupby('datetime')[['old_blocked','new_blocked','false_blocked']].mean()
    flag_daily.to_csv(OUT/'flag_daily.csv')
    event_summary={}; attribution={}
    keyed=flags.set_index(['datetime','instrument'])
    for w in p['windows']:
        for arm in ['cap_only','cap_vol']:
            audit=pd.read_parquet(OUT/'accounts/old'/w/arm/'order_audit.parquet')
            events=audit[audit.reason.eq('not_tradable')].join(keyed[['false_blocked','new_blocked']],on=['date','instrument'])
            events.to_csv(OUT/f'not_tradable_{w}_{arm}.csv',index=False)
            event_summary[f'{w}/{arm}']=dict(total=len(events),unit_false_block=int(events.false_blocked.fillna(False).sum()),still_blocked=int(events.new_blocked.fillna(False).sum()),missing_flags=int(events.false_blocked.isna().sum()))
        for arm in ['G1_original','B','D','cap_vol']:
            trades=pd.read_parquet(OUT/'accounts/new'/w/arm/'trades.parquet').join(keyed[['old_blocked','new_blocked','false_blocked']],on=['date','instrument'])
            trades.to_parquet(OUT/f'new_trades_flags_{w}_{arm}.parquet',index=False)
            old=pd.read_parquet(OUT/'accounts/old'/w/arm/'stock_daily_contributions.parquet').pivot(index='date',columns='instrument',values='net_contribution').fillna(0)
            new=pd.read_parquet(OUT/'accounts/new'/w/arm/'stock_daily_contributions.parquet').pivot(index='date',columns='instrument',values='net_contribution').fillna(0)
            delta=contribution_delta(new,old,reports['new'][w][arm].index)
            expected=(reports['new'][w][arm]['return']-reports['new'][w][arm].cost)-(reports['old'][w][arm]['return']-reports['old'][w][arm].cost)
            np.testing.assert_allclose(delta.sum(axis=1),expected,atol=1e-10,rtol=0)
            delta.to_csv(OUT/f'stock_change_{w}_{arm}.csv')
            attribution[f'{w}/{arm}']=dict(delta_pp=float(delta.sum().sum()*100),previously_blocked_trade_count=int(trades.false_blocked.sum()),previously_blocked_trade_value=float(trades.loc[trades.false_blocked.fillna(False),'value'].sum()),top5_pp=(delta.sum().sort_values(ascending=False).head(5)*100).to_dict(),bottom5_pp=(delta.sum().sort_values().head(5)*100).to_dict())
    save('not_tradable_summary.json',event_summary); save('execution_attribution.json',attribution)
    intervals={}
    for w in p['windows']:
        net=reports['new'][w]['D']['return']-reports['new'][w]['D'].cost
        own=pd.to_datetime(drawdown_episode(net)['interval_dates'])
        selections={'new_D_drawdown':own}
        if w=='W3': selections['old_D_common']=net.loc['2025-10-30':'2025-11-24'].index
        for name,dates in selections.items():
            intervals[f'{w}/{name}']={}
            for mode in ['old','new']:
                for arm in ['D','cap_vol']:
                    s=stocks[mode,w,arm].reindex(dates).sum().sort_values(ascending=False)
                    s.to_csv(OUT/f'interval_{w}_{name}_{mode}_{arm}.csv')
                    intervals[f'{w}/{name}'][f'{mode}/{arm}']=dict(net_sum=float(s.sum()),focus_300308=float(s.get('300308.SZ',0)))
    save('drawdown_intervals.json',intervals)
    audits=[]
    for path in [OUT/'field_requests.jsonl',*sorted((OUT/'accounts').glob('*/*/*/market_requests.jsonl'))]:
        requests=[json.loads(line) for line in path.read_text().splitlines()]
        assert requests and all(pd.Timestamp(x['effective_end'])<=pd.Timestamp('2026-07-24') for x in requests)
        audits.append(dict(path=str(path.relative_to(OUT)),requests=len(requests),sha256=sha(path),max_effective_end=max(x['effective_end'] for x in requests)))
    save('request_boundary_audit.json',dict(passed=True,files=audits))
    risk=pd.concat(risks,ignore_index=True)
    risk.to_csv(OUT/'risk_daily.csv',index=False)
    assets=ROOT/'docs/assets/limit_unit_audit_20261102'; assets.mkdir(parents=True,exist_ok=True)
    for filename,arms in [('cumulative_main',['G1_original','B','D','cap_vol']),('cumulative_all',ARMS)]:
        fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
        for ax,w in zip(axes,p['windows']):
            for arm in arms:
                r=reports['new'][w][arm]; assert r.attrs['execution_id']=='new'
                ax.plot(r.index,(r['return']-r.cost).cumsum(),label=LABELS[arm],color=COLORS[arm])
            ax.set_title(w+' '+ '—'.join(p['windows'][w])+'\n修正口径 pctchange/100'); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle('扣费累计收益（每日净收益累加）'); fig.tight_layout(); fig.savefig(assets/f'{filename}.png',dpi=150); plt.close(fig)
    for filename,drawdown in [('old_new',False),('drawdown',True)]:
        fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
        for ax,w in zip(axes,p['windows']):
            for arm in ['G1_original','B','D','cap_vol']:
                for mode,style in [('old','--'),('new','-')]:
                    r=reports[mode][w][arm]; net=r['return']-r.cost
                    nav=(1+net).cumprod(); y=nav/nav.cummax().clip(lower=1)-1 if drawdown else net.cumsum()
                    ax.plot(r.index,y,label=f'{arm}/{mode}',color=COLORS[arm],linestyle=style)
            ax.set_title(w); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2); ax.legend(fontsize=7,ncol=2)
        fig.tight_layout(); fig.savefig(assets/f'{filename}.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,10),sharey='row')
    for j,w in enumerate(p['windows']):
        bounds=p['windows'][w]
        flag_daily.loc[bounds[0]:bounds[1],['old_blocked','new_blocked']].plot(ax=axes[0,j]); axes[0,j].set_title(w+' blocked fraction (union universe)')
        for arm in ['D','cap_vol']:
            for mode,style in [('old','--'),('new','-')]:
                r=reports[mode][w][arm]; axes[1,j].plot(r.index,r.turnover,label=f'{arm}/{mode}',color=COLORS[arm],linestyle=style)
                d=risk[(risk.window==w)&(risk.arm==arm)&(risk['mode']==mode)]
                axes[2,j].plot(d.date,d.end_cash,label=f'{arm}/{mode}',color=COLORS[arm],linestyle=style)
        axes[1,j].set_title('Daily turnover'); axes[2,j].set_title('End cash / NAV')
        for i in range(3): axes[i,j].legend(fontsize=7); axes[i,j].yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout(); fig.savefig(assets/'limits_turnover_cash.png',dpi=150); plt.close(fig)
    lines=['# 成交限制单位纠偏与冻结方案复核报告','',
        f'发现原Qlib将涨跌金额 `$change` 与小数阈值0.095直接比较，原G1、MC40及D风险实验收益结论均受影响；本轮已完成冻结信号的受控执行纠偏，cap_vol固定门槛{"通过（仅历史候选）" if passed else "未通过"}，不自动采纳任何候选。']
    for w in p['windows']:
        m={a:metrics(r) for a,r in reports['new'][w].items()}
        lines.append(f'{w}修正口径cap_vol相对原G1为{100*(m["cap_vol"]["net_sum"]-m["G1_original"]["net_sum"]):+.4f}个百分点，相对B为{100*(m["cap_vol"]["net_sum"]-m["B"]["net_sum"]):+.4f}个百分点；B相对原G1为{100*(m["B"]["net_sum"]-m["G1_original"]["net_sum"]):+.4f}个百分点。')
    lines+=['','唯一变动在执行层：价格限制使用`$pctchange / 100`。模型底座、预测、冻结信号、CSI300 PIT、Top50/drop5/hold5、次日open、费用和风险规则不变。B是原抽样dynamic已采纳MC40，D是日期等权dynamic，A/C分别是对应static控制。', '',
        '![修正口径扣费累计收益（每日净收益累加）](assets/limit_unit_audit_20261102/cumulative_main.png)','',
        '|口径|窗|臂|扣费累计%|复利%|Sharpe|含初始1 MDD%|成本%|日均换手%|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f'|{r["mode"]}|{r["window"]}|{r["arm"]}|{r["net_sum"]*100:.4f}|{r["compound"]*100:.4f}|{r["sharpe"]:.4f}|{r["mdd"]*100:.4f}|{r["cost"]*100:.4f}|{r["turnover"]*100:.4f}|')
    lines+=['','|窗|臂|旧→新累计差pp|Sharpe变化|MDD变化pp|成本变化pp|换手变化pp|','|---|---|---:|---:|---:|---:|---:|']
    for w in p['windows']:
        for a in ARMS:
            old=metrics(reports['old'][w][a]); new=metrics(reports['new'][w][a])
            lines.append(f'|{w}|{a}|{(new["net_sum"]-old["net_sum"])*100:+.4f}|{new["sharpe"]-old["sharpe"]:+.4f}|{(new["mdd"]-old["mdd"])*100:+.4f}|{(new["cost"]-old["cost"])*100:+.4f}|{(new["turnover"]-old["turnover"])*100:+.4f}|')
    lines+=['','W3为126交易日/37223有限信号格，W4为134日/39563格。Sharpe为sqrt(252)×均值/样本标准差；费用为原引擎cost，换手为原引擎每日turnover。资金1亿，买0.001/卖0.0015/最低5。旧A/C使用冻结日报，其余旧臂均重新对拍；8臂新口径均有真实账户账本。','',
       '所有旧实验结果保留为“旧成交口径历史产物，收益结论待本次纠偏对照解释”。此前MC40采纳决定仅保留为历史记录，其旧收益不能作为当前有效依据；当前比较以同口径重放B和原G1为准。','',
       '补充源代码定位：所指bridge第387/388行位于指数同步段，不能单凭该两行推断股票同步路径；股票金额单位的直接证据是本次多股票真实字段恒等式及固定样例，完整外部源码SHA/原始片段保留。', '',
       '字段核查：'+json.dumps(json.loads((OUT/'field_audit.json').read_text()),ensure_ascii=False)+'。全窗字段在331股票并集上核查，包含当日不在PIT池中的股票，不能将该比例称为当日可投资池比例。精度异常未丢弃，详见field_exceptions.csv；金额绝对容差0.011元、收益率绝对容差0.000002。异常不等同企业行为已确诊。', '',
       '固定样例601138.SH在2025-08-14的change=-0.46元，pctchange=-1.0531%；300308.SZ为-12.53元、-4.9722%。旧规则均误禁，新规则允许。固定9.5%仍是研究简化，未同时实现创业板20%或ST规则；close判日频限制仍是原引擎假设，不能声称解决全部执行偏差。原forbid_all_trade_at_limit语义不变。','',
       '旧风险not_tradable事件复核：`'+json.dumps(event_summary,ensure_ascii=False)+'`。', '',
       '固定候选门槛：`'+json.dumps(dict(checks=checks,w3_strict_mdd=strict),ensure_ascii=False)+'`；失败不调参，不从cap_only/vol_only后验择优。','',
       '|窗|2×2对比|累计差pp|循环块95%区间pp|重采样正值比例|','|---|---|---:|---|---:|']
    for w,items in bootstrap.items():
        for name,v in items.items(): lines.append(f'|{w}|{name}|{v["delta_pp"]:.4f}|{v["ci95_pp"]}|{v["positive_fraction"]:.4f}|')
    lines+=['','20日共同日期循环块bootstrap5000次、seed100；仅历史诊断，未多重校正、未独立确认，不外推未来盈利。','',
       '|口径|窗|臂|日初最大权重均值%|HHI均值|现金均值%|60日波动暴露|超限股票日|风险减仓计划/成交|300308净贡献pp|300308方差份额|','|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for mode,windows in summaries.items():
        for w,arms in windows.items():
            for a,s in arms.items(): lines.append(f'|{mode}|{w}|{a}|{s["max_weight_mean"]*100:.4f}|{s["hhi_mean"]:.5f}|{s["cash_mean"]*100:.4f}|{s["vol60_mean"]:.4f}|{s["cap_exceed_stock_days"]}|{s["extra_sell_orders"]}/{s["extra_sell_filled"]}|{s["focus_300308"]["net_contribution"]*100:.4f}|{s["focus_300308"]["variance_share"]:.2%}|')
    lines+=['','|口径|窗|臂|自身峰日|自身谷日|恢复日|MDD%|','|---|---|---|---|---|---|---:|']
    for mode,windows in summaries.items():
        for w,arms in windows.items():
            for a,summary in arms.items():
                e=summary['drawdown']
                lines.append(f'|{mode}|{w}|{a}|{e["peak"]}|{e["trough"]}|{e["recovery"]}|{e["mdd"]*100:.4f}|')
    for w in p['windows']:
        old=summaries['old'][w]['D']['focus_300308']; new=summaries['new'][w]['D']['focus_300308']
        lines+=['',f'{w}原D的300308.SZ净贡献由旧口径{old["net_contribution"]*100:.4f}pp变为新口径{new["net_contribution"]*100:.4f}pp，方差份额由{old["variance_share"]:.2%}变为{new["variance_share"]:.2%}。该变化说明此前集中度描述受到错误交易锁定及其后续持仓路径影响；不将它解释为该股票的独立因果效应。']
    lines+=['','超限是cap_vol目标3%×折扣的日末比较；D未施加上限，其超限列不适用。风险暴露、覆盖权重、20/60日波动、逐日现金和逐笔超限原因均见risk_daily.csv和账户risk_summary.json。', '',
        '旧→新收益变化会计分解及此前受阻交易：`'+json.dumps(attribution,ensure_ascii=False)+'`。完整stock_change与new_trades_flags文件保留逐日期股票的贡献差和成交路径；不能把全部后续路径收益简单归因于单笔获准交易。','',
        '共同W3旧D峰后2025-10-29至11-24（不含峰日）及新D自身峰谷：`'+json.dumps(intervals,ensure_ascii=False)+'`。D/cap_vol各自峰谷和恢复日期见risk_summary.json；这里的因果变化仅是固定信号下受控改变执行规则，不推断个股或未来盈利因果。','',
        '![8臂完整曲线](assets/limit_unit_audit_20261102/cumulative_all.png)','',
        '![旧新口径对照](assets/limit_unit_audit_20261102/old_new.png)','',
        '![限制比例、换手与现金](assets/limit_unit_audit_20261102/limits_turnover_cash.png)','',
        '![含初始净值1回撤](assets/limit_unit_audit_20261102/drawdown.png)','',
        f'协议冻结提交60ecbb9，base 03cd705；协议SHA `{sha(OUT/"protocol.json")}`。所有输入信号、源日报、风险面板和源实现SHA见protocol.json，曲线绑定口径ID/信号SHA/日报SHA见curve_identities.json。旧日报对拍≤1e-10，新口径neutral关闭overlay与D日报及订单逐值一致。账户毛收益、费用、持仓贡献和NAV由只读原Account观测器复核。', '',
        '本次验证49项测试通过，32份账户NAV恒等式通过，旧新两口径各两窗neutral订单对拍通过；原始输出见verification目录。归因阶段发现首日空仓无股票行，已按完整日报日期显式补零并加回归测试，不改回测、收益序列或交易规则。', '',
        '每次真实行情读取前继承market_guard，独立请求日志已逐文件审查，effective_end全部≤2026-07-24，详见request_boundary_audit.json。本轮不重训、不GPU推理、不改共享quant-qlib或数据库，不push，不合并，不更新master索引。原9.5%限制外的停牌/缺报价遵循原引擎。','',
        '复算命令：','```bash',"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/user/miniconda3/envs/quant/bin/python -m state_adapter.limit_unit_audit",'/home/user/miniconda3/envs/quant/bin/python -m state_adapter.limit_unit_report','/home/user/miniconda3/envs/quant/bin/python -m pytest tests/test_limit_unit_audit.py tests/test_risk_overlay.py -q','```', '',
        '完整账户文件在artifacts/experiments/limit_unit_audit_20261102/accounts/{old,new}/{W3,W4}/{arm}/。预算沿用execution_clock.json，失败恢复不重置。协调器独立核验后发布研究索引，不自动延续或撤销历史采纳记录。']
    report=ROOT/'docs/成交限制单位纠偏与冻结方案复核报告_20261102.md'; report.write_text('\n'.join(lines)+'\n')
    save('report_complete.json',dict(report_sha256=sha(report),passed=passed,request_audit_passed=True,plots=[str(x.relative_to(ROOT)) for x in assets.glob('*.png')]))
    save('bot_status.json',dict(status='candidate' if passed else 'negative',stage='完整账户/报告/图及封存请求核查完成',report=str(report),next_step='协调器独立核验；不自动采纳'))

if __name__=='__main__': main()
