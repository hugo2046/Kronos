"""D组合层风险约束：真实账户归因、固定门槛与完整中文报告。"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from state_adapter.risk_overlay_run import ROOT,OUT,SRC,save,sha,load_protocol
from state_adapter.risk_overlay import acceptance
from state_adapter.risk_attribution import drawdown_episode,variance_contributions

ARMS=['G1_original','B','D','cap_only','vol_only','cap_vol']
COLORS={'G1_original':'#374b60','B':'#24965e','D':'#8061ad','cap_only':'#41a6b5','vol_only':'#be6682','cap_vol':'#ef8a23'}
LABELS={'G1_original':'Original G1','B':'Adopted state MC40 (B)','D':'Date-balanced dynamic MC40 (D)', 'cap_vol':'D + cap_vol (R)','cap_only':'D + cap_only','vol_only':'D + vol_only'}


def metrics(r: pd.DataFrame) -> dict[str,float]:
    """用同组日净收益计算固定指标。

    :param r: 原生日报。
    :returns: 指标。
    """
    net=r['return']-r.cost
    return dict(net_sum=float(net.sum()),compound=float((1+net).prod()-1),
        sharpe=float(np.sqrt(252)*net.mean()/net.std(ddof=1)),mdd=drawdown_episode(net)['mdd'],
        annual_vol=float(net.std(ddof=1)*np.sqrt(252)),cost=float(r.cost.sum()),turnover=float(r.turnover.mean()))


def bootstrap(a: pd.Series,b: pd.Series) -> dict[str,Any]:
    """20日配对移动块bootstrap，仅历史诊断。

    :param a: 候选日净收益。
    :param b: 参照日净收益。
    :returns: 累计差额百分点95%区间与重采样正值比例。
    """
    d=(a-b).to_numpy(); n=len(d); rng=np.random.default_rng(100)
    starts=rng.integers(0,n-20+1,size=(5000,int(np.ceil(n/20))))
    indices=(starts[:,:,None]+np.arange(20)).reshape(5000,-1)[:,:n]
    draws=d[indices].sum(axis=1)*100
    return dict(block=20,resamples=5000,seed=100,ci95_pp=np.quantile(draws,[.025,.975]).tolist(),positive_fraction=float((draws>0).mean()),method='配对非循环移动块，末块截断')


def account_evidence(window: str, arm: str, r: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame,dict[str,Any],pd.DataFrame]:
    """真实账户风险、费用及逐股方差对账。

    :param window: 窗口。
    :param arm: 策略臂。
    :param r: 日报。
    :param panel: 决策日风险面板。
    :returns: 每日风险、汇总与逐股贡献宽表。
    """
    directory=OUT/'accounts'/window/arm
    pos=pd.read_parquet(directory/'positions.parquet')
    balances=pd.read_parquet(directory/'balances.parquet').set_index('date')
    trades=pd.read_parquet(directory/'trades.parquet')
    contributions=pd.read_parquet(directory/'stock_daily_contributions.parquet')
    stocks=contributions.pivot(index='date',columns='instrument',values='net_contribution').fillna(0).reindex(r.index,fill_value=0)
    net=r['return']-r.cost
    np.testing.assert_allclose(stocks.sum(axis=1),net,atol=1e-10,rtol=0)
    np.testing.assert_allclose(trades.groupby('date').apply(lambda x:(x.cost/x.previous_nav).sum(),include_groups=False).reindex(r.index,fill_value=0),r.cost,atol=1e-10,rtol=0)
    np.testing.assert_allclose(balances.nav/balances.previous_nav-1,net,atol=1e-10,rtol=0)
    parts=variance_contributions(stocks,net).sort_values(ascending=False)
    np.testing.assert_allclose(parts.sum(),net.var(ddof=1),atol=1e-14,rtol=0)
    parts.rename('variance_contribution').to_csv(directory/'variance_contributions.csv')
    # 用完整交易日历映射u→t，不能在风险面板上随意shift后误配节假日。
    history=pd.read_parquet(OUT/'close_history.parquet').index
    rows=[]; over=[]
    auditpath=directory/'order_audit.parquet'
    audit=pd.read_parquet(auditpath) if auditpath.exists() else pd.DataFrame()
    for date in r.index:
        day=pos[pos.date==date].set_index('instrument')
        idx=history.get_indexer([date])[0]; t=history[idx-1] if idx>0 else pd.NaT
        risk=panel.xs(t,level='date') if t in panel.index.get_level_values('date') else pd.DataFrame()
        data=day.join(risk,how='left') if len(risk) else day.assign(**{c:np.nan for c in ['vol20','vol60','beta60','factor','price']})
        start=day.start_weight; finish=day.end_weight
        row=dict(date=date,window=window,arm=arm,max_weight=float(start.max()) if len(start) else 0.,
                 end_max_weight=float(finish.max()) if len(finish) else 0.,hhi=float((start**2).sum()),
                 end_hhi=float((finish**2).sum()),invested=float(start.sum()),end_invested=float(finish.sum()),
                 cash=float(balances.loc[date,'start_cash']/balances.loc[date,'previous_nav']),
                 end_cash=float(balances.loc[date,'end_cash']/balances.loc[date,'nav']))
        for field in ['vol20','vol60','beta60']:
            valid=data[field].notna(); covered=float(data.loc[valid,'start_weight'].sum())
            exposure=float((data.loc[valid,'start_weight']*data.loc[valid,field]).sum()) if covered>0 else np.nan
            row[field]=exposure; row[field+'_covered_weight']=covered
            row[field+'_invested_normalized']=exposure/covered if covered>0 else np.nan
        if arm in ['cap_only','cap_vol']:
            for code,stock in data.iterrows():
                f=1. if arm=='cap_only' else stock.factor
                limit=.03*f
                if np.isfinite(limit) and stock.end_weight>limit+1e-10:
                    reason='price_nav_move_or_lot_rounding'
                    if len(audit):
                        events=audit[(audit.date==date)&(audit.instrument==code)]
                        blocked=events[events.reason.isin(['hold_thresh','not_tradable','exchange_rejected','rounding','missing_valuation'])]
                        if len(blocked): reason=';'.join(sorted(blocked.reason.unique()))
                    over.append(dict(date=date,decision_date=t,instrument=code,end_weight=stock.end_weight,cap=limit,
                         excess=stock.end_weight-limit,reason=reason,start_amount=stock.start_amount,end_amount=stock.end_amount,
                         previous_price=stock.price,close=stock.end_price))
        rows.append(row)
    daily=pd.DataFrame(rows)
    daily.to_parquet(directory/'risk_daily.parquet',index=False)
    pd.DataFrame(over,columns=['date','decision_date','instrument','end_weight','cap','excess','reason','start_amount','end_amount','previous_price','close']).to_csv(directory/'actual_cap_exceedances.csv',index=False)
    extra_count=0; extra_cost=0.; extra_actual_count=0; extra_qty=0.
    if len(audit):
        sell=trades[trades.quantity<0].copy(); sell['actual_quantity']=-sell.quantity
        sell=sell.groupby(['date','instrument'])[['actual_quantity','cost','value']].sum()
        evidence=audit.join(sell,on=['date','instrument'],rsuffix='_actual')
        # 买卖分开与实际成交匹配，零成交显式记下，不把模拟成交当作执行。
        buy=trades[trades.quantity>0].groupby(['date','instrument'])[['quantity','cost','value']].sum()
        for direction,mask,actual in [('buy',evidence.kind=='buy',buy),('sell',evidence.kind=='risk_sell',sell)]:
            for i,row in evidence[mask].iterrows():
                key=(row.date,row.instrument)
                qty=float(actual.loc[key,'quantity' if direction=='buy' else 'actual_quantity']) if key in actual.index else 0.
                evidence.loc[i,'actual_quantity']=qty
                evidence.loc[i,'actual_cost']=float(actual.loc[key,'cost']) if key in actual.index else 0.
                evidence.loc[i,'actual_value']=float(actual.loc[key,'value']) if key in actual.index else 0.
                evidence.loc[i,'not_filled']=bool(row.after>0 and qty==0)
        x=evidence[(evidence.kind=='risk_sell')&(evidence.extra>0)].copy()
        extra_count=len(x)
        if len(x):
            extra_actual_count=int((x.actual_quantity>0).sum())
            allocation=(x.extra/x.after).clip(0,1)
            extra_cost=float((x.actual_cost*allocation).sum()); extra_qty=float((x.actual_quantity*allocation).sum())
        evidence.to_parquet(directory/'order_execution_audit.parquet',index=False)
    summary={}
    for field in ['max_weight','end_max_weight','hhi','end_hhi','invested','end_invested','cash','end_cash','vol20','vol60','beta60',
                  'vol20_invested_normalized','vol60_invested_normalized','beta60_invested_normalized','vol20_covered_weight','vol60_covered_weight','beta60_covered_weight']:
        summary[field+'_mean']=float(daily[field].mean()); summary[field+'_peak']=float(daily[field].max())
    summary.update(cap_exceed_stock_days=len(over),cap_exceed_days=len({x['date'] for x in over}),
        cap_exceed_reason_counts=pd.Series([x['reason'] for x in over],dtype=str).value_counts().to_dict(),
        exchange_suspended_rejections=int(audit.get('exchange_suspended',pd.Series(dtype=bool)).eq(True).sum()),
        exchange_limit_rejections=int(audit.get('exchange_limit',pd.Series(dtype=bool)).eq(True).sum()),
        extra_sell_orders=extra_count,extra_sell_filled=extra_actual_count,extra_sell_cost_currency=extra_cost,
        extra_adjusted_shares=extra_qty,variance=float(net.var(ddof=1)),variance_top10=parts.head(10).to_dict(),
        focus_300308=dict(net_contribution=float(stocks.get('300308.SZ',pd.Series(0.,index=stocks.index)).sum()),
                          variance_contribution=float(parts.get('300308.SZ',0.)),variance_share=float(parts.get('300308.SZ',0.)/net.var(ddof=1))),
        drawdown=drawdown_episode(net),contribution_max_abs=float((stocks.sum(axis=1)-net).abs().max()))
    save(str((directory/'risk_summary.json').relative_to(OUT)),summary)
    return daily,summary,stocks


def plots(reports: dict[str,dict[str,pd.DataFrame]], risk: pd.DataFrame) -> None:
    """完整两窗共享纵轴累计净收益及风险图。

    :param reports: 六臂两窗日报。
    :param risk: 每日真实风险。
    :returns: 无。
    """
    for name,arms in [('cumulative_main',['G1_original','B','D','cap_vol']),('cumulative_controls',ARMS),('drawdown',ARMS),('difference_vs_G1',['B','D','cap_vol'])]:
        fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
        for ax,(w,rr) in zip(axes,reports.items()):
            for a in arms:
                net=rr[a]['return']-rr[a].cost
                if name=='drawdown':
                    nav=(1+net).cumprod(); y=nav/np.maximum(nav.cummax(),1)-1
                elif name=='difference_vs_G1': y=(net-(rr['G1_original']['return']-rr['G1_original'].cost)).cumsum()
                else: y=net.cumsum()
                ax.plot(y.index,y,color=COLORS[a],label=LABELS[a],lw=2 if a=='cap_vol' else 1.5)
            ax.set_title(f'{w}: {rr["D"].index.min():%Y-%m-%d} to {rr["D"].index.max():%Y-%m-%d}')
            ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2); ax.tick_params(axis='x',rotation=30)
        titles={'cumulative_main':'Net cumulative return: cumsum(return - cost)', 'cumulative_controls':'All prespecified arms: cumsum(return - cost)',
                'drawdown':'Compounded drawdown (initial NAV = 1)','difference_vs_G1':'Cumulative net return difference vs Original G1'}
        fig.suptitle(titles[name]); handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9); fig.tight_layout(rect=(0,.13,1,.95)); fig.savefig(OUT/f'{name}.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,10),sharey='row')
    for j,w in enumerate(reports):
        for a in ['G1_original','B','D','cap_vol']:
            data=risk[(risk.window==w)&(risk.arm==a)].set_index('date')
            for i,field in enumerate(['cash','vol60','vol60_invested_normalized']):
                axes[i,j].plot(data.index,data[field],label=LABELS[a],color=COLORS[a])
                axes[i,j].yaxis.set_major_formatter(PercentFormatter(1)); axes[i,j].grid(alpha=.2)
                axes[i,j].set_title(f'{w}: '+{'cash':'Start-of-day cash / NAV','vol60':'NAV-weighted prior 60d stock volatility','vol60_invested_normalized':'Invested-normalized prior 60d volatility'}[field])
                axes[i,j].tick_params(axis='x',rotation=25)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2); fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(OUT/'cash_volatility.png',dpi=170); plt.close(fig)


def report() -> None:
    """核验所有产物后生成中文报告及最终阶段状态。

    :returns: 无。
    :raises RuntimeError: 缺失完整两窗或真实对账失败。
    """
    p=load_protocol(); reports={}; all_metrics={}; summary={}; risk_rows=[]; stocks={}; comparisons={}
    gate=json.loads((OUT/'replay_gate_passed.json').read_text())
    assert len(gate)==8 and all(x['max_abs']<=1e-10 for x in gate.values())
    for w in p['windows']:
        reports[w]={}; all_metrics[w]={}; summary[w]={}; stocks[w]={}
        panel=pd.read_parquet(OUT/f'risk_panel_{w}.parquet')
        for a in ARMS:
            directory=OUT/'accounts'/w/a
            assert json.loads((directory/'complete.json').read_text())['reconciliation']['exact']
            r=pd.read_parquet(OUT/f'report_{w}_{a}.parquet'); reports[w][a]=r; all_metrics[w][a]=metrics(r)
            if a!='G1_original': pd.testing.assert_index_equal(r.index,reports[w]['G1_original'].index)
            daily,s,c=account_evidence(w,a,r,panel); risk_rows.append(daily); summary[w][a]=s; stocks[w][a]=c
        comparisons[w]={}
        for ref in ['G1_original','B','D']:
            rr,dd,bb=all_metrics[w]['cap_vol'],all_metrics[w]['D'],all_metrics[w][ref]
            advantage=dd['net_sum']-bb['net_sum']
            comparisons[w][ref]=dict(delta_pp=(rr['net_sum']-bb['net_sum'])*100,
                 retained_advantage=(rr['net_sum']-bb['net_sum'])/advantage if ref!='D' and abs(advantage)>1e-8 else None,
                 d_original_advantage_pp=advantage*100,sharpe_delta=rr['sharpe']-bb['sharpe'],mdd_delta_pp=(rr['mdd']-bb['mdd'])*100,
                 bootstrap=bootstrap(reports[w]['cap_vol']['return']-reports[w]['cap_vol'].cost,reports[w][ref]['return']-reports[w][ref].cost))
    risk=pd.concat(risk_rows,ignore_index=True); risk.to_parquet(OUT/'risk_daily_all.parquet',index=False)
    result=acceptance(all_metrics); save('metrics.json',all_metrics); save('comparisons.json',comparisons); save('acceptance.json',result); save('risk_summary.json',summary)
    pd.DataFrame([{**m,'window':w,'arm':a} for w,aa in all_metrics.items() for a,m in aa.items()]).to_csv(OUT/'metrics.csv',index=False)
    # 共同回撤：峰日不计入，峰后到谷日含谷；候选自身区间另列。
    attribution={}
    for w in p['windows']:
        periods={'candidate_drawdown':summary[w]['cap_vol']['drawdown']['interval_dates']}
        if w=='W3': periods['D_common_drawdown']=[str(x.date()) for x in reports[w]['D'].index if pd.Timestamp('2025-10-29')<x<=pd.Timestamp('2025-11-24')]
        for period,dates in periods.items():
            difference=stocks[w]['cap_vol'].subtract(stocks[w]['D'],fill_value=0).reindex(pd.to_datetime(dates))
            bystock=difference.sum().sort_values()
            bystock.rename('R_minus_D_net_contribution').to_csv(OUT/f'stock_difference_{w}_{period}.csv')
            difference.to_parquet(OUT/f'stock_daily_difference_{w}_{period}.parquet')
            expected=(reports[w]['cap_vol']['return']-reports[w]['cap_vol'].cost-reports[w]['D']['return']+reports[w]['D'].cost).reindex(pd.to_datetime(dates)).sum()
            np.testing.assert_allclose(bystock.sum(),expected,atol=1e-10,rtol=0)
            attribution[w+'_'+period]=dict(total_pp=float(bystock.sum()*100),bottom10_pp=(bystock.head(10)*100).to_dict(),top10_pp=(bystock.tail(10).sort_values(ascending=False)*100).to_dict(),dates=dates)
    save('drawdown_attribution.json',attribution); plots(reports,risk)
    relative='../artifacts/experiments/d_risk_overlay_20261102/'
    passed='保留优势且改善风险门槛通过' if result['passed'] else '固定联合门槛未通过'
    above=all(result['checks'][w]['above_g1'] and result['checks'][w]['above_b'] for w in result['checks'])
    improved=all(result['checks'][w]['sharpe'] and result['checks'][w]['mdd'] for w in result['checks'])
    lines=['# D组合风险约束实验报告（2026-11-02）','',
      f'**结论：{passed}。** 主候选cap_vol两窗均保留对原G1和已采纳B的净累计收益优势：{"是" if above else "否"}；两窗Sharpe不低于原D且最大回撤不恶化：{"是" if improved else "否"}。完整两窗回测、真实账户风险归因和曲线已完成；未自动采纳。','',
      '本轮仅改变D（日期等权dynamic adapter MC40）的组合订单数量与额外部分减仓层，未训练新模型、未重新生成预测，不代表预测头或状态适配器进步。原G1主基准、B已采纳MC40和源封存产物保持不变。主候选始终为cap_vol，cap_only/vol_only仅为预设诊断对照，不从对照事后择优。','',
      f'![扣费累计收益（每日净收益累加）]({relative}cumulative_main.png)','',
      '主图为完整日报 `cumsum(return-cost)`，W3/W4共享纵轴；原G1深灰蓝、B绿、原D紫、本轮R橙。图中文字使用英文，避免服务器字体问题。','',
      '|窗口|方案|净累计%|净复利%|Sharpe|MDD%|年化波动%|总成本%|日均换手%|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for w,aa in all_metrics.items():
        for a,m in aa.items(): lines.append(f'|{w}|{a}|{m["net_sum"]*100:.4f}|{m["compound"]*100:.4f}|{m["sharpe"]:.4f}|{m["mdd"]*100:.4f}|{m["annual_vol"]*100:.4f}|{m["cost"]*100:.4f}|{m["turnover"]*100:.4f}|')
    lines+=['','W3为2025-07-01至2025-12-31共126日；W4为2026-01-05至2026-07-24共134日。冻结D信号有效格37223/39563。Sharpe用252日、样本标准差ddof1、零无风险利率；MDD以复利净值计算并含初始NAV=1。成本为日成本/NAV之和，换手为原引擎日成交金额/前日NAV（日内买卖双边合计）的日均值。','',
       '## 固定门槛和收益优势保留','',f'逐项判定：`{json.dumps(result,ensure_ascii=False)}`。任何一窗不满足即不通过，本轮不再调cap、波动窗或折扣强度。','',
       '|窗口|相对原D收益损失pp（正值=损失）|R−G1 pp|G1优势保留比例|R−B pp|B优势保留比例|R−D Sharpe|R−D MDD pp|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for w,cc in comparisons.items():
        lines.append(f'|{w}|{-cc["D"]["delta_pp"]:.4f}|{cc["G1_original"]["delta_pp"]:.4f}|{cc["G1_original"]["retained_advantage"]:.2%}|{cc["B"]["delta_pp"]:.4f}|{cc["B"]["retained_advantage"]:.2%}|{cc["D"]["sharpe_delta"]:.4f}|{cc["D"]["mdd_delta_pp"]:.4f}|')
    lines+=['','优势保留比例=(R−基准)/(原D−基准)，分母绝对值≤1e-8时记不可解释；负比例表示原优势被反转。MDD差为正表示回撤减轻。','',
        '|窗口|R减参照|原D原优势pp|20日配对块bootstrap95%区间pp|重采样差额>0比例|','|---|---|---:|---|---:|']
    for w,cc in comparisons.items():
        for ref,c in cc.items(): lines.append(f'|{w}|{ref}|{c["d_original_advantage_pp"]:.4f}|{c["bootstrap"]["ci95_pp"]}|{c["bootstrap"]["positive_fraction"]:.2%}|')
    lines+=['','bootstrap固定5000次、seed100，仅用于历史诊断。W3/W4此前反复查看，属于历史开发研究，不是独立盲测；重采样正值比例不是未经选择偏差影响的显著性证明。','',
        '## 真实仓位、现金与风险暴露','',
        '|窗口|方案|日初最大单股权重均值/峰值%|日末最大权重均值/峰值%|日初HHI均值|日初投资比例均值/峰值%|日初现金均值/峰值%|日末现金均值/峰值%|','|---|---|---|---|---|---|---|']
    for w,aa in summary.items():
        for a,s in aa.items(): lines.append(f'|{w}|{a}|{100*s["max_weight_mean"]:.2f}/{100*s["max_weight_peak"]:.2f}|{100*s["end_max_weight_mean"]:.2f}/{100*s["end_max_weight_peak"]:.2f}|{s["hhi_mean"]:.5f}|{100*s["invested_mean"]:.2f}/{100*s["invested_peak"]:.2f}|{100*s["cash_mean"]:.2f}/{100*s["cash_peak"]:.2f}|{100*s["end_cash_mean"]:.2f}/{100*s["end_cash_peak"]:.2f}|')
    lines+=['',f'W3原D/B日末最大单股权重均值复核为{100*summary["W3"]["D"]["end_max_weight_mean"]:.2f}%/{100*summary["W3"]["B"]["end_max_weight_mean"]:.2f}%，日末现金均值为{100*summary["W3"]["D"]["end_cash_mean"]:.2f}%/{100*summary["W3"]["B"]["end_cash_mean"]:.2f}%。日初口径与日末口径不同，保留两者，不混用。']
    lines+=['','现金峰值包含初始空仓日，不能忽略这一口径。风险暴露以真实日初权重乘截至前一交易日的历史值，20/60日股票波动均年化；beta为60日沪深300简单beta。','',
        '|窗口|方案|20日NAV加权波动%|60日NAV加权波动%|60日持仓内归一波动%|NAV加权beta|60日有效覆盖权重%|','|---|---|---:|---:|---:|---:|---:|']
    for w,aa in summary.items():
        for a,s in aa.items(): lines.append(f'|{w}|{a}|{100*s["vol20_mean"]:.2f}|{100*s["vol60_mean"]:.2f}|{100*s["vol60_invested_normalized_mean"]:.2f}|{s["beta60_mean"]:.3f}|{100*s["vol60_covered_weight_mean"]:.2f}|')
    for w in summary:
        d,r=summary[w]['D'],summary[w]['cap_vol']
        lines+=['',f'{w}：R相对D平均日初现金变化{100*(r["cash_mean"]-d["cash_mean"]):+.2f}pp，60日NAV加权波动变化{100*(r["vol60_mean"]-d["vol60_mean"]):+.2f}pp，持仓内归一波动变化{100*(r["vol60_invested_normalized_mean"]-d["vol60_invested_normalized_mean"]):+.2f}pp。现金效应与持股构成变化同时存在，不将低仓位带来的组合波动下降解释成选股或预测能力提高。']
    lines+=['',f'![现金与历史波动暴露]({relative}cash_volatility.png)','',
        '行业及流通市值缺可靠PIT接口，本轮不查询、不使用最新值替代。缺风险数据的买入折扣固定0.5；完整缺失审计见risk_features_W3/W4.json及risk_panel parquet。历史价格不向未来填补，风险值使用60个简单收益的ddof1标准差，至少61个收盘价。median_sigma仅在当日有限D信号的CSI300 PIT股票上计算。','',
        '## 风险减仓、超限与交易边界','',
        '|窗口|风险臂|额外减仓计划/成交笔数|归属风险减仓费用（元）|日末超限天数/股票日|超限原因计数|','|---|---|---|---:|---|---|']
    for w,aa in summary.items():
        for a in ['cap_only','vol_only','cap_vol']:
            s=aa[a]; lines.append(f'|{w}|{a}|{s["extra_sell_orders"]}/{s["extra_sell_filled"]}|{s["extra_sell_cost_currency"]:.2f}|{s["cap_exceed_days"]}/{s["cap_exceed_stock_days"]}|{s["cap_exceed_reason_counts"]}|')
    lines+=['','vol_only没有目标上限，其超限列为不适用（产物计数为0）。额外减仓成本来自实际成交，合并卖单按额外数量占比归属成本；不是反事实增量成本估计。order_execution_audit.parquet同时保留约束前后数量/估值金额、持有计数、原交易所拒绝及实际成交数量、费用和未成交标记。','',
       'not_tradable是继承Qlib交易所的可卖判定，不直接等同现实停牌；逐笔审计另存exchange_suspended/ exchange_limit，原引擎价格限制或缺行情均保持原样，不为风险层放宽。cap_vol实际目标为1.5%至3%（3%×f），不是统一3%硬上限。', '',
       '上限为过去信息形成的目标，不是每时每刻的硬上限。实际超限逐股逐日列在actual_cap_exceedances.csv：hold_thresh/交易所不可卖/取整为明确阻碍；price_nav_move_or_lot_rounding表示剩余超限可能来自执行日价格与NAV变动或整手余量，不将其武断归因为跳空。原排名换出仍最多5只，额外部分风险减仓是新增交易行为。风险交易改变真实账户路径后，后续原策略入选集合可能改变，最终持仓集合并非逐日固定。','',
       '成交保持Top50/drop5/hold5、t信号次日open、买0.001/卖0.0015/最低5、涨跌停0.095、资金1亿、CSI300 PIT、benchmark000300.SH。不增加shift，不使用engine_v2；risk cap=0.03×f，买单先乘f再限额，减仓所得留现金。无针对300308.SZ或其他事后贡献股规则，无行业黑名单。','',
       '## 共同回撤、候选自身回撤与个股风险','',
       '|窗口|方案|回撤峰日|谷日|恢复日|MDD%|','|---|---|---|---|---|---:|']
    for w in summary:
        for a in ['D','cap_vol']:
            e=summary[w][a]['drawdown']; lines.append(f'|{w}|{a}|{e["peak"]}|{e["trough"]}|{e["recovery"]}|{e["mdd"]*100:.4f}|')
    for name,x in attribution.items():
        lines+=['',f'{name}：R−D逐股净贡献合计{x["total_pp"]:.4f}pp。改善前五：`{dict(list(x["top10_pp"].items())[:5])}`；恶化前五：`{dict(list(x["bottom10_pp"].items())[:5])}`。完整逐股及逐日差额已保存。']
    lines+=['','W3共同区间固定为2025-10-29峰后至2025-11-24谷（不含峰日、含谷日）。候选自身峰谷单独列示，不能仅在原D区间观察后宣称风险消失。每个区间逐股净贡献差与组合日净收益差对账≤1e-10；贡献是实际账户会计分解，不是个股因果效应。','',
       '|窗口|方案|300308.SZ净收益贡献pp|该股方差贡献|该股方差份额|','|---|---|---:|---:|---:|']
    for w in summary:
        for a in ['D','cap_vol']:
            s=summary[w][a]['focus_300308']; lines.append(f'|{w}|{a}|{s["net_contribution"]*100:.4f}|{s["variance_contribution"]:.10f}|{s["variance_share"]:.2%}|')
        denominator=summary[w]['D']['focus_300308']['net_contribution']
        ratio=f"{summary[w]['cap_vol']['focus_300308']['net_contribution']/denominator:.2%}" if abs(denominator)>1e-8 else '不可解释（原D该股贡献为零或近零；W4未持有该股）'
        lines+=['',f'{w}该股净贡献保留比例{ratio}；候选方差贡献前五：`{dict(list(summary[w]["cap_vol"]["variance_top10"].items())[:5])}`。这些是事后风险复核，不反馈到规则选择。','']
    lines+=['W3候选最大回撤仍为10月29日至11月24日，没有把最大回撤移到别的日期；W4候选峰日提前到5月13日（原D为6月22日），谷日同为7月17日，峰谷跨度从19增至46个交易日，回撤幅度下降伴随更长的峰谷跨度。W3方差贡献首位从300308.SZ变为601138.SH，说明风险主导排序有所变化；W4首位仍为600176.SH。上述是风险分布变化的描述，不证明降低某股权重具有独立因果效应。', '']
    lines+=['方差按Cov(个股净贡献,组合净收益)分解，并保留账户残差项；逐股方差贡献总和与组合样本方差对账≤1e-14。风险是否转移需同时看候选自身峰谷和上述风险主导股，不能由单一股票降权直接推出总体改善。','',
       f'![所有预设风险臂完整曲线]({relative}cumulative_controls.png)','',f'![含初始净值1的回撤图]({relative}drawdown.png)','',f'![相对原G1累计差额]({relative}difference_vs_G1.png)','',
       '## 工程门禁、身份与复现','',
       f'协议SHA256：`{sha(OUT/"protocol.json")}`；冻结协议提交：`b24f78b`；base：`{p["base"]}`；分支：`codex/d-risk-overlay-20261102`；冻结D权重SHA256：`{p["source_model_sha256"]}`。本轮GPU新增用时0，未加载模型训练或推理。','',
       '最终执行代码提交为03d8ff5（完整SHA见各账户complete.json）；每次执行另记录risk_overlay/run、risk_attribution与原runner的文件SHA，缓存仅在代码及日报SHA匹配时复用。相关规则/既有回归/真实产物交付测试共51项通过，完整测试命令及输出摘要见verification.json；测试不运行训练或查询新行情。', '',
       '全部12份源输入SHA详见protocol.json；历史特征输入SHA和逐格最后历史日期详见risk_features_W3/W4.json与risk_panel_W3/W4.parquet；代码与小产物身份见delivery_manifest.json。信号直接复用源文件，三风险臂完整身份指向同一D信号，不另造模型身份。','',
       f'8次原G1/B/D/neutral日报重放最大误差{max(x["max_abs"] for x in gate.values()):.3e}，日期及列完全一致；neutral原始/最终订单与原D逐值一致。所有12臂窗口真实账户贡献、成本、NAV净收益通过对账；实际请求在读取前受market_guard约束，末日≤2026-07-24。','',
       '工程修订：首次风险臂尝试发现首日u前日t没有窗口信号，原策略空订单且空仓；增加空决策透传回归测试，不平移信号。协调器独立审查随后指出：退出有效信号池的持仓不等于缺风险，已修正为仅由合法历史sigma决定折扣，信号池只用于median；并将last_history_date与正有限估值价格mask对齐、补齐接口文档。修复均先通过失败回归测试再验证通过。初版六风险回测已经查看，保留在revision01_superseded；修正后重新跑全部8项源/neutral门禁及6项风险回测，结果与初版一致，不从前后版本择优。审查原文见D风险约束独立实现审查_20261102.md。执行计划后补了审查要求，原冻结protocol内完整原文与SHA不变。', '',
       '运行主机实际标识记录于runtime.json，为WS-EC500T；dev-debian别名在沙箱内无法解析，不能由此确认或否认外部SSH别名映射。使用用户指定工作树、quant环境及DolphinDB后端，保留此环境核验限制。','',
       '复现命令（仅CPU，已存在完整标记时复用产物；如需全量重跑请使用独立输出目录并先核验身份）：','',
       '```bash','CUDA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/user/miniconda3/envs/quant/bin/python -m state_adapter.risk_overlay_run --stage all',
       '/home/user/miniconda3/envs/quant/bin/python -m pytest tests/test_risk_overlay.py tests/test_risk_overlay_delivery.py tests/test_state_factorial.py tests/test_qlib_mean_comparison.py tests/test_date_balanced.py -q','```','',
       '原始账户产物在accounts/{W3,W4}/{方案}/：report、positions、trades、balances、stock_daily_contributions、reconciliation_audit、original_orders、风险臂final_orders/order_audit/order_execution_audit。完整数值与图端点由daily及metrics复算。主索引交由协调器处理，本轮不push、不切master、不自动采纳。']
    timings=[json.loads(f.read_text()) for f in (OUT/'accounts').glob('*/*/complete.json')]
    archived=[json.loads(f.read_text()) for f in (OUT/'revision01_superseded/accounts').glob('*/*/complete.json')]
    archived+=json.loads((OUT/'revision02_timing.json').read_text()) if (OUT/'revision02_timing.json').exists() else []
    timing=dict(archived_backtest_cpu_seconds=sum(x['cpu_seconds'] for x in archived),archived_backtest_wall_seconds=sum(x['wall_seconds'] for x in archived),backtest_cpu_seconds=sum(x['cpu_seconds'] for x in timings),backtest_wall_seconds=sum(x['wall_seconds'] for x in timings),gpu_seconds=0,
        elapsed_wall_seconds=time.time()-json.loads((OUT/'execution_clock.json').read_text())['started'])
    save('timing_summary.json',timing); lines+=['',f'累计回测CPU进程用时{timing["backtest_cpu_seconds"]:.2f}秒，回测墙钟用时{timing["backtest_wall_seconds"]:.2f}秒，报告生成时总墙钟{timing["elapsed_wall_seconds"]:.2f}秒；另保留归档复跑CPU{timing["archived_backtest_cpu_seconds"]:.2f}秒、墙钟{timing["archived_backtest_wall_seconds"]:.2f}秒；GPU新增用时0。开始/截止持久化在execution_clock.json，恢复不重置。']
    path=ROOT/'docs/D组合风险约束实验报告_20261102.md'; path.write_text('\n'.join(lines)+'\n')
    save('bot_status.json',dict(status='candidate' if result['passed'] else 'negative',summary=passed,report=str(path.relative_to(ROOT)),gpu_seconds=0,completed=time.time(),next_step='协调器审阅及主索引发布；不自动采纳'))

if __name__=='__main__': report()
