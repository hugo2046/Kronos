"""D组合风险约束CPU执行器：冻结身份、重放门禁、因果特征、两窗三臂。"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any
import numpy as np
import pandas as pd
from loguru import logger
from state_adapter.risk_overlay import RiskOverlayStrategy, build_risk_panel
from state_adapter.risk_attribution import replay_account
from state_adapter.market_guard import market_guard

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/experiments/d_risk_overlay_20261102'
SRC=Path('/home/user/workspace/Kronos-state-factorial/artifacts/experiments/state_factorial_20261102')


def save(name: str, value: Any) -> None:
    """原子持久化JSON。

    :param name: 输出相对路径。
    :param value: 可序列化数据。
    :returns: 无。
    """
    path=OUT/name; path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n'); temp.replace(path)


def sha(path: Path) -> str:
    """计算文件SHA256。

    :param path: 文件。
    :returns: 十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_budget() -> None:
    """检查持久化墙钟截止，不随恢复重置。

    :returns: 无。
    :raises RuntimeError: 预算到期。
    """
    if time.time() >= json.loads((OUT/'execution_clock.json').read_text())['deadline']:
        raise RuntimeError('三小时墙钟预算已到期')


def load_protocol() -> dict[str,Any]:
    """校验协议及全部源输入身份。

    :returns: 冻结协议。
    :raises RuntimeError: 协议或源数据漂移。
    """
    if sha(OUT/'protocol.json') != (OUT/'protocol.json.sha256').read_text().strip(): raise RuntimeError('协议漂移')
    p=json.loads((OUT/'protocol.json').read_text())
    for name,digest in p['source_sha256'].items():
        if sha(Path(name))!=digest: raise RuntimeError('源输入漂移: '+name)
    return p


def signal_for(window: str, arm: str) -> tuple[pd.DataFrame,pd.Series]:
    """读取冻结宽表并转换原生信号，不增加shift。

    :param window: 窗口。
    :param arm: 冻结信号臂。
    :returns: 宽表与原生信号。
    """
    wide=pd.read_parquet(SRC/f'signal_{window}_{arm}.parquet')
    signal=wide.stack(); signal.index.names=['datetime','instrument']
    return wide,signal.swaplevel().sort_index()


def execute(window: str, arm: str, bounds: list[str], panel: pd.DataFrame | None=None) -> tuple[pd.DataFrame,dict[str,Any]]:
    """执行一臂并保存真实交易、观测器对账、订单及身份。

    :param window: 窗口。
    :param arm: 臂名。
    :param bounds: 日期边界。
    :param panel: 过去信息风险面板。
    :returns: 日报及审计。
    """
    check_budget(); source=arm if arm in ['G1_original','B','D'] else 'D'
    wide,signal=signal_for(window,source)
    directory=OUT/'accounts'/window/arm; directory.mkdir(parents=True,exist_ok=True)
    completion=directory/'complete.json'
    code_identity={name:sha(ROOT/name) for name in ['state_adapter/risk_overlay.py','state_adapter/risk_overlay_run.py',
        'state_adapter/risk_attribution.py','qlib_mean_comparison/run.py']}
    if completion.exists():
        meta=json.loads(completion.read_text())
        if meta.get('code_sha256')==code_identity and meta.get('report_sha256')==sha(directory/'report.parquet'):
            return pd.read_parquet(directory/'report.parquet'),meta
        logger.info('代码身份或完成文件改变，重新执行 {} {}',window,arm)
    mode=arm if arm in ['cap_only','vol_only','cap_vol'] else 'neutral'
    strategy=RiskOverlayStrategy(topk=50,n_drop=5,hold_thresh=5,signal=signal,risk_panel=panel,mode=mode)
    # D原始重放另直接用原生Topk，neutral才使用风险继承类。
    if arm in ['G1_original','B','D']:
        from qlib.contrib.strategy import TopkDropoutStrategy
        strategy=TopkDropoutStrategy(topk=50,n_drop=5,hold_thresh=5,signal=signal)
        raw=[]; original=strategy.generate_trade_decision
        def record_original(execute_result=None):
            decision=original(execute_result)
            for o in decision.get_decision(): raw.append(dict(date=o.start_time.normalize(),instrument=o.stock_id,direction=int(o.direction),amount=float(o.amount)))
            return decision
        strategy.generate_trade_decision=record_original
    else: raw=strategy.raw_orders
    started=time.time(); cpu=time.process_time()
    logger.info('原生CPU回测 {} {}',window,arm)
    report,meta=replay_account(signal,*bounds,directory,strategy=strategy)
    if not meta['reconciliation']['exact']: raise RuntimeError('真实成交费用/持仓损益未对账')
    pd.DataFrame(raw).to_parquet(directory/'original_orders.parquet',index=False)
    if isinstance(strategy,RiskOverlayStrategy):
        pd.DataFrame(strategy.final_orders).to_parquet(directory/'final_orders.parquet',index=False)
        pd.DataFrame(strategy.order_audit).to_parquet(directory/'order_audit.parquet',index=False)
    report.to_parquet(OUT/f'report_{window}_{arm}.parquet')
    net=report['return']-report.cost
    pd.DataFrame(dict(net_daily=net,cumulative=net.cumsum(),nav=(1+net).cumprod())).to_parquet(OUT/f'daily_{window}_{arm}.parquet')
    meta.update(code_sha256=code_identity,report_sha256=sha(directory/'report.parquet'),wall_seconds=time.time()-started,cpu_seconds=time.process_time()-cpu,gpu_seconds=0,
        signal_sha256=sha(SRC/f'signal_{window}_{source}.parquet'),protocol_sha256=sha(OUT/'protocol.json'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_arm=source)
    save(str(completion.relative_to(OUT)),meta)
    save('bot_status.json',dict(status='running',stage=f'完成{window}/{arm}',pid=os.getpid(),next_step='继续门禁或固定三臂与报告'))
    return report,meta


def replay_gates(p: dict[str,Any]) -> None:
    """全部源日报和neutral逐值复现后才允许风险臂。

    :param p: 冻结协议。
    :returns: 无，门禁持久化。
    """
    audit={}
    for w,bounds in p['windows'].items():
        for arm in ['G1_original','B','D','neutral']:
            r,meta=execute(w,arm,bounds)
            expected=pd.read_parquet(SRC/f'report_{w}_{"D" if arm=="neutral" else arm}.parquet')
            if not r.index.equals(expected.index) or not r.columns.equals(expected.columns): raise RuntimeError('日报轴不一致')
            delta=float(np.abs(r-expected).to_numpy().max())
            if not np.isfinite(delta) or delta>1e-10: raise RuntimeError(f'日报门禁失败 {w} {arm}: {delta}')
            audit[f'{w}_{arm}']=dict(max_abs=delta,**meta)
            if arm=='neutral':
                a=pd.read_parquet(OUT/'accounts'/w/'D'/'original_orders.parquet')
                b=pd.read_parquet(OUT/'accounts'/w/'neutral'/'final_orders.parquet')
                pd.testing.assert_frame_equal(a,b,atol=1e-10,rtol=0)
                audit[f'{w}_{arm}']['orders_equal']=True
            save('replay_gate_progress.json',audit)
    save('replay_gate_passed.json',audit)


def risk_features(p: dict[str,Any]) -> dict[str,pd.DataFrame]:
    """守卫真实历史行情请求，计算逐决策日风险面板。

    :param p: 协议。
    :returns: 两窗面板。
    """
    from qlib.data import D
    wide={w:signal_for(w,'D')[0] for w in p['windows']}
    path=OUT/'close_history.parquet'
    if not path.exists():
        codes=sorted(set().union(*(set(v.columns) for v in wide.values()))|{'000300.SH'})
        with market_guard(OUT/'risk_market_requests.jsonl'):
            frame=D.features(codes,['$close'],start_time='2025-01-01',end_time='2026-07-24')
        close=frame.iloc[:,0].unstack('instrument').sort_index()
        close.to_parquet(path)
    close=pd.read_parquet(path)
    panels={}
    for w,s in wide.items():
        panel=build_risk_panel(close,s); panel.to_parquet(OUT/f'risk_panel_{w}.parquet'); panels[w]=panel
        valid=panel[panel.signal_valid]
        expected={'W3':(126,37223),'W4':(134,39563)}[w]
        if (len(s),len(valid))!=expected: raise RuntimeError('冻结信号覆盖不符')
        save(f'risk_features_{w}.json',dict(close_sha256=sha(path),signal_sha256=sha(SRC/f'signal_{w}_D.parquet'),panel_sha256=sha(OUT/f'risk_panel_{w}.parquet'),
             valid_cells=len(valid),missing_risk_cells=int(valid.missing_risk.sum()),missing_valuation_cells=int(valid.price.isna().sum()),
             historical_date_max=str(close.index.max()),timing='面板索引为决策日t，执行u取原日历shift=1，不额外shift'))
    return panels


def main() -> None:
    """互斥、CPU执行且保存可恢复阶段。

    :returns: 无。
    """
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['gates','arms','all','report'],default='all'); args=parser.parse_args()
    lock=(OUT/'runner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=load_protocol(); check_budget()
    save('runtime.json',dict(hostname=socket.gethostname(),pid=os.getpid(),python=os.sys.executable,gpu_seconds=0,started=time.time()))
    try:
        if args.stage!='report':
            replay_gates(p)
            if args.stage=='gates': return
            panels=risk_features(p)
            for w,bounds in p['windows'].items():
                for arm in ['cap_only','vol_only','cap_vol']: execute(w,arm,bounds,panels[w])
        if args.stage=='arms': return
        from state_adapter.risk_overlay_report import report
        report()
    except BaseException as exc:
        save('bot_status.json',dict(status='continue',summary=f'{type(exc).__name__}: {exc}',next_step='修复工程失败后原命令恢复，墙钟不重置',pid=os.getpid(),gpu_seconds=0)); raise

if __name__=='__main__': main()
