"""成交单位纠偏独立CPU执行器，恢复时校验冻结身份与固定截止。"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager
from typing import Any, Iterator
import numpy as np
import pandas as pd
from loguru import logger
from state_adapter.risk_attribution import replay_account
from state_adapter.risk_overlay import RiskOverlayStrategy
from state_adapter.market_guard import market_guard

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/experiments/limit_unit_audit_20261102'
SRC = Path('/home/user/workspace/Kronos-state-factorial/artifacts/experiments/state_factorial_20261102')
RISK = Path('/home/user/workspace/Kronos-d-risk/artifacts/experiments/d_risk_overlay_20261102')
LIMIT = ('$pctchange / 100 >= 0.095', '$pctchange / 100 <= -0.095')


def sha(path: Path) -> str:
    """计算身份。:param path: 文件。 :returns: SHA256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, value: Any) -> None:
    """原子记录。:param name: 相对路径。 :param value: 数据。 :returns: 无。"""
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temp.replace(path)


def budget() -> None:
    """继承墙钟。:raises RuntimeError: 到期。 :returns: 无。"""
    if time.time() >= json.loads((OUT / 'execution_clock.json').read_text())['deadline']:
        raise RuntimeError('原三小时墙钟已到期，不重置')


@contextmanager
def execution(mode: str) -> Iterator[None]:
    """仅新runner作用域替换配置并恢复。:param mode: old/new。 :yields: 执行上下文。"""
    from qlib_mean_comparison.run import BACKTEST_KWARGS
    old = BACKTEST_KWARGS['limit_threshold']
    BACKTEST_KWARGS['limit_threshold'] = LIMIT if mode == 'new' else .095
    try:
        yield
    finally:
        BACKTEST_KWARGS['limit_threshold'] = old


def fields() -> None:
    """保存真实字段、全窗flags及异常，不丢弃精度/企业行为异常。:returns: 无。"""
    from qlib.data import D
    if (OUT / 'field_audit.json').exists():
        return
    codes = sorted(set().union(*(set(pd.read_parquet(SRC / f'signal_{w}_D.parquet').columns) for w in ['W3','W4'])))
    columns = ['$close','$preclose','$factor','$change','$pctchange',*LIMIT]
    with market_guard(OUT / 'field_requests.jsonl'):
        sample = D.features(['601138.SH','300308.SZ'], columns, start_time='2025-08-11', end_time='2025-08-20')
        sample.to_csv(OUT / 'fixed_sample.csv')
        frame = D.features(codes, columns, start_time='2025-07-01', end_time='2026-07-24')
    frame['amount_expected'] = (frame['$close'] - frame['$preclose']) / frame['$factor']
    frame['return_expected'] = frame['$close'] / frame['$preclose'] - 1
    frame['amount_residual'] = frame['$change'] - frame.amount_expected
    frame['return_residual'] = frame['$pctchange'] / 100 - frame.return_expected
    frame['missing_fields'] = frame[columns[:5]].isna().any(axis=1)
    frame['amount_exception'] = frame.amount_residual.abs() > .011
    frame['return_exception'] = frame.return_residual.abs() > .000002
    suspended = frame['$close'].isna()
    for direction, compare in [('buy',lambda s:s.ge(.095)),('sell',lambda s:s.le(-.095))]:
        frame[f'old_{direction}'] = compare(frame['$change']) | suspended
        frame[f'new_{direction}'] = compare(frame['$pctchange']/100) | suspended
    for i, direction in enumerate(['buy','sell']):
        np.testing.assert_array_equal(frame[LIMIT[i]].astype(bool) | suspended, frame[f'new_{direction}'])
    frame['old_blocked'] = frame.old_buy | frame.old_sell
    frame['new_blocked'] = frame.new_buy | frame.new_sell
    frame['false_blocked'] = frame.old_blocked & ~frame.new_blocked
    frame.to_parquet(OUT / 'fields_flags.parquet')
    frame.to_csv(OUT / 'fields_flags.csv')
    frame[frame.missing_fields | frame.amount_exception | frame.return_exception].to_csv(OUT / 'field_exceptions.csv')
    save('field_audit.json', dict(rows=len(frame),stocks=len(codes),missing=int(frame.missing_fields.sum()),
        amount_exceptions=int(frame.amount_exception.sum()),return_exceptions=int(frame.return_exception.sum()),
        false_blocked=int(frame.false_blocked.sum()),new_blocked=int(frame.new_blocked.sum()),expression_verified=True,
        tolerance_amount=.011,tolerance_return=.000002))


def run(mode: str, window: str, arm: str, bounds: list[str]) -> pd.DataFrame:
    """重放单臂真实账户及订单。:param mode: 口径。 :param window: 窗口。 :param arm: 臂。 :param bounds: 日期。 :returns: 日报。"""
    budget()
    source = arm if arm in ['G1_original','A','B','C','D'] else 'D'
    path = SRC / f'signal_{window}_{source}.parquet'
    directory = OUT / 'accounts' / mode / window / arm
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(execution_id=mode,signal_sha256=sha(path),runner_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'))
    if (directory/'complete.json').exists():
        meta = json.loads((directory/'complete.json').read_text())
        if all(meta.get(k)==v for k,v in identity.items()) and meta['report_sha256']==sha(directory/'report.parquet'):
            return pd.read_parquet(directory/'report.parquet')
        raise RuntimeError('已完成产物身份漂移，拒绝静默覆盖')
    wide = pd.read_parquet(path)
    assert (len(wide),np.isfinite(wide.values).sum()) == {'W3':(126,37223),'W4':(134,39563)}[window]
    signal = wide.stack(); signal.index.names=['datetime','instrument']; signal=signal.swaplevel().sort_index()
    if source == arm:
        from qlib.contrib.strategy import TopkDropoutStrategy
        strategy = TopkDropoutStrategy(topk=50,n_drop=5,hold_thresh=5,signal=signal)
        raw = []; original = strategy.generate_trade_decision
        def record(execute_result: Any = None) -> Any:
            decision = original(execute_result)
            for order in decision.get_decision():
                raw.append(dict(date=order.start_time.normalize(),instrument=order.stock_id,direction=int(order.direction),amount=float(order.amount)))
            return decision
        strategy.generate_trade_decision = record
    else:
        panel = pd.read_parquet(RISK/f'risk_panel_{window}.parquet')
        strategy = RiskOverlayStrategy(topk=50,n_drop=5,hold_thresh=5,signal=signal,risk_panel=panel,mode=arm)
        raw = strategy.raw_orders
    logger.info('真实账户回放 {} {} {}',mode,window,arm)
    save('bot_status.json',dict(status='running',pid=os.getpid(),stage=f'{mode}/{window}/{arm}',backtest_started=True))
    with execution(mode):
        report,meta = replay_account(signal,*bounds,directory,strategy=strategy)
    if not meta['reconciliation']['exact']:
        raise RuntimeError('账户对账失败')
    pd.DataFrame(raw).to_parquet(directory/'original_orders.parquet',index=False)
    if isinstance(strategy,RiskOverlayStrategy):
        pd.DataFrame(strategy.final_orders).to_parquet(directory/'final_orders.parquet',index=False)
        pd.DataFrame(strategy.order_audit).to_parquet(directory/'order_audit.parquet',index=False)
    meta.update(identity,report_sha256=sha(directory/'report.parquet'))
    save(str((directory/'complete.json').relative_to(OUT)),meta)
    return report


def main() -> None:
    """带互斥门禁的可恢复执行。:returns: 无。"""
    lock = (OUT/'runner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p = json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'protocol.json') == (OUT/'protocol.json.sha256').read_text().strip()
    for name,digest in p['source_sha256'].items():
        if sha(Path(name)) != digest: raise RuntimeError('源身份改变: '+name)
    try:
        budget(); fields()
        gates = {}
        for w,bounds in p['windows'].items():
            for arm in ['G1_original','B','D','neutral','cap_only','vol_only','cap_vol']:
                actual = run('old',w,arm,bounds)
                src = SRC if arm in ['G1_original','B','D','neutral'] else RISK
                expected = pd.read_parquet(src/f'report_{w}_{"D" if arm=="neutral" else arm}.parquet')
                pd.testing.assert_frame_equal(actual,expected,atol=1e-10,rtol=0)
                gates[f'{w}/{arm}'] = float((actual-expected).abs().max().max())
                save('old_gates.json',gates)
        for w,bounds in p['windows'].items():
            for arm in [*p['arms'],'neutral']:
                run('new',w,arm,bounds)
            directory=OUT/'accounts/new'/w
            for name,other in [('report.parquet','report.parquet'),('original_orders.parquet','final_orders.parquet')]:
                pd.testing.assert_frame_equal(pd.read_parquet(directory/'D'/name),pd.read_parquet(directory/'neutral'/other),atol=1e-10,rtol=0)
        save('new_neutral_gate.json',dict(passed=True))
        save('bot_status.json',dict(status='continue',stage='全部旧新账户完成，待完整归因图表报告',pid=os.getpid()))
    except BaseException as exc:
        save('bot_status.json',dict(status='continue',stage='执行异常可恢复',error=f'{type(exc).__name__}: {exc}',pid=os.getpid()))
        raise

if __name__ == '__main__':
    main()
