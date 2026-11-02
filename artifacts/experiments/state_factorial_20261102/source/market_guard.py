"""Qlib行情入口在真实读取前执行封存边界，并持久化请求审计。"""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator
import pandas as pd

CUTOFF = pd.Timestamp('2026-07-24')


def bounded_features(fetch: Callable[..., Any], audit_path: Path) -> Callable[..., Any]:
    """包装行情读取，在读取前截断末日，拒绝无界请求。

    :param fetch: 原始D.features可调用对象。
    :param audit_path: 本进程请求审计JSONL。
    :returns: 保持返回对象不变的有界读取函数。
    """
    def guarded(instruments: Any, fields: Any, start_time: Any = None,
                end_time: Any = None, *args: Any, **kwargs: Any) -> Any:
        if end_time is None or (start_time is not None and pd.Timestamp(start_time).normalize() > CUTOFF):
            raise ValueError('拒绝无末日或起日越过封存线的行情请求')
        requested = pd.Timestamp(end_time)
        effective = min(requested, CUTOFF)
        # 先持久记录真正传给底层的边界；日历查询不在这里。
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open('a') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), fields=list(fields), start=str(start_time),
                requested_end=str(requested), effective_end=str(effective), guard_version=1)) + '\n')
            stream.flush()
        result = fetch(instruments, fields, start_time, effective, *args, **kwargs)
        if isinstance(result, pd.DataFrame) and not result.empty:
            if isinstance(result.index, pd.MultiIndex) and 'datetime' in result.index.names:
                dates = result.index.get_level_values('datetime')
            elif isinstance(result.index, pd.DatetimeIndex):
                dates = result.index
            else:
                dates = None
            if dates is not None and pd.Timestamp(dates.max()).normalize() > CUTOFF:
                raise RuntimeError('底层违反封存请求边界，返回未来行情')
        return result
    return guarded


@contextmanager
def market_guard(audit_path: Path) -> Iterator[None]:
    """对Qlib所有D.features调用安装读取前守卫并安全还原。

    :param audit_path: 当前进程请求审计路径。
    :yields: 守卫作用域。
    """
    from kronos_qlib import QlibProvider
    from qlib.data import D
    QlibProvider.init_qlib_once()
    # Wrapper没有__setattr__转发，必须恢复为原本的动态属性状态。
    existed = 'features' in D.__dict__
    previous = D.features
    D.features = bounded_features(previous, audit_path)
    try:
        yield
    finally:
        if existed:
            D.features = previous
        else:
            del D.features
