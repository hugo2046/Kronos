"""纯函数数据契约：日历期限与只用历史的尺度。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_labels(frame: pd.DataFrame, origins: pd.DatetimeIndex, calendar: pd.DatetimeIndex,
                label_end: pd.Timestamp) -> pd.DataFrame:
    """生成次日开盘至第十日收盘毛收益，不删除缺标签的原点。

    :param frame: 单股open/close，日期索引。
    :param origins: 已确定的预测原点。
    :param calendar: 市场交易日历。
    :param label_end: 当前分段允许的标签截止。
    :return: 保持原点索引的标签、有效mask与目标日期。
    """
    if not calendar.is_unique or not calendar.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError('日期必须有序且唯一')
    if pd.Timestamp(label_end)>pd.Timestamp('2026-07-24'):
        raise ValueError('标签上界越过封存')
    pos=calendar.get_indexer(origins)
    if (pos<0).any():raise ValueError('原点不在市场日历')
    out=pd.DataFrame(index=origins,columns=['entry_date','target_date','y'])
    available=pos+10<len(calendar)
    rows=np.flatnonzero(available)
    entry=calendar[pos[rows]+1];target=calendar[pos[rows]+10]
    out.loc[origins[rows],'entry_date']=entry.to_numpy()
    out.loc[origins[rows],'target_date']=target.to_numpy()
    allowed=target<=pd.Timestamp(label_end);rows=rows[allowed];entry=entry[allowed];target=target[allowed]
    op=frame.open.reindex(entry).to_numpy(dtype=float);cl=frame.close.reindex(target).to_numpy(dtype=float)
    valid=np.isfinite(op)&np.isfinite(cl)&(op>0)&(cl>0)
    out.loc[origins[rows[valid]],'y']=cl[valid]/op[valid]-1
    out['y']=pd.to_numeric(out.y);out['valid']=np.isfinite(out.y)
    out['entry_date']=pd.to_datetime(out.entry_date);out['target_date']=pd.to_datetime(out.target_date)
    return out


def history_scale(closes: np.ndarray) -> np.ndarray:
    """计算最近20个历史收益的尺度。

    :param closes: 样本乘历史日期矩阵，至少21个收盘价。
    :return: 每个样本的正尺度。
    """
    c=np.asarray(closes,dtype=np.float64)
    if c.ndim!=2 or c.shape[1]<21 or not np.isfinite(c).all() or (c<=0).any():
        raise ValueError('不完整或非法历史收盘价')
    c=c[:,-21:];returns=c[:,1:]/c[:,:-1]-1
    return np.maximum(returns.std(axis=1,ddof=1)*np.sqrt(10),1e-4)
