"""固定风险改序与报告指标，纯计算不初始化行情。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def risk_adjusted_score(g1: pd.Series,q10: pd.Series,penalty: float=.25) -> pd.Series:
    """在原G1有效股票格上施加固定风险排名扣分。

    :param g1: 已冻结原G1有效截面分数。
    :param q10: 相同股票的原收益尺度下分位数。
    :param penalty: 预登记扣分系数；零值精确走原分数路径。
    :return: 原索引顺序的分数。
    """
    if not g1.index.is_unique or not q10.index.is_unique or not g1.index.equals(q10.index):
        raise ValueError('风险预测股票格不一致')
    if not np.isfinite(g1).all() or not np.isfinite(q10).all():raise ValueError('风险格缺失')
    if penalty==0:return g1.copy()
    if penalty!=.25:raise ValueError('不允许搜索风险权重')
    risk=(-q10).clip(lower=0)
    return g1.rank(method='average',pct=True)-penalty*risk.rank(method='average',pct=True)


def annualized_return(net: np.ndarray) -> float:
    """复利年化净收益。

    :param net: 完整逐日扣费收益。
    :return: 252交易日口径年化。
    """
    x=np.asarray(net,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or (x<=-1).any():
        raise ValueError('非法日净收益')
    return float(np.expm1(np.log1p(x).sum()*252/len(x)))


def execution_score(scores: pd.Series) -> pd.Series:
    """为正惩罚新分数确定唯一执行次序，不依赖引擎稳定排序。

    :param scores: 固定公式计算的新分数。
    :return: 保持原索引，按分数降序/代码升序对应的唯一序数。
    """
    ordered=pd.DataFrame({'score':scores,'code':scores.index.astype(str)}).sort_values(['score','code'],ascending=[False,True],kind='stable')
    return pd.Series(np.arange(len(scores),0,-1,dtype=float),index=ordered.index).reindex(scores.index)
