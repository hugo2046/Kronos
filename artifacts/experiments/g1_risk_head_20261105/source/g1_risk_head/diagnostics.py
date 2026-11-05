"""风险预测指标；按日期聚合，不把股票当独立市场样本。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def auc(y: np.ndarray,risk: np.ndarray) -> float:
    """用平均秩计算带并列的AUROC。

    :param y: 0/1坏结果标签。
    :param risk: 越大风险越高的预测。
    :return: AUROC；只有一类时NaN。
    """
    y=np.asarray(y);risk=np.asarray(risk)
    if y.shape!=risk.shape or not np.isfinite(risk).all() or not np.isin(y,[0,1]).all():
        raise ValueError('AUROC输入非法')
    n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return float('nan')
    ranks=pd.Series(risk).rank(method='average').to_numpy()
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def summarize(y: np.ndarray,pred: np.ndarray,dates: np.ndarray) -> dict[str,float]:
    """真实收益尺度的日期等权pinball与覆盖率。

    :param y: 监督有效样本真实收益。
    :param pred: 同样本三个分位数预测。
    :param dates: 原点日期，不是标签到达日期。
    :return: 日期等权汇总。
    """
    y=np.asarray(y);pred=np.asarray(pred)
    if pred.shape!=(len(y),3) or len(dates)!=len(y) or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('评价样本异常')
    error=y[:,None]-pred;q=np.array([.1,.5,.9])
    loss=np.maximum(q*error,(q-1)*error)
    df=pd.DataFrame({'date':dates,'pinball':loss.mean(1),'p10':loss[:,0],
                     'coverage10':(y<=pred[:,0]).astype(float),
                     'coverage50':(y<=pred[:,1]).astype(float),
                     'coverage90':(y<=pred[:,2]).astype(float)})
    return df.groupby('date').mean().mean().to_dict()


def calibration_by_day(frame: pd.DataFrame,columns: list[str]) -> tuple[pd.DataFrame,dict]:
    """分组日期等权校准与循环块区间，保留逐日聚集证据。

    :param frame: 真实收益/日期/预测分位数表，可为Top50或历史状态分组。
    :param columns: 三分位列名。
    :return: 逐日指标与20日循环块5000次seed100均值区间。
    """
    rows=[]
    for date,g in frame.groupby('date'):
        rows.append(dict(date=date,samples=len(g),**summarize(g.y.to_numpy(),g[columns].to_numpy(),g.date.to_numpy())))
    daily=pd.DataFrame(rows)
    if daily.empty:return daily,dict(dates=0,samples=0,unavailable=True)
    n=len(daily);starts=np.random.default_rng(100).integers(0,n,size=(5000,(n+19)//20))
    idx=((starts[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
    result=dict(dates=n,samples=int(daily.samples.sum()),block=20,resamples=5000,seed=100)
    for col in ['pinball','p10','coverage10','coverage50','coverage90']:
        x=daily[col].to_numpy();result[col]=dict(mean=float(x.mean()),ci95=np.quantile(x[idx].mean(1),[.025,.975]).tolist())
    return daily,result
