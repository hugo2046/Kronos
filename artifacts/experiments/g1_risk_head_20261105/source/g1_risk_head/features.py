"""只读取预测原点历史的确定性G1表示预处理。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def preprocess(raw: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray,np.ndarray]:
    """按原G1窗口标准化并构造时间特征。

    :param raw: 样本乘90日乘六通道，严禁包含预测标签。
    :param dates: 此历史窗口的90个市场交易日。
    :return: float32标准化输入与逐样本时间戳矩阵。
    """
    raw=np.asarray(raw,dtype=np.float32);dates=pd.DatetimeIndex(dates)
    if raw.ndim!=3 or raw.shape[1:]!=(90,6) or len(dates)!=90 or not np.isfinite(raw).all():
        raise ValueError('历史输入几何或数值非法')
    if not dates.is_unique or not dates.is_monotonic_increasing:raise ValueError('历史日期异常')
    x=np.clip((raw-raw.mean(1,keepdims=True))/(raw.std(1,keepdims=True)+1e-5),-5,5).astype(np.float32)
    stamp=np.column_stack([dates.minute,dates.hour,dates.weekday,dates.day,dates.month]).astype(np.float32)
    return x,np.broadcast_to(stamp,(len(raw),90,5)).copy()
