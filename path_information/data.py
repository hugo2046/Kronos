"""PATH1 纯规则：日期边界、选股、标签、路径变换、秩相关（计划 §2/§3/§5）。

全部函数不触 GPU / 不加载模型 / 不读数据库，为合成可测的冻结规则。
样本选择的唯一权威实现在 :mod:`path_information.paths`（PIT 成分 +
fetch.end≤t 的 90 日窗口 + 基线有限信号交集后按 SHA256 排序取 64），
本模块只提供纯函数，避免出现第二条抽样路径。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from path_information import config as C


def segment_days(calendar: pd.DatetimeIndex, start: str, label_end: str,
                 predict_len: int = C.PREDICT_LEN) -> pd.DatetimeIndex:
    """段内**全部**交易日中，``calendar[t+10] ≤ 段标签末日`` 的决策日。

    规则（计划 §2）：候选区间 ``[start, label_end]`` 内逐交易日遍历，排除
    10 日标签窗跨段边界的决策日；不做 stride 抽样。精确日期与数量由
    交易日日历生成（不能用股票历史行号替代日历）。

    :param calendar: 完整交易日历（升序）。
    :param start: 段起始日 ``YYYY-MM-DD``（含）。
    :param label_end: 段标签最晚观测日 ``YYYY-MM-DD``。
    :param predict_len: 标签天数 H（默认 10）。
    :returns: 决策日 ``DatetimeIndex``（升序）。
    """
    cal = pd.DatetimeIndex(calendar).sort_values()
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(label_end)
    days = []
    for i, t in enumerate(cal):
        if t < lo:
            continue
        if t > hi:
            break
        j = i + predict_len
        if j >= len(cal) or cal[j] > hi:
            break                      # 日历升序：后续决策日必然更晚，截断
        days.append(t)
    return pd.DatetimeIndex(days)


def label_dates(calendar: pd.DatetimeIndex, t: pd.Timestamp,
                predict_len: int = C.PREDICT_LEN) -> pd.DatetimeIndex:
    """决策日 ``t`` 之后的 ``predict_len`` 个**精确交易日**。"""
    cal = pd.DatetimeIndex(calendar)
    pos = cal.get_loc(pd.Timestamp(t))
    return cal[pos + 1: pos + 1 + predict_len]


def mean_label(close_t: float, future_closes: np.ndarray) -> float:
    """``y_mean = mean(close[t+1..t+10]) / close_t − 1``（与 mean 信号同量纲）。

    :raises ValueError: 长度不符 / 含非有限值 / 现价非法——调用方判该格无标签。
    """
    arr = np.asarray(future_closes, dtype=np.float64)
    if arr.shape != (C.PREDICT_LEN,):
        raise ValueError(f"future_closes 长度应为 {C.PREDICT_LEN}，实际 {arr.shape}")
    if not np.isfinite(arr).all() or not np.isfinite(close_t) or close_t <= 0:
        raise ValueError("标签价格缺失/非有限/现价非法 → 该格无标签")
    return float(arr.mean() / close_t - 1.0)


def valid_label_mask(y: np.ndarray) -> np.ndarray:
    """有效标签格掩码（有限值）；缺标签只屏蔽 loss/评价格，不重选股票。"""
    arr = np.asarray(y, dtype=np.float64)
    return np.isfinite(arr)


def select_codes(eligible: list[str], date: str,
                 per_day: int = C.PER_DAY) -> list[str]:
    """按 ``SHA256('PATH1|17|YYYY-MM-DD|code')`` 升序取前 ``per_day`` 只。

    确定性选择：与处理顺序无关、断点续跑不漂移；不以未来收益/停牌/标签
    完整性挑选。不足 ``per_day`` 抛错（计划 §3：停止排查，不补替）。
    """
    pool = sorted(eligible)
    if len(pool) < per_day:
        raise ValueError(f"合格股票 {len(pool)} < {per_day}，停止排查")
    ds = pd.Timestamp(date).strftime("%Y-%m-%d")

    def digest(code: str) -> str:
        return hashlib.sha256(
            C.SELECT_KEY.format(date=ds, code=code).encode()).hexdigest()

    keyed = sorted(pool, key=lambda c: (digest(c), c))
    return keyed[:per_day]


def to_relative_paths(close_paths: np.ndarray, close_t: np.ndarray) -> np.ndarray:
    """真实价格样本 → 相对当前 close 的未来路径收益 ``p[n,h] = pred/close_t − 1``。

    :param close_paths: ``[B, N, H]`` 预测 close（真实价格维）。
    :param close_t: ``[B]`` 决策日后复权 close（窗口最后一行）。
    """
    p = np.asarray(close_paths, dtype=np.float64)
    ct = np.asarray(close_t, dtype=np.float64)
    return p / ct[..., None, None] - 1.0


def path_shape(p: np.ndarray) -> np.ndarray:
    """形状增量：``shape = p − p.mean(axis=(-2,-1), keepdims=True)``。

    对每股减去其全部样本×期限的均值：股票特定常数平移不变（计划 §7）。
    """
    arr = np.asarray(p, dtype=np.float64)
    return arr - arr.mean(axis=(-2, -1), keepdims=True)


def _avg_rank(a: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均；0-based）。"""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0        # 1-based 平均秩
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_avg_rank(a: np.ndarray, b: np.ndarray) -> float:
    """平均秩 Spearman 相关；任一侧常数（秩无定义）记 NaN。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx, ry = _avg_rank(x), _avg_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


__all__ = ["segment_days", "label_dates", "mean_label", "valid_label_mask",
           "select_codes", "to_relative_paths", "path_shape",
           "spearman_avg_rank"]
