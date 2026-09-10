"""样本规则、标签与归一化统计（计划 §3/§4 的纯函数部分）。

全部函数不触 GPU / 不加载模型；``decision_days`` / ``mean_label`` /
``select_codes`` / ``fit_norm_stats`` 为合成可测的冻结规则。

样本选择的**唯一权威路径**在 :mod:`sae_residual.cache`（PIT 成分 +
90 日窗口 + 标签完整性共同决定资格后才进入 rng17 抽样），本模块只提供
纯规则，避免出现第二条抽样路径令 RNG 消费序列分叉。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sae_residual import config as C


def decision_days(calendar: pd.DatetimeIndex, start: str,
                  label_end: str, stride: int = C.STRIDE,
                  predict_len: int = C.PREDICT_LEN) -> pd.DatetimeIndex:
    """从 ``start``（含）起每 ``stride`` 个交易日取一个决策日。

    规则（计划 §4）：定位 ``start`` 当日或其后的第一个交易日，按下标
    ``0, stride, 2*stride, ...`` 取决策日；再要求其后 ``predict_len`` 个
    交易日（10 日标签终点）≤ ``label_end``，否则截断。

    :param calendar: 完整交易日历（升序）。
    :param start: 起始日 ``YYYY-MM-DD``。
    :param label_end: 10 日标签终点上界。
    :param stride: 抽样间隔（默认 5）。
    :param predict_len: 标签天数 H（默认 10）。
    :returns: 决策日 ``DatetimeIndex``（升序）。
    """
    cal = pd.DatetimeIndex(calendar).sort_values()
    ge_start = cal[cal >= pd.Timestamp(start)]
    if len(ge_start) == 0:
        return pd.DatetimeIndex([])
    anchor_pos = cal.get_loc(ge_start[0])
    bound = pd.Timestamp(label_end)
    days = []
    for i in range(anchor_pos, len(cal), stride):
        end_pos = i + predict_len
        if end_pos >= len(cal):
            break
        if cal[end_pos] > bound:
            break
        days.append(cal[i])
    return pd.DatetimeIndex(days)


def label_dates(calendar: pd.DatetimeIndex, t: pd.Timestamp,
                predict_len: int = C.PREDICT_LEN) -> pd.DatetimeIndex:
    """决策日 ``t`` 之后的 ``predict_len`` 个**精确交易日**（不跳停牌凑数）。"""
    cal = pd.DatetimeIndex(calendar)
    pos = cal.get_loc(pd.Timestamp(t))
    return cal[pos + 1: pos + 1 + predict_len]


def mean_label(close_t: float, future_closes: np.ndarray) -> float:
    """``y_mean = mean(close[t+1..t+10]) / close_t − 1``（与 mean 信号同量纲）。

    :param close_t: 决策日 t 的后复权 close（窗口最后一行，同 teacher 分母）。
    :param future_closes: 10 个精确标签日的 close（长度必须 = H，有限）。
    :raises ValueError: 长度不符或含非有限值——调用方应把该样本判为无标签。
    """
    arr = np.asarray(future_closes, dtype=np.float64)
    if arr.shape != (C.PREDICT_LEN,):
        raise ValueError(f"future_closes 长度应为 {C.PREDICT_LEN}，实际 {arr.shape}")
    if not np.isfinite(arr).all() or not np.isfinite(close_t) or close_t <= 0:
        raise ValueError("标签价格缺失/非有限/现价非法 → 该样本无标签")
    return float(arr.mean() / close_t - 1.0)


def select_codes(eligible: list[str], rng: np.random.Generator,
                 per_day: int = C.PER_DAY) -> list[str]:
    """从排序后的合格股票中无放回选 ``per_day`` 只（数据 RNG seed17）。

    :param eligible: 当日合格代码（内部先排序，保证顺序稳定）。
    :param rng: 调用方持有的 ``np.random.Generator``（按日期顺序推进）。
    :returns: 选中的代码（rng 输出顺序）；不足 ``per_day`` 抛 ``ValueError``。
    """
    pool = sorted(eligible)
    if len(pool) < per_day:
        raise ValueError(f"合格股票 {len(pool)} < {per_day}，该日跳过")
    picked = rng.choice(pool, size=per_day, replace=False)
    return [str(c) for c in picked]


def make_day_rng(date: pd.Timestamp, seed: int = C.DATA_RNG_SEED
                 ) -> np.random.Generator:
    """按日期派生的稳定抽样 RNG（seed17 ⊕ YYYYMMDD）。

    派生算法固定并记录：``default_rng([17, yyyymmdd])``。任意日期的选择
    只依赖当日合格集合与该派生种子，与处理顺序/断点续跑无关——顺序推进
    的单一 RNG 在跳过已完成日期时不消费随机数，会令后续日期选择漂移，
    故弃用。
    """
    t = pd.Timestamp(date)
    return np.random.default_rng([seed, t.year * 10000 + t.month * 100 + t.day])


def fit_norm_stats(h_train: np.ndarray) -> dict:
    """只用**训练**隐状态估计 μh/σh（不看验证/测试，计划 §3）。

    :param h_train: 训练隐状态 ``[N, D]``（float32/float64 均可）。
    :returns: ``{"mu": [D], "sigma": [D]}``，sigma 已套 ``max(σ, 1e-6)`` 地板。
    """
    arr = np.asarray(h_train, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"h_train 应为非空 [N,D]，实际 {arr.shape}")
    mu = arr.mean(axis=0)
    sigma = np.maximum(arr.std(axis=0), C.STATS_FLOOR)
    return {"mu": mu, "sigma": sigma}


def residual_std(e_train: np.ndarray) -> float:
    """训练残差标准差 ``σe = max(std(e), 1e-6)``，**不减均值**（§3）。

    σe 触到数值地板（std < 1e-6）时说明训练残差退化，调用方必须停实验
    并记录，不得放大噪声继续训练（计划 §3）。
    """
    arr = np.asarray(e_train, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("e_train 为空")
    std = float(arr.std())
    if std < C.STATS_FLOOR:
        raise ValueError(
            f"训练残差退化：std(e)={std:.3e} < 地板 {C.STATS_FLOOR:.0e}，停止实验")
    return std


def normalize_hidden(h: np.ndarray, stats: dict) -> np.ndarray:
    """``x = (h − μh) / max(σh, 1e-6)``（μh/σh 只来自训练）。"""
    arr = np.asarray(h, dtype=np.float32)
    mu = np.asarray(stats["mu"], dtype=np.float32)
    sigma = np.asarray(stats["sigma"], dtype=np.float32)
    return ((arr - mu) / sigma).astype(np.float32)


def normalize_window(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐列 z-score + clip±5，与 ``KronosPredictor.predict_batch`` 逐字一致。

    :param x: 原始 OHLCVA 窗口 ``[L, 6]``。
    :returns: ``(x_norm, mean, std)``——mean/std 为逐列原始量纲（预测反变换用）。
    """
    arr = np.asarray(x, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    xn = np.clip((arr - mean) / (std + C.EPS_STD), -C.NORM_CLIP, C.NORM_CLIP)
    return xn.astype(np.float32), mean, std


def teacher_mean_signal(pred_close: np.ndarray, close_t: float) -> float:
    """teacher mean 信号：``mean(pred_close[t+1..t+10]) / close_t − 1``。

    与 ``baseline_suite.signal.compute_variants_from_preds`` 的 mean 分支
    逐字一致（``np.mean``，非 nanmean）。
    """
    return float(np.mean(pred_close) / close_t - 1.0)


__all__ = ["decision_days", "label_dates", "mean_label", "select_codes",
           "make_day_rng", "fit_norm_stats", "residual_std",
           "normalize_hidden", "normalize_window", "teacher_mean_signal"]
