"""c4_temporal：C4 时间维因子（三值化 + 半衰期加权）变换层（C4 计划 §1，20260820）。

C4 = 既有 G1 逐日 mean 信号 parquet 的**确定函数**（零训练零推理）：

1. 三值化 ``e_i(t) = +1 若 G1_mean > +2%；−1 若 < −2%；否则 0``（恰 ±2% 取 0）；
2. 半衰期加权 ``C4_i(t) = Σ_{k=0..29} λ^k · e_i(t−k)``，λ = 0.5^(1/10)
   （半衰期 10 交易日、窗口 30 交易日，威科夫研报惯例值）；
3. 评估自合并窗信号首日后第 30 个交易日起（预热 30 日如实烧掉）。

冻结超参（§5 纪律：跑后不得扫描）：阈值 ±2% / 半衰期 10 / 窗口 30。
冻结 NaN 语义（跑前执行决策，见 tests/test_c4_temporal.py 模块 docstring）：
NaN 不投票、加权和中跳过（贡献 0、权重照丢）、30 项全 NaN → C4 = NaN。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# —— 冻结超参（C4 计划 §1；跑后禁扫描）——
TRINARIZE_THRESHOLD = 0.02  # ±2%
HALF_LIFE_DAYS = 10  # 半衰期 10 交易日
C4_WINDOW = 30  # 加权窗口 30 交易日
WARMUP_DAYS = 30  # 预热烧 30 交易日（评估自首日后第 30 个交易日起）

C4_LAMBDA = 0.5 ** (1.0 / HALF_LIFE_DAYS)  # λ = 0.5^(1/10) ≈ 0.933033

REPO_ROOT = Path(__file__).resolve().parent.parent
R4 = REPO_ROOT / "finetune_suite" / "data"
G5_DATA = REPO_ROOT / "g5_head" / "data"

# G1 族三种子两窗 mean 信号 parquet（G7 封盘同款映射；G1 信号只读）
G1_MEAN_PARQUETS = {
    100: {
        "2025h2": G5_DATA / "daily_signals_2025h2_G1_mean.parquet",
        "backtest": R4 / "g1" / "daily_signals_backtest_G1_mean.parquet",
    },
    101: {
        "2025h2": R4 / "g2" / "s101" / "daily_signals_2025h2_G2S101_mean.parquet",
        "backtest": R4 / "g2" / "s101" / "daily_signals_backtest_G2S101_mean.parquet",
    },
    102: {
        "2025h2": R4 / "g2" / "s102" / "daily_signals_2025h2_G2S102_mean.parquet",
        "backtest": R4 / "g2" / "s102" / "daily_signals_backtest_G2S102_mean.parquet",
    },
}

MERGED_WINDOW = ("2025-07-01", "2026-07-24")  # 两子窗合并的连续区间


def trinarize(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """三值化：> +2% → +1；< −2% → −1；否则（含恰 ±2%）→ 0；NaN → NaN。

    :param x: G1 mean 信号（宽表 index=date/columns=code 或 Series）。
    :return: 同形状三值宽表/序列（float，+1.0/0.0/−1.0/NaN）。
    """
    if isinstance(x, pd.Series):
        return pd.Series(
            np.where(x.isna(), np.nan, np.where(x > TRINARIZE_THRESHOLD, 1.0,
                                                np.where(x < -TRINARIZE_THRESHOLD, -1.0, 0.0))),
            index=x.index, name=x.name,
        )
    return pd.DataFrame(
        np.where(x.isna(), np.nan, np.where(x > TRINARIZE_THRESHOLD, 1.0,
                                            np.where(x < -TRINARIZE_THRESHOLD, -1.0, 0.0))),
        index=x.index, columns=x.columns,
    )


def c4_weights() -> np.ndarray:
    """λ 幂级数权重 ``[λ^0, λ^1, …, λ^29]``（今日权重 1，滞后 10 日恰 0.5）。"""
    return C4_LAMBDA ** np.arange(C4_WINDOW)


def c4_ew_sum(e: pd.DataFrame) -> pd.DataFrame:
    """半衰期加权滚动和 ``C4_i(t) = Σ_{k=0..29} λ^k · e_i(t−k)``（NaN 跳过）。

    向量化实现：``e.fillna(0)`` 逐滞后移位累加 + 有效项计数；
    30 项全 NaN 的 (t, i) → NaN（整窗无票 = 不可买）。

    :param e: 三值化后的宽表（+1/0/−1/NaN）。
    :return: 同形状加权宽表（float；理论值域 ±(1−λ^30)/(1−λ) ≈ ±13.066）。
    """
    w = c4_weights()
    e0 = e.fillna(0.0)
    valid = e.notna().astype(float)
    val = pd.DataFrame(0.0, index=e.index, columns=e.columns)
    cnt = pd.DataFrame(0.0, index=e.index, columns=e.columns)
    for k in range(C4_WINDOW):
        # shift 头部（序列前 k 日无历史）与 NaN 同语义 = 无票跳过（贡献 0、权重照丢）
        val = val + w[k] * e0.shift(k).fillna(0.0)
        cnt = cnt + valid.shift(k).fillna(0.0)
    out = val.where(cnt > 0)
    return out.astype(float)


def build_c4(g1_mean_merged: pd.DataFrame) -> pd.DataFrame:
    """C4 全变换：G1 mean 合并宽表 → 三值化 → 半衰期加权（确定性纯函数）。

    :param g1_mean_merged: 合并 260 日连续宽表（:func:`load_g1_mean_merged`）。
    :return: C4 信号宽表（同 index/columns；预热期行保留、由评估层剔除）。
    """
    return c4_ew_sum(trinarize(g1_mean_merged))


def load_g1_mean_merged(seed: int) -> pd.DataFrame:
    """载入某 G1 族种子的两窗 mean 信号并合并为连续 260 日宽表。

    两窗列集外连接（跨窗缺信号的格 = NaN，按冻结 NaN 语义跳过）。

    :param seed: 100/101/102。
    :return: 260 行（2025-07-01~2026-07-24）宽表，列 = 两窗列并集。
    """
    paths = G1_MEAN_PARQUETS[seed]
    h2 = pd.read_parquet(paths["2025h2"])
    bt = pd.read_parquet(paths["backtest"])
    assert len(h2) == 126 and len(bt) == 134, (
        f"s{seed} 窗长漂移：2025h2={len(h2)} backtest={len(bt)}（期望 126+134）"
    )
    merged = pd.concat([h2, bt])  # index 连续（2025-12-31 → 2026-01-05 无缺口）
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    assert len(merged) == 260, f"s{seed} 合并后 {len(merged)} 日 ≠ 260"
    assert str(merged.index[0].date()) == MERGED_WINDOW[0]
    assert str(merged.index[-1].date()) == MERGED_WINDOW[1]
    return merged


def eval_rebalance_dates(all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """评估调仓日 = 全部信号日剔除预热前 30 个交易日（首日后第 30 个交易日起）。

    C4 需 30 日信号历史；预热期（首月）如实烧掉不进评估（~230 个交易日）。

    :param all_dates: 合并窗全部信号日（260 日）。
    :return: 评估调仓日（230 日）。
    """
    ev = all_dates[WARMUP_DAYS:]
    assert len(all_dates) - len(ev) == WARMUP_DAYS, "预热期必须整段 30 日烧掉"
    return pd.DatetimeIndex(ev)
