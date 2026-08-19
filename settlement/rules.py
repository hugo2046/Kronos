"""组合规则确定性推导（组合规则预注册_20260816.md §2 + C3混合规则预注册_20260818.md §1）。

全部为 G3 登记表已有列的确定性函数，结算时完全可复算；本模块零绩效计算：

- C1 = mean(s100_mean, s101_mean, s102_mean)，缺任一种子值剔除该股当日；
- C2 = MA200 门控切换：gate=True → M，False → C1（迟补日沿用盖章 gate 值，
  结算时另做剔除敏感性——见 executor 的 C2_excl_late）；
- C3 = 0.5·zscore(C1) + 0.5·zscore(M)，当日有效池（两侧均在册）上
  均值 0 方差 1（总体方差 ddof=0），权重 0.5/0.5 永久冻结；
- R1 原版 = M/F0 切换：gate=True → M，False → F0_mean
  （improve_suite/regime_switch.py R1 同构，"mean" 腿在结算语境为 F0）。
"""
from __future__ import annotations

import pandas as pd

C1_SEED_COLS = ("s100_mean", "s101_mean", "s102_mean")
REQUIRED_COLS = C1_SEED_COLS + ("M",)


def _check_day_wide(day_wide: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in day_wide.columns]
    if missing:
        raise KeyError(f"登记表缺输入列: {missing}")
    if not day_wide.index.is_unique:
        raise ValueError("code 索引有重复——同日推导无法无歧义进行")


def derive_c1_day(day_wide: pd.DataFrame) -> pd.Series:
    """C1：三种子 mean 集成（s103/104 诊断种子不进；缺任一种子剔除该股当日）。"""
    _check_day_wide(day_wide)
    return day_wide[list(C1_SEED_COLS)].mean(axis=1, skipna=False).dropna()


def derive_c2_day(day_wide: pd.DataFrame, *, gate: bool) -> pd.Series:
    """C2：MA200 门控切换——True→M（趋势市动量），False→C1（集成）。"""
    if gate:
        return day_wide["M"].dropna()
    return derive_c1_day(day_wide)


def derive_c3_day(day_wide: pd.DataFrame) -> pd.DataFrame:
    """C3：0.5·z_C1 + 0.5·z_M（权重冻结；有效池=两侧均在册；ddof=0）。"""
    _check_day_wide(day_wide)
    c1 = day_wide[list(C1_SEED_COLS)].mean(axis=1, skipna=False)
    m = day_wide["M"].astype("float64")
    valid = c1.notna() & m.notna()
    if int(valid.sum()) < 2:
        raise ValueError("当日有效股票池 < 2，截面 zscore 无法定义")

    def _z(x: pd.Series) -> pd.Series:
        sd = x.std(ddof=0)  # 总体方差：均值 0 方差 1 精确成立（无歧义口径）
        if sd == 0:
            raise ValueError("当日截面标准差为 0，zscore 无法无歧义定义")
        return (x - x.mean()) / sd

    out = pd.DataFrame(
        {"C1": c1[valid], "z_C1": _z(c1[valid]), "z_M": _z(m[valid])}
    )
    out["C3"] = 0.5 * out["z_C1"] + 0.5 * out["z_M"]
    return out


def r1_assemble(m: pd.Series, f0_mean: pd.Series, *, gate: bool) -> pd.Series:
    """R1 原版组装：gate=True→M 动量，False→F0_mean（第 2 轮状态切换前瞻复检）。"""
    return m if gate else f0_mean
