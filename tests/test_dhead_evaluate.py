"""任务5（方案 §7）：评价器测试——保真/真实标签/门禁拒绝（纯离线合成）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dhead_distill.evaluate import (
    block_bootstrap_paired_diff,
    daily_rank_ic,
    evaluate_economic_gate,
    fidelity_gate,
    fidelity_metrics,
)


def _days(n_days: int = 10, n: int = 40, seed: int = 0):
    """合成逐日截面：pred [N,10]、replicas [R,N,10]、y_real [N,10]。"""
    rng = np.random.default_rng(seed)
    days = []
    for d in range(n_days):
        base = rng.normal(0, 0.02, (n, 1))
        y = base + rng.normal(0, 0.01, (n, 10))
        teacher = np.stack([
            y + rng.normal(0, 0.003, (n, 10)) for _ in range(3)
        ])  # [R,N,10]
        days.append(dict(date=f"2025-01-{d + 1:02d}", y=y, teacher=teacher))
    return days


def test_fidelity_zero_when_predictions_match_teacher() -> None:
    """学生预测 == replica1：E = 0；相同预测保真误差为 0。"""
    days = _days(5)
    scale = np.full(10, 0.02)
    out = fidelity_metrics(
        days=days,
        student=lambda day: day["teacher"][1],   # 学生完美复刻 replica1
        scale=scale,
    )
    assert out["E"] == pytest.approx(0.0, abs=1e-12)
    assert out["R"] > 0  # 教师自身波动为正
    assert out["ratio"] == pytest.approx(0.0, abs=1e-9)


def test_fidelity_constant_baseline_no_spurious_correlation() -> None:
    """独立常量基线：日 Spearman 无效（零方差）被显式计数，不产生虚假相关。"""
    days = _days(5)
    scale = np.full(10, 0.02)
    out = fidelity_metrics(
        days=days,
        student=lambda day: np.zeros_like(day["teacher"][1]),
        scale=scale,
    )
    assert out["n_days_invalid_spearman"] == out["n_days"]
    assert out["mean_spearman_valid"] is None  # 无有效日不得给均值
    assert out["E"] > 0


def test_fidelity_gate_rules() -> None:
    """门禁：E/R≤2 且有效日 Spearman 均值≥0.8 且有效率≥80% 才通过。"""
    days = _days(10, seed=1)
    scale = np.full(10, 0.02)
    # 完美学生：通过
    good = fidelity_metrics(days=days, student=lambda day: day["teacher"][1],
                            scale=scale)
    assert fidelity_gate(good, ratio_max=2.0, spearman_min=0.8,
                         valid_frac_min=0.8)["passed"]
    # 纯噪声学生：不通过
    rng = np.random.default_rng(7)
    bad = fidelity_metrics(
        days=days,
        student=lambda day: rng.normal(0, 0.02, day["teacher"][1].shape),
        scale=scale,
    )
    assert not fidelity_gate(bad, ratio_max=2.0, spearman_min=0.8,
                             valid_frac_min=0.8)["passed"]


def test_daily_rank_ic_counts_invalid_days() -> None:
    """无效日期（标签或预测零方差）显式计数，不混入均值。"""
    days = _days(6, seed=2)
    # 把最后 2 天预测改成常量 → 无效
    calls = {"i": -1}

    def student(day):
        calls["i"] += 1
        if calls["i"] >= 4:
            return np.zeros_like(day["teacher"][1])
        return day["y"]

    out = daily_rank_ic(days=days, student=student)
    assert out["n_days"] == 6
    assert out["n_days_valid"] == 4
    assert out["n_days_invalid"] == 2
    assert -1.0 <= out["mean_rank_ic"] <= 1.0


def test_block_bootstrap_paired_diff_ci() -> None:
    """区块 bootstrap（按日期整体抽样）：确定性 seed + 合理区间。"""
    rng = np.random.default_rng(0)
    n_days = 30
    diff_by_day = rng.normal(0.001, 0.005, n_days)  # 真实微正差
    ci1 = block_bootstrap_paired_diff(diff_by_day, block=10, n_boot=500,
                                      seed=20260905)
    ci2 = block_bootstrap_paired_diff(diff_by_day, block=10, n_boot=500,
                                      seed=20260905)
    assert ci1["lo"] == ci2["lo"] and ci1["hi"] == ci2["hi"]  # 确定性
    assert ci1["lo"] < ci1["hi"]
    assert ci1["lo"] < 0.001 < ci1["hi"]  # 覆盖真值


def test_economic_gate_refuses_without_executor() -> None:
    """未通过回测门禁/无经验证执行器：economic 拒绝输出可交易结论。"""
    verdict = evaluate_economic_gate(
        engine_verified=False, executor_available=False)
    assert not verdict["allowed"]
    assert "拒绝" in verdict["reason"] or "不可" in verdict["reason"]
    # 门禁通过 + 执行器可用才允许
    ok = evaluate_economic_gate(engine_verified=True, executor_available=True)
    assert ok["allowed"]
