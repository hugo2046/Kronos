"""评价器：保真/真实标签/速度与阶段门禁（方案 §8）。

- 保真（§8.2）：teacher-repeat 误差 R = MSE_norm(replica1, replica2)、
  学生独立误差 E = MSE_norm(student, replica1)，同验证清单按日等权；
  门禁 E/(R+1e-8) ≤ 2.0 且日 Spearman 均值 ≥ 0.8 且有效日占比 ≥ 80%；
- 真实标签（§8.4）：日均 RankIC、各 horizon 误差、期限均值误差；无效日
  （标签/预测零方差）显式计数；
- 区块 bootstrap：按**日期整体**抽样（块长 10 交易日、2000 次、
  seed 20260905），配对差 95% 区间；
- 经济门禁：未核验引擎/无经验证执行器时拒绝输出可交易结论。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import stats as sps


def _nmse(pred: np.ndarray, target: np.ndarray, scale: np.ndarray) -> float:
    """normalized_mse 的 numpy 版（评价路径，无 torch 依赖）。"""
    return float((((pred - target) / scale) ** 2).mean())


def fidelity_metrics(
    *,
    days: list[dict],
    student: Callable[[dict], np.ndarray],
    scale: np.ndarray,
) -> dict:
    """逐日保真指标：E（学生 vs replica1）、R（replica1 vs replica2）、Spearman。

    :param days: 逐日截面列表，每项含 ``teacher`` [R,N,10]（R≥2）。
    :param student: day → pred [N,10] 的学生预测函数。
    :param scale: [10] 期限尺度。
    :returns: 含 E/R/ratio、逐日 Spearman 统计（零方差日计为无效）。
    """
    e_list, r_list, spear_list = [], [], []
    n_invalid = 0
    for day in days:
        rep = day["teacher"]  # [R,N,10]
        pred = np.asarray(student(day), dtype=np.float64)
        e_list.append(_nmse(pred, rep[1], scale))
        r_list.append(_nmse(rep[1], rep[2], scale))
        p_sig = pred.mean(axis=-1)
        t_sig = rep[1].mean(axis=-1)
        if np.std(p_sig) < 1e-12 or np.std(t_sig) < 1e-12:
            n_invalid += 1
            continue
        spear_list.append(float(sps.spearmanr(p_sig, t_sig).statistic))
    n = max(len(days), 1)
    e = float(np.mean(e_list)) if e_list else 0.0
    r = float(np.mean(r_list)) if r_list else 0.0
    return {
        "E": e, "R": r, "ratio": e / (r + 1e-8),
        "n_days": len(days),
        "n_days_invalid_spearman": n_invalid,
        "n_days_valid_spearman": len(spear_list),
        "mean_spearman_valid": (
            float(np.mean(spear_list)) if spear_list else None
        ),
        "spearman_day_weighted": e / n,  # 占位对齐字段（按日等权由 e_list 均值保证）
    }


def fidelity_gate(
    metrics: dict, *, ratio_max: float, spearman_min: float,
    valid_frac_min: float,
) -> dict:
    """保真门禁判定（§8.2；工程容差，非统计显著性）。"""
    mean_sp = metrics["mean_spearman_valid"]
    valid_frac = metrics["n_days_valid_spearman"] / max(metrics["n_days"], 1)
    ratio_ok = metrics["ratio"] <= ratio_max
    spearman_ok = mean_sp is not None and mean_sp >= spearman_min
    frac_ok = valid_frac >= valid_frac_min
    return {
        "passed": bool(ratio_ok and spearman_ok and frac_ok),
        "ratio_ok": bool(ratio_ok),
        "spearman_ok": bool(spearman_ok),
        "valid_frac_ok": bool(frac_ok),
        "ratio": metrics["ratio"], "mean_spearman": mean_sp,
        "valid_frac": valid_frac,
        "thresholds": {"ratio_max": ratio_max,
                       "spearman_min": spearman_min,
                       "valid_frac_min": valid_frac_min},
    }


def daily_rank_ic(
    *, days: list[dict], student: Callable[[dict], np.ndarray],
) -> dict:
    """真实标签日均 RankIC（期限均值信号 vs 标签期限均值）+ 无效日计数。"""
    ics = []
    n_invalid = 0
    per_horizon_err: list[np.ndarray] = []
    mean_err: list[np.ndarray] = []
    for day in days:
        pred = np.asarray(student(day), dtype=np.float64)
        y = day["y"]
        p_sig, y_sig = pred.mean(axis=-1), y.mean(axis=-1)
        if np.std(p_sig) < 1e-12 or np.std(y_sig) < 1e-12:
            n_invalid += 1
        else:
            ics.append(float(sps.spearmanr(p_sig, y_sig).statistic))
        per_horizon_err.append(np.abs(pred - y).mean(axis=0))  # [10]
        mean_err.append(np.abs(p_sig - y_sig).mean())
    return {
        "n_days": len(days), "n_days_valid": len(ics),
        "n_days_invalid": n_invalid,
        "mean_rank_ic": float(np.mean(ics)) if ics else None,
        "per_horizon_mae": (
            np.mean(per_horizon_err, axis=0).tolist()
            if per_horizon_err else None
        ),
        "horizon_mean_mae": float(np.mean(mean_err)) if mean_err else None,
    }


def block_bootstrap_paired_diff(
    diff_by_day: np.ndarray, *, block: int = 10, n_boot: int = 2000,
    seed: int = 20260905,
) -> dict:
    """按日期整体的区块 bootstrap：配对差均值 95% 区间（块长假设已知）。"""
    x = np.asarray(diff_by_day, dtype=np.float64)
    n = len(x)
    if n < block:
        raise ValueError(f"日数 {n} < 块长 {block}，无法区块 bootstrap")
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    start_max = n - block
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.sort(rng.integers(0, start_max + 1, n_blocks))
        segs = [x[s: s + block] for s in idx]
        sample = np.concatenate(segs)[:n]
        boots[b] = sample.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "lo": float(lo), "hi": float(hi),
        "point": float(x.mean()), "n_days": n, "block": block,
        "n_boot": n_boot, "seed": seed,
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm 校正（§8.4：两项配对检验同时做时）。"""
    order = np.argsort(p_values)
    m = len(p_values)
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p_values[i]
        prev = max(prev, val)
        adj[i] = min(1.0, prev)
    return adj


def evaluate_economic_gate(*, engine_verified: bool,
                           executor_available: bool) -> dict:
    """经济结论门禁（§8.4）：无核验引擎/执行器 → 拒绝输出可交易结论。"""
    allowed = engine_verified and executor_available
    return {
        "allowed": bool(allowed),
        "engine_verified": engine_verified,
        "executor_available": executor_available,
        "reason": (
            "引擎与执行器均已核验，允许输出经济回放指标"
            if allowed else
            "拒绝输出可交易/胜出结论：回测引擎争议未独立复核或缺少经验验证执行器"
            "（方案 §8.4——不得把'已有引擎'自动视为门禁通过）"
        ),
    }


def d2_unlock_condition(
    d0_history: list[dict], d1_history: list[dict],
    d0_fidelity: dict, d1_fidelity: dict,
) -> dict:
    """D2 解锁三条件（§8.3，v1 修复 #2：**统一 val_task 口径**比较）。

    v1 缺陷：D0 按 val_d、D1 按 val_task 各取最优再直接比较——跨口径。
    正确语义：D1 验证 ``S+0.05I`` 优于 D0 的验证 ``S+0.05I``（两臂 history
    均记录 val_task，无需换算）。

    :param d0_history/d1_history: 两臂逐 epoch 训练历史（含 val_task 键）。
    :param d0_fidelity/d1_fidelity: fidelity_metrics 输出（含 ratio/gate）。
    """
    best_d0 = min(h["val_task"] for h in d0_history)
    best_d1 = min(h["val_task"] for h in d1_history)
    conds = {
        "d0_fidelity_passed": bool(d0_fidelity["gate"]["passed"]),
        "d1_beats_d0_val_task": bool(best_d1 < best_d0),
        "d1_ratio_le_2": bool(d1_fidelity["ratio"] <= 2.0),
    }
    return {
        "conditions": conds,
        "unlocked": all(conds.values()),
        "best_d0_val_task": best_d0,
        "best_d1_val_task": best_d1,
    }


__all__ = [
    "fidelity_metrics", "fidelity_gate", "daily_rank_ic",
    "block_bootstrap_paired_diff", "holm_correction", "evaluate_economic_gate",
    "d2_unlock_condition",
]
