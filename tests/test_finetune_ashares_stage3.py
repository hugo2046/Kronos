"""finetune_ashares 阶段 3 判据单元测试（计划 §5，20260815 计划）。

预注册判据 1~5 的**逻辑**用合成数字验证（与真实结果无关，跑前可测）：

- 判据 1（机制）：best epoch ≥ 2 或 val 序列存在 epoch≥2 改善（非单调恶化）；
- 判据 2（改善）：G1_mean AER(等权) ≥ F1_mean(-4.24%) + 3pp = -1.24%（含等于）；
- 判据 3（存活）：G1_mean AER(等权) > 0 且 AER(指数) > 0（严格大于）；
- 判据 4（G0 稳定性）：G0_mean AER(等权) ≥ F0_mean@2025H2 + 5pp（含等于）；
- 判据 5（路线关闭）：判据 1、2 均失败才触发。
"""
from __future__ import annotations

import pandas as pd

from finetune_suite.run_g1_backtest import (
    F1_MEAN_AER_EW_FROZEN,
    judge_criterion1_mechanism,
    judge_criteria,
)


def _epoch_table(val_losses: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"epoch": range(1, len(val_losses) + 1), "val_loss": val_losses}
    )


def test_c1_best_epoch_ge2():
    ok, note = judge_criterion1_mechanism(_epoch_table([3.4, 3.3, 3.35]), best_epoch=2)
    assert ok, note


def test_c1_val_improvement_after_epoch2():
    # best_epoch=1 但 val 非单调恶化（epoch3 比 epoch2 改善）→ 判据 1 通过
    ok, note = judge_criterion1_mechanism(_epoch_table([3.36, 3.40, 3.38, 3.41]), best_epoch=1)
    assert ok, note


def test_c1_monotonic_worsening_best1_fails():
    ok, _ = judge_criterion1_mechanism(_epoch_table([3.36, 3.40, 3.44]), best_epoch=1)
    assert not ok


def test_c1_all_tie_is_monotonic_nonimprovement():
    # 全持平 = 无 epoch≥2 改善（严格小于才算改善）
    ok, _ = judge_criterion1_mechanism(_epoch_table([3.36, 3.36, 3.36]), best_epoch=1)
    assert not ok


def test_c2_boundary_inclusive():
    v = judge_criteria(
        g1_mean_aer_ew=-0.0124, g1_mean_aer_idx=-0.05,
        g0_mean_aer_ew=0.0, g0_f0_mean_aer_ew=0.0,
        c1_passed=False,
    )
    assert v["criterion_2_improvement"]["passed"]  # 恰好 -1.24% → ≥ 成立


def test_c3_strict_positive():
    v = judge_criteria(0.0, 0.01, 0.0, 0.0, False)
    assert not v["criterion_3_survival"]["passed"]  # AER(等权)=0 不算 > 0


def test_c4_boundary_inclusive():
    v = judge_criteria(0.0, 0.0, 0.05, 0.0, False)
    assert v["criterion_4_g0_stability"]["passed"]  # 恰好 +5pp → ≥ 成立


def test_c5_requires_both_c1_c2_fail():
    both_fail = judge_criteria(-0.10, -0.10, -0.10, -0.10, False)
    assert both_fail["criterion_5_route_closed"]["triggered"]
    c1_only = judge_criteria(-0.10, -0.10, -0.10, -0.10, True)
    assert not c1_only["criterion_5_route_closed"]["triggered"]
    c2_only = judge_criteria(-0.01, -0.10, -0.10, -0.10, False)
    assert not c2_only["criterion_5_route_closed"]["triggered"]


def test_frozen_baseline_constant():
    assert F1_MEAN_AER_EW_FROZEN == -0.0424  # 计划 §5 判据 2 冻结基线
