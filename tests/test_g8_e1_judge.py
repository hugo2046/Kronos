"""G8+E1 统一判读逻辑合成数字单元测试（计划 §1/§2 判据冻结，20260820）。"""
from __future__ import annotations

from finetune_suite.run_g8_e1_judge import judge_e1, judge_g8


def _e1_bt(drops, net_delta):
    """构造 judge_e1 的 backtest 块（drops/net_delta 为小数）。"""
    s = ("s100", "s101", "s102")
    ds = sorted(drops)
    return {
        "table": {seed: {"delta": {"turnover_drop_pct": d}} for seed, d in zip(s, drops)},
        "medians": {
            "turnover_drop_pct_median": ds[1],
            "turnover_drop_pct_min": ds[0],
            "net_aer_ew_median_delta": net_delta,
        },
    }


def test_g8_all_positive_registers():
    v = judge_g8(0.15, 0.10, 0.05, {"s100": 0.02, "s101": 0.05, "s102": -0.01})
    assert v["G8_1_survival_register"]["passed"]
    assert not v["G8_2_significance"]["passed"]  # +5pp < +26pp
    assert not v["G8_4_negative"]["triggered"]
    # 带内配对差一律"不可判"
    assert all("不可判" in n for n in v["G8_3_paired_diffs"]["notes"].values())


def test_g8_negative_closes_arm():
    v = judge_g8(-0.05, 0.10, -0.10, {"s100": -0.1, "s101": 0.0, "s102": -0.2})
    assert not v["G8_1_survival_register"]["passed"]
    assert v["G8_4_negative"]["triggered"]
    assert "不做终点日期搜索" in v["G8_4_negative"]["note"]


def test_g8_significance_threshold():
    assert not judge_g8(0.5, 0.4, 0.26, {})["G8_2_significance"]["passed"]  # 恰在线下
    assert judge_g8(0.5, 0.4, 0.2601, {})["G8_2_significance"]["passed"]


def test_e1_mechanical_pass_registers():
    v = judge_e1(_e1_bt([0.35, 0.40, 0.33], 0.05))
    e = v["E1_1_mechanical"]
    assert e["passed"] and e["median_drop"] == 0.35
    assert v["E1_2_harmless"]["passed"]
    assert v["E1_3_register"]["triggered"]
    assert not v["E1_4_negative"]["triggered"]


def test_e1_turnover_floor_triggers_close():
    v = judge_e1(_e1_bt([-0.02, -0.05, -0.03], 0.04))
    assert not v["E1_1_mechanical"]["passed"]
    assert v["E1_4_negative"]["triggered"]
    assert "不做阈值搜索" in v["E1_4_negative"]["note"]
    # E1-2 即便通过也不注册（E1-3 需两者同时）
    assert v["E1_2_harmless"]["passed"]
    assert not v["E1_3_register"]["triggered"]


def test_e1_harmless_boundary():
    v = judge_e1(_e1_bt([0.35, 0.36, 0.37], -0.26))
    assert not v["E1_2_harmless"]["passed"]  # 恰在 −26pp 线上不通过（> 才通过）
    assert not v["E1_3_register"]["triggered"]
