"""TAS1 协议测试（计划 §8「必须覆盖的评价与工程测试」）。

覆盖：IC 列置换不变、同信号零配对差、缺失覆盖门禁、HAC 日历空档、
Holm 与判决分支、checkpoint 选择只用验证、resume 校验 manifest、
forward 封存边界。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tas_state.config import TASConfig
from tas_state.evaluate import (
    daily_rank_ic,
    hac_tvalue,
    holm_adjust,
    contiguous_segments,
    paired_difference,
    stage_a_decision,
)


def _wide(rng, dates, codes, seed_offset=0.0):
    r = np.random.default_rng(7)
    data = r.normal(size=(len(dates), len(codes))) + seed_offset
    return pd.DataFrame(data, index=dates, columns=codes)


@pytest.fixture()
def frames():
    dates = pd.bdate_range("2025-07-01", periods=40)
    codes = [f"C{i:03d}" for i in range(60)]
    r = np.random.default_rng(1)
    sig = pd.DataFrame(r.normal(size=(40, 60)), index=dates, columns=codes)
    fwd = pd.DataFrame(r.normal(size=(40, 60)), index=dates, columns=codes)
    return sig, fwd, dates, codes


# ============================================================
# 1. IC 对列置换不变（计划 §8）
# ============================================================
def test_ic_invariant_to_column_permutation(frames):
    sig, fwd, dates, codes = frames
    ic0 = daily_rank_ic(sig, fwd, dates)
    perm = list(reversed(codes))
    ic1 = daily_rank_ic(sig[perm], fwd[perm], dates)
    ic2 = daily_rank_ic(sig, fwd[list(np.random.default_rng(0).permutation(codes))], dates)
    pd.testing.assert_series_equal(ic0, ic1)
    pd.testing.assert_series_equal(ic0, ic2)


# ============================================================
# 2. 同一信号两份配对差严格为 0（不能数值退化显著）
# ============================================================
def test_identical_signals_have_zero_paired_difference(frames):
    sig, fwd, dates, _ = frames
    ic_a = daily_rank_ic(sig, fwd, dates)
    ic_b = daily_rank_ic(sig.copy(), fwd.copy(), dates)
    d = paired_difference(ic_a, ic_b)
    assert (d == 0).all()
    cal = pd.bdate_range(dates[0] - pd.Timedelta(days=10), dates[-1] + pd.Timedelta(days=10))
    segs = contiguous_segments(d.index, cal)
    t, p = hac_tvalue(d.values, lag=9, segments=segs)
    assert np.isnan(t) or abs(t) < 1e-8  # 方差为 0 → NaN 或 0，不得显著


# ============================================================
# 3. 缺失预测触发覆盖门禁（不允许缩池续评）
# ============================================================
def test_missing_prediction_fails_coverage_gate(frames):
    sig, fwd, dates, codes = frames
    # 删除一只合法股票的预测（设 NaN）→ 合并后 < 阈值时报错，不缩池
    sig2 = sig.copy()
    sig2.iloc[0, 31:] = np.nan  # 仅保留 31 只（>30，作为对照）
    ic = daily_rank_ic(sig2, fwd, dates[:1])  # 不应报错
    assert len(ic) == 1
    sig3 = sig.copy()
    sig3.iloc[0, 30:] = np.nan  # 恰好 30 只 → 边界允许
    daily_rank_ic(sig3, fwd, dates[:1])
    sig4 = sig.copy()
    sig4.iloc[0, 29:] = np.nan  # 29 只 → 覆盖门禁失败
    with pytest.raises(ValueError, match="覆盖门禁"):
        daily_rank_ic(sig4, fwd, dates[:1])
    sig5 = sig.copy()
    sig5.iloc[0, 5] = np.nan  # 人为删除一只（60→59 只仍 >30，但前向缺同股票）
    # 缺失被 notna 掩码剔除：59 只仍在阈值上，此处验证列对齐不崩溃
    ic5 = daily_rank_ic(sig5, fwd, dates[:1])
    assert not np.isnan(ic5.iloc[0])


# ============================================================
# 4. HAC 尊重日历空档（缺失前后不得视为连续 lag-1）
# ============================================================
def test_hac_respects_calendar_gaps():
    # 构造 AR(1) 序列；在中间挖掉 1 个交易日 → 分段
    r = np.random.default_rng(3)
    n = 100
    x = np.zeros(n)
    z = r.normal(size=n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + z[i]
    cal = pd.bdate_range("2025-01-01", periods=n)
    # 完整：一段
    segs_full = contiguous_segments(cal, cal)
    assert len(segs_full) == 1
    t_full, _ = hac_tvalue(x, 9, segs_full)
    # 挖空第 50 个交易日（样本 49 与 50 在日历上不再相邻）
    idx_gap = cal.delete(50)
    x_gap = np.delete(x, 50)
    segs_gap = contiguous_segments(idx_gap, cal)
    assert len(segs_gap) == 2
    t_gap, _ = hac_tvalue(x_gap, 9, segs_gap)
    # 两段协方差与整段不同 → t 值不同（空档被尊重，不是简单串接）
    assert not np.isclose(t_full, t_gap)
    # 对照：若把空档样本硬串接（错误做法），应等于整段删一样本的结果
    t_naive, _ = hac_tvalue(x_gap, 9, [slice(0, n - 1)])
    assert not np.isclose(t_gap, t_naive)


# ============================================================
# 5. Holm 玩具例与判决分支（计划 §8）
# ============================================================
def test_holm_toy_example():
    adj = holm_adjust([0.001, 0.04, 0.2])
    assert np.allclose(adj, [0.003, 0.08, 0.2])


def _ic_series(dates, mean):
    r = np.random.default_rng(5)
    return pd.Series(mean + 0.05 * r.normal(size=len(dates)), index=dates)


def test_stage_a_branches():
    d1 = pd.bdate_range("2025-07-01", periods=30)
    d2 = pd.bdate_range("2026-01-01", periods=30)
    # 全过：两窗 IC>0 且 Δ>0
    v = stage_a_decision(
        _ic_series(d1, 0.02), _ic_series(d2, 0.03),
        _ic_series(d1, 0.0), _ic_series(d2, 0.0),
    )
    assert v.pass_all and not v.reasons
    # IC 一窗负 → 拒
    v2 = stage_a_decision(
        _ic_series(d1, -0.01), _ic_series(d2, 0.03),
        _ic_series(d1, 0.0), _ic_series(d2, 0.0),
    )
    assert not v2.pass_all and any("W1 T-PRE IC" in r for r in v2.reasons)
    # Δ 一窗负 → 拒
    v3 = stage_a_decision(
        _ic_series(d1, 0.02), _ic_series(d2, 0.03),
        _ic_series(d1, 0.05), _ic_series(d2, 0.0),
    )
    assert not v3.pass_all and any("ΔIC" in r for r in v3.reasons)


# ============================================================
# 6. checkpoint 选择只用验证（历史文件替换不改选点）
# ============================================================
def test_checkpoint_selection_uses_validation_only(tmp_path):
    from tas_state.train import select_checkpoint

    val_scores = {1000: 0.011, 2000: 0.0132, 4000: 0.0131}  # 平局 <1e-6 取早
    ck = select_checkpoint(val_scores)
    assert ck == 2000
    # 验证分完全平局 → 取较早
    assert select_checkpoint({1000: 0.01, 2000: 0.01, 4000: 0.01}) == 1000
    # 与"历史评估"无关：同输入必同输出（无隐藏状态）
    assert select_checkpoint(val_scores) == ck


# ============================================================
# 7. resume 校验 manifest（tokenizer 哈希/臂/seed/股票顺序变化拒绝复用）
# ============================================================
def test_resume_checks_manifest(tmp_path):
    from tas_state.signals import check_resume_compatible, SignalManifest

    m = SignalManifest(
        config_sha256="cfg1", tokenizer_sha256="tok1", arm="T-PRE", seed=42,
        window="W1", columns_hash="abc",
    )
    check_resume_compatible(m, config_sha256="cfg1", tokenizer_sha256="tok1",
                            arm="T-PRE", seed=42, window="W1", columns_hash="abc")
    base = dict(config_sha256="cfg1", tokenizer_sha256="tok1",
                arm="T-PRE", seed=42, window="W1")
    for k, v in (
        ("tokenizer_sha256", "tok2"), ("arm", "T-POST"), ("seed", 43),
        ("window", "W2"), ("columns_hash", "xyz"), ("config_sha256", "cfg2"),
    ):
        with pytest.raises(ValueError, match="manifest"):
            check_resume_compatible(m, **{**base, k: v})


# ============================================================
# 8. forward 封存：旧 forward 不可读、新 forward 不可提前评价
# ============================================================
def test_forward_seal():
    from tas_state.register import FORWARD_SEAL_START, assert_forward_seal

    # 白名单历史结算缓冲（价格数据允许晚于 seal；非 forward 信号）
    assert_forward_seal(
        pd.Timestamp("2026-08-01"), what="历史结算缓冲", settlement_buffer=True
    )
    # ≥ seal 的旧 forward 访问被拒
    with pytest.raises(PermissionError, match="封存"):
        assert_forward_seal(pd.Timestamp("2026-07-25"), what="旧forward信号")
    # 未成熟（未来日期）的提前评价被拒
    with pytest.raises(PermissionError, match="未成熟"):
        assert_forward_seal(
            pd.Timestamp("2026-09-23"), what="新forward信号", early_eval=True
        )
