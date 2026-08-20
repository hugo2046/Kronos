"""c4_temporal 单测（C4 计划 §3 3.1，20260820）。

先于实现写成（模块未建时 import 失败 → FAIL；实现后 → PASS）。
冻结超参（C4 计划 §1，跑前定死禁止扫描）：阈值 ±2%、半衰期 10 交易日、
窗口 30 交易日、预热烧 30 交易日（评估自信号首日后第 30 个交易日起）。

冻结 NaN 语义（执行决策，跑前定案）：
    - 三值化：NaN 信号 → NaN（不投 +1/0/−1 任何一票）；
    - 半衰期加权和：窗口内 NaN 项**跳过**（贡献 0、权重照丢）；
      30 项全 NaN → C4 = NaN（整窗无票 = 不可买）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEEDS = (100, 101, 102)
THR = 0.02  # 三值化阈值 ±2%（冻结）
HALF_LIFE = 10  # 半衰期 10 交易日（冻结）
C4_WINDOW = 30  # 加权窗口 30 交易日（冻结）
WARMUP = 30  # 预热烧 30 交易日（冻结：首日后第 30 个交易日起评估）


class TestTrinarizeBoundary:
    """三值化边界（恰 ±2% 取 0——严格不等式）。"""

    def test_boundary_exactly_pm2pct_is_zero(self) -> None:
        from c4_temporal.transform import trinarize

        x = pd.Series([0.02, -0.02, 0.0200001, -0.0200001, 0.0, 0.0199999, -0.0199999])
        got = trinarize(x)
        assert list(got) == [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], list(got)

    def test_nan_propagates(self) -> None:
        from c4_temporal.transform import trinarize

        got = trinarize(pd.Series([np.nan, 0.5, -0.5]))
        assert np.isnan(got.iloc[0])
        assert list(got.iloc[1:]) == [1.0, -1.0]

    def test_wide_frame_elementwise(self) -> None:
        from c4_temporal.transform import trinarize

        wide = pd.DataFrame(
            {"A": [0.03, -0.01], "B": [np.nan, -0.9]}, index=pd.date_range("2025-01-01", periods=2)
        )
        got = trinarize(wide)
        assert got.loc[wide.index[0], "A"] == 1.0
        assert got.loc[wide.index[0], "B"] != got.loc[wide.index[0], "B"]  # NaN
        assert got.loc[wide.index[1], "A"] == 0.0
        assert got.loc[wide.index[1], "B"] == -1.0


class TestHalflifeWeights:
    """半衰期权重和（λ 幂级数手算对拍）。"""

    def test_lambda_hand_computed(self) -> None:
        from c4_temporal.transform import C4_LAMBDA, c4_weights

        # λ = 0.5^(1/10) 手算 = 0.9330329915368074…
        assert C4_LAMBDA == 0.5 ** (1.0 / 10.0)
        assert abs(C4_LAMBDA - 0.9330329915368074) < 1e-15
        # λ 幂级数手算：λ^k = 0.5^(k/10)，故 λ^10=0.5 / λ^20=0.25 / λ^30=0.125
        w = c4_weights()
        assert len(w) == C4_WINDOW
        np.testing.assert_allclose(w, [0.5 ** (k / 10) for k in range(C4_WINDOW)], rtol=0, atol=1e-15)
        assert abs(w[10] - 0.5) < 1e-12  # 半衰期性质：滞后 10 日权重恰减半
        assert abs(w[20] - 0.25) < 1e-12
        assert abs(w[29] - 0.5 ** 2.9) < 1e-12

    def test_constant_series_matches_closed_form(self) -> None:
        """恒 +1 序列 → C4 = Σ λ^k = (1−λ^30)/(1−λ)（手算闭式对拍）。"""
        from c4_temporal.transform import C4_LAMBDA, c4_ew_sum

        idx = pd.date_range("2025-01-01", periods=60, freq="B")
        e = pd.DataFrame(1.0, index=idx, columns=["A"])
        got = c4_ew_sum(e)
        closed = (1 - C4_LAMBDA**30) / (1 - C4_LAMBDA)  # 手算：0.875/0.0669670085
        assert abs(closed - 13.0661354) < 1e-5  # 手算量级锚（修正版：13.0661354）
        # 满窗日（第 30 日起）逐日等于闭式；前 29 日为部分和 Σ_{k=0..t}
        for t in range(29, 60):
            assert abs(got["A"].iloc[t] - closed) < 1e-9, t
        partial_0 = 1.0  # 首日只有 k=0 一项
        assert abs(got["A"].iloc[0] - partial_0) < 1e-12

    def test_single_spike_decay(self) -> None:
        """单日 +1 脉冲 → C4(t) = λ^(t−t0)，滞后 10 日恰 0.5，30 日后归零。"""
        from c4_temporal.transform import C4_LAMBDA, c4_ew_sum

        idx = pd.date_range("2025-01-01", periods=70, freq="B")
        e = pd.DataFrame(0.0, index=idx, columns=["A"])
        t0 = 5
        e.iloc[t0, 0] = 1.0
        got = c4_ew_sum(e)["A"]
        assert abs(got.iloc[t0] - 1.0) < 1e-12  # λ^0
        assert abs(got.iloc[t0 + 10] - 0.5) < 1e-12  # λ^10 = 0.5（半衰期）
        assert abs(got.iloc[t0 + 20] - 0.25) < 1e-12
        assert abs(got.iloc[t0 + 29] - C4_LAMBDA**29) < 1e-12
        assert got.iloc[t0 + 30] == 0.0  # 窗口滑出 → 该票彻底出窗
        assert got.iloc[t0 - 1] == 0.0  # 脉冲日前无信息

    def test_nan_skip_semantics(self) -> None:
        """NaN 项跳过（贡献 0）；全 NaN 窗 → NaN。"""
        from c4_temporal.transform import c4_ew_sum

        idx = pd.date_range("2025-01-01", periods=40, freq="B")
        # A：30 项窗里恰一项 NaN，其余 +1 → 手算 = 闭式 − λ^(滞后)
        e = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
        e.iloc[10, 0] = np.nan
        # B：全 NaN → 整列 NaN
        e["B"] = np.nan
        got = c4_ew_sum(e)
        from c4_temporal.transform import C4_LAMBDA

        closed = (1 - C4_LAMBDA**30) / (1 - C4_LAMBDA)
        # 第 40 日（t=39）：窗 = 10..39，NaN 在滞后 29 处
        hand = closed - C4_LAMBDA**29
        assert abs(got["A"].iloc[39] - hand) < 1e-9
        assert got["B"].isna().all()


class TestWarmupExclusion:
    """预热期剔除断言（评估自信号首日后第 30 个交易日起）。"""

    def test_eval_dates_drop_first_30(self) -> None:
        from c4_temporal.transform import eval_rebalance_dates

        idx = pd.bdate_range("2025-07-01", periods=260)
        ev = eval_rebalance_dates(idx)
        assert len(ev) == 260 - WARMUP == 230
        assert ev[0] == idx[WARMUP]  # 首日后第 30 个交易日（index 30）
        burned = set(idx[:WARMUP])
        assert not (burned & set(ev)), "预热期 30 日必须整段烧掉"
        assert list(ev) == list(idx[WARMUP:])

    def test_merged_shape_260_continuous(self) -> None:
        from c4_temporal.transform import load_g1_mean_merged

        for s in SEEDS:
            merged = load_g1_mean_merged(s)
            assert len(merged) == 260, f"s{s} 合并后 {len(merged)} 日 ≠ 126+134"
            assert merged.index.is_monotonic_increasing and merged.index.is_unique
            assert str(merged.index[0].date()) == "2025-07-01"
            assert str(merged.index[-1].date()) == "2026-07-24"


class TestDeterministicRecompute:
    """同输入重算逐位一致（C4 = 确定性纯函数）。"""

    def test_bitwise_identical(self) -> None:
        from c4_temporal.transform import build_c4, load_g1_mean_merged

        src = load_g1_mean_merged(100)
        a, b = build_c4(src), build_c4(src)
        assert a.index.equals(b.index) and a.columns.equals(b.columns)
        np.testing.assert_array_equal(a.values, b.values)  # NaN 逐位相等

    def test_trinarize_ew_compose_matches_build(self) -> None:
        """build_c4 = trinarize → c4_ew_sum（两步手工组合与一键构建逐位一致）。"""
        from c4_temporal.transform import build_c4, c4_ew_sum, load_g1_mean_merged, trinarize

        src = load_g1_mean_merged(102)
        manual = c4_ew_sum(trinarize(src))
        np.testing.assert_array_equal(build_c4(src).values, manual.values)
