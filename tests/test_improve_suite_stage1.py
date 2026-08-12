"""improve_suite 阶段 1 单测（计划 §3）。

两条覆盖（纯 DataFrame / Series，不触发 qlib / GPU）：

    1. ``test_build_switch_signal``：3 日×3 股玩具宽表 + 已知 gate，断言逐行选择正确
       （含 gate 缺失日回退 False 分支）；
    2. ``test_gate_ma200_known_series``：250 日合成指数（前 200 日横盘、后 50 日拉升），
       断言拉升段 gate 为 True、横盘段无 MA200 时为 NaN。

先于实现写成：模块未建时 import 失败 → FAIL；实现后 → PASS。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# §3.1.1 test_build_switch_signal
# ============================================================


class TestBuildSwitchSignal:
    """逐行选择：gate[t]==True 取 sig_a 行，否则取 sig_b 行（计划 §3）。"""

    def test_rowwise_selection(self) -> None:
        from improve_suite.regime_switch import build_switch_signal

        dates = pd.date_range("2024-07-01", periods=3, freq="B")
        cols = ["A", "B", "C"]
        sig_a = pd.DataFrame([[10, 20, 30], [11, 21, 31], [12, 22, 32]], index=dates, columns=cols)
        sig_b = pd.DataFrame([[100, 200, 300], [101, 201, 301], [102, 202, 302]], index=dates, columns=cols)
        # day0 True, day1 False, day2 True
        gate = pd.Series([True, False, True], index=dates)

        out = build_switch_signal(sig_a, sig_b, gate)
        # day0 (True) → a; day1 (False) → b; day2 (True) → a
        assert list(out.iloc[0]) == [10, 20, 30], "day0 应取 sig_a（gate True）"
        assert list(out.iloc[1]) == [101, 201, 301], "day1 应取 sig_b（gate False）"
        assert list(out.iloc[2]) == [12, 22, 32], "day2 应取 sig_a（gate True）"

    def test_gate_missing_day_falls_back_to_false_branch(self) -> None:
        """gate 缺失日回退 False 分支（sig_b）——计划 §3.1.1。"""
        from improve_suite.regime_switch import build_switch_signal

        dates = pd.date_range("2024-07-01", periods=3, freq="B")
        cols = ["A", "B"]
        sig_a = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=dates, columns=cols)
        sig_b = pd.DataFrame([[10, 20], [30, 40], [50, 60]], index=dates, columns=cols)
        # 只给 day0 设 True，day1/day2 缺失 → 应回退 sig_b
        gate = pd.Series([True], index=[dates[0]])

        out = build_switch_signal(sig_a, sig_b, gate)
        assert list(out.iloc[0]) == [1, 2], "day0 gate True → sig_a"
        assert list(out.iloc[1]) == [30, 40], "day1 gate 缺失 → 回退 sig_b"
        assert list(out.iloc[2]) == [50, 60], "day2 gate 缺失 → 回退 sig_b"

    def test_handles_disjoint_columns(self) -> None:
        """sig_a / sig_b 列不完全一致时取并集、缺失填 NaN。"""
        from improve_suite.regime_switch import build_switch_signal

        dates = pd.date_range("2024-07-01", periods=1, freq="B")
        sig_a = pd.DataFrame([[1.0]], index=dates, columns=["A"])
        sig_b = pd.DataFrame([[2.0]], index=dates, columns=["B"])
        gate = pd.Series([True], index=dates)

        out = build_switch_signal(sig_a, sig_b, gate)
        assert "A" in out.columns and "B" in out.columns
        # gate True → sig_a 行：A=1, B=NaN
        assert out.loc[dates[0], "A"] == 1.0
        assert pd.isna(out.loc[dates[0], "B"])


# ============================================================
# §3.1.2 test_gate_ma200_known_series
# ============================================================


class TestGateMA200:
    """000300.SH 收盘 > MA200 的门控（计划 §3 R1/R1' 的 gate）。"""

    def test_rising_segment_gate_true(self) -> None:
        """前 200 日横盘、后 50 日拉升 → 拉升段 gate True。"""
        from improve_suite.regime_switch import ma200_gate_from_close

        # 前 200 日 close=100（横盘），后 50 日从 101 线性升到 150
        close = pd.Series(
            [100.0] * 200 + [100.0 + i for i in range(1, 51)],
            index=pd.date_range("2023-01-01", periods=250, freq="B"),
        )
        gate = ma200_gate_from_close(close, window=200)

        # 前 199 日无 MA200（rolling 200 需满 200 个）→ NaN
        assert pd.isna(gate.iloc[199 - 1]), "第 199 日（<200）应无 MA200 → NaN"
        # 第 200 日（index 199）：MA200=100，close=100 → 不大于 → False
        assert gate.iloc[199] == False, "第 200 日 close=MA200=100 → gate False（不严格大于）"
        # 拉升段：close > MA200（MA200 缓慢上移但远低于 close）
        rising = gate.iloc[200:]
        assert rising.notna().all(), "拉升段应有 MA200（满 200 窗）"
        assert (rising > 0).all(), "拉升段 close 全程 > MA200 → gate 全 True"
        assert rising.sum() == 50, f"拉升段应全 True（50 日），实得 {rising.sum()}"

    def test_declining_segment_gate_false(self) -> None:
        """前 200 日高位、后 50 日下跌 → 下跌段 gate False。"""
        from improve_suite.regime_switch import ma200_gate_from_close

        close = pd.Series(
            [100.0] * 200 + [100.0 - i for i in range(1, 51)],
            index=pd.date_range("2023-01-01", periods=250, freq="B"),
        )
        gate = ma200_gate_from_close(close, window=200)
        declining = gate.iloc[200:]
        # 下跌段 close < MA200 → 全 False
        assert (declining == False).all(), "下跌段 close < MA200 → gate 全 False"
