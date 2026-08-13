"""improve_suite 阶段 3 单测——L/H/T 网格配置（计划 §5.3.1）。

断言 ``ImproveConfig`` 网格覆盖字段生效且未覆盖字段与 canonical 逐字一致。
"""
from __future__ import annotations

import pytest


class TestGridConfigLoad:
    """网格覆盖字段生效，未覆盖字段不变（计划 §5.3.1）。"""

    def test_overrides_apply(self) -> None:
        from improve_suite.common import ImproveConfig

        canonical = ImproveConfig.load(window="paper")
        c1 = ImproveConfig.load(window="paper", lookback=8, predict_len=5, T=1.0)
        assert c1.lookback == 8
        assert c1.predict_len == 5
        assert c1.T == 1.0
        # 未覆盖字段不变
        assert c1.top_p == canonical.top_p
        assert c1.sample_count == canonical.sample_count
        assert c1.seed == canonical.seed
        assert c1.pool == canonical.pool
        assert c1.top_k == canonical.top_k

    def test_c3_low_temperature(self) -> None:
        """C3 = L90/H10/T0.6（低温，论文点预测建议）。"""
        from improve_suite.common import ImproveConfig

        c3 = ImproveConfig.load(window="oos", lookback=90, predict_len=10, T=0.6)
        assert c3.lookback == 90
        assert c3.predict_len == 10
        assert c3.T == 0.6

    def test_canonical_label(self) -> None:
        from improve_suite.common import ImproveConfig

        c1 = ImproveConfig.load(window="paper", lookback=8, predict_len=5, T=1.0)
        assert c1.canonical_label() == "L8_H5_T1.0"
        c3 = ImproveConfig.load(window="paper", lookback=90, predict_len=10, T=0.6)
        assert c3.canonical_label() == "L90_H10_T0.6"

    def test_frozen_three_configs_match_plan(self) -> None:
        """跑前冻结的三配置（§5）：C1=8/5/1.0, C2=30/5/1.0, C3=90/10/0.6。"""
        from improve_suite.common import ImproveConfig

        specs = {"C1": (8, 5, 1.0), "C2": (30, 5, 1.0), "C3": (90, 10, 0.6)}
        for name, (L, H, T) in specs.items():
            cfg = ImproveConfig.load(window="paper", lookback=L, predict_len=H, T=T)
            assert (cfg.lookback, cfg.predict_len, cfg.T) == (L, H, T), f"{name} 配置错"
