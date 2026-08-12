"""improve_suite 阶段 2 单测——分布统计信号（计划 §4.2）。

构造 2 股 × 已知路径矩阵，手算断言 S1/S2/S3 三值。

三条信号（跑前冻结，§4.2）：对每 (date, code)，由 N 条路径的 H 日平均收益
``r_i = mean(path_i) / close_t − 1`` 计算：

    - S1 ``neg_std``    = ``-std(r_1..r_N)``
    - S2 ``sharpe_like`` = ``mean(r_i) / std(r_i)``
    - S3 ``q10``         = ``quantile(r_i, 0.1)``

先于实现写成：模块未建时 import 失败 → FAIL；实现后 → PASS。
"""
from __future__ import annotations

import numpy as np
import pytest


class TestDistSignals:
    """手算已知路径矩阵，断言 S1/S2/S3（计划 §4.2）。"""

    def test_single_path_degenerate(self) -> None:
        """单路径：std=0 → neg_std=0、sharpe 未定义（nan/inf）、q10=该路径 r。"""
        from improve_suite.dist_signals import compute_dist_signals

        # 1 条路径 H=3：[110,120,130]，close=100 → r=120/100-1=0.2
        path_close = np.array([[110.0, 120.0, 130.0]])
        s = compute_dist_signals(path_close, last_close=100.0)
        assert s["mean"] == pytest.approx(0.2)
        assert s["neg_std"] == pytest.approx(0.0, abs=1e-12)
        assert s["q10"] == pytest.approx(0.2)
        # 单路径 std=0 → sharpe 无定义
        assert np.isnan(s["sharpe_like"]) or np.isinf(s["sharpe_like"])

    def test_two_paths_hand_computed(self) -> None:
        """2 路径手算：r=[0.2, 0.0]，mean=0.1，std=0.1，sharpe=1.0。"""
        from improve_suite.dist_signals import compute_dist_signals

        # path0=[110,130] → mean=120 → r=0.2；path1=[90,110] → mean=100 → r=0.0
        path_close = np.array(
            [
                [110.0, 130.0],
                [90.0, 110.0],
            ]
        )  # shape (N=2, H=2)
        s = compute_dist_signals(path_close, last_close=100.0)
        assert s["mean"] == pytest.approx(0.1)
        # std([0.2, 0.0], ddof=0) = 0.1
        assert s["neg_std"] == pytest.approx(-0.1)
        assert s["sharpe_like"] == pytest.approx(1.0)
        # quantile([0.0, 0.2], 0.1) = 0.0 + 0.1*(0.2-0.0) = 0.02（linear 插值）
        assert s["q10"] == pytest.approx(0.02)

    def test_three_paths_hand_computed(self) -> None:
        """3 路径手算全字段（验证 std/ddof=0 与 quantile 逻辑）。"""
        from improve_suite.dist_signals import compute_dist_signals

        # path_i 全 H 步恒定 → r_i = v_i/close - 1
        # path0=[120,120] → r=0.2；path1=[80,80] → r=-0.2；path2=[100,100] → r=0.0
        path_close = np.array(
            [
                [120.0, 120.0],
                [80.0, 80.0],
                [100.0, 100.0],
            ]
        )  # (N=3, H=2)
        s = compute_dist_signals(path_close, last_close=100.0)
        r = np.array([0.2, -0.2, 0.0])
        assert s["mean"] == pytest.approx(r.mean())  # 0.0
        # std ddof=0
        assert s["neg_std"] == pytest.approx(-r.std(ddof=0))
        assert s["sharpe_like"] == pytest.approx(r.mean() / r.std(ddof=0))
        assert s["q10"] == pytest.approx(np.quantile(r, 0.1))

    def test_mean_equals_canonical_signal(self) -> None:
        """mean(r_i) 必须等于 canonical mean 信号（对拍一致性）。

        canonical mean = mean(所有路径所有步) / close - 1 = mean(r_i)。
        """
        from improve_suite.dist_signals import compute_dist_signals

        rng = np.random.default_rng(42)
        path_close = rng.uniform(80, 120, size=(20, 10))  # (N=20, H=10)
        close = 100.0
        s = compute_dist_signals(path_close, last_close=close)
        canonical = path_close.mean() / close - 1.0
        assert s["mean"] == pytest.approx(canonical, abs=1e-12)
