"""baseline_suite 单测（计划 §5.1）。

两条覆盖（均不触发 qlib / GPU，纯张量 / DataFrame）：

    1. 四聚合正确性：构造已知预测路径，断言四个信号值；
    2. mean 对拍门禁：构造新 mean 与既有 K 组**合成一致**的场景断言通过，
       构造**带差异**的场景断言抛 AssertionError。

mean 对拍的"逐位一致"语义在 run_signals.cmd_gate 里实现（读真实 parquet），
本测试用 monkeypatch 把"读 parquet"换成内存 DataFrame，验证门禁逻辑本身。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# §5.1.1 四聚合正确性
# ============================================================


class TestComputeVariants:
    """构造已知预测路径，断言四个信号值（计划 §1）。"""

    def test_four_variants_on_known_path(self) -> None:
        """预测 close 路径 [10, 11, 12, 13, 14]，现价 10 → 四变体可手算。

        last = 14/10 - 1 = 0.4
        mean = mean([10,11,12,13,14])/10 - 1 = 12/10 - 1 = 0.2
        max  = 14/10 - 1 = 0.4
        min  = 10/10 - 1 = 0.0
        """
        from baseline_suite.signal import compute_variants_from_preds

        pred_close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        v = compute_variants_from_preds(pred_close, last_close=10.0)
        assert v["last"] == pytest.approx(0.4)
        assert v["mean"] == pytest.approx(0.2)
        assert v["max"] == pytest.approx(0.4)
        assert v["min"] == pytest.approx(0.0)

    def test_all_divide_by_current_price(self) -> None:
        """四变体必须除以现价（§1 核心修正）。

        同一条预测路径、不同现价 → 信号按 1/close 缩放。
        现价翻倍 → 四个信号约减半（近似，因 -1 项不缩放）。
        """
        from baseline_suite.signal import compute_variants_from_preds

        pred = pd.Series([10.0, 10.0, 10.0])  # 恒定预测
        v1 = compute_variants_from_preds(pred, last_close=10.0)
        v2 = compute_variants_from_preds(pred, last_close=20.0)
        # 恒定预测 close=10，现价 10 → 全 0；现价 20 → 10/20-1 = -0.5
        assert v1["mean"] == pytest.approx(0.0)
        assert v2["mean"] == pytest.approx(-0.5)
        assert v2["last"] == v2["mean"]  # 恒定路径四变体应相等

    def test_mean_matches_paper_replication_exactly(self) -> None:
        """对拍门禁的核心：本模块 mean 必须与 paper_replication 逐字一致。

        ``paper_replication.signal.compute_signal_from_preds``：
            ``np.mean(values)/last_close - 1``
        本模块 mean：同公式。跨多个随机路径断言逐位相等。
        """
        from paper_replication.signal import compute_signal_from_preds
        from baseline_suite.signal import compute_variants_from_preds

        rng = np.random.default_rng(0)
        for _ in range(20):
            path = pd.Series(rng.uniform(5, 50, size=10))
            close = float(rng.uniform(5, 50))
            ref = compute_signal_from_preds(path, close)
            v = compute_variants_from_preds(path, close)
            assert v["mean"] == pytest.approx(ref, abs=1e-12), (
                f"mean 与 K 组口径不一致：path={path.values}, close={close}, "
                f"mean={v['mean']}, ref={ref}"
            )

    def test_variant_relations_hold(self) -> None:
        """min ≤ mean ≤ max 恒成立；last 落在 [min, max] 内（§1 定义）。"""
        from baseline_suite.signal import compute_variants_from_preds

        rng = np.random.default_rng(1)
        for _ in range(50):
            path = pd.Series(rng.uniform(1, 100, size=10))
            close = float(rng.uniform(1, 100))
            v = compute_variants_from_preds(path, close)
            assert v["min"] <= v["mean"] <= v["max"], (
                f"min≤mean≤max 不成立：{v}"
            )
            assert v["min"] <= v["last"] <= v["max"], (
                f"last 应落在 [min,max] 内：{v}"
            )


# ============================================================
# §5.1.2 mean 对拍门禁（逻辑层）
# ============================================================


class TestMeanGate:
    """对拍门禁逻辑（cmd_gate 的核心断言，不读真实 parquet）。"""

    @staticmethod
    def _patch_fs(monkeypatch, new, ref):
        """monkeypatch read_parquet + Path.exists，让 cmd_gate 不碰真实文件。"""
        import pathlib

        def fake_read(p):
            return new if "paper_mean" in str(p) else ref

        def fake_exists(self):
            return True  # 测试场景下两个路径都"存在"

        monkeypatch.setattr(pd, "read_parquet", fake_read)
        monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    def test_gate_passes_when_identical(self, monkeypatch) -> None:
        """新 mean 与既有 K 组逐位一致 → 门禁通过（不抛异常）。"""
        from baseline_suite import run_signals

        dates = pd.bdate_range("2024-07-01", periods=10)
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(7)
        data = rng.standard_normal((10, 3))
        ref = pd.DataFrame(data, index=dates, columns=cols)
        new = ref.copy()
        self._patch_fs(monkeypatch, new, ref)

        from baseline_suite.common import BaselineConfig
        cfg = BaselineConfig.load(window="paper")
        run_signals.cmd_gate(cfg)  # 不抛异常即通过

    def test_gate_fails_on_value_diff(self, monkeypatch) -> None:
        """新 mean 与既有 K 组存在 >1e-8 的值差异 → 抛 AssertionError。"""
        from baseline_suite import run_signals

        dates = pd.bdate_range("2024-07-01", periods=10)
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(7)
        ref = pd.DataFrame(rng.standard_normal((10, 3)), index=dates, columns=cols)
        new = ref.copy()
        new.iloc[0, 0] += 1e-3  # 注入小差异
        self._patch_fs(monkeypatch, new, ref)

        from baseline_suite.common import BaselineConfig
        cfg = BaselineConfig.load(window="paper")
        with pytest.raises(AssertionError, match="对拍门禁未通过"):
            run_signals.cmd_gate(cfg)

    def test_gate_fails_on_nan_mismatch(self, monkeypatch) -> None:
        """新 mean 与既有 K 组 NaN 位置不一致 → 抛 AssertionError。"""
        from baseline_suite import run_signals

        dates = pd.bdate_range("2024-07-01", periods=10)
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(7)
        ref = pd.DataFrame(rng.standard_normal((10, 3)), index=dates, columns=cols)
        new = ref.copy()
        new.iloc[0, 0] = np.nan  # 新侧 NaN、参考侧有值
        self._patch_fs(monkeypatch, new, ref)

        from baseline_suite.common import BaselineConfig
        cfg = BaselineConfig.load(window="paper")
        with pytest.raises(AssertionError, match="对拍门禁未通过"):
            run_signals.cmd_gate(cfg)


# ============================================================
# §5.1.3 判读函数（judge_oos 四条判据）
# ============================================================


class TestJudgeOos:
    """样本外预注册判定（§4 四条判据，跑前冻结的逻辑层验证）。"""

    def _make_perf(self, aer: float, ir: float = 0.5):
        from paper_replication.engine import PerfStats
        return PerfStats(name="x", n_days=240, aer=aer, ir=ir,
                         max_drawdown=-0.1, daily_turnover=0.1)

    def test_main_holds_when_both_positive(self) -> None:
        """判据1：mean AER(等权)>0 且 AER(指数)>0 → 成立。"""
        from baseline_suite.pipeline import judge_oos
        v = judge_oos(
            mean_perf_ew=self._make_perf(0.05, 0.6),
            mean_perf_idx=self._make_perf(0.08, 0.6),
            min_perf_ew=self._make_perf(0.02),
            placeholder_perf_ew=self._make_perf(-0.01),
        )
        assert v["criterion_main"]["both_positive"] is True
        assert "正超额在样本外成立" in v["verdict"]

    def test_main_fails_when_ew_negative(self) -> None:
        """判据3：mean AER(等权)≤0 → 疑似窗口运气。"""
        from baseline_suite.pipeline import judge_oos
        v = judge_oos(
            mean_perf_ew=self._make_perf(-0.02, -0.1),
            mean_perf_idx=self._make_perf(0.03, 0.2),
            min_perf_ew=self._make_perf(-0.01),
            placeholder_perf_ew=self._make_perf(-0.01),
        )
        assert v["criterion_main"]["both_positive"] is False
        assert "不能外推" in v["verdict"]

    def test_engine_gate_stops_when_placeholder_alpha(self) -> None:
        """引擎门禁：P 组 AER(等权)≥3% → 停止。"""
        from baseline_suite.pipeline import judge_oos
        v = judge_oos(
            mean_perf_ew=self._make_perf(0.05),
            mean_perf_idx=self._make_perf(0.08),
            min_perf_ew=self._make_perf(0.02),
            placeholder_perf_ew=self._make_perf(0.05),  # 5% ≥ 3%
        )
        assert v["engine_gate"]["passed"] is False
        assert v["verdict"] == "引擎门禁未通过，停止"

    def test_strength_band(self) -> None:
        """判据2：IR(指数)≥0.3 → 强度未塌方；<0.3 但 AER>0 → 强度衰减。"""
        from baseline_suite.pipeline import judge_oos
        v1 = judge_oos(
            mean_perf_ew=self._make_perf(0.05, 0.6),
            mean_perf_idx=self._make_perf(0.08, 0.5),
            min_perf_ew=self._make_perf(0.02),
            placeholder_perf_ew=self._make_perf(-0.01),
        )
        assert "强度未塌方" in v1["criterion_strength"]["note"]
        v2 = judge_oos(
            mean_perf_ew=self._make_perf(0.05, 0.2),
            mean_perf_idx=self._make_perf(0.02, 0.15),
            min_perf_ew=self._make_perf(0.02),
            placeholder_perf_ew=self._make_perf(-0.01),
        )
        assert "强度衰减" in v2["criterion_strength"]["note"]

    def test_min_candidate_discovery(self) -> None:
        """判据4：min 两段都强于 mean（差>3pp）→ 候选发现。"""
        from baseline_suite.pipeline import judge_oos
        v = judge_oos(
            mean_perf_ew=self._make_perf(0.02),     # 样本外 mean
            mean_perf_idx=self._make_perf(0.05),
            min_perf_ew=self._make_perf(0.06),      # 样本外 min 强 4pp
            placeholder_perf_ew=self._make_perf(-0.01),
            paper_min_perf_ew=self._make_perf(0.06),  # 论文窗 min
            paper_mean_perf_ew=self._make_perf(0.01),  # 论文窗 mean，差 5pp
        )
        assert v["criterion_min"]["both_strong"] is True
        assert "候选发现" in v["criterion_min"]["note"]

    def test_min_not_candidate_when_one_side(self) -> None:
        """判据4：min 只有一段强 → 不构成候选发现。"""
        from baseline_suite.pipeline import judge_oos
        v = judge_oos(
            mean_perf_ew=self._make_perf(0.02),
            mean_perf_idx=self._make_perf(0.05),
            min_perf_ew=self._make_perf(0.06),        # 样本外强
            placeholder_perf_ew=self._make_perf(-0.01),
            paper_min_perf_ew=self._make_perf(0.015),  # 论文窗差 <3pp
            paper_mean_perf_ew=self._make_perf(0.01),
        )
        assert v["criterion_min"]["both_strong"] is False
        assert "不构成候选发现" in v["criterion_min"]["note"]
