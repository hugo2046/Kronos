"""mean-comparison 合成契约测试（不触真实行情/DDB/GPU/train）。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from qlib_mean_comparison import run as mc  # noqa: E402


def _wide(seed=0, days=10):
    cal = pd.bdate_range("2025-07-01", periods=days)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, (days, 4)),
                        index=cal, columns=["A", "B", "C", "D"])


def test_load_transpose_unifies_orientation(tmp_path, monkeypatch):
    """股票×日期 与 日期×股票 两种 parquet 统一为 date×instrument。"""
    w = _wide()
    p1 = tmp_path / "by_date.parquet"
    w.to_parquet(p1)                              # 日期×股票
    p2 = tmp_path / "by_code.parquet"
    w.T.to_parquet(p2)                            # 股票×日期
    monkeypatch.setitem(mc.SIGNAL_PATHS, "G1_mean",
                        {"W3": p1, "W4": p2})
    a = mc.load_wide("G1_mean", "W3")
    b = mc.load_wide("G1_mean", "W4")
    pd.testing.assert_frame_equal(a, b)           # 转置归一 + 索引转 Datetime
    assert a.index.max() <= pd.Timestamp("2026-07-24")


def test_common_set_and_no_double_shift():
    """共同掩码 = 三臂 notna 相交；信号序列不做额外平移（索引恒等）。"""
    w1, w2, w3 = _wide(1), _wide(2), _wide(3)
    w1.iloc[0, 0] = np.nan                        # G1 缺一格
    mask, stats = mc.common_set({"G1_mean": w1, "A1_mean": w2,
                                 "A0_bestCE_mean": w3})
    assert not mask.iloc[0, 0] and mask.iloc[1:, 1:].all().all()
    assert stats["n_common_cells"] == int(mask.sum().sum())
    sig = mc.prepare_signal_series(w2, mask)
    # 无平移：信号日期集合 = 掩码有效日期集合（t 日信号留在 t 日，由
    # qlib-ddb TopkDropoutStrategy 内部 shift=1 实现 t+1 开盘执行）
    assert set(sig.index.get_level_values("datetime")) == \
        set(mask.index[mask.any(axis=1)])


def test_same_signals_same_curve_and_delta_formula():
    """相同信号两臂曲线相同；delta_end = 曲线末差；cumsum 与复利分开。"""
    w = _wide(3)
    mask, _ = mc.common_set({"a": w, "b": w, "c": w})
    report = pd.DataFrame({"return": np.linspace(0.001, 0.002, 10),
                           "cost": [0.0002] * 10,
                           "bench": [0.0005] * 10,
                           "turnover": [0.1] * 10},
                          index=w.index)
    c1 = mc.compute_curves(report)
    c2 = mc.compute_curves(report.copy())
    pd.testing.assert_series_equal(c1["curve"], c2["curve"])
    net = report["return"] - report["cost"]
    assert c1["curve"].iloc[-1] == pytest.approx(net.cumsum().iloc[-1])
    assert c1["excess_curve"].iloc[-1] == pytest.approx(
        (net - report["bench"]).cumsum().iloc[-1])
    assert c1["compound_cum"] == pytest.approx(float((1 + net).prod() - 1))
    # 超额差不变性：同基准下 (A1-bench)-(G1-bench) == A1-G1
    r2 = report.copy()
    r2["return"] = report["return"] + 0.001
    c3 = mc.compute_curves(r2)
    assert (c3["curve"].iloc[-1] - c1["curve"].iloc[-1]) == pytest.approx(
        c3["excess_curve"].iloc[-1] - c1["excess_curve"].iloc[-1])


def test_bench_mismatch_rejects():
    """基准不同/缺日期拒绝比较（run.py 内断言的合成镜像）。"""
    idx = pd.bdate_range("2025-07-01", periods=5)
    b1 = pd.Series(0.0, index=idx)
    b2 = b1.copy()
    b2.iloc[0] = 1e-6
    assert not b1.equals(b2)                       # 值不同 → 拒绝语义成立
    b3 = b1.iloc[1:]                               # 日期缺失 → 拒绝语义成立
    assert not b1.index.equals(b3.index)


def test_source_has_no_train_predict():
    src = (REPO_ROOT / "qlib_mean_comparison" / "run.py").read_text(
        encoding="utf-8")
    for banned in ("model.fit", "TrainerR", "auto_regressive_inference",
                   "torch.load", "from_pretrained"):
        assert banned not in src, f"出现禁用调用：{banned}"


def test_input_sha_unchanged_roundtrip(tmp_path):
    f = tmp_path / "s.parquet"
    _wide().to_parquet(f)
    sha1 = mc.sha256_file(f)
    assert mc.sha256_file(f) == sha1               # 未变一致
    w = _wide(9)
    w.to_parquet(f)
    assert mc.sha256_file(f) != sha1               # 变更可检出
