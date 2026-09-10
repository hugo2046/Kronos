"""g1_fixed_candidates 合成契约测试（不触真实行情/DDB/GPU/train）。

覆盖计划 §6 要求：C1 按 (date,code) 对齐不按位置；缺 seed 拒绝主比较；
mean 是信号均值非收益均值；max≥mean 文件门禁；相同输入相同曲线；基准
日期不同拒绝；不裁尾日；源码不调用训练/推理/forward 读取。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from g1_fixed_candidates import run as fc  # noqa: E402


def _wide(seed=0, days=10, cols=("A", "B", "C", "D")):
    cal = pd.bdate_range("2025-07-01", periods=days)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, (days, len(cols))),
                        index=cal, columns=list(cols))


def test_c1_aligns_by_date_code_not_position():
    """C1 逐 (date,code) 键对齐：乱序列/乱序行不改变结果（位置配对会错）。"""
    s100 = _wide(1)
    s101 = _wide(2)
    s102 = _wide(3)
    shuffled = s101[list(reversed(s101.columns))].iloc[::-1]
    c1_a = fc.build_c1({"G1_mean": s100, "G2S101_mean": s101,
                        "G2S102_mean": s102})
    c1_b = fc.build_c1({"G1_mean": s100, "G2S101_mean": shuffled,
                        "G2S102_mean": s102})
    pd.testing.assert_frame_equal(c1_a, c1_b)
    manual = (s100 + s101 + s102) / 3
    pd.testing.assert_frame_equal(c1_a.reindex_like(manual), manual)


def test_missing_seed_rejects_main():
    """任一 seed 在基线池上缺格 → C1 为 NaN → 主比较被拒绝（不缩池）。"""
    s100 = _wide(1)
    s101 = _wide(2)
    s102 = _wide(3)
    s101.iloc[3, 1] = np.nan                     # s101 缺一格
    mask = fc.g1_pool({"G1_mean": s100})
    c1 = fc.build_c1({"G1_mean": s100, "G2S101_mean": s101,
                      "G2S102_mean": s102})
    assert c1.iloc[3, 1] != c1.iloc[3, 1]        # skipna=False → NaN 传播
    with pytest.raises(ValueError):
        fc.require_main_ready(c1, mask)          # 缺 seed → 拒绝主比较
    fixed = s101.copy()
    fixed.iloc[3, 1] = 0.5
    c1_ok = fc.build_c1({"G1_mean": s100, "G2S101_mean": fixed,
                         "G2S102_mean": s102})
    fc.require_main_ready(c1_ok, mask)           # 完整 → 放行


def test_c1_is_signal_mean_not_return_mean():
    """C1 平均的是同日同股**信号**，不是三条策略收益/NAV。"""
    s100 = pd.DataFrame(0.01, index=_wide().index, columns=["A", "B"])
    s101 = pd.DataFrame(0.03, index=_wide().index, columns=["A", "B"])
    s102 = pd.DataFrame(0.05, index=_wide().index, columns=["A", "B"])
    c1 = fc.build_c1({"G1_mean": s100, "G2S101_mean": s101,
                      "G2S102_mean": s102})
    # 逐格等于信号算术平均；且是宽信号表（与收益序列不同物）
    assert isinstance(c1, pd.DataFrame)
    assert c1.shape == s100.shape
    assert np.allclose(c1.to_numpy(), 0.03)


def test_max_ge_mean_gate():
    """同一 s100 有效格上 G1_max ≥ G1_mean（容差 1e-6），违反即拦截。"""
    mean = _wide(4)
    good = mean + 0.01
    stats = fc.max_ge_mean_gate(mean, good)
    assert stats["violations"] == 0 and fc.gate_ok(stats)
    bad = mean.copy()
    bad.iloc[2, 2] = mean.iloc[2, 2] - 0.01      # 违反 > 容差
    stats_bad = fc.max_ge_mean_gate(mean, bad)
    assert stats_bad["violations"] == 1 and not fc.gate_ok(stats_bad)
    tiny = mean - 5e-7                            # 容差内 → 通过
    assert fc.gate_ok(fc.max_ge_mean_gate(mean, tiny))


def test_same_inputs_same_curve():
    """相同输入得到相同曲线；delta 公式与 cumsum/复利分开。"""
    report = pd.DataFrame({"return": np.linspace(0.001, 0.002, 10),
                           "cost": [0.0002] * 10,
                           "bench": [0.0005] * 10,
                           "turnover": [0.1] * 10},
                          index=_wide().index)
    c1 = fc.compute_curves(report)
    c2 = fc.compute_curves(report.copy())
    pd.testing.assert_series_equal(c1["curve"], c2["curve"])
    net = report["return"] - report["cost"]
    assert c1["curve"].iloc[-1] == pytest.approx(net.cumsum().iloc[-1])
    assert c1["compound_cum"] == pytest.approx(float((1 + net).prod() - 1))


def test_baseline_reproduction_rejects_mismatch():
    """基线复现：日期不同/数值超容差(1e-10)拒绝。"""
    idx = pd.bdate_range("2025-07-01", periods=6)
    base = pd.DataFrame({"return": 0.001, "cost": 0.0001,
                         "bench": 0.0005, "turnover": 0.1},
                        index=idx)
    fc.verify_baseline_reproduction(base.copy(), base, tol=1e-10)
    value_off = base.copy()
    value_off.iloc[0, 0] += 1e-9
    with pytest.raises(AssertionError):
        fc.verify_baseline_reproduction(value_off, base, tol=1e-10)
    date_off = base.iloc[:-1]
    with pytest.raises(AssertionError):
        fc.verify_baseline_reproduction(date_off, base, tol=1e-10)


def test_no_tail_truncation():
    """决策窗口用完整交易日，禁止去掉末尾 20 日。"""
    full = _wide(days=30)
    bounds = (str(full.index.min().date()), str(full.index.max().date()))
    fc.assert_full_window(full, bounds, 30)
    truncated = full.iloc[:-20]
    with pytest.raises(AssertionError):
        fc.assert_full_window(truncated, bounds, 30)


def test_source_has_no_train_predict_forward():
    """源码不含训练/推理/forward 读取调用；截止日固定 2026-07-24。"""
    src = (REPO_ROOT / "g1_fixed_candidates" / "run.py").read_text(
        encoding="utf-8")
    for banned in ("model.fit", "TrainerR", "auto_regressive_inference",
                   "torch.load", "from_pretrained", "predict_batch_chunked"):
        assert banned not in src, f"出现禁用调用：{banned}"
    assert fc.FORWARD_CUTOFF == "2026-07-24"
    # 主比较臂固定三候选：不新增 seed/权重搜索
    assert fc.ARMS_MAIN == ("G1_mean", "G1_max", "C1_mean_ensemble")
    assert set(fc.C1_SEED_ARMS) == {"G1_mean", "G2S101_mean", "G2S102_mean"}


def test_figure_artifacts_are_actual_png_files(tmp_path):
    """实际Qlib recorder回读PNG字节，防止把Path对象序列化成图片产物。"""
    import base64

    import mlflow
    from qlib.workflow.recorder import MLflowRecorder

    upload = getattr(fc, "save_figure_artifacts", None)
    assert callable(upload), "缺少实际文件图片归档入口"
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII="
    )
    figures = tmp_path / "figures_source"
    figures.mkdir()
    names = [f"fig_{window}_{kind}.png"
             for window in ("W3", "W4") for kind in ("main", "appendix")]
    for name in names:
        (figures / name).write_bytes(png)
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        "figure-upload-test", artifact_location=(tmp_path / "artifacts").as_uri())
    run = client.create_run(experiment_id)
    recorder = MLflowRecorder(experiment_id, uri, mlflow_run=run)

    upload(recorder, figures)

    stored = client.list_artifacts(run.info.run_id, "figures")
    assert {Path(item.path).name for item in stored} == set(names)
    for item in stored:
        downloaded = Path(client.download_artifacts(run.info.run_id, item.path))
        assert downloaded.read_bytes() == png
