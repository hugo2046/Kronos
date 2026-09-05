"""Codex 复核五项修复的回归测试（v1 诊断修复，2026-09-05）。

每项修复至少一例：
1. 教师 eval：ensure_teacher_eval 强制 eval；teacher 身份含 model_eval，
   旧身份（无该字段）拒绝混用；
2. D2 损失比较：d2_unlock_condition 统一 val_task 口径（不用 D0 的 val_d）；
3. LoRA device/dtype：A/B 跟随被包装权重的 device/dtype；
4. 缓存身份：清单装载协议过滤 + 内容校验；TeacherRunner.load_verified
   逐字段核验（protocol/replicas/predict_len/权重指纹）；
5. 阶段门禁：D2 门禁 seed 一致性；臂前置依赖显式报错。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from tests.test_dhead_teacher import FakePredictor, _tiny_manifest, _runner


# ================================================================ 修复 #1 ====

class _StubTeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.2)
        self.lin = nn.Linear(4, 4)


class _StubPredictor:
    def __init__(self):
        self.model = _StubTeacherModel()
        self.tokenizer = _StubTeacherModel()


def test_ensure_teacher_eval_forces_eval() -> None:
    """装载后的教师必须 eval（G1 dropout=0.2，train 态污染目标）。"""
    from dhead_distill.teacher import ensure_teacher_eval

    p = _StubPredictor()
    assert p.model.training and p.tokenizer.training  # 构造默认 train 态
    ensure_teacher_eval(p)
    assert not p.model.training and not p.tokenizer.training


def test_teacher_identity_carries_model_eval(tmp_path, monkeypatch) -> None:
    """身份含 model_eval：v1（无该字段）的 teacher_run.json 拒绝复用。"""
    m = _tiny_manifest()
    r = _runner(tmp_path, monkeypatch, m)
    r.run()
    run_json = r.run_dir / "teacher_run.json"
    doc = json.loads(run_json.read_text("utf-8"))
    assert doc["model_eval"] is True

    # 模拟 v1 旧身份：去掉 model_eval 字段 → 新 runner 拒绝混用
    doc_v1 = {k: v for k, v in doc.items() if k != "model_eval"}
    doc_v1.pop("days", None)
    run_json.write_text(json.dumps(doc_v1, ensure_ascii=False), "utf-8")
    with pytest.raises(RuntimeError, match="身份不一致"):
        _runner(tmp_path, monkeypatch, m)


# ================================================================ 修复 #2 ====

def _fid(gate_pass: bool, ratio: float) -> dict:
    return {"gate": {"passed": gate_pass}, "ratio": ratio}


def test_d2_unlock_compares_val_task_not_val_d() -> None:
    """D1 解锁比较用两臂 val_task：D0 的 val_d 再小也不参与比较。"""
    from dhead_distill.evaluate import d2_unlock_condition

    # D0 的 val_d 极小（0.01）但 val_task 差（0.6）；D1 val_task 好（0.5）
    d0_hist = [{"epoch": 0, "val_loss": 0.01, "val_task": 0.6}]
    d1_hist = [{"epoch": 0, "val_loss": 0.50, "val_task": 0.5}]
    v = d2_unlock_condition(d0_hist, d1_hist, _fid(True, 1.0), _fid(True, 1.0))
    assert v["conditions"]["d1_beats_d0_val_task"] is True
    assert v["unlocked"] is True

    # 反例：D1 val_task（0.7）劣于 D0（0.6）→ 不解锁（即使 D1 val_loss 更小）
    d1_hist2 = [{"epoch": 0, "val_loss": 0.001, "val_task": 0.7}]
    v2 = d2_unlock_condition(d0_hist, d1_hist2, _fid(True, 1.0), _fid(True, 1.0))
    assert v2["conditions"]["d1_beats_d0_val_task"] is False
    assert v2["unlocked"] is False

    # 保真条件独立生效：D0 门禁 FAIL → 不解锁
    v3 = d2_unlock_condition(d0_hist, d1_hist, _fid(False, 1.0), _fid(True, 1.0))
    assert v3["unlocked"] is False
    v4 = d2_unlock_condition(d0_hist, d1_hist, _fid(True, 1.0), _fid(True, 2.5))
    assert v4["unlocked"] is False


# ================================================================ 修复 #3 ====

def test_lora_follows_base_dtype_and_device() -> None:
    """LoRA A/B 与被包装权重同 device/dtype（D2 在 CUDA 上的设备失配修复）。"""
    from dhead_distill.backbone import LoRALinear

    base = nn.Linear(832, 832).double()  # dtype=float64（device=cpu）
    l = LoRALinear(base, rank=8, alpha=8)
    assert l.lora_A.weight.dtype == torch.float64
    assert l.lora_B.weight.dtype == torch.float64
    assert l.lora_A.weight.device == base.weight.device

    x = torch.randn(2, 4, 832, dtype=torch.float64)
    y = l(x)
    assert y.dtype == torch.float64  # 前向不发生 dtype 降级


# ================================================================ 修复 #4 ====

def test_manifest_load_verify_detects_tamper(tmp_path, monkeypatch) -> None:
    """DayManifest.load(verify=True)：内容被改 → 拒绝。"""
    from dhead_distill.data import DayManifest

    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art"))
    (tmp_path / "base").mkdir(exist_ok=True)
    m = _tiny_manifest()
    d = m.save(f"prepare-pilot-train-{m.content_hash[:12]}")
    m2 = DayManifest.load(f"prepare-pilot-train-{m.content_hash[:12]}",
                          verify=True)  # 未篡改 → 通过
    assert len(m2.samples) == len(m.samples)

    arr_path = d / "arrays.npz"
    z = dict(np.load(arr_path, allow_pickle=False))
    z["close_t"] = z["close_t"] + 1.0  # 篡改数组内容
    np.savez(arr_path, **z)
    with pytest.raises(RuntimeError, match="内容校验失败"):
        DayManifest.load(f"prepare-pilot-train-{m.content_hash[:12]}",
                         verify=True)


def test_load_manifest_protocol_filter(tmp_path, monkeypatch) -> None:
    """_load_manifest：协议不匹配的候选被过滤；无匹配 → 显式报错。"""
    import dataclasses

    from dhead_distill.cli import _load_manifest
    from dhead_distill.config import DHeadConfig, protocol_hash
    from dhead_distill.data import content_hash

    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art"))
    (tmp_path / "base").mkdir(exist_ok=True)
    cfg_match = dataclasses.replace(DHeadConfig(), profile="pilot")
    m = _tiny_manifest()
    m.profile = "pilot"
    m.protocol = protocol_hash(cfg_match)   # 让清单协议与 cfg 对齐
    m.content_hash = content_hash(m)        # 协议入内容 hash，须重算
    m.save(f"prepare-pilot-train-{m.content_hash[:12]}")

    got = _load_manifest("pilot", "train", cfg_match)
    assert got.content_hash == m.content_hash

    cfg_other = dataclasses.replace(DHeadConfig(), train_start="2015-01-01")
    with pytest.raises(FileNotFoundError, match="协议不匹配"):
        _load_manifest("pilot", "train", cfg_other)


def test_teacher_load_verified_rejects_mismatch(tmp_path, monkeypatch) -> None:
    """load_verified：replicas/predict_len/权重指纹不一致 → 拒绝。"""
    from dhead_distill.teacher import TeacherRunner

    m = _tiny_manifest()
    r = _runner(tmp_path, monkeypatch, m)
    r.run()
    w = r.weight_hash

    ok = TeacherRunner.load_verified(m, replicas=3, predict_len=10,
                                     expected_weight_hash=w)
    arr, keys = ok.load_targets_array()
    assert arr.shape[0] == len(m.samples) and arr.shape[1] == 3

    with pytest.raises(RuntimeError, match="replicas"):
        TeacherRunner.load_verified(m, replicas=2, predict_len=10)
    with pytest.raises(RuntimeError, match="predict_len"):
        TeacherRunner.load_verified(m, replicas=3, predict_len=5)
    with pytest.raises(RuntimeError, match="weight_hash"):
        TeacherRunner.load_verified(m, replicas=3, predict_len=10,
                                    expected_weight_hash="0" * 64)


# ================================================================ 修复 #5 ====

def _write_result(monkeypatch, tmp_path, profile, arm, seed) -> None:
    from dhead_distill.data import safe_artifact_dir

    d = safe_artifact_dir(f"train-{profile}-{arm}-s{seed}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(
        {"arm": arm, "seed": seed, "best_epoch": 0, "history": []}), "utf-8")


def test_d2_gate_seed_must_match(tmp_path, monkeypatch) -> None:
    """D2 门禁文件 seed 与请求 seed 不一致 → 拒绝（每 seed 独立过门禁）。"""
    from dhead_distill.cli import _require_d2_gate
    from dhead_distill.data import safe_artifact_dir

    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art"))
    (tmp_path / "base").mkdir(exist_ok=True)
    g = safe_artifact_dir("eval-pilot-fidelity")
    g.mkdir(parents=True, exist_ok=True)
    (g / "summary.json").write_text(json.dumps(
        {"seed": 42, "d2_unlocked": True, "d2_reason": "ok"}), "utf-8")

    _require_d2_gate("pilot", 42)  # 匹配 → 通过
    with pytest.raises(RuntimeError, match="seed 不匹配"):
        _require_d2_gate("pilot", 43)


def test_arm_prerequisites_enforced(tmp_path, monkeypatch) -> None:
    """臂前置依赖缺失 → 显式门禁报错（不裸抛 FileNotFoundError）。"""
    from dhead_distill.cli import _require_arm_ready

    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art"))
    (tmp_path / "base").mkdir(exist_ok=True)

    with pytest.raises(RuntimeError, match="前置结果缺失"):
        _require_arm_ready("pilot", "D1", 42)  # D0 未训
    _write_result(monkeypatch, tmp_path, "pilot", "D0", 42)
    _require_arm_ready("pilot", "D1", 42)      # D0 在 → 通过
    with pytest.raises(RuntimeError, match="前置结果缺失"):
        _require_arm_ready("pilot", "S-long", 42)  # 还缺 D1
    _write_result(monkeypatch, tmp_path, "pilot", "D1", 42)
    _require_arm_ready("pilot", "S-long", 42)


# ========================================================= v1.1 接口注册 ====

def test_scale_source_interface_and_hash_neutrality() -> None:
    """v1.1 方案 A：默认值保持 v1 协议 hash；非默认=新协议；口径计算正确。"""
    import dataclasses

    from dhead_distill.cli import _train_scale
    from dhead_distill.config import DHeadConfig, protocol_hash

    # v1 基线 hash（a70bab6 实测，加字段不得改变）
    V1_BASE = "cf4fb6803cc4a52b00851c2e1f1b22dde1feb9aa03899803d49e0f65eed3877b"
    V1_PILOT = "47915bc0c3d3854869e92031470c69370e9c11169763041a136feb7b9518581f"
    assert protocol_hash(DHeadConfig()) == V1_BASE
    assert protocol_hash(DHeadConfig.with_profile("pilot")) == V1_PILOT

    # 非默认口径 → hash 变化（新协议，产物隔离）
    cfg_t = dataclasses.replace(DHeadConfig(), scale_source="teacher_r0_std")
    assert protocol_hash(cfg_t) != V1_BASE

    # 非法取值 → 构造报错
    with pytest.raises(ValueError, match="scale_source"):
        dataclasses.replace(DHeadConfig(), scale_source="nonsense")

    # 口径计算：train_real_std vs teacher_r0_std
    class _M:  # 最小 manifest 替身
        samples = [type("S", (), {"y_real": np.full(10, 0.01 * (i + 1), np.float32)})()
                   for i in range(4)]

    m = _M()
    s1 = _train_scale(m)  # 默认 train_real_std
    y = np.stack([s.y_real for s in m.samples])
    np.testing.assert_allclose(s1, np.maximum(y.std(0), 0.01), rtol=1e-6)
    rng = np.random.default_rng(0)
    teach = rng.normal(0, 0.05, (4, 3, 10)).astype(np.float32)  # [N,R,H]
    s2 = _train_scale(m, cfg_t, teacher_targets=teach)
    np.testing.assert_allclose(
        s2, np.maximum(teach[:, 0, :].std(0), 0.01), rtol=1e-6)
    with pytest.raises(ValueError, match="需要教师目标"):
        _train_scale(m, cfg_t, teacher_targets=None)
