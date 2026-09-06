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
    """R1：按实际选中 checkpoint 比较——多 epoch 反例（复核 20260906 §R1）。

    D0 epoch0 (val_d=.1, val_task=.5)、epoch1 (val_d=.2, val_task=.3)；
    D0 按 val_d 选中 epoch0 → 比较基线是 .5；D1 选定点 task=.4 → .4<.5 通过。
    旧实现取 min(val_task)=.3 会误拒（.4<.3 不成立）。
    """
    from dhead_distill.evaluate import d2_unlock_condition

    d0_res = {
        "best_epoch": 0,
        "history": [
            {"epoch": 0, "val_d": 0.1, "val_task": 0.5, "val_loss": 0.1},
            {"epoch": 1, "val_d": 0.2, "val_task": 0.3, "val_loss": 0.2},
        ],
    }
    d1_res = {
        "best_epoch": 0,
        "history": [{"epoch": 0, "val_d": 0.9, "val_task": 0.4,
                     "val_loss": 0.4}],
    }
    v = d2_unlock_condition(d0_res, d1_res, _fid(True, 1.0), _fid(True, 1.0))
    assert v["d0_selected_epoch"] == 0 and v["d0_selected_val_task"] == 0.5
    assert v["conditions"]["d1_beats_d0_val_task"] is True
    assert v["unlocked"] is True   # 旧 min(val_task) 实现在此会误拒

    # 反例：D1 选定点 task=.7 > D0 选中点 .5 → 不解锁
    d1_bad = {"best_epoch": 0,
              "history": [{"epoch": 0, "val_task": 0.7}]}
    v2 = d2_unlock_condition(d0_res, d1_bad, _fid(True, 1.0), _fid(True, 1.0))
    assert v2["unlocked"] is False

    # 保真条件独立生效
    v3 = d2_unlock_condition(d0_res, d1_res, _fid(False, 1.0), _fid(True, 1.0))
    assert v3["unlocked"] is False
    v4 = d2_unlock_condition(d0_res, d1_res, _fid(True, 1.0), _fid(True, 2.5))
    assert v4["unlocked"] is False

    # R1 错误形态：best_epoch 缺失 / epoch 重复 / 对应行不存在
    with pytest.raises(ValueError, match="best_epoch"):
        d2_unlock_condition({"history": d0_res["history"]}, d1_res,
                            _fid(True, 1), _fid(True, 1))
    with pytest.raises(ValueError, match="重复"):
        d2_unlock_condition({"best_epoch": 0, "history": [
            {"epoch": 0, "val_task": .5}, {"epoch": 0, "val_task": .4}]},
            d1_res, _fid(True, 1), _fid(True, 1))
    with pytest.raises(ValueError, match="不存在"):
        d2_unlock_condition({"best_epoch": 9, "history": d0_res["history"]},
                            d1_res, _fid(True, 1), _fid(True, 1))


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

    # R3：学生级字段（output_space=affine，B 臂）不改变清单选取——A/B 共用清单
    cfg_arm_b = dataclasses.replace(cfg_match,
                                    output_space="normalized_close_affine_return")
    got_b = _load_manifest("pilot", "train", cfg_arm_b)
    assert got_b.content_hash == m.content_hash

    cfg_other = dataclasses.replace(DHeadConfig(), train_start="2015-01-01")
    with pytest.raises(FileNotFoundError, match="数据集协议不匹配"):
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
    """R4：D2 门禁核验 seed + 协议 + 数据集 + 输出语义全身份。"""
    import dataclasses

    from dhead_distill.cli import _require_d2_gate
    from dhead_distill.config import (
        DHeadConfig, dataset_protocol_hash, protocol_hash,
    )
    from dhead_distill.data import safe_artifact_dir

    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art"))
    (tmp_path / "base").mkdir(exist_ok=True)
    cfg = DHeadConfig.with_profile("pilot")
    g = safe_artifact_dir("eval-pilot-fidelity")
    g.mkdir(parents=True, exist_ok=True)
    (g / "summary.json").write_text(json.dumps({
        "seed": 42, "d2_unlocked": True, "d2_reason": "ok",
        "protocol": protocol_hash(cfg),
        "dataset_protocol": dataset_protocol_hash(cfg),
        "output_space": cfg.output_space,
    }), "utf-8")

    _require_d2_gate("pilot", 42, cfg)  # 全身份匹配 → 通过
    with pytest.raises(RuntimeError, match="seed 不匹配"):
        _require_d2_gate("pilot", 43, cfg)
    # 协议变化（output_space=affine）→ 旧门禁文件失配
    cfg_b = dataclasses.replace(cfg, output_space="normalized_close_affine_return")
    with pytest.raises(RuntimeError, match="不匹配"):
        _require_d2_gate("pilot", 42, cfg_b)
    # 数据集变化（窗口起点）→ 失配
    cfg_c = dataclasses.replace(cfg, train_start="2015-01-01")
    with pytest.raises(RuntimeError, match="不匹配"):
        _require_d2_gate("pilot", 42, cfg_c)


def test_main_gate_requires_pilot_pass(tmp_path, monkeypatch) -> None:
    """R4：main 入口实检 pilot 门禁——存在结果文件 ≠ 门禁通过。"""
    from dhead_distill.cli import _require_profile_gate
    from dhead_distill.data import safe_artifact_dir

    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art"))
    (tmp_path / "base").mkdir(exist_ok=True)

    _require_profile_gate("pilot")  # pilot 不受限
    with pytest.raises(RuntimeError, match="main 入口被拒"):
        _require_profile_gate("main")  # 无 pilot 门禁文件
    g = safe_artifact_dir("eval-pilot-fidelity")
    g.mkdir(parents=True, exist_ok=True)
    (g / "summary.json").write_text(json.dumps(
        {"arms": {"D0": {"gate": {"passed": False}}}}), "utf-8")
    with pytest.raises(RuntimeError, match="保真门禁未通过"):
        _require_profile_gate("main")  # 文件在但门禁 FAIL → 仍拒
    (g / "summary.json").write_text(json.dumps(
        {"arms": {"D0": {"gate": {"passed": True}}}}), "utf-8")
    _require_profile_gate("main")  # PASS → 放行


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


# ================================================================ R2 修复 ====

def _mk_window(close_t: float, spread: float, n: int = 90, seed: int = 0):
    """合成 90 日窗口：close 终值 close_t，日间波动幅度 spread。"""
    rng = np.random.default_rng(seed)
    closes = close_t + np.linspace(-spread, spread, n) + \
        rng.normal(0, spread / 50, n)
    closes[-1] = close_t
    w = np.tile(closes[:, None], (1, 6))
    w[:, 0] = closes * 1.001
    w[:, 1] = closes * 1.002
    w[:, 2] = closes * 0.998
    w[:, 4] = np.abs(rng.normal(1e6, 1e4, n))
    w[:, 5] = w[:, 4] * closes
    return w


def test_affine_restore_teacher_formula_parity() -> None:
    """R2 对拍：a/b 还原与教师 close 反归一化公式代数/数值一致。"""
    from dhead_distill.data import affine_restore_params

    for close_t, spread in [(100.0, 5.0), (1001.0, 3.0), (12.5, 0.4)]:
        w = _mk_window(close_t, spread, seed=int(close_t))
        a, b = affine_restore_params(w)
        # 教师公式：z=1（预测 close = 1 个标准化单位）
        c64 = w[:, 3].astype(np.float64)
        close_pred_teacher = 1.0 * (c64.std() + 1e-5) + c64.mean()
        r_teacher = close_pred_teacher / close_t - 1.0
        r_restore = a * 1.0 + b
        assert r_restore == pytest.approx(r_teacher, rel=1e-12, abs=1e-14)


def test_affine_restore_additive_shift_counterexample() -> None:
    """R2 反例（复核 20260906 原文）：+900 平移的两窗口——标准化输入相同、
    close_t=101/1001——相同 z 语义下还原须区分收益 1/101 vs 1/1001。"""
    from dhead_distill.data import affine_restore_params, window_zscore_clip

    w1 = _mk_window(101.0, 4.0, seed=7)
    w2 = w1 + 900.0          # 加法平移：z-score 后输入完全相同
    z1 = window_zscore_clip(w1.astype(np.float64), eps=1e-5, clip=5.0)
    z2 = window_zscore_clip(w2.astype(np.float64), eps=1e-5, clip=5.0)
    np.testing.assert_allclose(z1, z2, atol=1e-6)  # 标准化输入相同

    a1, b1 = affine_restore_params(w1)
    a2, b2 = affine_restore_params(w2)
    # 相同 z_hat → 还原收益不同（能区分两个价格尺度）
    r1 = a1 * 0.01 + b1
    r2 = a2 * 0.01 + b2
    assert abs(r1 - r2) > 1e-6
    # 精确验证教师语义：z = (close_pred − mu)/denom 使预测 close 恰为 close_t+1
    c1, c2 = w1[:, 3].astype(np.float64), w2[:, 3].astype(np.float64)
    z_plus1_w1 = (101.0 + 1.0 - c1.mean()) / (c1.std() + 1e-5)
    z_plus1_w2 = (1001.0 + 1.0 - c2.mean()) / (c2.std() + 1e-5)
    assert (a1 * z_plus1_w1 + b1) == pytest.approx(1.0 / 101.0, rel=1e-9)
    assert (a2 * z_plus1_w2 + b2) == pytest.approx(1.0 / 1001.0, rel=1e-9)


def test_affine_restore_invariant_to_future_data() -> None:
    """R2：a/b 只由历史 90 行算出（函数确定性 + 低波动/末日离群样本有限）。"""
    from dhead_distill.data import affine_restore_params

    w = _mk_window(200.0, 6.0, seed=3)
    a1, b1 = affine_restore_params(w)
    a2, b2 = affine_restore_params(w.copy())
    assert (a1, b1) == (a2, b2)  # 未来数据不进入函数：同输入同输出
    w_low = _mk_window(50.0, 0.05, seed=1)          # 低波动
    a_low, b_low = affine_restore_params(w_low)
    assert np.isfinite(a_low) and np.isfinite(b_low) and a_low > 0
    w_out = _mk_window(80.0, 2.0, seed=2)
    w_out[-2, 3] = 80.0 * 1.5                       # 末日离群（属历史）
    a_out, b_out = affine_restore_params(w_out)
    assert np.isfinite(a_out) and np.isfinite(b_out)


def test_head_affine_forward_and_guards() -> None:
    """R2：affine 头前向 r=a·z+b；raw_return 头拒绝 a/b；affine 头缺 a/b 报错。"""
    from dhead_distill.head import MultiHorizonHead

    def _stamp(B):
        s = torch.zeros(B, 10, 5, dtype=torch.long)
        s[..., 3] = 5
        s[..., 4] = 6
        return s

    head_a = MultiHorizonHead(d_model=32, head_dim=16,
                              output_space="normalized_close_affine_return")
    head_a.eval()
    hidden = torch.randn(4, 90, 32)
    with torch.no_grad():
        out = head_a(hidden, _stamp(4), torch.full((4,), 0.5),
                     torch.full((4,), 0.01))
    assert out.shape == (4, 10)
    head_r = MultiHorizonHead(d_model=32, head_dim=16)  # raw_return
    head_r.eval()
    with pytest.raises(ValueError, match="raw_return"):
        head_r(hidden, _stamp(4), torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError, match="必须传入"):
        head_a(hidden, _stamp(4))


def test_materialize_days_carries_affine_fields() -> None:
    """R2：_materialize_days 产出逐样本 a/b（训练与推理同一还原函数）。"""
    from dhead_distill.data import affine_restore_params
    from dhead_distill.train import _materialize_days
    from tests.test_dhead_teacher import _tiny_manifest

    m = _tiny_manifest()
    teacher = np.zeros((len(m.samples), 3, 10), dtype=np.float32)
    days = _materialize_days(m, teacher)
    n_checked = 0
    for batch in days:
        assert batch.a.shape == (len(batch.y_real),)
        assert batch.b.shape == (len(batch.y_real),)
        for j in range(len(batch.y_real)):
            code = [s.code for s in m.samples
                    if s.date.strftime("%Y-%m-%d") == batch.date_iso][j]
            a_ref, b_ref = affine_restore_params(
                m.x_raw[(batch.date_iso, code)])
            assert float(batch.a[j]) == pytest.approx(a_ref, rel=1e-6)
            assert float(batch.b[j]) == pytest.approx(b_ref, rel=1e-6)
            n_checked += 1
    assert n_checked == len(m.samples)


# ================================================================ R3 修复 ====

def test_teacher_shard_tamper_detected(tmp_path, monkeypatch) -> None:
    """R3：有效 NPZ 内只改一个数 → 装载层必须发现（内容 hash）。"""
    m = _tiny_manifest()
    r = _runner(tmp_path, monkeypatch, m)
    r.run()
    shard = sorted(r.run_dir.glob("day-*.npz"))[0]
    z = dict(np.load(shard, allow_pickle=False))
    z["y_teacher"][0, 0, 0] += np.float32(1e-4)   # 只改一个数
    z["content_sha256"] = np.array("0" * 64)      # 旧 hash 对不上新内容
    np.savez(shard, **z)
    assert r._load_verified_shard(shard.name[4:-4]) is None  # 视为损坏


def test_teacher_namespace_isolation(tmp_path, monkeypatch) -> None:
    """R3：namespace 隔离——v11-rev2 目录独立，v1 目录零改动。"""
    from tests.test_dhead_teacher import FakePredictor
    from dhead_distill.teacher import TeacherRunner

    m = _tiny_manifest()
    r1 = _runner(tmp_path, monkeypatch, m)   # namespace 默认 v1
    r1.run()
    v1_files = sorted(p.name for p in r1.run_dir.glob("*.npz"))

    close_t = {(s.date.strftime("%Y-%m-%d"), s.code): 100.0 for s in m.samples}
    r2 = TeacherRunner(
        manifest=m, predict_fn=FakePredictor(close_t).predict,
        weight_hash=r1.weight_hash, n_paths=r1.n_paths, replicas=r1.replicas,
        teacher_T=r1.teacher_T, teacher_top_p=r1.teacher_top_p,
        teacher_top_k=r1.teacher_top_k, predict_len=r1.predict_len,
        namespace="v11-rev2", model_eval_verified=True,
    )
    assert r2.run_dir != r1.run_dir and "v11-rev2" in r2.run_dir.name
    r2.run()
    assert sorted(p.name for p in r1.run_dir.glob("*.npz")) == v1_files
    with pytest.raises(FileNotFoundError, match="namespace"):
        TeacherRunner.load_verified(m, replicas=3, predict_len=10,
                                    namespace="nope")


def test_eval_generation_requires_verified_eval(tmp_path, monkeypatch) -> None:
    """R3：生成路径必须传 model_eval_verified=True（不能只信写死标记）。"""
    from tests.test_dhead_teacher import FakePredictor
    from dhead_distill.teacher import TeacherRunner

    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art"))
    (tmp_path / "base").mkdir(exist_ok=True)
    m = _tiny_manifest()
    close_t = {(s.date.strftime("%Y-%m-%d"), s.code): 100.0 for s in m.samples}
    with pytest.raises(ValueError, match="ensure_teacher_eval"):
        TeacherRunner(
            manifest=m, predict_fn=FakePredictor(close_t).predict,
            weight_hash="w" * 64, n_paths=4, replicas=3,
            teacher_T=1.0, teacher_top_p=0.9, teacher_top_k=0, predict_len=10,
        )


def test_bootstrap_unsorted_starts_and_unit_note() -> None:
    """R4：bootstrap 起点不排序（抽样修正）+ 抽样单位声明 + 确定性。"""
    from dhead_distill.evaluate import block_bootstrap_paired_diff

    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.005, 40)
    ci = block_bootstrap_paired_diff(x, block=10, n_boot=300, seed=1)
    assert ci["sampling_unit"].startswith("trading_day_series")
    ci2 = block_bootstrap_paired_diff(x, block=10, n_boot=300, seed=1)
    assert ci == ci2
