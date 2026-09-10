"""生产链身份门禁：全部使用合成缓存、权重和信号，不访问数据库。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from peer_residual import config as C
from peer_residual import data as PD
from peer_residual import evaluate as E
from peer_residual import run as R
from peer_residual import train as T
from sae_residual.cache_io import sha256_file




def _day():
    return PD.DayData("2025-01-02", ["a", "b"],
                      np.ones((2, 16), dtype=np.float32),
                      np.ones(2, dtype=bool), np.ones(2, dtype=np.float32),
                      np.array([0.1, 0.2]))


def _freeze(rt):
    def infer(arm, seed, w, stats, heads_dir, weight_shas):
        # 真正调用生产加载函数，避免伪造 best 元数据绕过身份检查。
        _, best = T.load_best(heads_dir, arm, seed, 16)
        return rt["wide"].copy(), best
    return R.freeze_signals_stage(rt["path"], {"d_in": 16, "sigma_e": 0.1},
                                  [100], PD.g1_weight_shas(), rt["path"],
                                  infer_signal_fn=infer)


def test_train_writes_identity_and_inference_rejects_changed_base(peer_runtime,
                                                                 monkeypatch):
    rt = peer_runtime
    # 避免覆盖 fixture 权重：生产入口对新 seed 做极小合成更新。
    out = T.train_arm([_day()], [_day()], "OFF", 101, rt["path"], epochs=1)
    ck = torch.load(T.head_file(rt["path"], "OFF", 101, 1), weights_only=True)
    assert ck["identity"] == {**rt["identity"], "arm": "OFF", "seed": 101, "epoch": 1}
    assert out["epoch_sha256"]["1"]
    monkeypatch.setattr(PD, "load_split_days", lambda *a, **k: [_day()])
    wide, _ = E.infer_test_signal("OFF", 101, "W3", {"d_in": 16, "sigma_e": 0.1}, rt["path"])
    assert wide.shape == (1, 2)
    (C.G1_PREDICTOR / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        E.infer_test_signal("OFF", 101, "W3", {"d_in": 16, "sigma_e": 0.1}, rt["path"])


def test_reuse_failure_never_retrains_or_overwrites(peer_runtime, monkeypatch):
    monkeypatch.setattr(PD, "load_frozen_stats", lambda: {"d_in": 16})
    monkeypatch.setattr(PD, "load_split_days", lambda *a, **k: [_day()])
    victim = T.head_file(peer_runtime["path"], "OFF", 100, 1)
    victim.write_bytes(b"broken")
    def forbidden(*a, **k):
        pytest.fail("复用失败不允许进入训练")
    monkeypatch.setattr(T, "train_arm", forbidden)
    with pytest.raises(RuntimeError):
        R._train_missing([100], "pilot")
    assert victim.read_bytes() == b"broken"


@pytest.mark.parametrize("field", ["run_id", "protocol_digest", "weight_shas",
                                    "head_sha256", "identity", "baseline"])
def test_tampered_manifest_zero_backtests(peer_runtime, field):
    _freeze(peer_runtime)
    path = peer_runtime["path"] / "frozen_signals_manifest.json"
    m = json.loads(path.read_text())
    if field == "head_sha256":
        m["signals"][0][field] = "wrong"
    else:
        m[field] = "wrong"
    path.write_text(json.dumps(m))
    calls = []
    with pytest.raises(RuntimeError):
        R.backtest_frozen_stage(peer_runtime["path"], [100], runner_fn=lambda *a: calls.append(1))
    assert calls == []


def test_valid_freeze_reads_exact_verified_files_and_refuses_overwrite(peer_runtime,
                                                                     monkeypatch):
    _freeze(peer_runtime)
    calls = []
    def runner(path, seeds, read, backtest):
        calls.append(1)
        pd.testing.assert_frame_equal(read("PEER_OFF_s100", "W3"), peer_runtime["wide"], check_freq=False)
        return {"ok": True}
    assert R.backtest_frozen_stage(peer_runtime["path"], [100], runner_fn=runner) == {"ok": True}
    assert calls == [1]
    with pytest.raises(RuntimeError):
        _freeze(peer_runtime)


def test_production_cache_loader_rejects_changed_bytes(peer_runtime, monkeypatch):
    """元数据未变但 chunk 字节变化时，真实 split 加载入口必须拒绝。"""
    root = peer_runtime["path"]
    for label, mf in (("sae", C.SAE_CACHE_MANIFEST),
                      ("peer", root / "peer_cache_manifest.json")):
        d = root / label / "W3"
        d.mkdir(parents=True)
        p = d / ("W3_20250102.npz" if label == "sae" else "peer_W3_20250102.npz")
        p.write_bytes(b"original")
        mf.write_text(json.dumps({"splits": {"W3": {"chunks": [
            {"file": p.name, "sha256": sha256_file(p)}]}}}))
        if label == "sae":
            p.write_bytes(b"tampered")
    monkeypatch.setattr(PD, "load_day", lambda *a, **k: _day())
    with pytest.raises(RuntimeError):
        PD.load_split_days("W3", {}, sae_dir=root / "sae", peer_dir=root / "peer")


def test_orphan_head_blocks_all_training(peer_runtime, monkeypatch):
    """另一臂存在中断产物时，不能先训练第一臂再发现问题。"""
    root = peer_runtime["path"]
    T.head_file(root, "ON", 101, 1).write_bytes(b"partial")
    monkeypatch.setattr(PD, "load_frozen_stats", lambda: {"d_in": 16})
    monkeypatch.setattr(PD, "load_split_days", lambda *a, **k: [_day()])
    monkeypatch.setattr(T, "train_arm", lambda *a, **k: pytest.fail("不允许训练任何臂"))
    with pytest.raises(RuntimeError):
        R._train_missing([101], "pilot")
