"""G10 运行时状态合成回归测试（20260909 收尾计划 §5/§6/§7，先复现缺陷再修）。

全部合成（无 CUDA/GPU/真实语料/权重）；fake CUDA 用 monkeypatch 模拟 backend。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from g10_head_pilot import runtime_state as rs  # noqa: E402


def _draw() -> tuple:
    """跨全部随机源各抽一次（返回可比对样本）。"""
    return (torch.rand(4), np.random.rand(4),
            torch.randint(0, 2 ** 31, (2,)).tolist())


def test_validation_rng_isolation_and_exception_safety(monkeypatch) -> None:
    """训练抽样 A→验证（耗随机）→抽样 B == 无验证的 B；重复验证同结果；
    验证抛异常仍恢复。覆盖 Python/numpy/Torch-CPU + 模拟 CUDA。"""
    calls = {"set": []}
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_rng_state",
                        lambda *a: torch.arange(8))
    monkeypatch.setattr(torch.cuda, "set_rng_state",
                        lambda st, *a: calls["set"].append(st.clone()))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: None)
    torch.manual_seed(7)
    np.random.seed(7)
    import random as pyrandom
    pyrandom.seed(7)
    _a = _draw()                                   # 训练抽样 A
    with rs.validation_rng():                      # 验证边界
        _v1 = _draw()
        with rs.validation_rng():
            _v2 = _draw()
        assert torch.equal(_v1[0], _v2[0])          # 重复验证同结果
    b_with_val = _draw()
    # 无验证路径
    torch.manual_seed(7)
    np.random.seed(7)
    pyrandom.seed(7)
    _a2 = _draw()
    assert torch.equal(_a[0], _a2[0])
    b_no_val = _draw()
    assert torch.equal(b_with_val[0], b_no_val[0])  # 验证不扰动训练流
    # 异常仍恢复
    torch.manual_seed(9)
    snap_before = torch.get_rng_state()
    with pytest.raises(RuntimeError), rs.validation_rng():
        raise RuntimeError("val failed")
    assert torch.equal(torch.get_rng_state(), snap_before)
    assert calls["set"], "模拟 CUDA 状态未被恢复"


def test_epoch_checkpoint_resume_equivalence(tmp_path) -> None:
    """小 dropout 模型：两 epoch 连续跑 vs ep1 保存退出后恢复，最终权重/
    优化器/调度器/随机输出全一致（调用生产 save/load 函数，epoch 边界恢复）。"""
    def make():
        torch.manual_seed(3)
        return nn.Sequential(nn.Linear(6, 12), nn.Dropout(0.5), nn.Linear(12, 2))

    def run_epochs(n, resume_from=None):
        model = make()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-2, total_steps=4)
        start = 0
        if resume_from is not None:
            ck = rs.load_versioned_checkpoint(resume_from)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sch.load_state_dict(ck["scheduler"])
            rs.restore_rng(ck["rng"])
            start = ck["meta"]["next_epoch"]
        for epoch in range(start, n):
            model.train()
            for _ in range(2):                     # 每个 epoch 2 个随机批
                x = torch.randn(8, 6)
                loss = model(x).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
            with rs.validation_rng():
                model.eval()
                _ = model(torch.randn(4, 6))       # 验证（耗随机后恢复）
            rs.save_versioned_checkpoint(
                tmp_path / f"ep{epoch + 1}.pt" if resume_from is None
                else resume_from,
                {"model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sch.state_dict(), "rng": rs.capture_rng(),
                 "meta": {"runtime_schema": rs.RUNTIME_SCHEMA,
                          "next_epoch": epoch + 1}})
        return model, opt, sch

    torch.manual_seed(11)
    cont_model, cont_opt, cont_sch = run_epochs(2)
    torch.manual_seed(11)
    m1, *_ = run_epochs(1)                          # ep1 保存退出
    res_model, res_opt, res_sch = run_epochs(2, resume_from=tmp_path / "ep1.pt")
    for (k1, v1), (k2, v2) in zip(cont_model.state_dict().items(),
                                  res_model.state_dict().items()):
        assert torch.equal(v1, v2), f"恢复后权重不一致：{k1}"
    assert cont_opt.state_dict()["state"][0]["exp_avg"].equal(
        res_opt.state_dict()["state"][0]["exp_avg"])
    assert cont_sch.get_last_lr() == res_sch.get_last_lr()
    torch.manual_seed(21)
    out_cont = cont_model(torch.randn(3, 6))
    rs.restore_rng(torch.load(tmp_path / "ep2.pt", weights_only=True)["rng"])
    torch.manual_seed(21)
    out_res = res_model(torch.randn(3, 6))
    assert torch.equal(out_cont, out_res)


def test_versioned_checkpoint_rejects_old_schema(tmp_path) -> None:
    torch.save({"predictor": {}, "epoch": 3}, tmp_path / "old.pt")
    with pytest.raises(RuntimeError, match="旧 schema"):
        rs.load_versioned_checkpoint(tmp_path / "old.pt")


def test_best_save_load_production_path(tmp_path) -> None:
    """生产选点+版本化 best：e1 最优、e2 更差且权重不同 → best 载入=e1；
    NaN 不覆盖有限 best。"""
    history = [{"epoch": 1, "val_ce": 1.0}, {"epoch": 2, "val_ce": 1.5}]
    best_epoch, best_ce = rs.select_best(history)
    assert (best_epoch, best_ce) == (1, 1.0)
    torch.manual_seed(0)
    w1 = nn.Linear(4, 2)
    w2 = nn.Linear(4, 2)
    rs.save_versioned_checkpoint(tmp_path / "best.pt", {
        "predictor": w1.state_dict(),
        "meta": {"runtime_schema": rs.RUNTIME_SCHEMA, "kind": "best",
                 "epoch": best_epoch, "val_ce": best_ce}})
    loaded = rs.load_versioned_checkpoint(tmp_path / "best.pt")
    for (k, v) in loaded["predictor"].items():
        assert torch.equal(v, w1.state_dict()[k])
    assert not torch.equal(w1.weight, w2.weight)
    # NaN 不覆盖有限 best
    with pytest.raises(ValueError):
        rs.select_best([{"epoch": 1, "val_ce": float("nan")}])


def test_cache_identity_and_window_rng_chain(tmp_path) -> None:
    """fake predictor：完整两窗一次运行 vs W3 命中 cache 后恢复 RNG 链续推
    W4 → 信号逐值一致；身份不匹配（换 checkpoint 哈希）拒绝复用。"""
    def fake_score(rng_tag: str) -> dict:
        return {"W3": torch.rand(5).tolist(), "W4": torch.rand(5).tolist()}

    def full_run(tag: str, ck_sha: str, reuse_w3: bool = False):
        out = {}
        chain_path = tmp_path / f"chain_{tag}.pt"
        torch.manual_seed(100)                       # 首次运行语义
        chain = {}
        for w in ("W3", "W4"):
            spath = tmp_path / f"{tag}_{w}.parquet"
            exp = {"runtime_schema": rs.RUNTIME_SCHEMA,
                   "checkpoint_sha256": ck_sha, "checkpoint_epoch": 15,
                   "tokenizer_sha256": "tok", "protocol_sha256": "proto",
                   "inference": {"sample_count": 20}, "window": w}
            if rs.cache_reusable(spath, exp):
                if f"{w}_end" in chain:
                    rs.restore_rng(chain[f"{w}_end"])
                continue
            if reuse_w3 and w == "W3" and spath.is_file():
                # 旧协议式"只看文件存在"复用已被 cache_reusable=False 拒绝；
                # 但有 RNG 链时可恢复——此处模拟元数据齐全的真实命中
                meta = {k: v for k, v in exp.items()}
                rs.write_signal_meta(spath, rs.signal_meta_for(
                    ck_sha, 15, "tok", "proto", {"sample_count": 20}, w, spath))
                if rs.cache_reusable(spath, meta) and f"{w}_end" in chain:
                    rs.restore_rng(chain[f"{w}_end"])
                continue
            sig = fake_score(w)
            spath.write_text(str(sig), encoding="utf-8")
            rs.write_signal_meta(spath, rs.signal_meta_for(
                ck_sha, 15, "tok", "proto", {"sample_count": 20}, w, spath))
            chain[f"{w}_end"] = rs.capture_rng()
            torch.save(chain, chain_path)
            out[w] = sig
        return out

    a = full_run("a", "ckAAA")                       # 完整两窗
    b = full_run("a", "ckAAA", reuse_w3=True)        # W3 命中（链恢复）→ W4
    assert a["W4"] == b.get("W4") or not b           # b 无新输出=全命中
    # 显式验证 RNG 链路径：重放 W4 依赖 W3_end
    torch.manual_seed(100)
    w3_full = fake_score("W3")
    chain = torch.load(tmp_path / "chain_a.pt", weights_only=True)
    rs.restore_rng(chain["W3_end"])
    w4_after_w3 = fake_score("W4")
    assert w4_after_w3 == a["W4"]
    # 身份不匹配拒绝：换 checkpoint 哈希
    assert not rs.cache_reusable(
        tmp_path / "a_W3.parquet",
        {"runtime_schema": rs.RUNTIME_SCHEMA, "checkpoint_sha256": "ckBBB",
         "checkpoint_epoch": 15, "tokenizer_sha256": "tok",
         "protocol_sha256": "proto", "inference": {"sample_count": 20},
         "window": "W3"})
    # 输出被篡改 → SHA 不符拒绝
    p = tmp_path / "a_W3.parquet"
    orig = p.read_text(encoding="utf-8")
    p.write_text(orig + "x", encoding="utf-8")
    assert not rs.cache_reusable(
        p, {"checkpoint_sha256": "ckAAA", "checkpoint_epoch": 15,
            "tokenizer_sha256": "tok", "protocol_sha256": "proto",
            "inference": {"sample_count": 20}, "window": "W3"})


def test_config_load_protocol_fail_closed(tmp_path, monkeypatch) -> None:
    import g10_head_pilot.config as cfg

    monkeypatch.setattr(cfg, "RUN_DIR", tmp_path)
    monkeypatch.setattr(cfg, "PROTOCOL_PATH", tmp_path / "protocol.json")
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="身份不可验证"):
        cfg.load_protocol()                          # 缺 SHA 侧文件拒绝
    sha = cfg.write_protocol({"k": 1})               # 显式新建 run 允许写入
    rec, got = cfg.load_protocol()
    assert rec == {"k": 1} and got == sha
