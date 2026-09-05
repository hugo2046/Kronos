"""任务4（方案 §7）：统一训练器测试（stub 底座 + 可学规律合成数据，无 GPU）。

覆盖：loss 下降；中断恢复下一步与不中断一致；checkpoint 选择只能访问
train/val；D2 不能读取冻结末层 hidden 缓存；各臂损失组合正确。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from dhead_distill.config import DHeadConfig, replace_profile_budget
from dhead_distill.data import DayManifest, Sample, _calc_stamps, content_hash


def _learnable_manifest(n_days: int = 6, n_stocks: int = 40, seed: int = 3):
    """合成有明确可学规律的清单：y_real 由 close_t 的线性函数决定。

    头可从隐状态读出 close_t → 学到 y = 0.001 * close_t 的规律。
    """
    rng = np.random.default_rng(seed)
    m = DayManifest(split="train", pool="ashares", profile="pilot",
                    protocol="proto-hash-train", list_seed=20260905)
    base = pd.Timestamp("2024-06-03")
    cal = pd.bdate_range(base - pd.Timedelta(days=200), base + pd.Timedelta(days=30))
    for di in range(n_days):
        d = cal[100 + di]
        d_iso = d.strftime("%Y-%m-%d")
        x_cal = cal[11 + di:101 + di]
        y_cal = cal[101 + di:111 + di]
        m.x_stamp[d_iso] = _calc_stamps(x_cal)
        m.y_stamp[d_iso] = _calc_stamps(y_cal)
        m.x_cal[d_iso] = x_cal.values.astype("datetime64[D]")
        m.y_cal[d_iso] = y_cal.values.astype("datetime64[D]")
        for si in range(n_stocks):
            code = f"SZ{si:04d}"
            key = (d_iso, code)
            x = np.tile(np.linspace(1.0, 2.0, 90)[:, None], (1, 6))
            x[:, 3] = 100.0 + si  # close_t 可由窗口末行读出
            m.x_raw[key] = x.astype(np.float32)
            m.close_t[key] = float(x[-1, 3])
            y = np.full(10, 0.001 * (100.0 + si), dtype=np.float32)
            y = y + rng.normal(0, 1e-4, 10).astype(np.float32)
            m.samples.append(Sample(date=d, code=code, y_real=y, label_ok=True))
    m.content_hash = content_hash(m)
    return m


def _stub_backbone_and_head(d_model: int = 32):
    from dhead_distill.backbone import StudentBackbone
    from dhead_distill.head import MultiHorizonHead
    from tests.test_dhead_model import _StubKronos, _StubTokenizer

    torch.manual_seed(11)
    bb = StudentBackbone(_StubTokenizer(), _StubKronos(d_model=d_model, n_layers=4),
                         d_model_expected=d_model, n_trainable_layers=2)
    head = MultiHorizonHead(d_model=d_model, head_dim=16, n_heads=4, n_horizons=10,
                            calendar_cardinalities=(60, 24, 7, 32, 13))
    return bb, head


def _teacher_from_labels(m: DayManifest, noise: float = 0.0) -> np.ndarray:
    """教师目标 = 真实标签 + 小噪声（测试用确定性伪教师，3 组）。"""
    rng = np.random.default_rng(5)
    return np.stack([
        s.y_real + (rng.normal(0, noise, 10).astype(np.float32) if noise else 0.0)
        for s in m.samples
    ])[:, None, :].repeat(3, axis=1)  # [N, 3, 10]


def _cfg(**kw) -> DHeadConfig:
    import dataclasses

    return dataclasses.replace(
        replace_profile_budget(DHeadConfig(), per_day=40, min_per_day=8),
        max_epochs=kw.pop("max_epochs", 4), patience=kw.pop("patience", 3),
        lr=3e-3, **kw,
    )


def _make_trainer(tmp_path, monkeypatch, arm="D0", seed=42, max_epochs=4,
                  patience=3, teacher_noise=0.0, head=None, bb=None,
                  max_epochs_override=None, disable_early_stop=False,
                  art="art"):
    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / art))
    (tmp_path / "base").mkdir(exist_ok=True)
    from dhead_distill.train import UnifiedTrainer

    train_m = _learnable_manifest(n_days=6, n_stocks=40)
    val_m = _learnable_manifest(n_days=3, n_stocks=40, seed=4)
    bb, head_ = _stub_backbone_and_head()
    teach_tr = _teacher_from_labels(train_m, teacher_noise)
    teach_va = _teacher_from_labels(val_m, teacher_noise)
    cfg = _cfg(max_epochs=max_epochs, patience=patience)
    return UnifiedTrainer(
        arm=arm, cfg=cfg, backbone=bb, head=head or head_,
        scale=np.full(10, 0.01),
        train_manifest=train_m, val_manifest=val_m,
        train_teacher=teach_tr, val_teacher=teach_va,
        run_name=f"unit-{arm}-{seed}", seed=seed, device="cpu",
        max_epochs_override=max_epochs_override,
        disable_early_stop=disable_early_stop,
    )


def test_arm_loss_decreases(tmp_path, monkeypatch) -> None:
    """合成可学规律：训练若干 epoch 后训练损失显著低于首 epoch。"""
    tr = _make_trainer(tmp_path, monkeypatch, arm="D0", max_epochs=3)
    res = tr.fit()
    assert res.history[0]["train_loss"] > res.history[-1]["train_loss"] * 1.5 + 1e-4


def test_unknown_arm_rejected(tmp_path, monkeypatch) -> None:
    """未知臂显式报错。"""
    with pytest.raises(ValueError, match="未知臂"):
        _make_trainer(tmp_path, monkeypatch, arm="X99")


def test_checkpoint_resume_consistency(tmp_path, monkeypatch) -> None:
    """中断恢复：resume 后下一步（参数与损失）与不中断连续跑一致。"""
    # 连续跑 3 epoch
    tr_full = _make_trainer(tmp_path, monkeypatch, arm="D0", max_epochs=3,
                            patience=10, seed=99)
    res_full = tr_full.fit()

    # 断点跑：先 2 epoch（独立产物根），再 resume 到 3——恢复用
    # max_epochs_override 延长，cfg 协议不变（身份一致才许续写）
    tr_part = _make_trainer(tmp_path, monkeypatch, arm="D0", max_epochs=2,
                            patience=10, seed=99, art="art-resume")
    tr_part.fit()
    from dhead_distill.train import UnifiedTrainer

    tr_resume = UnifiedTrainer(
        arm="D0", cfg=_cfg(max_epochs=2, patience=10), backbone=tr_part.backbone,
        head=tr_part.head, scale=np.full(10, 0.01),
        train_manifest=tr_part.train_manifest, val_manifest=tr_part.val_manifest,
        train_teacher=tr_part.train_teacher, val_teacher=tr_part.val_teacher,
        run_name=tr_part.run_name, seed=99, device="cpu",
        max_epochs_override=3,
    )
    res_resume = tr_resume.fit()

    assert len(res_resume.history) == 3
    for a, b in zip(res_full.history, res_resume.history):
        assert a["train_loss"] == pytest.approx(b["train_loss"], abs=1e-6)
        assert a["val_loss"] == pytest.approx(b["val_loss"], abs=1e-6)
    fa = dict(tr_full.head.state_dict())
    fb = dict(tr_resume.head.state_dict())
    for k in fa:
        torch.testing.assert_close(fa[k], fb[k])


def test_selection_uses_only_train_val() -> None:
    """checkpoint 选择只许访问 train/val 指标：带回放收益的 history 被拒绝。"""
    from dhead_distill.train import select_best_epoch

    good = [
        {"epoch": 0, "train_loss": 1.0, "val_loss": 0.9, "val_task": 0.9},
        {"epoch": 1, "train_loss": 0.8, "val_loss": 0.7, "val_task": 0.7},
        {"epoch": 2, "train_loss": 0.7, "val_loss": 0.8, "val_task": 0.8},
    ]
    assert select_best_epoch(good) == 1  # val 最小；tie 取更早
    poisoned = [dict(g, replay_aer=0.5) for g in good]
    with pytest.raises(ValueError, match="非法指标"):
        select_best_epoch(poisoned)


def test_d2_rejects_hidden_cache(tmp_path, monkeypatch) -> None:
    """D2 禁止读取冻结末层 hidden 缓存：传入缓存必须报错。"""
    tr = _make_trainer(tmp_path, monkeypatch, arm="D0", max_epochs=1)
    from dhead_distill.train import UnifiedTrainer

    train_m = tr.train_manifest
    fake_cache = {d: torch.zeros(1) for d in
                  {s.date.strftime("%Y-%m-%d") for s in train_m.samples}}
    with pytest.raises(ValueError, match="缓存"):
        UnifiedTrainer(
            arm="D2", cfg=_cfg(max_epochs=1), backbone=tr.backbone,
            head=tr.head, scale=np.full(10, 0.01),
            train_manifest=train_m, val_manifest=tr.val_manifest,
            train_teacher=tr.train_teacher, val_teacher=tr.val_teacher,
            run_name="unit-d2-cache", seed=42, device="cpu",
            hidden_cache=fake_cache,
        )


def test_arm_loss_composition() -> None:
    """各臂任务损失组合：D0 纯 D；S/S-long = S+0.05I；D1/D2/D1-cont = 0.5S+0.5D+0.05I。"""
    from dhead_distill.train import arm_loss

    s, d, i = torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)
    assert arm_loss("D0", s, d, i).item() == pytest.approx(2.0)
    assert arm_loss("S", s, d, i).item() == pytest.approx(1.0 + 0.05 * 3.0)
    assert arm_loss("S-long", s, d, i).item() == pytest.approx(1.0 + 0.05 * 3.0)
    assert arm_loss("D1-cont", s, d, i).item() == pytest.approx(0.5 + 1.0 + 0.15)
    assert arm_loss("D1", s, d, i).item() == pytest.approx(0.5 + 1.0 + 0.15)
    assert arm_loss("D2", s, d, i).item() == pytest.approx(0.5 + 1.0 + 0.15)
    with pytest.raises(ValueError):
        arm_loss("X99", s, d, i)  # 未知臂禁用
