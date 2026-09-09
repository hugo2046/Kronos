"""LoRA-mean 训练（计划 §3）：G1 同源生成 CE，仅 LoRA A/B 更新。

批次流/验证隔离复用 G10 已验证语义：QlibDataset 自持采样 RNG、
DistributedSampler(seed0)、tokenizer 在线 encode（eval/no_grad）、
``use_teacher_forcing=False``、AdamW(0.9,0.95,wd0.1)+OneCycle(4e-5)+clip3、
FP32、batch50、15×2000 更新、验证 400 批固定采样（``validation_rng`` 隔离，
提取自 g10 runtime_state 的已验证纯函数）。本轮**无断点续训**：失败保留
原 run 报告剩余预算，不自动重训。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from g10_head_pilot.runtime_state import select_best, validation_rng
from lora_mean_pilot.adapter import (
    adapter_state, base_state_hash, inject_lora, save_adapter,
)

REPO = Path(__file__).resolve().parent.parent
sys_paths = [str(REPO / "finetune"), str(REPO)]


def _paths() -> None:
    import sys
    for p in sys_paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def make_config_cls():
    """ashares 语料软链 Config（同 G10 语义，输出指向本包 data）。"""
    _paths()
    from finetune_suite.config import Config

    class LoRAConfig(Config):
        def __init__(self):
            super().__init__()
            self.dataset_path = str(Path(__file__).resolve().parent
                                    / "data" / "corpus_link")
            self.use_comet = False

    link = Path(LoRAConfig().dataset_path)
    link.mkdir(parents=True, exist_ok=True)
    src = Path("/home/user/workspace/Kronos/finetune_suite/data/ashares")
    for split in ("train_data.pkl", "val_data.pkl"):
        dst = link / split
        if not dst.exists():
            dst.symlink_to(src / split)
    return LoRAConfig


def train_lora(seed: int, *, out_dir: Path, device: str = "cuda:0",
               g1_tokenizer_path: Path, g1_predictor_path: Path,
               epochs: int = 15, smoke: bool = False, smoke_steps: int = 2):
    """单 seed LoRA 训练：G1 权重冻结 + LoRA 注入 → 生成 CE。"""
    _paths()
    import dataset as official_dataset
    from finetune.utils.training_utils import set_seed
    from model.kronos import Kronos, KronosTokenizer

    set_seed(seed, 0)
    cfg_cls = make_config_cls()
    official_dataset.Config = cfg_cls
    from dataset import QlibDataset

    tokenizer = KronosTokenizer.from_pretrained(str(g1_tokenizer_path))
    tokenizer.eval().to(device)
    predictor = Kronos.from_pretrained(str(g1_predictor_path)).to(device)
    trainable = inject_lora(predictor, rank=8, alpha=8, seed=seed)
    n_trainable = sum(p.numel() for p in predictor.parameters()
                      if p.requires_grad)
    base_sha_before = base_state_hash(predictor)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = QlibDataset("train"), QlibDataset("val")
    t_sampler = DistributedSampler(train_ds, num_replicas=1, rank=0,
                                   shuffle=True)
    v_sampler = DistributedSampler(val_ds, num_replicas=1, rank=0,
                                   shuffle=False)
    t_loader = DataLoader(train_ds, batch_size=50, sampler=t_sampler,
                          num_workers=2, pin_memory=True, drop_last=True)
    v_loader = DataLoader(val_ds, batch_size=50, sampler=v_sampler,
                          num_workers=2, pin_memory=True, drop_last=False)
    opt = torch.optim.AdamW([p for p in predictor.parameters()
                             if p.requires_grad], lr=4e-5, betas=(0.9, 0.95),
                            weight_decay=0.1)
    n_epochs = 1 if smoke else epochs
    steps_per = smoke_steps if smoke else len(t_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=4e-5, total_steps=15 * steps_per, pct_start=0.03,
        div_factor=10)

    history: list[dict] = []
    compute = 0.0
    for epoch in range(n_epochs):
        t0 = time.time()
        predictor.train()
        t_sampler.set_epoch(epoch)
        train_ds.set_epoch_seed(epoch * 10000)
        loss_sum, n_steps = 0.0, 0
        for bx, bstamp in t_loader:
            bx = bx.to(device, non_blocking=True)
            bstamp = bstamp.to(device, non_blocking=True)
            with torch.no_grad():
                s1, s2 = tokenizer.encode(bx, half=True)
            logits = predictor(s1[:, :-1], s2[:, :-1], bstamp[:, :-1, :])
            loss, _, _ = predictor.head.compute_loss(
                logits[0], logits[1], s1[:, 1:], s2[:, 1:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in predictor.parameters() if p.requires_grad], 3.0)
            opt.step()
            sched.step()
            loss_sum += float(loss.item())
            n_steps += 1
            if smoke and n_steps >= smoke_steps:
                break
        predictor.eval()
        val_ds.set_epoch_seed(0)
        vl, vb = 0.0, 0
        with validation_rng(), torch.no_grad():
            for bx, bstamp in v_loader:
                bx = bx.to(device, non_blocking=True)
                bstamp = bstamp.to(device, non_blocking=True)
                s1, s2 = tokenizer.encode(bx, half=True)
                logits = predictor(s1[:, :-1], s2[:, :-1], bstamp[:, :-1, :])
                vloss, _, _ = predictor.head.compute_loss(
                    logits[0], logits[1], s1[:, 1:], s2[:, 1:])
                vl += float(vloss.item())
                vb += 1
                if smoke and vb >= 2:
                    break
                if not smoke and vb >= 400:
                    break
        ce = vl / max(vb, 1)
        compute += time.time() - t0
        history.append({"epoch": epoch + 1,
                        "train_loss": loss_sum / max(n_steps, 1),
                        "val_ce": ce, "seconds": compute})
        if not smoke:   # 每 epoch 唯一文件 + SHA（best 只是索引）
            sha = save_adapter(
                out_dir / f"adapter_epoch_{epoch + 1:03d}.pt",
                {"adapter": adapter_state(predictor), "epoch": epoch + 1,
                 "val_ce": ce, "seed": seed,
                 "base_state_sha256": base_sha_before,
                 "trainable_names": trainable})
        logger.info(f"[LoRA{seed}] epoch {epoch + 1:02d} train="
                    f"{history[-1]['train_loss']:.4f} val_CE={ce:.4f} "
                    f"({vb} 批, {compute:.0f}s 累计)")
    assert base_state_hash(predictor) == base_sha_before, "基底权重被改动！"
    best_epoch, best_ce = select_best(history)
    (out_dir / "history.json").write_text(
        json.dumps({"seed": seed, "n_trainable": n_trainable,
                    "trainable_names": trainable,
                    "base_state_sha256": base_sha_before,
                    "best_epoch": best_epoch, "best_ce": best_ce,
                    "history": history, "compute_seconds": compute},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    return {"best_epoch": best_epoch, "best_ce": best_ce,
            "best_adapter": str(out_dir / f"adapter_epoch_{best_epoch:03d}.pt"),
            "n_trainable": n_trainable, "base_state_sha256": base_sha_before,
            "history": history, "compute_seconds": compute}
