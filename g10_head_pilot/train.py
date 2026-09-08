"""G10H-pilot 训练：A0 全参数 / A1 仅 DualHead 投影（计划 §3）。

复用官方 ``finetune_suite.train_predictor`` 的世界语义（world_size=1）：
QlibDataset 自持采样 RNG、DistributedSampler(seed=0, 按 epoch 重洗)、
tokenizer 在线 encode（eval/no_grad）、``use_teacher_forcing=False`` 的
DualHead CE、AdamW(0.9,0.95,wd0.1)+OneCycle(4e-5,pct.03,div10)+clip3、
FP32、num_workers=2。两臂先后各自**完整重置**（set_seed(100) → 重建数据
→ 重载官方 Kronos-base 起点），批次流由数据侧 RNG 决定、与模型随机流隔离，
配对以**逐 epoch 批次内容摘要**审计（比样本键更直接）。

纪律：不运行写 G1 目录的 ``train_g1.py``；A1 参数范围精确匹配
``A1_ALLOWED``；每 epoch 保存可续跑 checkpoint；验证前保存训练 RNG、
验证采样固定（set_epoch_seed(0)）、结束后恢复。
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

from g10_head_pilot.config import (
    A1_ALLOWED, ADAM_BETAS, ADAM_WD, BATCH, EPOCHS, GRAD_CLIP, LR,
    N_VAL_BATCHES_PER_EPOCH, N_TRAIN_ITER, N_VAL_ITER, NUM_WORKERS,
    PCT_START, DIV_FACTOR, RUN_DIR, SEED, sha256_file, state_hash,
)


def apply_param_mask(predictor, arm: str) -> list[str]:
    """按计划 §3 精确限制梯度更新；返回可训练参数名排序列表。"""
    allowed = set(A1_ALLOWED)
    names = {name for name, _ in predictor.named_parameters()}
    assert allowed <= names, f"A1 清单含不存在参数：{allowed - names}"
    for name, parameter in predictor.named_parameters():
        parameter.requires_grad_(arm == "A0" or name in allowed)
    return sorted(n for n, p in predictor.named_parameters() if p.requires_grad)


class _BatchDigest:
    """逐批累积的批次内容摘要（配对审计：两臂同 epoch 摘要必须一致）。"""

    def __init__(self) -> None:
        import hashlib

        self._h = hashlib.sha256()
        self.n = 0

    def update(self, x: torch.Tensor, stamp: torch.Tensor) -> None:
        self._h.update(x.detach().cpu().numpy().tobytes())
        self._h.update(stamp.detach().cpu().numpy().tobytes())
        self.n += 1

    def hexdigest(self) -> str:
        return self._h.hexdigest()[:16]


def _finetune_paths() -> None:
    import sys
    root = Path(__file__).resolve().parent.parent
    for p in (str(root / "finetune"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def make_g10h_config_cls():
    """指向 ashares 语料软链的 Config 子类（运行时注入官方 dataset 模块）。"""
    _finetune_paths()
    from finetune_suite.config import Config
    from g10_head_pilot.config import ASHARES_DATA

    class G10HConfig(Config):
        """G10H：ashares 同源语料 + 本包输出目录，超参零改动。"""

        def __init__(self):
            super().__init__()
            self.dataset_path = str(Path(__file__).resolve().parent
                                    / "data" / "corpus_link")
            self.use_comet = False

    link = Path(G10HConfig().dataset_path)
    link.mkdir(parents=True, exist_ok=True)
    for split in ("train_data.pkl", "val_data.pkl"):
        dst = link / split
        if not dst.exists():
            dst.symlink_to(ASHARES_DATA / split)
    return G10HConfig


def set_seed_100() -> None:
    _finetune_paths()
    from utils.training_utils import set_seed

    set_seed(SEED, 0)


def wait_out_guard(modules: list, on_boundary: str) -> None:
    """16:30 登记 cron 守卫：epoch 边界把模块卸到 CPU 释放 GPU。"""
    from g10_head_pilot.config import in_guard

    if not in_guard():
        return
    logger.warning(f"[guard] {on_boundary} 进入登记守卫窗，卸载 GPU")
    for m in modules:
        m.to("cpu")
    torch.cuda.empty_cache()
    n = 0
    while in_guard():
        time.sleep(30)
        n += 1
    for m in modules:
        m.to("cuda:0")
    logger.info(f"[guard] 守卫结束（{n * 0.5:.0f}min），回载继续")


def train_arm(
    arm: str, *, device: str = "cuda:0", smoke: bool = False,
    smoke_steps: int = 2,
) -> dict:
    """单臂完整训练（两臂先后调用，各自从头重置随机世界）。"""
    _finetune_paths()
    import dataset as official_dataset
    from model.kronos import Kronos, KronosTokenizer

    from g10_head_pilot.config import G1_TOKENIZER, OFFICIAL_PREDICTOR

    set_seed_100()
    cfg_cls = make_g10h_config_cls()
    official_dataset.Config = cfg_cls
    from dataset import QlibDataset

    tokenizer = KronosTokenizer.from_pretrained(str(G1_TOKENIZER))
    tokenizer.eval().to(device)
    tok_sha = sha256_file(G1_TOKENIZER / "model.safetensors")

    predictor = Kronos.from_pretrained(OFFICIAL_PREDICTOR).to(device)
    init_hash = state_hash(predictor.state_dict())
    trainable = apply_param_mask(predictor, arm)
    n_trainable = sum(p.numel() for p in predictor.parameters() if p.requires_grad)

    train_ds, val_ds = QlibDataset("train"), QlibDataset("val")
    train_sampler = DistributedSampler(train_ds, num_replicas=1, rank=0,
                                       shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=1, rank=0,
                                      shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, sampler=val_sampler,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=False)

    steps_per_epoch = smoke_steps if smoke else len(train_loader)
    epochs = 1 if smoke else EPOCHS
    opt = torch.optim.AdamW([p for p in predictor.parameters() if p.requires_grad],
                            lr=LR, betas=ADAM_BETAS, weight_decay=ADAM_WD)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * steps_per_epoch,
        pct_start=PCT_START, div_factor=DIV_FACTOR)

    arm_dir = RUN_DIR / ("smoke" if smoke else arm)
    arm_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = arm_dir / "resume.pt"
    history: list[dict] = []
    start_epoch = 0
    best_ce, best_epoch = float("inf"), None
    if ckpt_path.is_file() and not smoke:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if ck.get("done"):
            logger.info(f"[{arm}] 已完成，跳过")
            return {"best_epoch": ck["best_epoch"], "best_ce": ck["best_ce"],
                    "skipped": True}
        predictor.load_state_dict(ck["predictor"])
        opt.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        history = ck["history"]
        best_ce, best_epoch = ck["best_ce"], ck["best_epoch"]
        logger.info(f"[{arm}] 续跑 epoch {start_epoch}")
    logger.info(f"[{arm}] 可训练 {n_trainable:,} 参数（{len(trainable)} 张量）| "
                f"起点哈希 {init_hash[:12]}… | tokenizer {tok_sha[:12]}… | "
                f"训练 {len(train_loader)} 批/epoch × {epochs}")

    lr_track: list[float] = []
    compute_s = 0.0
    for epoch in range(start_epoch, epochs):
        wait_out_guard([predictor, tokenizer], f"[{arm}] epoch {epoch + 1}")
        t0 = time.time()
        predictor.train()
        train_sampler.set_epoch(epoch)
        train_ds.set_epoch_seed(epoch * 10000)
        train_rng = (random.getstate(), torch.get_rng_state(),
                     np.random.get_state())
        torch.manual_seed(SEED)          # 验证前重置全局 RNG（计划 §3 约定）
        digest = _BatchDigest()
        n_steps, loss_sum = 0, 0.0
        for bx, bstamp in train_loader:
            digest.update(bx, bstamp)
            bx = bx.to(device, non_blocking=True)
            bstamp = bstamp.to(device, non_blocking=True)
            with torch.no_grad():
                s1, s2 = tokenizer.encode(bx, half=True)
            logits = predictor(s1[:, :-1], s2[:, :-1], bstamp[:, :-1, :])
            loss, _a, _b = predictor.head.compute_loss(
                logits[0], logits[1], s1[:, 1:], s2[:, 1:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP)
            opt.step()
            scheduler.step()
            loss_sum += float(loss.item())
            n_steps += 1
            lr_track.append(opt.param_groups[0]["lr"])
            if smoke and n_steps >= smoke_steps:
                break

        predictor.eval()
        val_ds.set_epoch_seed(0)
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for bx, bstamp in val_loader:
                bx = bx.to(device, non_blocking=True)
                bstamp = bstamp.to(device, non_blocking=True)
                s1, s2 = tokenizer.encode(bx, half=True)
                logits = predictor(s1[:, :-1], s2[:, :-1], bstamp[:, :-1, :])
                vloss, _, _ = predictor.head.compute_loss(
                    logits[0], logits[1], s1[:, 1:], s2[:, 1:])
                val_loss_sum += float(vloss.item())
                val_batches += 1
                if smoke and val_batches >= 2:
                    break
                if not smoke and val_batches >= N_VAL_BATCHES_PER_EPOCH:
                    break
        val_ce = val_loss_sum / max(val_batches, 1)
        elapsed = time.time() - t0
        compute_s += elapsed
        improved = val_ce < best_ce
        history.append({"epoch": epoch + 1,
                        "train_loss": loss_sum / max(n_steps, 1),
                        "val_ce": val_ce, "val_batches": val_batches,
                        "seconds": elapsed, "batch_digest": digest.hexdigest(),
                        "lr_last": lr_track[-1] if lr_track else None})
        marker = "★" if improved else ""
        logger.info(f"[{arm}] epoch {epoch + 1:02d}/{epochs} "
                    f"train={history[-1]['train_loss']:.4f} val_CE={val_ce:.4f} "
                    f"({val_batches} 批, {elapsed:.0f}s) 批摘要 "
                    f"{digest.hexdigest()[:8]} {marker}")
        if improved:
            best_ce, best_epoch = val_ce, epoch + 1
            if not smoke:
                torch.save(predictor.state_dict(), arm_dir / "best.pt")
        if not smoke:
            torch.save({
                "predictor": predictor.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch, "history": history,
                "best_ce": best_ce, "best_epoch": best_epoch,
                "init_hash": init_hash, "arm": arm, "done": False,
            }, ckpt_path)
        random.setstate(train_rng[0])
        torch.set_rng_state(train_rng[1])
        np.random.set_state(train_rng[2])

    import hashlib

    lr_sha = hashlib.sha256(
        b"".join(np.float64(v).tobytes() for v in lr_track)).hexdigest()[:16]
    out = {"arm": arm, "init_hash": init_hash, "n_trainable": n_trainable,
           "trainable_names": trainable, "best_ce": best_ce,
           "best_epoch": best_epoch, "history": history,
           "lr_sha": lr_sha, "compute_seconds": compute_s}
    if not smoke:
        assert best_epoch is not None, "无有限验证 CE（实验无效）"
        torch.save({**out, "done": True, "predictor": predictor.state_dict()},
                   arm_dir / "final.pt")
        # resume 标记完成：幂等跳过后续 --stage train
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ck["done"] = True
        torch.save(ck, ckpt_path)
        (arm_dir / "history.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8")
    return out
