"""R1 六训编排（计划 §2/§4.2，20260903 L1与R1计划）：缓存 + IC 损失训练。

**臂/种子冻结**：R-lin（H-lin 同构，833 参数）× {42,43,44} +
R-kda（H-kda 同构，1.21M）× {42,43,44}，共 6 训。

**唯一变量 = 损失函数**（MSE → IC）之外的一切逐字镜像 G5（``g5_head.run_g5_head``）：
同一冻结 G1 底座隐状态缓存（只读，经 ``cache_loader`` 安全装载）、同一训练协议
常量（AdamW 3e-4 / wd 0.01 / epochs ≤50 / patience 5，逐字 import 零漂移）、
同一 train/早停窗（2022-01-04~2023-12-15 / 2024-01-02~2024-06-14）、同一早停
判据（早停段日均 RankIC，直接复用 ``g5_head.run_g5_head._evaluate_rankic_cached``）。

**批结构随损失按 qlib 惯例改为每日截面**（IC 损失的定义要求）：一个优化步 =
一个交易日的全截面（~280 只）；日边界由 ``day_groups`` 从数据层确定性重建并与
冻结缓存逐位对拍。训练只看 train/早停曲线（§5：评估数字在 6 checkpoint 定型前
不看——本模块无 eval 入口）。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m r1_objective.run_r1_train
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# —— 协议常量：逐字镜像 g5_head（G5_* 即冻结源，test_r1_protocol_frozen 对拍防漂移）——
from g5_head.run_g5_head import (
    G5_BATCH, G5_EPOCHS, G5_ES_END, G5_ES_START, G5_LR, G5_PATIENCE,
    G5_PURGE, G5_TRAIN_END, G5_TRAIN_START, G5_WD,
)

R1_LR = G5_LR                # 3e-4
R1_WD = G5_WD                # 0.01
R1_BATCH = G5_BATCH          # 128（协议镜像对拍用；IC 批结构下批 = 每日截面）
R1_EPOCHS = G5_EPOCHS        # 50
R1_PATIENCE = G5_PATIENCE    # 5
R1_TRAIN_START = G5_TRAIN_START
R1_TRAIN_END = G5_TRAIN_END
R1_ES_START = G5_ES_START
R1_ES_END = G5_ES_END
R1_PURGE = G5_PURGE

# —— 臂 / 种子 / 窗（计划 §2 表冻结）——
R1_ARMS = ("R-lin", "R-kda")
R1_SEEDS = (42, 43, 44)
R1_WINDOWS = ("backtest", "2025h2")

LOSS_NAME = "IC(-Pearson, 每日截面批内)"
R1_BATCHING = "per-day-cross-section"

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"


def _make_head(arm: str, backbone) -> nn.Module:
    """按臂名构造头（与 G5 头族一一对应，同构只读 import）。"""
    from cross_section_kda.models import B2LinearProbe
    from g5_head.heads import G5KdaHead

    if arm == "R-kda":
        return G5KdaHead(backbone)
    if arm == "R-lin":
        return B2LinearProbe(backbone)
    raise ValueError(f"未知臂 {arm}")


def _make_loss_fn():
    """R1 损失：IC（每日截面批内 −Pearson）。与 MSE 分支互斥（仿射不变性）。"""
    from r1_objective.ic_loss import ic_loss

    return ic_loss


def train_one(arm: str, backbone, cache: dict, day_groups: list[dict],
              *, seed: int, device: str) -> tuple[nn.Module, dict]:
    """单臂单种子训练（缓存 hidden → 每日截面 IC 损失；早停协议逐字同 G5）。"""
    from cross_section_kda.train import set_seed
    from g5_head.heads import decode_score
    from g5_head.run_g5_head import _evaluate_rankic_cached
    from r1_objective.ic_loss import ic_loss

    set_seed(seed)
    model = _make_head(arm, backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{arm} s{seed}] 可训练参数 {n_trainable:,} | lr={R1_LR} wd={R1_WD} "
                f"epochs≤{R1_EPOCHS} patience={R1_PATIENCE} | 损失={LOSS_NAME}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=R1_LR, weight_decay=R1_WD)
    loss_fn = _make_loss_fn()

    xs = cache["train"]["hidden"]           # [Ntr, 90, 832]（CPU，按日切片上 GPU）
    ys = cache["train"]["y"]                # [Ntr]
    es_days = cache["es"]
    n_days = len(day_groups)
    logger.info(f"[{arm} s{seed}] 训练样本 {xs.size(0)} 条 / {n_days} 个截面日"
                f"（G1 底座缓存 hidden，g5 只读）")

    best_es = -np.inf
    best_state = None
    history: list[dict] = []
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, R1_EPOCHS + 1):
        model.train()
        day_order = torch.randperm(n_days)
        epoch_loss = 0.0
        n_steps = 0
        for di in day_order.tolist():
            g = day_groups[di]
            h = xs[g["start"]:g["end"]].to(device)
            y = ys[g["start"]:g["end"]].to(device)
            opt.zero_grad()
            score = decode_score(model, h)
            loss = loss_fn(score, y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        train_loss = epoch_loss / max(n_steps, 1)

        es_rankic = _evaluate_rankic_cached(model, es_days, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "es_rankic": es_rankic})
        improved = np.isfinite(es_rankic) and es_rankic > best_es
        marker = "★" if improved else ""
        logger.info(f"[{arm} s{seed}] epoch {epoch:02d}/{R1_EPOCHS} "
                    f"train_ICloss={train_loss:.5f} es_RankIC={es_rankic:+.4f} {marker}")
        if improved:
            best_es = es_rankic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= R1_PATIENCE:
                logger.info(f"[{arm} s{seed}] 早停：{R1_PATIENCE} 个 epoch 无改善，"
                            f"最佳 es_RankIC={best_es:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    logger.info(f"[{arm} s{seed}] 训练完成 用时 {elapsed:.1f}s 最佳 es_RankIC={best_es:+.4f}")

    info = {
        "arm": arm, "seed": seed, "n_trainable": n_trainable, "loss": LOSS_NAME,
        "batching": R1_BATCHING, "best_es_rankic": best_es, "history": history,
        "n_epochs_run": len(history), "elapsed_s": elapsed,
        "cfg": {"lr": R1_LR, "weight_decay": R1_WD, "epochs": R1_EPOCHS,
                "patience": R1_PATIENCE, "seed": seed, "n_train_days": n_days},
    }
    return model, info


def _save_ckpt(model: nn.Module, name: str, info: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = DATA_DIR / f"{name}_best.pt"
    torch.save({"state_dict": model.state_dict(), "info": info}, ckpt)
    (DATA_DIR / f"{name}_history.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    logger.info(f"[{name}] checkpoint → {ckpt.name} | history → {name}_history.json")
    return ckpt


def run_train(device: str = "cuda:0") -> dict:
    """4.2 六训：R-lin×3 + R-kda×3（IC 损失）。"""
    from g5_head.backbone_g1 import load_g1_backbone
    from r1_objective.cache_loader import load_hidden_cache
    from r1_objective.day_groups import HIDDEN_CACHE_PATH, load_day_groups

    backbone = load_g1_backbone(device)
    cache = load_hidden_cache(HIDDEN_CACHE_PATH)
    doc = load_day_groups()  # 首次运行触发构建 + 与冻结缓存逐位对拍
    day_groups = doc["groups"]
    logger.info(f"日边界 {doc['verify']['n_days']} 日 / {doc['n_train_samples']} 样本"
                f"（对拍 max|Δy|={doc['verify']['max_abs_diff']:.1e}）")

    jobs = [(arm, seed) for arm in R1_ARMS for seed in R1_SEEDS]
    records = {}
    for arm, seed in jobs:
        name = f"{arm}_s{seed}"
        logger.info("=" * 70)
        logger.info(f"4.2 训练 {name}（唯一变量=损失：MSE→IC）")
        logger.info("=" * 70)
        model, info = train_one(arm, backbone, cache, day_groups, seed=seed, device=device)
        _save_ckpt(model, name, info)
        records[name] = {"n_trainable": info["n_trainable"],
                         "loss": LOSS_NAME,
                         "best_es_rankic": info["best_es_rankic"],
                         "n_epochs_run": info["n_epochs_run"],
                         "elapsed_s": info["elapsed_s"]}

    out = {"experiment": "r1_stage1_train", "date": "2026-09-03", "records": records}
    (DATA_DIR / "r1_stage1_records.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    logger.info("==== 阶段 4.2 六训全部完成 ====")
    return out


def main() -> int:
    run_train("cuda:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
