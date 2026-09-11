"""PATH1 训练：train-only 统计 + 双臂同构训练 + e0 候选（计划 §5）。

统计（``mu_g1/sigma_g1/sigma_e``）只用本轮 train 有效标签格计算（ddof0），
冻结后 val/dev_eval 不重拟合；σe 触数值地板停止实验。训练全程 CPU FP32、
AdamW lr1e-3 / betas(0.9,0.999) / wd0.01、clip_grad_norm3、batch_days4、
每日监督股 loss 均值再对日期等权；固定 50 epochs 无早停；未训练 e0 是
合法候选，按 val 日等权 MSE 最低选 best（严格更低才更新，并列取更早）。
所有阶段同时报告 G1 零残差 MSE。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from path_information import config as C
from path_information import model as PM


def fit_stats(train_days: list[dict]) -> dict:
    """只用 train 有效标签格拟合 ``mu_g1/sigma_g1/sigma_e``（ddof0）。

    :param train_days: 逐日 ``{"date", "shape"[N,H10], "s_g1"[N], "y"[N],
        "valid"[N]}``（shape 已是 path_shape 输出）。
    :raises ValueError: 有效格为空或 σe 触地板（退化停止，计划 §3/§5）。
    """
    s_parts, e_parts = [], []
    for d in train_days:
        v = np.asarray(d["valid"], dtype=bool)
        s_parts.append(np.asarray(d["s_g1"], dtype=np.float64)[v])
        e_parts.append((np.asarray(d["y"], dtype=np.float64)
                        - np.asarray(d["s_g1"], dtype=np.float64))[v])
    s_all = np.concatenate(s_parts) if s_parts else np.array([])
    e_all = np.concatenate(e_parts) if e_parts else np.array([])
    if s_all.size == 0:
        raise ValueError("train 无有效标签格")
    sigma_e = float(e_all.std())
    if sigma_e < C.STATS_FLOOR:
        raise ValueError(f"训练残差退化：std(e)={sigma_e:.3e} < 地板 "
                         f"{C.STATS_FLOOR:.0e}，停止实验")
    return {"mu_g1": float(s_all.mean()), "sigma_g1": float(s_all.std()),
            "sigma_e": sigma_e, "n_train_cells": int(s_all.size)}


def day_equal_weight_loss(head: PM.PathHead, days: list[dict], stats: dict,
                          arm: str | None = None) -> float:
    """每日监督股 loss 均值 → 日期等权（评价用，无梯度）。

    :param arm: None = 直接对给定 head 前向（测试用）；"PATH"/"MEAN" 按臂语义喂。
    """
    head.eval()
    per_day = []
    with torch.no_grad():
        for d in days:
            v = np.asarray(d["valid"], dtype=bool)
            if not v.any():
                continue
            shape = torch.from_numpy((np.asarray(d["shape"])
                                      / stats["sigma_e"]).astype(np.float32))
            if arm is not None:
                shape = PM.arm_input(shape, arm)
            b = torch.from_numpy(((np.asarray(d["s_g1"]) - stats["mu_g1"])
                                  / stats["sigma_g1"]).astype(np.float32))
            r_hat = head(shape, b).numpy()
            r = ((np.asarray(d["y"]) - np.asarray(d["s_g1"]))
                 / stats["sigma_e"]).astype(np.float32)
            per_day.append(float(np.mean((r_hat[v] - r[v]) ** 2)))
    return float(np.mean(per_day))


def g1_zero_residual_mse(days: list[dict], stats: dict) -> float:
    """G1 零残差基准 MSE（归一化）：``mean_daily(mean_cells(((y−s)/σe)²)``。"""
    per_day = []
    for d in days:
        v = np.asarray(d["valid"], dtype=bool)
        if not v.any():
            continue
        r = ((np.asarray(d["y"]) - np.asarray(d["s_g1"]))
             / stats["sigma_e"])
        per_day.append(float(np.mean(r[v] ** 2)))
    return float(np.mean(per_day))


def head_file(out_dir: Path, arm: str, seed: int, epoch: int) -> Path:
    return out_dir / f"head_{arm}_s{seed}_e{epoch:03d}.pt"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def train_arm(arm: str, train_days: list[dict], val_days: list[dict],
              stats: dict, out_dir: Path, seed: int = C.SEED,
              epochs: int = C.TRAIN["epochs"], lr: float | None = None,
              context: dict | None = None) -> dict:
    """训练一个臂并落盘 e0..eN 检查点 + history + best 标记。

    :param epochs: 默认冻结 50；测试可注入小值（契约验证用）。
    :param lr: 默认冻结 1e-3；测试可注入 0 验证 e0 并列选点。
    :param context: 生产身份上下文（base/cache SHA 等），嵌入 best 标记。
    :returns: ``{"epochs": [...], "best_epoch", "best_val_mse", ...}``。
    """
    lr_use = C.TRAIN["lr"] if lr is None else lr
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)                  # 两臂同初始化
    head = PM.PathHead()
    opt = torch.optim.AdamW(head.parameters(), lr=lr_use,
                            betas=tuple(C.TRAIN["betas"]),
                            weight_decay=C.TRAIN["weight_decay"])
    n_days = len(train_days)
    g = torch.Generator()
    g.manual_seed(seed)                      # 两臂相同日期 shuffle
    metrics: list[dict] = []
    t0 = time.perf_counter()

    def val_mse() -> float:
        return day_equal_weight_loss(head, val_days, stats, arm=arm)

    def train_mse() -> float:
        return day_equal_weight_loss(head, train_days, stats, arm=arm)

    def save(epoch: int) -> str:
        f = head_file(out_dir, arm, seed, epoch)
        torch.save({"state_dict": head.state_dict(),
                    "meta": {"arm": arm, "seed": seed, "epoch": epoch,
                             "protocol": C.PROTOCOL_VERSION}}, f)
        return f.name

    # e0：未训练即候选（选到 e0 = 如实退化为 G1，计划 §5）
    metrics.append({"epoch": 0, "val_mse": val_mse(), "train_mse": train_mse()})
    save(0)
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n_days, generator=g)
        for i in range(0, n_days, C.TRAIN["batch_days"]):
            idx = perm[i:i + C.TRAIN["batch_days"]]
            batch_days = [train_days[int(k)] for k in idx]
            losses = []
            for d in batch_days:
                v = np.asarray(d["valid"], dtype=bool)
                if not v.any():
                    continue
                shape = PM.arm_input(torch.from_numpy(
                    (np.asarray(d["shape"]) / stats["sigma_e"]).astype(
                        np.float32)), arm)
                b = torch.from_numpy(((np.asarray(d["s_g1"]) - stats["mu_g1"])
                                     / stats["sigma_g1"]).astype(np.float32))
                r_hat = head(shape, b)
                r = torch.from_numpy(((np.asarray(d["y"]) - np.asarray(d["s_g1"]))
                                      / stats["sigma_e"]).astype(np.float32))
                losses.append(((r_hat[v] - r[v]) ** 2).mean())
            if not losses:
                continue
            loss = torch.stack(losses).mean()   # 日等权（批内平均）
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),
                                           C.TRAIN["grad_clip"])
            opt.step()
        metrics.append({"epoch": epoch, "val_mse": val_mse(),
                        "train_mse": train_mse()})
        save(epoch)
    # 唯一选点：val 最低，严格更低才更新（并列取更早 = min 的元组序）
    best = min(metrics, key=lambda m: (m["val_mse"], m["epoch"]))
    best_name = head_file(out_dir, arm, seed, best["epoch"]).name
    result = {"arm": arm, "seed": seed, "epochs_run": epochs,
              "best_epoch": int(best["epoch"]),
              "best_val_mse": float(best["val_mse"]),
              "final_val_mse": float(metrics[-1]["val_mse"]),
              "g1_zero_val_mse": g1_zero_residual_mse(val_days, stats),
              "g1_zero_train_mse": g1_zero_residual_mse(train_days, stats),
              "best_head_file": best_name,
              "best_head_sha256": _sha256_file(out_dir / best_name),
              "context": dict(context or {}),
              "wall_s": round(time.perf_counter() - t0, 1), "device": "cpu"}
    (out_dir / f"epochs_metrics_{arm}_s{seed}.json").write_text(
        json.dumps({"summary": result, "epochs": metrics},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"best_{arm}_s{seed}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{arm} s{seed}] best e{best['epoch']} "
                f"val_mse={best['val_mse']:.6f} "
                f"(G1 零残差 val={result['g1_zero_val_mse']:.6f})")
    return {**result, "epochs": metrics}


def load_head(out_dir: Path, arm: str, seed: int, epoch: int) -> PM.PathHead:
    """载入指定 epoch 头并校验 meta（错 arm/seed/epoch/protocol 拒绝）。"""
    f = head_file(out_dir, arm, seed, epoch)
    if not f.is_file():
        raise FileNotFoundError(f"头文件缺失：{f}")
    ckpt = torch.load(f, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    expect = {"arm": arm, "seed": seed, "epoch": epoch,
              "protocol": C.PROTOCOL_VERSION}
    for k, v in expect.items():
        if meta.get(k) != v:
            raise RuntimeError(f"头文件 meta 不匹配 {k}={meta.get(k)!r}!={v!r}")
    head = PM.PathHead()
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head


def load_best(out_dir: Path, arm: str, seed: int,
              expect_context: dict | None = None) -> tuple[PM.PathHead, dict]:
    """按 best 标记载入最佳 epoch（含 e0），校验文件 SHA 与生产上下文。"""
    p = out_dir / f"best_{arm}_s{seed}.json"
    best = json.loads(p.read_text(encoding="utf-8"))
    f = out_dir / best["best_head_file"]
    if _sha256_file(f) != best["best_head_sha256"]:
        raise RuntimeError(f"{arm} best 头文件 SHA 不一致：{f}")
    for k, v in (expect_context or {}).items():
        if best.get("context", {}).get(k) != v:
            raise RuntimeError(
                f"{arm} best 头上下文不匹配 {k}="
                f"{best.get('context', {}).get(k)!r}!={v!r}")
    return load_head(out_dir, arm, seed, int(best["best_epoch"])), best


__all__ = ["fit_stats", "day_equal_weight_loss", "g1_zero_residual_mse",
           "train_arm", "load_head", "load_best", "head_file"]
