"""PEER1 头训练（计划 §5 冻结预算）与逐 epoch 唯一检查点。

AdamW lr=3e-4 / betas(0.9,0.999) / wd=0.01、每批最多 4 个日期（日期维 =
batch 维，padding 到批内 max N）、clip_grad_norm 3、FP32、100 epochs、不
early-stop；OFF/ON 同 seed → 同初始参数字节、同日期 shuffle（torch.Generator
(seed) 驱动）。唯一选点 = 验证日等权预测 MSE 最低 epoch（严格更低才更新，
并列取最早）；损失 = 每日期在 LossSet 上 mean MSE，再对批内日期等权 mean。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from peer_residual import config as C
from peer_residual import data as PD
from peer_residual.model import build_head


def masked_day_mse(r_hat: torch.Tensor, r: torch.Tensor,
                   loss: torch.Tensor) -> torch.Tensor:
    """每日期在 LossSet 上 mean MSE → 对批内日期等权 mean。

    :param r_hat: ``[B, N]`` 预测。
    :param r: ``[B, N]`` 目标（padding 处 0）。
    :param loss: ``[B, N]`` bool 监督掩码（LossSet）。
    """
    err = (r_hat - r).pow(2) * loss
    per_day = err.sum(dim=1) / loss.sum(dim=1).clamp_min(1)
    return per_day.mean()


def train_step(model, opt, x: torch.Tensor, valid: torch.Tensor,
               loss: torch.Tensor, r: torch.Tensor, arm: str) -> tuple[float, float]:
    """单批更新（纯函数，机制测试直接调用）。

    :returns: ``(batch_loss, grad_norm_before_clip)``。
    """
    r_hat = model(x, valid, arm)
    obj = masked_day_mse(r_hat, r, loss)
    opt.zero_grad(set_to_none=True)
    obj.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), C.TRAIN["grad_clip"])
    opt.step()
    return float(obj.detach()), float(gn)


def _batches(n: int, batch_days: int):
    for i in range(0, n, batch_days):
        yield list(range(i, min(i + batch_days, n)))


def val_loss_days(model, val_days: list, arm: str,
                  device: str = "cpu") -> float:
    """验证日等权预测 MSE（各日 mean over LossSet，再跨日 mean）。"""
    model.eval()
    per_day_all = []
    with torch.no_grad():
        for ids in _batches(len(val_days), C.TRAIN["batch_days"]):
            days = [val_days[i] for i in ids]
            x, valid, loss, r = PD.collate_batch(days)
            r_hat = model(x.to(device), valid.to(device), arm)
            err = (r_hat.to("cpu") - r).pow(2) * loss
            per_day = err.sum(dim=1) / loss.sum(dim=1).clamp_min(1)
            per_day_all.extend(per_day.tolist())
    return float(np.mean(per_day_all))


def head_file(out_dir: Path, arm: str, seed: int, epoch: int) -> Path:
    return out_dir / f"head_{arm}_s{seed}_e{epoch:03d}.pt"


def train_arm(train_days: list, val_days: list, arm: str, seed: int,
              out_dir: Path, device: str = "cpu",
              epochs: int = C.TRAIN["epochs"]) -> dict:
    """训练一个 (arm, seed) 头并落盘逐 epoch 检查点与指标。

    :returns: ``{"best_epoch","best_val_mse","epoch_files","metrics",...}``。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    d_in = train_days[0].x.shape[1]
    model = build_head(seed, d_in).to(device)
    # 零残差门禁：初始 r_hat 必须逐值为 0（s_final 逐值等于 G1 的前提）
    x0, v0, _, _ = PD.collate_batch(train_days[:1])
    with torch.no_grad():
        r0 = model(x0.to(device), v0.to(device), arm)
    assert bool((r0 == 0).all()), "初始残差输出非零：零残差门禁失败"
    opt = torch.optim.AdamW(model.parameters(), lr=C.TRAIN["lr"],
                            betas=tuple(C.TRAIN["betas"]),
                            weight_decay=C.TRAIN["weight_decay"])
    n = len(train_days)
    g = torch.Generator()
    g.manual_seed(seed)                     # OFF/ON 相同日期 shuffle
    metrics: list[dict] = []
    files: list[str] = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        if epoch == 1:
            first_perm = perm.clone()
        sums = {"l_pred": 0.0, "grad_norm": 0.0}
        nb = 0
        for ids in _batches(n, C.TRAIN["batch_days"]):
            days = [train_days[int(perm[i])] for i in ids]
            x, valid, loss, r = PD.collate_batch(days)
            bl, gn = train_step(model, opt, x.to(device), valid.to(device),
                                loss.to(device), r.to(device), arm)
            sums["l_pred"] += bl
            sums["grad_norm"] += gn
            nb += 1
        v_loss = val_loss_days(model, val_days, arm, device)
        metrics.append({"epoch": epoch, "train_l_pred": sums["l_pred"] / nb,
                       "grad_norm": sums["grad_norm"] / nb,
                       "val_l_pred": v_loss})
        f = head_file(out_dir, arm, seed, epoch)
        torch.save({"state_dict": model.state_dict(),
                    "meta": {"arm": arm, "seed": seed, "epoch": epoch,
                             "val_l_pred": v_loss, "d_in": int(d_in),
                             "head": dict(C.HEAD),
                             "protocol": C.PROTOCOL_VERSION}}, f)
        files.append(f.name)
    # 唯一选点：验证等权 MSE 最低，严格更低才更新（并列取最早）
    best = min(metrics, key=lambda m: (m["val_l_pred"], m["epoch"]))
    result = {"arm": arm, "seed": seed,
              "best_epoch": best["epoch"], "best_val_l_pred": best["val_l_pred"],
              "final_val_l_pred": metrics[-1]["val_l_pred"],
              "epoch_files": files,
              "day_keys_digest": PD.day_keys_digest(train_days),
              "first_epoch_perm": first_perm.tolist()[:16],
              "wall_s": time.perf_counter() - t0, "device": device}
    (out_dir / f"epochs_metrics_{arm}_s{seed}.json").write_text(
        json.dumps({"summary": {k: v for k, v in result.items()
                                if k != "first_epoch_perm"},
                    "epochs": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / f"best_{arm}_s{seed}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{arm} s{seed}] best e{best['epoch']} "
                f"val_mse={best['val_l_pred']:.6f} "
                f"(final e{epochs}={metrics[-1]['val_l_pred']:.6f})")
    return result


def state_dict_hash(model_or_sd) -> str:
    """state_dict 字节级 SHA256（同 seed OFF/ON 初始一致的门禁）。"""
    import hashlib

    sd = (model_or_sd.state_dict() if hasattr(model_or_sd, "state_dict")
          else model_or_sd)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_head(out_dir: Path, arm: str, seed: int, epoch: int, d_in: int):
    """载入指定 epoch 头并校验 meta（错权重/错身份拒绝，计划 §6）。"""
    f = head_file(out_dir, arm, seed, epoch)
    if not f.is_file():
        raise FileNotFoundError(f"头文件缺失：{f}")
    ckpt = torch.load(f, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    expect = {"arm": arm, "seed": seed, "epoch": epoch,
              "protocol": C.PROTOCOL_VERSION, "d_in": int(d_in)}
    for k, v in expect.items():
        if meta.get(k) != v:
            raise RuntimeError(f"头文件 meta 不匹配 {k}={meta.get(k)!r}!={v!r}")
    model = build_head(seed, d_in)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_best(out_dir: Path, arm: str, seed: int, d_in: int):
    """按 best 标记载入最佳 epoch 头。"""
    p = out_dir / f"best_{arm}_s{seed}.json"
    best = json.loads(p.read_text(encoding="utf-8"))
    return load_head(out_dir, arm, seed, best["best_epoch"], d_in), best


__all__ = ["train_arm", "train_step", "val_loss_days", "masked_day_mse",
           "load_head", "load_best", "head_file", "state_dict_hash"]
