"""head 训练（计划 §5 冻结预算）与逐 epoch 唯一检查点。

AdamW lr=3e-4 / betas(0.9,0.999) / wd=0.01、batch 256、clip_grad_norm 3、
FP32、100 epochs、不 early-stop、不 OneCycle；每 epoch 完整遍历冻结缓存，
shuffle 由 ``torch.Generator(seed)`` 驱动——AE/SAE 同 seed 消费完全相同的
批序列。唯一选点指标 = 验证 ``L_pred`` 最低（并列取最早），验证为各日期
等权（先日内 mean 再跨日 mean）。

默认 CPU 训练小头（计划 §7：优先复用缓存、GPU 预算留给 teacher/hidden）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from sae_residual import config as C
from sae_residual import data as D
from sae_residual.cache_io import load_split_arrays
from sae_residual.model import (activation_diagnostics, build_head, head_loss)


def prepare_training_data(expect: dict | None = None
                          ) -> tuple[dict, dict, dict]:
    """加载缓存 → 归一化统计（只用 train）→ 训练张量。

    :param expect: 逐 chunk 身份校验字段（如当前 G1 权重 SHA）；缺省只校验
        协议与段名。
    :returns: ``(train, val, stats)``：
        train/val = ``{"x": float32[N,D], "r": float32[N], "date": [N]}``；
        stats = ``{"mu","sigma","sigma_e","n_train","n_val","d_in"}``。
    """
    base = dict(expect or {})
    tr = load_split_arrays("train", {**base, "split": "train"})
    va = load_split_arrays("val", {**base, "split": "val"})
    stats_raw = D.fit_norm_stats(tr["h"])
    e_train = tr["y_mean"] - tr["s_g1"]
    sigma_e = D.residual_std(e_train)          # 退化 → 抛错停止（§3）
    x_tr = D.normalize_hidden(tr["h"], stats_raw)
    x_va = D.normalize_hidden(va["h"], stats_raw)
    e_val = va["y_mean"] - va["s_g1"]
    stats = {"mu": stats_raw["mu"], "sigma": stats_raw["sigma"],
             "sigma_e": sigma_e, "n_train": int(len(x_tr)),
             "n_val": int(len(x_va)), "d_in": int(x_tr.shape[1]),
             "e_train_std": float(e_train.std()),
             "e_train_mean": float(e_train.mean()),
             "e_val_std": float(e_val.std())}
    train = {"x": x_tr, "r": (e_train / sigma_e).astype(np.float32),
             "date": tr["dates"]}
    val = {"x": x_va, "r": (e_val / sigma_e).astype(np.float32),
           "date": va["dates"]}
    return train, val, stats


def save_stats(stats: dict, out_dir: Path) -> Path:
    p = out_dir / "norm_stats.json"
    payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in stats.items()}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def val_pred_loss(model, val: dict) -> float:
    """验证 ``L_pred``：各日期等权（先日内 mean，再跨日 mean）。"""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(val["x"])
        r_hat, _, _ = model(x)
        err = (r_hat - torch.from_numpy(val["r"])).pow(2).numpy()
    df = pd.DataFrame({"date": val["date"], "err": err})
    per_day = df.groupby("date")["err"].mean()
    return float(per_day.mean())


def head_file(out_dir: Path, arm: str, seed: int, epoch: int) -> Path:
    return out_dir / f"head_{arm}_s{seed}_e{epoch:03d}.pt"


def train_arm(train: dict, val: dict, arm: str, seed: int,
              out_dir: Path, device: str = "cpu",
              epochs: int = C.TRAIN["epochs"]) -> dict:
    """训练一个 (arm, seed) 头并落盘逐 epoch 检查点与指标。

    :returns: ``{"best_epoch", "best_val_l_pred", "epoch_files", "metrics"}``。
    """
    beta = C.LOSS["beta_sae"] if arm == "SAE" else C.LOSS["beta_ae"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_head(seed, train["x"].shape[1]).to(device)
    # 零残差门禁：初始 r_hat 必须逐值为 0（s_final 逐值等于 G1 的实现前提）
    with torch.no_grad():
        r0 = model(torch.from_numpy(train["x"][:8]).to(device))[0]
        assert bool((r0 == 0).all()), "初始残差输出非零：零残差门禁失败"
    opt = torch.optim.AdamW(model.parameters(), lr=C.TRAIN["lr"],
                            betas=tuple(C.TRAIN["betas"]),
                            weight_decay=C.TRAIN["weight_decay"])
    x_all = torch.from_numpy(train["x"]).to(device)
    r_all = torch.from_numpy(train["r"]).to(device)
    n = len(x_all)
    g = torch.Generator()
    g.manual_seed(seed)                       # AE/SAE 相同批序列
    metrics: list[dict] = []
    files: list[str] = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        sums = {"l_pred": 0.0, "l_recon": 0.0, "l_sparse": 0.0, "grad_norm": 0.0,
                "mean_act": 0.0, "low_act": 0.0}
        nb = 0
        for i in range(0, n, C.TRAIN["batch_size"]):
            idx = perm[i:i + C.TRAIN["batch_size"]]
            x, r = x_all[idx], r_all[idx]
            r_hat, x_hat, z = model(x)
            losses = head_loss(r_hat, r, x_hat, x, z, beta)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), C.TRAIN["grad_clip"])
            opt.step()
            bs = len(idx)
            sums["l_pred"] += float(losses["l_pred"]) * bs
            sums["l_recon"] += float(losses["l_recon"]) * bs
            sums["l_sparse"] += float(losses["l_sparse"]) * bs
            sums["grad_norm"] += float(gn)
            diag = activation_diagnostics(z)
            sums["mean_act"] += diag["mean_activation"] * bs
            sums["low_act"] += diag["low_activation_frac"] * bs
            nb += 1
        v_loss = val_pred_loss(model, val)
        m = {"epoch": epoch,
             "train_l_pred": sums["l_pred"] / n,
             "train_l_recon": sums["l_recon"] / n,
             "train_l_sparse": sums["l_sparse"] / n,
             "grad_norm": sums["grad_norm"] / nb,
             "mean_activation": sums["mean_act"] / n,
             "low_activation_frac": sums["low_act"] / n,
             "val_l_pred": v_loss}
        metrics.append(m)
        f = head_file(out_dir, arm, seed, epoch)
        torch.save({"state_dict": model.state_dict(),
                    "meta": {"arm": arm, "seed": seed, "epoch": epoch,
                             "beta": beta, "val_l_pred": v_loss,
                             "d_in": int(train["x"].shape[1]),
                             "protocol": C.PROTOCOL_VERSION}}, f)
        files.append(f.name)
    # 唯一选点：验证 L_pred 最低，并列最早（min 天然取最早）
    best = min(metrics, key=lambda m: (m["val_l_pred"], m["epoch"]))
    result = {"arm": arm, "seed": seed, "beta": beta,
              "best_epoch": best["epoch"],
              "best_val_l_pred": best["val_l_pred"],
              "final_val_l_pred": metrics[-1]["val_l_pred"],
              "epoch_files": files,
              "wall_s": time.perf_counter() - t0, "device": device}
    (out_dir / f"epochs_metrics_{arm}_s{seed}.json").write_text(
        json.dumps({"summary": result, "epochs": metrics},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"best_{arm}_s{seed}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{arm} s{seed}] best e{best['epoch']} "
                f"val_l_pred={best['val_l_pred']:.6f} "
                f"(final e{epochs}={metrics[-1]['val_l_pred']:.6f})")
    return result


def load_head(out_dir: Path, arm: str, seed: int, epoch: int, d_in: int):
    """载入指定 epoch 头并校验 meta（错权重拒绝，计划 §7）。"""
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
    model = build_head(seed, d_in)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_best(out_dir: Path, arm: str, seed: int, d_in: int):
    """按 best 标记载入最佳 epoch 头。"""
    p = out_dir / f"best_{arm}_s{seed}.json"
    best = json.loads(p.read_text(encoding="utf-8"))
    return load_head(out_dir, arm, seed, best["best_epoch"], d_in), best


__all__ = ["prepare_training_data", "save_stats", "train_arm", "load_head",
           "load_best", "val_pred_loss", "head_file"]
