"""B1/B2/B3 训练管线（计划 §2 超参预注册）。

预注册超参（**只许在训练/早停段内调，最终验证段一次定型**）：
    AdamW，lr=1e-3（B2）/3e-4（B1,B3），weight_decay=0.01，batch=1024，
    epochs≤50，早停 patience=5（按早停验证段的日均 RankIC），seed=42。

回归目标：MSE（标签已按日截面 z-score）。每个 epoch 末在早停验证段算逐调仓日
RankIC（模型输出作因子，对未截面化的 fwd_ret 做 Spearman），取日均 RankIC 最大
checkpoint 作为定型产物。

最终验证段（2024-07-01 之后）本模块**不触碰**——跑它需显式调
``evaluate_arms.run_final_validation``，且只跑一次。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from scipy import stats

from cross_section.common import DATA_DIR as CS_DATA_DIR
from cross_section_kda import (
    B1SupervisedHead,
    B2LinearProbe,
    B3KronosKdaHead,
    KronosFrozenBackbone,
)
from cross_section_kda.data import (
    EARLY_STOP_END,
    EARLY_STOP_START,
    FINAL_START,
    LOOKBACK,
    PREDICT_LEN,
    PURGE,
    SampleTensorBatch,
    TRAIN_END,
    TRAIN_START,
    assert_purge_intervals,
    build_daily_samples,
    make_splits,
)
from cross_section_kda.models import count_trainable

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"          # 训练产物不入库（*.pt / *.parquet 已 gitignore）
DATA_DIR.mkdir(parents=True, exist_ok=True)
B0_SIGNALS_PATH = CS_DATA_DIR / "signals_with_baselines.parquet"

# 预注册超参（计划 §2）
SEED = 42
# 计划 §2 预注册 batch=1024，但实测 d_model=832 下 KDA 递归状态 [B,H,K,V] 在
# batch=256 即 OOM（RTX 5090 32GB，峰值 19.6GB@128）。这里落到硬件上限 128，
# 属硬件必要偏差（batch 仍是 2 的幂、训练动态同量级），如实写入结果文档局限。
BATCH_SIZE = 128
EPOCHS = 50
PATIENCE = 5
WEIGHT_DECAY = 0.01
LR = {"B1": 3e-4, "B2": 1e-3, "B3": 3e-4}


@dataclass
class TrainConfig:
    """训练超参（预注册）。"""

    arm: str                       # "B1" | "B2" | "B3"
    seed: int = SEED
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    patience: int = PATIENCE
    weight_decay: float = WEIGHT_DECAY
    lr: float = field(default=0.0)

    def __post_init__(self) -> None:
        if not self.lr:
            self.lr = LR[self.arm]


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据缓存：逐日构造样本（cpu 张量），按 split 分组
# ============================================================


def build_split_cache(
    provider,
    *,
    start: str,
    end: str,
    pool: str = "csi300",
    rebalance_only: bool = False,
) -> list[SampleTensorBatch]:
    """在 [start, end] 内逐交易日构造样本，返回日级 batch 列表。

    :param rebalance_only: True 时只取 B0 的调仓日网格（final 段用，每 10 交易日
        一期）；False 时取全部交易日（训练/早停段用，扩大训练集）。
    """
    cal = provider.trading_days()
    ts0, ts1 = pd.Timestamp(start), pd.Timestamp(end)
    days = cal[(cal >= ts0) & (cal <= ts1)]
    if rebalance_only:
        b0 = pd.read_parquet(B0_SIGNALS_PATH)
        grid = set(pd.Timestamp(d) for d in b0["date"].unique())
        days = [d for d in days if d in grid]
        days = pd.DatetimeIndex(sorted(days))

    batches: list[SampleTensorBatch] = []
    n_skip = 0
    for d in days:
        b = build_daily_samples(provider, date=d.strftime("%Y-%m-%d"), pool=pool,
                                lookback=LOOKBACK, predict_len=PREDICT_LEN)
        if b is None:
            n_skip += 1
            continue
        batches.append(b)
    logger.info(
        f"build_split_cache[{start}~{end}] rebalance_only={rebalance_only}: "
        f"{len(batches)} 日（跳过 {n_skip} 日），"
        f"样本合计 {sum(len(b.codes) for b in batches)} 条"
    )
    return batches


# ============================================================
# 训练循环
# ============================================================


def _evaluate_rankic(model: nn.Module, batches: list[SampleTensorBatch], device: str) -> float:
    """在给定 split 上算日均 RankIC（模型输出作因子，对原始 fwd_ret 做 Spearman）。

    :returns: 逐日 RankIC 的均值（早停选 checkpoint 用）。
    """
    model.eval()
    rankics: list[float] = []
    with torch.no_grad():
        for b in batches:
            x = b.x_norm.to(device)
            s = b.stamp.to(device)
            score = model(x, s).cpu().numpy()
            valid = np.isfinite(score) & np.isfinite(b.fwd_ret_raw)
            if valid.sum() < 5:
                continue
            rho, _ = stats.spearmanr(score[valid], b.fwd_ret_raw[valid])
            if np.isfinite(rho):
                rankics.append(float(rho))
    return float(np.mean(rankics)) if rankics else float("nan")


def train_arm(
    arm: str,
    backbone: KronosFrozenBackbone | None,
    train_batches: list[SampleTensorBatch],
    es_batches: list[SampleTensorBatch],
    *,
    device: str,
    cfg: TrainConfig | None = None,
) -> tuple[nn.Module, dict]:
    """训练单臂，返回（带最佳 checkpoint 的模型, 训练历史）。

    :param backbone: B2/B3 传入冻结主干；B1 传 None。
    """
    assert arm in ("B1", "B2", "B3"), f"未知 arm={arm}"
    set_seed(cfg.seed if cfg else SEED)
    cfg = cfg or TrainConfig(arm=arm)

    if arm == "B1":
        model = B1SupervisedHead().to(device)
    elif arm == "B2":
        model = B2LinearProbe(backbone).to(device)
    else:
        model = B3KronosKdaHead(backbone).to(device)

    n_trainable = count_trainable(model)
    logger.info(f"[{arm}] 可训练参数 {n_trainable:,} | lr={cfg.lr} wd={cfg.weight_decay} "
                f"batch={cfg.batch_size} epochs≤{cfg.epochs} patience={cfg.patience}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()

    # flatten 训练样本到 (x, stamp, y_z) 列表，便于 shuffle
    xs = torch.cat([b.x_norm for b in train_batches], dim=0)
    stamps = torch.cat([b.stamp for b in train_batches], dim=0)
    ys = torch.from_numpy(np.concatenate([b.y_z for b in train_batches], axis=0))
    n = xs.size(0)
    logger.info(f"[{arm}] 训练样本 {n} 条 / {len(train_batches)} 日")

    best_es_rankic = -np.inf
    best_state = None
    history: list[dict] = []
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            x = xs[idx].to(device)
            st = stamps[idx].to(device)
            y = ys[idx].to(device)
            opt.zero_grad()
            score = model(x, st)
            loss = loss_fn(score, y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        es_rankic = _evaluate_rankic(model, es_batches, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "es_rankic": es_rankic})
        improved = np.isfinite(es_rankic) and es_rankic > best_es_rankic
        marker = "★" if improved else ""
        logger.info(f"[{arm}] epoch {epoch:02d}/{cfg.epochs} "
                    f"train_loss={train_loss:.5f} es_RankIC={es_rankic:+.4f} {marker}")

        if improved:
            best_es_rankic = es_rankic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                logger.info(f"[{arm}] 早停：{cfg.patience} 个 epoch 无改善，"
                            f"最佳 es_RankIC={best_es_rankic:+.4f} @ "
                            f"epoch {max(h['epoch'] for h in history if h['es_rankic']==best_es_rankic)}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    logger.info(f"[{arm}] 训练完成 用时 {elapsed:.1f}s 最佳 es_RankIC={best_es_rankic:+.4f}")

    info = {
        "arm": arm,
        "n_trainable": n_trainable,
        "best_es_rankic": best_es_rankic,
        "history": history,
        "n_epochs_run": len(history),
        "cfg": {"lr": cfg.lr, "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size, "epochs": cfg.epochs,
                "patience": cfg.patience, "seed": cfg.seed},
    }
    return model, info


def save_checkpoint(model: nn.Module, arm: str, info: dict) -> Path:
    """保存最佳 checkpoint 与训练历史（不入库，gitignore 覆盖 *.pt/*.json 在 data 下）。"""
    ckpt = DATA_DIR / f"{arm}_best.pt"
    torch.save({"state_dict": model.state_dict(), "info": info}, ckpt)
    hist = DATA_DIR / f"{arm}_history.json"
    with open(hist, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=float)
    logger.info(f"[{arm}] checkpoint → {ckpt} | history → {hist}")
    return ckpt


__all__ = [
    "TrainConfig", "set_seed", "build_split_cache",
    "train_arm", "save_checkpoint",
    "DATA_DIR", "B0_SIGNALS_PATH",
    "SEED", "BATCH_SIZE", "EPOCHS", "PATIENCE", "WEIGHT_DECAY", "LR",
]
