"""G5 换头臂训练编排（计划 §2，20260817）：缓存 + 七训。

**纪律（计划 §6）**：
    - 臂/种子冻结：H-kda×{42,43,44} + H-mamba×{42,43,44} + H-lin×{42}（共 7）；
    - 训练协议常量逐字 import 自 cross_section_kda（``G5_*`` 镜像，
      ``test_protocol_constants_match`` 对拍防漂移）；
    - 训练只看 train/早停曲线（es 日均 RankIC）；评估数字在 7 checkpoint 定型前不看
      （评估在阶段 2 的独立脚本，本模块无 eval 入口）；
    - 只写 ``g5_head/data/``；G1 权重只读；不改任何既有目录。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m g5_head.run_g5_head cache   # 1.2
    /home/user/miniconda3/envs/quant/bin/python -m g5_head.run_g5_head train   # 1.3
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
from scipy import stats

# —— 训练协议常量：逐字 import 自 cross_section_kda（冻结源，禁止漂移）——
from cross_section_kda.data import (
    EARLY_STOP_END,
    EARLY_STOP_START,
    LOOKBACK,
    PREDICT_LEN,
    PURGE,
    TRAIN_END,
    TRAIN_START,
)
from cross_section_kda.train import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    PATIENCE,
    WEIGHT_DECAY,
    build_split_cache,
    set_seed,
)

# G5 镜像（供 test_protocol_constants_match 对拍；值来自上面的 import，零漂移）
G5_LR = LR["B3"]            # 3e-4
G5_WD = WEIGHT_DECAY        # 0.01
G5_BATCH = BATCH_SIZE       # 128
G5_EPOCHS = EPOCHS          # 50
G5_PATIENCE = PATIENCE      # 5
G5_TRAIN_START = TRAIN_START
G5_TRAIN_END = TRAIN_END
G5_ES_START = EARLY_STOP_START
G5_ES_END = EARLY_STOP_END
G5_PURGE = PURGE

# 臂 / 种子（计划 §2，冻结）
G5_SEEDS = (42, 43, 44)
HLIN_SEED = 42

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
HIDDEN_CACHE_PATH = DATA_DIR / "hidden_cache_train_es.pt"


# ============================================================
# 1.2：G1 底座隐状态缓存（train+es 一次前向落盘）
# ============================================================


def _extract_day_hidden(backbone, x_norm: torch.Tensor, stamp: torch.Tensor,
                        device: str, chunk: int = 128) -> torch.Tensor:
    """对一个交易日的 [N,90,6] 分块过冻结 G1 底座，返回 CPU 上 [N,90,832]。

    逐日 ``empty_cache`` 释放碎片（与第 3 轮同理由：长序列逐日前向的显存碎片）。
    """
    hiddens: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(x_norm), chunk):
            xb = x_norm[i:i + chunk].to(device)
            sb = stamp[i:i + chunk].to(device)
            hiddens.append(backbone.extract(xb, sb).cpu())
            del xb, sb
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(hiddens, dim=0)


def build_and_save_cache(backbone, device: str, *, verify_n: int = 3) -> dict:
    """预提取 train+es 全样本日的 G1 底座隐状态并落盘（结构同第 3 轮缓存）。

    缓存结构::

        {
          "train": {"hidden": Tensor[Ntr,90,832], "y": Tensor[Ntr]},  # 扁平
          "es":    [{"date", "hidden": Tensor[n,90,832], "fwd_ret_raw": ndarray[n],
                     "codes": [...]}, ...],                            # 按日（日均 RankIC 用）
          "refs":  [{"x": Tensor[1,90,6], "stamp": Tensor[1,90,5], "hidden": Tensor[1,90,832]}, ...]
        }

    落盘后做缓存 vs 在线对拍（前若干日各 1 样本，max|Δ| < 1e-5 断言）。
    """
    from kronos_qlib import QlibProvider

    p = QlibProvider("csi300", TRAIN_START, EARLY_STOP_END)
    train_batches = build_split_cache(p, start=TRAIN_START, end=TRAIN_END,
                                      pool="csi300", rebalance_only=False)
    es_batches = build_split_cache(p, start=EARLY_STOP_START, end=EARLY_STOP_END,
                                   pool="csi300", rebalance_only=False)
    logger.info(f"样本日数：train {len(train_batches)} 日 / es {len(es_batches)} 日")

    tr_hidden, tr_y = [], []
    refs: list[dict] = []
    n_train_days = len(train_batches)
    for di, b in enumerate(train_batches):
        h = _extract_day_hidden(backbone, b.x_norm, b.stamp, device)
        tr_hidden.append(h)
        tr_y.append(torch.from_numpy(b.y_z))
        if len(refs) < verify_n:
            idx = 0
            refs.append({"x": b.x_norm[idx:idx + 1].clone(),
                         "stamp": b.stamp[idx:idx + 1].clone(),
                         "hidden": h[idx:idx + 1].clone()})
        if (di + 1) % 50 == 0 or di == 0:
            logger.info(f"  cache train [{di + 1}/{n_train_days}] {b.date.date()} "
                        f"GPU={torch.cuda.memory_allocated() / 1e9:.1f}GB")
    train_hidden = torch.cat(tr_hidden, dim=0)
    train_y = torch.cat(tr_y, dim=0)
    del tr_hidden, tr_y

    es_days: list[dict] = []
    n_es_days = len(es_batches)
    for di, b in enumerate(es_batches):
        h = _extract_day_hidden(backbone, b.x_norm, b.stamp, device)
        es_days.append({"date": b.date, "hidden": h, "fwd_ret_raw": b.fwd_ret_raw, "codes": b.codes})
        if (di + 1) % 20 == 0 or di == 0:
            logger.info(f"  cache es [{di + 1}/{n_es_days}] {b.date.date()} "
                        f"GPU={torch.cuda.memory_allocated() / 1e9:.1f}GB")

    cache = {"train": {"hidden": train_hidden, "y": train_y}, "es": es_days, "refs": refs}

    # —— 对拍：缓存 vs 在线（refs 样本逐位一致）——
    max_diff = 0.0
    for r in refs:
        with torch.no_grad():
            h_online = backbone.extract(r["x"].to(device), r["stamp"].to(device)).cpu()
        d = (h_online - r["hidden"]).abs().max().item()
        max_diff = max(max_diff, d)
    logger.info(f"隐状态缓存对拍：{len(refs)} 样本 max|Δ|={max_diff:.3e}（G1 底座确定性保证）")
    assert max_diff < 1e-5, f"缓存与在线不一致：max|Δ|={max_diff}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, HIDDEN_CACHE_PATH)
    logger.info(f"隐状态缓存落盘 → {HIDDEN_CACHE_PATH} | train {train_hidden.shape[0]} 样本、"
                f"es {len(es_days)} 日 | 文件大小 {(HIDDEN_CACHE_PATH.stat().st_size / 1e9):.1f}GB")
    return cache


# ============================================================
# 1.3：七训（缓存 hidden → 头训练；协议同 T1）
# ============================================================


def _evaluate_rankic_cached(head, es_days: list[dict], device: str) -> float:
    """早停段日均 RankIC（缓存 hidden 喂 decode_score；口径同第 3 轮）。"""
    from g5_head.heads import decode_score

    head.eval()
    rankics: list[float] = []
    with torch.no_grad():
        for day in es_days:
            score = decode_score(head, day["hidden"].to(device)).cpu().numpy()
            fwd = day["fwd_ret_raw"]
            valid = np.isfinite(score) & np.isfinite(fwd)
            if valid.sum() < 5:
                continue
            rho, _ = stats.spearmanr(score[valid], fwd[valid])
            if np.isfinite(rho):
                rankics.append(float(rho))
    return float(np.mean(rankics)) if rankics else float("nan")


def _make_head(arm: str, backbone) -> nn.Module:
    """按臂名构造头（H-lin 复用 B2LinearProbe / H-mamba 复用 T1 / H-kda 本包）。"""
    from cross_section_kda.models import B2LinearProbe
    from g5_head.heads import G5KdaHead
    from improve_suite.mamba_head import MambaTemporalHead

    if arm == "H-kda":
        return G5KdaHead(backbone)
    if arm == "H-mamba":
        return MambaTemporalHead(backbone)
    if arm == "H-lin":
        return B2LinearProbe(backbone)
    raise ValueError(f"未知臂 {arm}")


def train_one(arm: str, backbone, cache: dict, *, seed: int, device: str) -> tuple[nn.Module, dict]:
    """单臂单种子训练（缓存 hidden；AdamW/MSE/早停，协议逐字同第 3 轮 T1）。"""
    from g5_head.heads import decode_score

    set_seed(seed)
    model = _make_head(arm, backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{arm} s{seed}] 可训练参数 {n_trainable:,} | lr={G5_LR} wd={G5_WD} "
                f"batch={G5_BATCH} epochs≤{G5_EPOCHS} patience={G5_PATIENCE}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=G5_LR, weight_decay=G5_WD)
    loss_fn = nn.MSELoss()

    xs = cache["train"]["hidden"]
    ys = cache["train"]["y"]
    es_days = cache["es"]
    n = xs.size(0)
    logger.info(f"[{arm} s{seed}] 训练样本 {n} 条（G1 底座缓存 hidden）")

    best_es = -np.inf
    best_state = None
    history: list[dict] = []
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, G5_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, G5_BATCH):
            idx = perm[i:i + G5_BATCH]
            h = xs[idx].to(device)
            y = ys[idx].to(device)
            opt.zero_grad()
            score = decode_score(model, h)
            loss = loss_fn(score, y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        es_rankic = _evaluate_rankic_cached(model, es_days, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "es_rankic": es_rankic})
        improved = np.isfinite(es_rankic) and es_rankic > best_es
        marker = "★" if improved else ""
        logger.info(f"[{arm} s{seed}] epoch {epoch:02d}/{G5_EPOCHS} "
                    f"train_loss={train_loss:.5f} es_RankIC={es_rankic:+.4f} {marker}")
        if improved:
            best_es = es_rankic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= G5_PATIENCE:
                logger.info(f"[{arm} s{seed}] 早停：{G5_PATIENCE} 个 epoch 无改善，"
                            f"最佳 es_RankIC={best_es:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    logger.info(f"[{arm} s{seed}] 训练完成 用时 {elapsed:.1f}s 最佳 es_RankIC={best_es:+.4f}")

    info = {
        "arm": arm, "seed": seed, "n_trainable": n_trainable,
        "best_es_rankic": best_es, "history": history, "n_epochs_run": len(history),
        "elapsed_s": elapsed,
        "cfg": {"lr": G5_LR, "weight_decay": G5_WD, "batch_size": G5_BATCH,
                "epochs": G5_EPOCHS, "patience": G5_PATIENCE, "seed": seed},
    }
    return model, info


def _save_ckpt(model: nn.Module, name: str, info: dict) -> Path:
    """保存最佳 checkpoint + history 到 g5_head/data/（不入库）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = DATA_DIR / f"{name}_best.pt"
    torch.save({"state_dict": model.state_dict(), "info": info}, ckpt)
    hist = DATA_DIR / f"{name}_history.json"
    with open(hist, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=float)
    logger.info(f"[{name}] checkpoint → {ckpt.name} | history → {hist.name}")
    return ckpt


def run_train(device: str) -> dict:
    """1.3 七训：H-kda×3 + H-mamba×3 + H-lin×1。"""
    from g5_head.backbone_g1 import load_g1_backbone

    backbone = load_g1_backbone(device)
    if not HIDDEN_CACHE_PATH.exists():
        raise FileNotFoundError(f"隐状态缓存缺失：{HIDDEN_CACHE_PATH}（先跑 cache 相）")
    logger.info(f"加载隐状态缓存 {HIDDEN_CACHE_PATH.name}")
    cache = torch.load(HIDDEN_CACHE_PATH, map_location="cpu", weights_only=False)

    jobs = [(arm, seed) for arm in ("H-kda", "H-mamba") for seed in G5_SEEDS] + [("H-lin", HLIN_SEED)]
    records = {}
    for arm, seed in jobs:
        name = f"{arm}_s{seed}"
        logger.info("=" * 70)
        logger.info(f"1.3 训练 {name}")
        logger.info("=" * 70)
        model, info = train_one(arm, backbone, cache, seed=seed, device=device)
        _save_ckpt(model, name, info)
        records[name] = {"n_trainable": info["n_trainable"],
                         "best_es_rankic": info["best_es_rankic"],
                         "n_epochs_run": info["n_epochs_run"],
                         "elapsed_s": info["elapsed_s"]}

    out = {"experiment": "g5_stage1_train", "date": "2026-08-17", "records": records}
    with open(DATA_DIR / "g5_stage1_records.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    logger.info("==== 阶段 1 七训全部完成 ====")
    return out


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    device = "cuda:0"
    if phase == "cache":
        from g5_head.backbone_g1 import load_g1_backbone

        backbone = load_g1_backbone(device)
        build_and_save_cache(backbone, device)
    elif phase == "train":
        run_train(device)
    else:
        logger.error(f"未知 phase={phase!r}，可选 cache / train")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
