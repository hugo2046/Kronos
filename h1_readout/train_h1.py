"""H1 四臂训练编排（计划 §1/§3.2，20260905 H1 计划）。

**臂（冻结）**：H1a-lin / H1a-kda（2014-01-02~2024-12-17 × csi300 PIT）、
H1b-lin / H1b-kda（同期 × 全 A PIT 并集，G1 同语料）——唯一变量 = 训练数据。

**逐字复用 R1**（零漂移 import 对拍，``test_protocol_frozen``）：G1 s100 底座冻结
（``g5_head.backbone_g1.load_g1_backbone``，只读）、头结构（``r1_objective`` 的
_make_head 同构：lin=B2LinearProbe 833 / kda=G5KdaHead 1,209,937）、IC 损失
（``r1_objective.ic_loss``，每日截面批内 −Pearson）、AdamW 3e-4 / wd 0.01、
batch 128（同日截面随机抽）、patience 5（早停段日均 RankIC）。

**H1 侧冻结**：在线过冻结底座（11 年全 A 缓存不可行 → 每步在线前向）、
每 epoch 2000 步 × epochs ≤ 15、早停段 = csi300 × 2025-01-01~2025-06-30。

纪律（§4）：训练只看 train/早停曲线；评估数字在四臂打分全部落盘前不看
（本模块无 eval 入口，评估在 ``run_h1_signals``）。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m h1_readout.train_h1 --arm H1a-lin
    /home/user/miniconda3/envs/quant/bin/python -m h1_readout.train_h1 --seed 43 --arm H1b-kda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# —— 协议常量：逐字镜像 R1（零漂移；test_protocol_frozen 对拍）——
from r1_objective.run_r1_train import (
    R1_BATCH, R1_LR, R1_PATIENCE, R1_WD, _make_loss_fn,
)

H1_LR = R1_LR               # 3e-4
H1_WD = R1_WD               # 0.01
H1_BATCH = R1_BATCH         # 128（同日截面随机抽 128 股）
H1_PATIENCE = R1_PATIENCE   # 5
LOSS_NAME = "IC(-Pearson, 每日截面批内)"

# —— H1 侧冻结（计划 §1）——
H1_STEPS_PER_EPOCH = 2000   # G1 式步数封顶
H1_EPOCHS = 15
H1_ARMS = ("H1a-lin", "H1a-kda", "H1b-lin", "H1b-kda")
ARM_POOL = {"H1a-lin": "csi300", "H1a-kda": "csi300",
            "H1b-lin": "ashares", "H1b-kda": "ashares"}
H1_SEEDS = (42,)            # 两阶段：seed 42 先行，HC1 通过者 3.5 补 43/44
ES_POOL = "csi300"          # 早停段恒 csi300（R1 早停口径；a/b 仅训练数据不同)

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"


def _make_head(arm: str, backbone) -> nn.Module:
    """头结构与 R1 一一同构（lin → B2LinearProbe / kda → G5KdaHead）。"""
    from r1_objective.run_r1_train import _make_head as r1_make_head

    return r1_make_head("R-" + arm.split("-")[1], backbone)


def _cache_es_hidden(backbone, es_days, device: str, chunk: int = 128) -> list[dict]:
    """早停日在线过底座一次 → 缓存 hidden（冻结底座确定性保证，g5 同构）。"""
    out: list[dict] = []
    for i, d in enumerate(es_days):
        hiddens: list[torch.Tensor] = []
        with torch.no_grad():
            for s in range(0, len(d.codes), chunk):
                h = backbone.extract(d.x_norm[s:s + chunk].to(device),
                                     d.stamp[s:s + chunk].to(device))
                hiddens.append(h.cpu())
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        out.append({"date": d.date, "hidden": torch.cat(hiddens, dim=0),
                    "fwd_ret_raw": d.fwd_raw, "codes": d.codes})
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  es hidden [{i + 1}/{len(es_days)}] {d.date.date()} "
                        f"{len(d.codes)} 只")
    return out


def train_one(arm: str, *, seed: int, device: str = "cuda:0") -> tuple[nn.Module, dict]:
    """单臂训练：在线底座前向 + 每日截面 IC 损失 + 早停段 RankIC 早停。"""
    from cross_section_kda.train import set_seed
    from g5_head.backbone_g1 import load_g1_backbone
    from g5_head.heads import decode_score
    from g5_head.run_g5_head import _evaluate_rankic_cached
    from h1_readout.corpus import build_corpus
    from h1_readout.sampler import DailyBatchSampler

    pool = ARM_POOL[arm]
    logger.info(f"[{arm} s{seed}] 语料 = {pool}（在线前向；缓存不可行 ~TB 级）")
    corpus = build_corpus(pool)
    logger.info(f"[{arm} s{seed}] 语料统计：{corpus.stats}")

    backbone = load_g1_backbone(device)
    set_seed(seed)
    model = _make_head(arm, backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{arm} s{seed}] 可训练参数 {n_trainable:,}（底座冻结只读）| "
                f"lr={H1_LR} wd={H1_WD} batch={H1_BATCH} steps×{H1_STEPS_PER_EPOCH} "
                f"epochs≤{H1_EPOCHS} patience={H1_PATIENCE} | 损失={LOSS_NAME}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=H1_LR, weight_decay=H1_WD)
    loss_fn = _make_loss_fn()

    es_hidden = _cache_es_hidden(backbone, corpus.es_days, device)
    sampler = DailyBatchSampler(corpus, seed=seed, batch_size=H1_BATCH)

    best_es = -np.inf
    best_state = None
    history: list[dict] = []
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, H1_EPOCHS + 1):
        model.train()
        epoch_loss, n_steps = 0.0, 0
        for x, stamp, y, _day in sampler.steps(H1_STEPS_PER_EPOCH):
            xb = x.to(device, non_blocking=True)
            sb = stamp.to(device, non_blocking=True)
            yb = y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.no_grad():
                hidden = backbone.extract(xb, sb)
            score = decode_score(model, hidden)
            loss = loss_fn(score, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        train_loss = epoch_loss / max(n_steps, 1)

        es_rankic = _evaluate_rankic_cached(model, es_hidden, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "es_rankic": es_rankic})
        improved = np.isfinite(es_rankic) and es_rankic > best_es
        marker = "★" if improved else ""
        logger.info(f"[{arm} s{seed}] epoch {epoch:02d}/{H1_EPOCHS} "
                    f"train_ICloss={train_loss:.5f} es_RankIC={es_rankic:+.4f} {marker}")
        if improved:
            best_es = es_rankic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= H1_PATIENCE:
                logger.info(f"[{arm} s{seed}] 早停：{H1_PATIENCE} 个 epoch 无改善，"
                            f"最佳 es_RankIC={best_es:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    logger.info(f"[{arm} s{seed}] 训练完成 用时 {elapsed / 60:.1f}min "
                f"最佳 es_RankIC={best_es:+.4f}")

    info = {
        "arm": arm, "seed": seed, "pool": pool, "loss": LOSS_NAME,
        "n_trainable": n_trainable, "best_es_rankic": best_es,
        "history": history, "n_epochs_run": len(history), "elapsed_s": elapsed,
        "corpus_stats": corpus.stats,
        "cfg": {"lr": H1_LR, "weight_decay": H1_WD, "batch": H1_BATCH,
                "steps_per_epoch": H1_STEPS_PER_EPOCH, "epochs": H1_EPOCHS,
                "patience": H1_PATIENCE, "seed": seed},
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


def main() -> int:
    parser = argparse.ArgumentParser(description="H1 单臂训练（数据广度唯一变量）")
    parser.add_argument("--arm", choices=list(H1_ARMS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model, info = train_one(args.arm, seed=args.seed)
    _save_ckpt(model, f"{args.arm}_s{args.seed}", info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
