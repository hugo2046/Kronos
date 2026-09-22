"""TAS1 训练器（计划 §5.2 固定配方 + §4 STATIC 固定平均 Z）。

用法（P2）::

    /home/user/miniconda3/envs/quant/bin/python -m tas_state.train \\
        --family dynamic --seed 42

固定配方（冻结，不 warmup、不 lr 网格）：AdamW lr=3e-4 wd=0.01、
梯度裁剪 1.0、有效 batch=64（microbatch × 梯度累积）、4000 步、
1000/2000/4000 三 checkpoint；按 optimizer step 奇偶交替 PRE/POST（1:1）。

验证选择（§5.2）：每个 checkpoint 在 2025H1 合法决策日自由生成 N=20，
k=10 截面 RankIC 均值选点，平局到 1e-6 取较早 checkpoint。T 族只按
PRE 验证分选点；POST 用同一 checkpoint；STATIC 按自己的 PRE 分选。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from tas_state.config import PKG_DIR, REPO_ROOT, TASConfig
from tas_state.data import TASCorpus, load_corpora
from tas_state.model import ConditionalKronos, StateEncoder

OUT_DIR = PKG_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# STATIC 固定平均 Z 缓存（三种子共用；key=corpus+config+底座哈希）
STATIC_Z_PATH = PKG_DIR / "data" / "static_z_mean.npz"


# ============================================================
# checkpoint 选择规则（冻结；测试锁定）
# ============================================================
def select_checkpoint(val_scores: dict[int, float]) -> int:
    """按验证 RankIC 均值选点；平局（|Δ|≤1e-6）取较早 step。

    只读验证分——历史评估文件不进入本函数（协议测试锁定的边界）。
    """
    best_step, best_score = None, -np.inf
    for step in sorted(val_scores):  # 升序遍历：平局自然保留较早者
        s = val_scores[step]
        if s > best_score + 1e-6:
            best_step, best_score = step, s
    if best_step is None:
        raise ValueError(f"无有效验证分：{val_scores}")
    return best_step


# ============================================================
# STATIC 固定平均 Z（计划 §4）
# ============================================================
def build_static_z(
    ck: ConditionalKronos, corpus: TASCorpus, cfg: TASConfig
) -> np.ndarray:
    """训练集固定 4096 样本的平均 Z（SHA256 字典序；三种子共用）。

    不遍历全量隐状态缓存，不从验证/评估窗口估计；不足 4096 用全量并记录。
    """
    keys = corpus.static_sample_keys(cfg.static_sample_count)
    logger.info(f"STATIC 平均 Z：{len(keys)} 样本（规则键 SHA256 字典序）")
    zs = []
    micro = 64
    with torch.no_grad():
        for s in range(0, len(keys), micro):
            b = corpus.batch(keys[s : s + micro])
            x = torch.tensor(b["X"]).to(next(ck.parameters()).device)
            st = torch.tensor(b["stamp"]).to(x.device)
            H, _ = ck.first_pass(x, st)
            zs.append(ck.summarize(H).cpu().numpy())
    z_mean = np.concatenate(zs, axis=0).mean(axis=0)  # [4, 2d]
    np.savez(
        STATIC_Z_PATH,
        z_mean=z_mean,
        n_samples=len(keys),
        config_sha256=cfg.sha256(),
        corpus_sha256=_file_sha(REPO_ROOT / cfg.train_corpus_path),
    )
    logger.info(f"STATIC 平均 Z 已冻结：{STATIC_Z_PATH}（n={len(keys)}）")
    return z_mean


def load_static_z(cfg: TASConfig) -> np.ndarray:
    """读缓存；身份不匹配（config/tokenizer/corpus 变化）拒绝复用。"""
    if not STATIC_Z_PATH.exists():
        raise FileNotFoundError(f"STATIC Z 缓存不存在：{STATIC_Z_PATH}（先构建）")
    d = np.load(STATIC_Z_PATH, allow_pickle=False)
    expect = {
        "config_sha256": cfg.sha256(),
        "corpus_sha256": _file_sha(REPO_ROOT / cfg.train_corpus_path),
    }
    for k, v in expect.items():
        got = str(d[k]) if k in d else "<missing>"
        if got != v:
            raise ValueError(f"STATIC Z 身份不匹配：{k} {got} != {v}（必须重建）")
    return d["z_mean"]


def _file_sha(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ============================================================
# 训练循环
# ============================================================
def train_family(
    family: str,
    seed: int,
    cfg: TASConfig,
    *,
    max_steps: int | None = None,
    device: str | None = None,
) -> Path:
    """训练一个模型族（dynamic / static），返回输出目录。

    :param max_steps: 调试截断（正式运行 = cfg.total_steps）。
    """
    from model.kronos import Kronos, KronosTokenizer

    if family not in ("dynamic", "static"):
        raise ValueError(f"未知族 {family!r}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or cfg.device)

    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    kronos = Kronos.from_pretrained(cfg.model_name)
    encoder = StateEncoder(
        kronos.d_model, hidden=cfg.state_hidden, n_states=cfg.n_states
    )
    ck = ConditionalKronos(kronos, tokenizer, encoder, cfg).to(dev)

    out_dir = OUT_DIR / f"{family}_s{seed}"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_corpus, _ = load_corpora(cfg)
    sampler = train_corpus.sampler(seed=seed)

    z_mean = None
    if family == "static":
        # STATIC Z 身份锚定 = config + corpus（tokenizer 权重哈希由 preflight
        # manifest 记录，signals 阶段统一校验）
        try:
            z_mean = load_static_z(cfg)
        except (FileNotFoundError, ValueError):
            z_mean = build_static_z(ck, train_corpus, cfg)

    opt = torch.optim.AdamW(
        encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    micro = cfg.microbatch
    accum = max(1, cfg.effective_batch // micro)
    total = max_steps or cfg.total_steps
    ckpt_steps = sorted(
        s for s in cfg.ckpt_steps if s <= total
    ) or [total]

    ck.train()
    step = 0
    t0 = time.perf_counter()
    loss_val = float("nan")
    while step < total:
        opt.zero_grad()
        loss_acc = 0.0
        for _ in range(accum):
            layout = "pre" if step % 2 == 0 else "post"  # 奇偶交替（§4）
            keys = sampler.draw(micro)
            b = train_corpus.batch(keys)
            x = torch.tensor(b["X"]).to(dev)
            st = torch.tensor(b["stamp"]).to(dev)
            yst = torch.tensor(b["y_stamp"]).to(dev)
            with torch.no_grad():
                ys1 = tokenizer.encode(torch.tensor(b["Y"]).to(dev), half=True)
                if family == "dynamic":
                    S, _ = ck.encode_state(x, st)
                else:
                    zt = torch.tensor(z_mean, dtype=torch.float32).to(dev)
                    zt = zt.unsqueeze(0).expand(x.shape[0], -1, -1)
                    S = ck.encoder(zt)
                    _, hist_unused = ck.first_pass(x, st)
            fwd = ck.teacher_forced_forward(
                x, st, ys1[0], ys1[1], yst, S=S, layout=layout
            )
            # L = mean_k [CE(s1)+CE(s2)] / 2（§3.4；DualHead.compute_loss 同式）
            loss = F.cross_entropy(
                fwd.s1_logits.reshape(-1, fwd.s1_logits.shape[-1]),
                ys1[0].reshape(-1),
            )
            loss = (loss + F.cross_entropy(
                fwd.s2_logits.reshape(-1, fwd.s2_logits.shape[-1]),
                ys1[1].reshape(-1),
            )) / 2
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        loss_val = loss_acc
        if step % 100 == 0 or step in ckpt_steps:
            el = time.perf_counter() - t0
            logger.info(
                f"[{family}/s{seed}] step {step}/{total} loss={loss_val:.4f} "
                f"({el:.0f}s, layout 奇偶交替)"
            )
        if step in ckpt_steps:
            torch.save(
                {
                    "step": step,
                    "encoder_state": encoder.state_dict(),
                    "family": family,
                    "seed": seed,
                    "config_sha256": cfg.sha256(),
                    "loss": loss_val,
                },
                out_dir / "checkpoints" / f"step{step:05d}.pt",
            )
    meta = {
        "family": family, "seed": seed, "config_sha256": cfg.sha256(),
        "total_steps": total, "microbatch": micro, "accum": accum,
        "final_loss": loss_val, "wall_seconds": round(time.perf_counter() - t0, 1),
        "static_z_n": (int(z_mean.shape[0]) if z_mean is not None else None),
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[{family}/s{seed}] 训练完成：{out_dir}")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PKG_DIR / "config.json"))
    ap.add_argument("--family", choices=["dynamic", "static"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None, help="调试截断")
    args = ap.parse_args()
    cfg = TASConfig.load(args.config)
    if args.seed not in (42, 43, 44):
        raise ValueError(f"seed 必须是 42/43/44，收到 {args.seed}")
    train_family(args.family, args.seed, cfg, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
