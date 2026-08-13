"""阶段 4 监督臂：冻结 tokenizer 词表 + 小 Mamba（计划 §6）。

三级消融闸门（逐级解锁，不回头调超参）：

    - R-1 单向 Mamba：valid 日均 RankIC ≥ 0.02 → 解锁 R-2 且 R-1 过引擎；
    - R-2 双向（同块反转序列二跑、输出相加）：valid RankIC 较 R-1 提升 ≥ 0.005 → 解锁 R-3；
    - R-3 +AGC 特征图卷积（SAMBA 式）：同样提升门槛。

过闸级别的引擎判据：oos1 AER(等权) > 0 且 > B1 oos1（−13.74%）+10pp → 存活。

训练协议（冻结）：d_model=64/n_layer=2/d_state=16/expand=2/lr=1e-4/wd=1e-3/
batch=2048/MSE/早停=valid 日均 RankIC 连续 5 epoch 不升/seed=42。

切分（冻结）：train 2019-01-01~2024-06-30，valid = paper 窗，judge = oos1 窗。
**训练与早停只看 train/valid，oos1 只在臂定型后跑一次。**

用法::

    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_stage4 --encode
    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_stage4 --train
    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_stage4 --engine
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from baseline_suite.common import DATA_DIR as BL_DATA_DIR
from improve_suite.common import DATA_DIR
from improve_suite.token_dataset import OHLCVA, HORIZON, LOOKBACK, build_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
B1_OOS_EW = -0.1374  # §0/既有：B1 oos AER(等权)
TOKEN_DIR = DATA_DIR / "tokens"

# 切分（冻结）
TRAIN_START, TRAIN_END = "2019-01-01", "2024-06-30"


# ============================================================
# token 编码（批量按日，落盘）
# ============================================================


def cmd_encode(device: str) -> None:
    """编码 train/valid/oos 三段 token 样本，落盘 .pt。"""
    from improve_suite.common import ImproveConfig
    from kronos_qlib import QlibProvider
    from model import KronosTokenizer

    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ImproveConfig.load(window="paper")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name).to(device)

    provider = QlibProvider("csi300", "2018-06-01", "2026-07-24")
    cal = provider.trading_days("2018-06-01", "2026-07-24")

    segments = {
        "train": cal[(cal >= TRAIN_START) & (cal <= TRAIN_END)],
        "valid": cal[(cal >= "2024-07-01") & (cal <= "2025-06-30")],
        "judge": cal[(cal >= "2025-07-01") & (cal <= "2026-07-24")],
    }
    for name, dates in segments.items():
        out = TOKEN_DIR / f"tokens_{name}.pt"
        if out.exists():
            logger.info(f"{name} 已存在，跳过：{out}")
            continue
        logger.info(f"编码 {name}：{len(dates)} 日")
        samples = build_samples(provider, tokenizer, dates=dates, pool="csi300", device=device)
        torch.save(samples, out)
        logger.info(f"{name} 落盘 {out}：{len(samples['y'])} 样本")


# ============================================================
# RankIC 评估
# ============================================================


def daily_rankic(scores: np.ndarray, ys: np.ndarray, dates: list[str]) -> float:
    """逐日截面 Spearman RankIC 的均值。

    :param scores / ys: (N,) 模型分数 / 真实标签。
    :param dates: 每样本的决策日（分组键）。
    """
    from scipy.stats import spearmanr

    df = pd.DataFrame({"date": dates, "score": scores, "y": ys})
    ics = []
    for d, g in df.groupby("date"):
        if len(g) < 5:
            continue
        rho, _ = spearmanr(g["score"], g["y"])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


# ============================================================
# 训练
# ============================================================


class _Bidirectional(nn.Module):
    """R-2：同块反转序列二跑、输出相加。"""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, s1, s2):
        fwd = self.base(s1, s2)
        bwd = self.base(s1.flip(1), s2.flip(1))
        return fwd + bwd


def _iterate_batches(s1, s2, y, batch_size, shuffle=True):
    n = len(y)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, batch_size):
        b = idx[s : s + batch_size]
        yield s1[b], s2[b], y[b]


def train_level(
    level: str, train_data, valid_data, *, cfg, device, max_epochs=50, patience=5
):
    """训练某一闸门级别（R-1 单向 / R-2 双向）。返回 (model, best_valid_rankic)。"""
    from improve_suite.mamba_min import MambaSeqRegressor

    torch.manual_seed(42)
    model = MambaSeqRegressor(cfg).to(device)
    if level == "R-2":
        model = _Bidirectional(model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    s1_tr, s2_tr, y_tr = train_data["s1"], train_data["s2"], train_data["y"]
    s1_va, s2_va, y_va = valid_data["s1"], valid_data["s2"], valid_data["y"]
    va_dates = [m[0] for m in valid_data["meta"]]

    best_ic = -1.0
    no_improve = 0
    best_state = None
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        n_b = 0
        for s1b, s2b, yb in _iterate_batches(s1_tr, s2_tr, y_tr, 2048):
            s1b, s2b, yb = s1b.to(device), s2b.to(device), yb.to(device)
            pred = model(s1b, s2b)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss)
            n_b += 1
        # valid RankIC
        model.eval()
        with torch.no_grad():
            scores = []
            for s in range(0, len(y_va), 4096):
                sc = model(s1_va[s : s + 4096].to(device), s2_va[s : s + 4096].to(device))
                scores.append(sc.cpu().numpy())
            scores = np.concatenate(scores)
        ic = daily_rankic(scores, y_va.numpy(), va_dates)
        logger.info(f"[{level}] epoch {epoch + 1}: train_loss={total_loss / max(n_b,1):.6f} valid_RankIC={ic:+.4f}")
        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[{level}] 早停 @epoch {epoch + 1}（{patience} epoch 无提升）")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ic


def cmd_train(device: str) -> None:
    """R-1 → 闸门 → R-2（→ 闸门 → R-3 记录未实现）。"""
    from improve_suite.mamba_min import MambaConfig

    train_data = torch.load(TOKEN_DIR / "tokens_train.pt", weights_only=False)
    valid_data = torch.load(TOKEN_DIR / "tokens_valid.pt", weights_only=False)

    # 从 tokenizer 读 vocab
    from improve_suite.common import ImproveConfig
    from model import KronosTokenizer

    cfg_q = ImproveConfig.load(window="paper")
    tokenizer = KronosTokenizer.from_pretrained(cfg_q.tokenizer_name)
    s1_vocab = 2 ** tokenizer.s1_bits
    s2_vocab = 2 ** tokenizer.s2_bits
    mcfg = MambaConfig(s1_vocab=s1_vocab, s2_vocab=s2_vocab)
    logger.info(f"vocab: s1={s1_vocab} (2^{tokenizer.s1_bits}), s2={s2_vocab} (2^{tokenizer.s2_bits})")

    gates = {}
    # R-1
    r1, ic1 = train_level("R-1", train_data, valid_data, cfg=mcfg, device=device)
    gates["R-1"] = {"valid_rankic": ic1, "threshold": 0.02, "passed": ic1 >= 0.02}
    torch.save(r1.state_dict(), DATA_DIR / "stage4_R1.pt")
    logger.info(f"R-1 valid RankIC={ic1:+.4f} → {'过闸' if ic1 >= 0.02 else '未过闸（停在 R-1）'}")

    if not gates["R-1"]["passed"]:
        final_level, final_model = "R-1", r1
    else:
        # R-2
        r2, ic2 = train_level("R-2", train_data, valid_data, cfg=mcfg, device=device)
        improve = ic2 - ic1
        gates["R-2"] = {"valid_rankic": ic2, "improvement": improve, "threshold": 0.005, "passed": improve >= 0.005}
        torch.save(r2.state_dict(), DATA_DIR / "stage4_R2.pt")
        logger.info(f"R-2 valid RankIC={ic2:+.4f}（提升 {improve:+.4f}）→ {'过闸' if improve >= 0.005 else '未过闸（停在 R-1）'}")
        final_level, final_model = ("R-2", r2) if improve >= 0.005 else ("R-1", r1)
        # R-3（+AGC）：本计划记录为"未实现"——AGC 跨截面图卷积需另立工程，闸门逻辑就位
        gates["R-3"] = {"note": "AGC 图卷积未实现（SAMBA 式跨截面），如 R-2 过闸则另立计划"}

    torch.save({"level": final_level, "state_dict": final_model.state_dict(), "gates": gates}, DATA_DIR / "stage4_final.pt")

    out = {"stage": 4, "gates": gates, "final_level": final_level,
           "protocol": {"d_model": 64, "n_layer": 2, "d_state": 16, "expand": 2,
                        "lr": 1e-4, "wd": 1e-3, "batch": 2048, "patience": 5, "seed": 42}}
    with open(DATA_DIR / "stage4_train_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"==== 阶段 4 训练：最终级别 {final_level} ====")


# ============================================================
# 引擎评估（最高过闸级，oos1 只跑一次）
# ============================================================


def cmd_engine(device: str) -> None:
    """最高过闸级过 oos1 引擎。"""
    from baseline_suite.common import BaselineConfig
    from baseline_suite.pipeline import build_dual_benchmarks, run_group
    from baseline_suite.signal import build_px_tradeable
    from improve_suite.mamba_min import MambaConfig, MambaSeqRegressor

    ckpt = torch.load(DATA_DIR / "stage4_final.pt", weights_only=False)
    level = ckpt["level"]
    is_bi = level == "R-2"

    from improve_suite.common import ImproveConfig
    from model import KronosTokenizer

    cfg_q = ImproveConfig.load(window="paper")
    tokenizer = KronosTokenizer.from_pretrained(cfg_q.tokenizer_name)
    mcfg = MambaConfig(s1_vocab=2 ** tokenizer.s1_bits, s2_vocab=2 ** tokenizer.s2_bits)
    model = MambaSeqRegressor(mcfg).to(device)
    if is_bi:
        model = _Bidirectional(model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    judge_data = torch.load(TOKEN_DIR / "tokens_judge.pt", weights_only=False)
    with torch.no_grad():
        scores = []
        for s in range(0, len(judge_data["y"]), 4096):
            sc = model(judge_data["s1"][s:s+4096].to(device), judge_data["s2"][s:s+4096].to(device))
            scores.append(sc.cpu().numpy())
        scores = np.concatenate(scores)

    # 组装 oos 信号宽表
    meta = judge_data["meta"]
    rows = {}
    for (ds, code), sc in zip(meta, scores):
        rows.setdefault(ds, {})[code] = float(sc)
    sig = pd.DataFrame(rows).T.sort_index()
    sig.index = pd.to_datetime(sig.index)

    bc = BaselineConfig.load(window="oos")
    from kronos_qlib import QlibProvider

    provider = QlibProvider(bc.pool, bc.backtest_start, bc.backtest_end)
    rebalances = provider.trading_days(bc.backtest_start, bc.backtest_end)
    sig = sig.reindex(rebalances)
    all_cols = sorted(sig.columns)
    px, trd = build_px_tradeable(provider, bc, rebalances, all_cols)
    bench_idx, bench_ew, _ = build_dual_benchmarks(provider, bc, px, trd)
    pi, pe, dr, _, _ = run_group(sig, px, trd, bench_idx, bench_ew, cfg=bc, name=f"stage4_{level}_oos")

    survives = pe.aer > 0 and pe.aer > B1_OOS_EW + 0.10
    verdict = "监督臂存活，待前向确认" if survives else f"未达引擎判据（oos AER(等权)={pe.aer:+.2%} ≤ {B1_OOS_EW+0.10:+.2%}）"
    logger.info(f"==== 阶段 4 引擎 [{level}] oos：AER(等权)={pe.aer:+.2%} → {verdict} ====")

    out = {"stage": "4_engine", "level": level,
           "oos_perf_idx": pi.to_dict(), "oos_perf_ew": pe.to_dict(),
           "b1_oos_ew": B1_OOS_EW, "survives": bool(survives), "verdict": verdict}
    with open(DATA_DIR / "stage4_engine_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 4 监督臂")
    parser.add_argument("--encode", action="store_true", help="编码 token 样本")
    parser.add_argument("--train", action="store_true", help="训练 + 闸门")
    parser.add_argument("--engine", action="store_true", help="最高过闸级过 oos 引擎")
    args = parser.parse_args()
    device = "cuda:0"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.encode:
        cmd_encode(device)
    elif args.train:
        cmd_train(device)
    elif args.engine:
        cmd_engine(device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
