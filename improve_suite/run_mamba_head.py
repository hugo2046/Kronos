"""Mamba 时序头（B3 延伸）· 训练 + 评估编排（计划 §3~§5）。

**纪律（计划 §9）**：
    - 臂/种子/超参冻结：T1（MambaTemporalHead）一结构 × {42,43,44} 三种子 + D1（B3 结构）
      × {43,44} 两种子；跑后不追加；
    - 训练协议常量**逐字 import** 自 :mod:`cross_section_kda.train` / :mod:`cross_section_kda.data`
      （下方 ``T1_*`` 镜像，``test_protocol_constants_match`` 对拍防漂移）；
    - 训练只看 train/早停段曲线（早停段日均 RankIC）；oos1 数字在 5 checkpoint 全定型后
      才过引擎，判读一次封盘；
    - forward 窗（2026-07-25 后）零接触；
    - **不改** ``cross_section_kda/`` 任何文件；B3 既有 checkpoint 只读（D1 复用 ``train_arm``
      但**不调** ``save_checkpoint``，自存到 ``improve_suite/data/``，绝不覆盖 ``B3_best.pt``）。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_mamba_head cache   # 阶段2.1
    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_mamba_head train   # 阶段2.2~2.3
    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_mamba_head eval    # 阶段3
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from scipy import stats

# —— 训练协议常量：逐字 import 自 cross_section_kda（冻结源，禁止漂移）——
from cross_section_kda.data import (
    EARLY_STOP_END,
    EARLY_STOP_START,
    FINAL_START,
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
    SEED,
    WEIGHT_DECAY,
    TrainConfig,
    build_split_cache,
    save_checkpoint,  # noqa: F401  （仅对照，本模块不用于 D1——见 _save_ckpt）
    set_seed,
    train_arm,
)

# T1 镜像（供 test_protocol_constants_match 对拍；值来自上面的 import，零漂移）
T1_LR = LR["B3"]            # 3e-4
T1_WD = WEIGHT_DECAY        # 0.01
T1_BATCH = BATCH_SIZE       # 128
T1_EPOCHS = EPOCHS          # 50
T1_PATIENCE = PATIENCE      # 5
T1_TRAIN_START = TRAIN_START
T1_TRAIN_END = TRAIN_END
T1_ES_START = EARLY_STOP_START
T1_ES_END = EARLY_STOP_END
T1_PURGE = PURGE

# 臂 / 种子（计划 §2，冻结）
T1_SEEDS = (42, 43, 44)
D1_SEEDS = (43, 44)

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"
REPO_ROOT = PKG_DIR.parent
KDA_DATA_DIR = REPO_ROOT / "cross_section_kda" / "data"
HIDDEN_CACHE_PATH = DATA_DIR / "hidden_cache_train_es.pt"

# 引擎配置（计划 §0，跑前冻结）
ENGINE_CFG: dict[str, dict] = {
    "csi300": dict(top_k=50, drop_n=5, min_hold=5, cost_bps=15.0),
    "csi500": dict(top_k=100, drop_n=10, min_hold=5, cost_bps=15.0),
}
_CSI_INDEX = {"csi300": "000300.SH", "csi500": "000905.SH"}


# ============================================================
# 主干加载
# ============================================================


def _load_backbone(device: str):
    """加载冻结 Kronos 主干（B3/T1/D1 共用同一实例）。"""
    from cross_section_kda import KronosFrozenBackbone
    from improve_suite.common import ImproveConfig
    from model import Kronos, KronosTokenizer

    cfg = ImproveConfig.load(window="paper")  # 仅取 model/tokenizer 名
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name).to(device)
    kronos = Kronos.from_pretrained(cfg.model_name).to(device)
    return KronosFrozenBackbone(tokenizer=tokenizer, kronos=kronos, device=device)


# ============================================================
# 阶段 2.1：隐状态缓存
# ============================================================


def _extract_day_hidden(backbone, x_norm: torch.Tensor, stamp: torch.Tensor,
                        device: str, chunk: int = 128) -> torch.Tensor:
    """对一个调仓日的 [N,90,6] 分块过冻结主干，返回 CPU 上 [N,90,832]。"""
    hiddens: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(x_norm), chunk):
            h = backbone.extract(x_norm[i:i + chunk].to(device), stamp[i:i + chunk].to(device))
            hiddens.append(h.cpu())
    return torch.cat(hiddens, dim=0)


def build_and_save_cache(backbone, device: str, *, verify_n: int = 3) -> dict:
    """预提取 train+早停段全样本日的 [B,90,832] 隐状态并落盘。

    缓存结构::

        {
          "train": {"hidden": Tensor[Ntr,90,832], "y": Tensor[Ntr]},  # 扁平
          "es":    [{"date", "hidden": Tensor[n,90,832], "fwd_ret_raw": ndarray[n],
                     "codes": [...]}, ...],                            # 按日（日均 RankIC 用）
          "refs":  [{"x": Tensor[1,90,6], "stamp": Tensor[1,90,5], "hidden": Tensor[1,90,832]}, ...]
        }

    主干确定性已由 §0 保证（eval + no_grad），缓存与在线逐位等价（下方对拍验证）。
    """
    from kronos_qlib import QlibProvider

    p = QlibProvider("csi300", TRAIN_START, EARLY_STOP_END)
    train_batches = build_split_cache(p, start=TRAIN_START, end=TRAIN_END,
                                      pool="csi300", rebalance_only=False)
    es_batches = build_split_cache(p, start=EARLY_STOP_START, end=EARLY_STOP_END,
                                   pool="csi300", rebalance_only=False)

    # —— train 扁平 ——
    tr_hidden, tr_y = [], []
    refs: list[dict] = []
    for b in train_batches:
        h = _extract_day_hidden(backbone, b.x_norm, b.stamp, device)  # [N,90,832]
        tr_hidden.append(h)
        tr_y.append(torch.from_numpy(b.y_z))
        # 收集对拍参考样本（前若干日各抽 1）
        if len(refs) < verify_n:
            idx = 0
            refs.append({"x": b.x_norm[idx].clone(), "stamp": b.stamp[idx].clone(), "hidden": h[idx].clone()})
    train_hidden = torch.cat(tr_hidden, dim=0)
    train_y = torch.cat(tr_y, dim=0)

    # —— es 按日 ——
    es_days: list[dict] = []
    for b in es_batches:
        h = _extract_day_hidden(backbone, b.x_norm, b.stamp, device)
        es_days.append({"date": b.date, "hidden": h, "fwd_ret_raw": b.fwd_ret_raw, "codes": b.codes})

    cache = {"train": {"hidden": train_hidden, "y": train_y}, "es": es_days, "refs": refs}

    # —— 对拍：缓存 vs 在线（3 样本逐位一致）——
    max_diff = 0.0
    for r in refs:
        with torch.no_grad():
            h_online = backbone.extract(r["x"].to(device), r["stamp"].to(device)).cpu()
        d = (h_online - r["hidden"]).abs().max().item()
        max_diff = max(max_diff, d)
    logger.info(f"隐状态缓存对拍：{len(refs)} 样本 max|Δ|={max_diff:.3e}（主干确定性保证）")
    assert max_diff < 1e-5, f"缓存与在线不一致：max|Δ|={max_diff}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, HIDDEN_CACHE_PATH)
    logger.info(
        f"隐状态缓存落盘 → {HIDDEN_CACHE_PATH.name} | "
        f"train {train_hidden.shape[0]} 样本、es {len(es_days)} 日"
    )
    return cache


# ============================================================
# 阶段 2.2：T1 训练（缓存）
# ============================================================


def _evaluate_rankic_cached(head, es_days: list[dict], device: str) -> float:
    """早停段日均 RankIC（缓存 hidden 喂 _decode；口径同 train._evaluate_rankic）。"""
    head.eval()
    rankics: list[float] = []
    with torch.no_grad():
        for day in es_days:
            score = head._decode(day["hidden"].to(device)).cpu().numpy()
            fwd = day["fwd_ret_raw"]
            valid = np.isfinite(score) & np.isfinite(fwd)
            if valid.sum() < 5:
                continue
            rho, _ = stats.spearmanr(score[valid], fwd[valid])
            if np.isfinite(rho):
                rankics.append(float(rho))
    return float(np.mean(rankics)) if rankics else float("nan")


def train_mamba_head(backbone, cache: dict, *, seed: int, device: str) -> tuple[nn.Module, dict]:
    """T1 单种子训练（喂缓存 hidden 给 _decode；AdamW/MSE/早停，协议同 B3）。

    缓存与在线 ``forward`` 数学等价（主干冻结 + no_grad）；唯一差别是跳过主干前向，
    数值不变（同 seed 同结果）。
    """
    from improve_suite.mamba_head import MambaTemporalHead

    set_seed(seed)
    model = MambaTemporalHead(backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[T1 s{seed}] 可训练参数 {n_trainable:,} | lr={T1_LR} wd={T1_WD} "
                f"batch={T1_BATCH} epochs≤{T1_EPOCHS} patience={T1_PATIENCE}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=T1_LR, weight_decay=T1_WD)
    loss_fn = nn.MSELoss()

    xs = cache["train"]["hidden"]          # [N,90,832] CPU
    ys = cache["train"]["y"]               # [N]
    es_days = cache["es"]
    n = xs.size(0)
    logger.info(f"[T1 s{seed}] 训练样本 {n} 条（缓存 hidden）")

    best_es = -np.inf
    best_state = None
    history: list[dict] = []
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, T1_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, T1_BATCH):
            idx = perm[i:i + T1_BATCH]
            h = xs[idx].to(device)             # [b,90,832]
            y = ys[idx].to(device)
            opt.zero_grad()
            score = model._decode(h)
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
        logger.info(f"[T1 s{seed}] epoch {epoch:02d}/{T1_EPOCHS} "
                    f"train_loss={train_loss:.5f} es_RankIC={es_rankic:+.4f} {marker}")
        if improved:
            best_es = es_rankic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= T1_PATIENCE:
                logger.info(f"[T1 s{seed}] 早停：{T1_PATIENCE} 个 epoch 无改善，最佳 es_RankIC={best_es:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    logger.info(f"[T1 s{seed}] 训练完成 用时 {elapsed:.1f}s 最佳 es_RankIC={best_es:+.4f}")

    info = {
        "arm": "T1", "seed": seed, "n_trainable": n_trainable,
        "best_es_rankic": best_es, "history": history, "n_epochs_run": len(history),
        "cfg": {"lr": T1_LR, "weight_decay": T1_WD, "batch_size": T1_BATCH,
                "epochs": T1_EPOCHS, "patience": T1_PATIENCE, "seed": seed},
    }
    return model, info


# ============================================================
# checkpoint 落盘（improve_suite/data，不动 cross_section_kda/data）
# ============================================================


def _save_ckpt(model: nn.Module, name: str, info: dict) -> Path:
    """保存最佳 checkpoint + history 到 improve_suite/data/（不入库，gitignore *.pt）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = DATA_DIR / f"{name}_best.pt"
    torch.save({"state_dict": model.state_dict(), "info": info}, ckpt)
    hist = DATA_DIR / f"{name}_history.json"
    with open(hist, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=float)
    logger.info(f"[{name}] checkpoint → {ckpt.name} | history → {hist.name}")
    return ckpt


# ============================================================
# 阶段 2.3：D1（B3 结构）新训（在线，复用 train_arm）
# ============================================================


def train_d1(backbone, train_batches, es_batches, *, seed: int, device: str) -> tuple[nn.Module, dict]:
    """D1：B3 结构按原协议新训（seed 43/44，种子稳健性诊断）。

    复用 :func:`cross_section_kda.train.train_arm`（``arm="B3"``），**不调** ``save_checkpoint``
    （它写 ``cross_section_kda/data/B3_best.pt`` 会覆盖既有 T0 checkpoint——严禁）。
    B3KronosKdaHead.forward 不可改，无法吃缓存 → 在线训练骨干（如实记录此受限点）。
    """
    model, info = train_arm("B3", backbone, train_batches, es_batches,
                            device=device, cfg=TrainConfig(arm="B3", seed=seed))
    info = dict(info)
    info["arm"] = "D1"
    info["cfg"] = dict(info["cfg"])
    info["cfg"]["seed"] = seed
    logger.info(f"[D1 s{seed}] 完成（B3 结构重训）可训练参数 {info['n_trainable']:,} "
                f"最佳 es_RankIC={info['best_es_rankic']:+.4f}")
    return model, info


# ============================================================
# 阶段 2 编排
# ============================================================


def run_stage2(device: str) -> dict:
    """阶段 2 全流程：建缓存 → T1×3 → D1×2。"""
    backbone = _load_backbone(device)

    # 2.1 缓存（已存在则加载）
    if HIDDEN_CACHE_PATH.exists():
        logger.info(f"隐状态缓存已存在，加载 {HIDDEN_CACHE_PATH.name}")
        cache = torch.load(HIDDEN_CACHE_PATH, map_location="cpu", weights_only=False)
    else:
        cache = build_and_save_cache(backbone, device)

    # D1 在线训练需 SampleTensorBatch（与缓存同源，复用 build_split_cache）
    from kronos_qlib import QlibProvider

    p = QlibProvider("csi300", TRAIN_START, EARLY_STOP_END)
    train_batches = build_split_cache(p, start=TRAIN_START, end=TRAIN_END,
                                      pool="csi300", rebalance_only=False)
    es_batches = build_split_cache(p, start=EARLY_STOP_START, end=EARLY_STOP_END,
                                   pool="csi300", rebalance_only=False)

    records = {}
    # 2.2 T1 × {42,43,44}
    for seed in T1_SEEDS:
        logger.info("=" * 70)
        logger.info(f"阶段 2.2：T1 训练 seed={seed}")
        logger.info("=" * 70)
        model, info = train_mamba_head(backbone, cache, seed=seed, device=device)
        _save_ckpt(model, f"T1_s{seed}", info)
        records[f"T1_s{seed}"] = {
            "n_trainable": info["n_trainable"], "best_es_rankic": info["best_es_rankic"],
            "n_epochs_run": info["n_epochs_run"],
        }

    # 2.3 D1 × {43,44}
    for seed in D1_SEEDS:
        logger.info("=" * 70)
        logger.info(f"阶段 2.3：D1（B3 结构）训练 seed={seed}")
        logger.info("=" * 70)
        model, info = train_d1(backbone, train_batches, es_batches, seed=seed, device=device)
        _save_ckpt(model, f"D1_s{seed}", info)
        records[f"D1_s{seed}"] = {
            "n_trainable": info["n_trainable"], "best_es_rankic": info["best_es_rankic"],
            "n_epochs_run": info["n_epochs_run"],
        }

    out = {"stage": "2_train", "records": records,
           "note": "D1 在线训练骨干（B3KronosKdaHead.forward 不可改，无法吃缓存）"}
    with open(DATA_DIR / "mamba_head_stage2_records.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    logger.info("==== 阶段 2 训练全部完成 ====")
    return out


# ============================================================
# 阶段 3：引擎评估
# ============================================================


def _model_signals_wide(model, provider, rebalances, pool: str, device: str) -> pd.DataFrame:
    """逐日构样本 → 模型全路径 forward（在线骨干）→ 信号宽表 [date × code]。"""
    from cross_section_kda.data import build_daily_samples

    model.eval()
    rows: list[dict] = []
    for i, d in enumerate(rebalances):
        ds = d.strftime("%Y-%m-%d")
        b = build_daily_samples(provider, date=ds, pool=pool)
        if b is None:
            rows.append({})
            continue
        with torch.no_grad():
            score = model(b.x_norm.to(device), b.stamp.to(device)).cpu().numpy()
        rows.append({c: float(s) for c, s in zip(b.codes, score)})
        if (i + 1) % 30 == 0 or i == 0:
            logger.info(f"  signal [{i + 1}/{len(rebalances)}] {ds}: {len(b.codes)} 只")
    return pd.DataFrame(rows, index=rebalances)


def _index_benchmark(pool: str, start: str, end: str) -> pd.Series:
    """池对应指数日收益（csi300=000300.SH / csi500=000905.SH）。"""
    from kronos_qlib import QlibProvider

    code = _CSI_INDEX[pool]
    p = QlibProvider([code], start, end)
    df = p.fetch(["$close"], freq="day")
    if len(df) == 0:
        raise RuntimeError(f"指数 {code} 在 {start}~{end} 无数据")
    if "instrument" in df.index.names:
        df = df.xs(code, level="instrument")
    close = df["close"].sort_index()
    return close.pct_change(fill_method=None).dropna()


def backtest_model(model, window: str, pool: str, device: str) -> dict:
    """单模型单池单窗：信号 → 引擎 → 双基准（指数 + 同池等权）AER/IR/MDD/换手。"""
    from dataclasses import replace

    from baseline_suite.common import BaselineConfig
    from baseline_suite.signal import build_px_tradeable
    from kronos_qlib import QlibProvider
    from paper_replication.benchmark import build_pool_equal_weight_benchmark
    from paper_replication.engine import (
        EngineConfig,
        attach_benchmark,
        compute_perf,
        run_portfolio,
    )

    base = BaselineConfig.load(window=window)
    overrides = {"csi300": dict(pool="csi300", top_k=50, drop_n=5),
                 "csi500": dict(pool="csi500", top_k=100, drop_n=10)}[pool]
    cfg = replace(base, **overrides)
    start, end = cfg.backtest_start, cfg.backtest_end

    provider = QlibProvider(pool, start, end)
    rebalances = provider.trading_days(start, end)
    sig = _model_signals_wide(model, provider, rebalances, pool, device)
    all_cols = sorted(sig.columns)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)

    bench_idx = _index_benchmark(pool, start, end)
    bench_ew = build_pool_equal_weight_benchmark(px, trd)

    ec = EngineConfig(**ENGINE_CFG[pool])
    sig_a = sig.reindex(index=px.index, columns=px.columns)
    daily_ret, _, trades = run_portfolio(sig_a, px, trd, cfg=ec)
    perf_idx = compute_perf(attach_benchmark(daily_ret, bench_idx), trades,
                            name=f"{pool}_{window}_idx")
    perf_ew = compute_perf(attach_benchmark(daily_ret, bench_ew), trades,
                           name=f"{pool}_{window}_ew")
    return {"perf_idx": perf_idx.to_dict(), "perf_ew": perf_ew.to_dict()}


def _load_all_models(backbone, device: str) -> dict[str, nn.Module]:
    """载入 6 个 checkpoint：T0(B3 既有) + T1×3 + D1×2。"""
    from cross_section_kda import B3KronosKdaHead
    from improve_suite.mamba_head import MambaTemporalHead

    models: dict[str, nn.Module] = {}

    def _load(head_cls, name, path):
        m = head_cls(backbone).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models[name] = m
        logger.info(f"载入 {name} ← {path.name}")

    # T0：B3 既有 checkpoint（只读）
    _load(B3KronosKdaHead, "T0", KDA_DATA_DIR / "B3_best.pt")
    # T1 × 3
    for seed in T1_SEEDS:
        _load(MambaTemporalHead, f"T1_s{seed}", DATA_DIR / f"T1_s{seed}_best.pt")
    # D1 × 2
    for seed in D1_SEEDS:
        _load(B3KronosKdaHead, f"D1_s{seed}", DATA_DIR / f"D1_s{seed}_best.pt")
    return models


def _judge(results: dict) -> dict:
    """预注册判据 1~5 一次性判读（计划 §5）。

    中位种子：T1 三种子按 oos1 两池 AER(等权) 均值排序取中位（防单种子运气/挑最好）。
    """
    pools = ("csi300", "csi500")

    def oos_aer_ew(model_key, pool):
        return results[model_key]["oos"][pool]["perf_ew"]["aer"]

    def oos_aer_idx(model_key, pool):
        return results[model_key]["oos"][pool]["perf_idx"]["aer"]

    seed_scores = {s: np.mean([oos_aer_ew(f"T1_s{s}", p) for p in pools]) for s in T1_SEEDS}
    median_seed = sorted(T1_SEEDS, key=lambda s: seed_scores[s])[1]  # 3 个的中位

    # 判据 1（主：存活）
    c1 = (all(oos_aer_ew(f"T1_s{median_seed}", p) > 0 for p in pools) and
          all(oos_aer_idx(f"T1_s{median_seed}", p) > 0 for p in pools))
    # 判据 2（强：改进）—— T1 中位种子 oos1 两池 AER(等权) ≥ B3(T0) + 2pp
    b3_ew = {p: oos_aer_ew("T0", p) for p in pools}
    c2 = all(oos_aer_ew(f"T1_s{median_seed}", p) >= b3_ew[p] + 0.02 for p in pools)
    # 判据 3（表示层）—— 两头族各自 ≥2 种子 oos1 csi300 AER(等权)>0
    kda_pos = sum(1 for k in ("T0", "D1_s43", "D1_s44") if oos_aer_ew(k, "csi300") > 0)
    mamba_pos = sum(1 for s in T1_SEEDS if oos_aer_ew(f"T1_s{s}", "csi300") > 0)
    c3 = kda_pos >= 2 and mamba_pos >= 2
    # 判据 4（脆弱性 D1）—— D1 两种子 oos1 csi300 AER(等权) 均 ≤ 0
    c4 = all(oos_aer_ew(f"D1_s{s}", "csi300") <= 0 for s in D1_SEEDS)
    # 判据 5（全败）
    t1_all_neg = all(oos_aer_ew(f"T1_s{s}", p) <= 0 for s in T1_SEEDS for p in pools)
    d1_all_neg = all(oos_aer_ew(f"D1_s{s}", "csi300") <= 0 for s in D1_SEEDS)
    c5 = t1_all_neg and d1_all_neg

    return {
        "median_seed": median_seed,
        "seed_scores_oos_aer_ew_mean": {f"s{s}": float(v) for s, v in seed_scores.items()},
        "b3_t0_oos_aer_ew": {p: float(b3_ew[p]) for p in pools},
        "criterion_1_survive": bool(c1),
        "criterion_2_improve_2pp": bool(c2),
        "criterion_3_representation": bool(c3),
        "criterion_4_d1_fragile": bool(c4),
        "criterion_5_all_fail": bool(c5),
        "kda_positive_seeds_csi300": int(kda_pos),
        "mamba_positive_seeds_csi300": int(mamba_pos),
    }


def run_stage3(device: str) -> dict:
    """阶段 3：6 模型 × 2 池 × 2 窗过引擎 → 全表 → 判据 1~5 一次性判读。"""
    backbone = _load_backbone(device)
    models = _load_all_models(backbone, device)

    pools = ("csi300", "csi500")
    windows = ("paper", "oos")
    results: dict[str, dict] = {k: {"paper": {}, "oos": {}} for k in models}

    for name, m in models.items():
        for window in windows:
            for pool in pools:
                logger.info(f"---- {name} | {window} | {pool} ----")
                perf = backtest_model(m, window, pool, device)
                results[name][window][pool] = perf

    judgment = _judge(results)
    logger.info("==== 阶段 3 判读（一次性封盘）====")
    logger.info(f"中位种子 = T1_s{judgment['median_seed']}")
    logger.info(f"判据1(存活)={judgment['criterion_1_survive']} "
                f"判据2(改进+2pp)={judgment['criterion_2_improve_2pp']} "
                f"判据3(表示层)={judgment['criterion_3_representation']}")
    logger.info(f"判据4(D1脆弱)={judgment['criterion_4_d1_fragile']} "
                f"判据5(全败)={judgment['criterion_5_all_fail']}")

    out = {
        "experiment": "mamba_head_b3_extension",
        "date": "2026-08-13",
        "models": list(models.keys()),
        "engine": {p: dict(cfg=ENGINE_CFG[p], index=_CSI_INDEX[p], equal_weight="同池等权") for p in pools},
        "windows": {"paper": "2024-07-01~2025-06-30", "oos": "2025-07-01~2026-07-24"},
        "results": results,
        "judgment": judgment,
        "capacity_note": ("B3/D1 head 实测 15.3M 可训练参数（非计划 §0 所述 ~1M）；"
                          "T1=1.09M。T1↔B3/D1 对照被容量混淆（~14×），判读降级为"
                          "'机制+容量联合差异'，详见结果文档。"),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "mamba_head_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"阶段 3 全表 + 判读 → {DATA_DIR / 'mamba_head_results.json'}")
    return out


# ============================================================
# main
# ============================================================


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else "stage2"
    device = "cuda:0"
    if phase == "cache":
        backbone = _load_backbone(device)
        build_and_save_cache(backbone, device)
    elif phase == "train":
        run_stage2(device)
    elif phase == "eval":
        run_stage3(device)
    else:
        logger.error(f"未知 phase={phase}，可选 cache / train / eval")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
