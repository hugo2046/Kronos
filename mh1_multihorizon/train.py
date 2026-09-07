"""MH1 训练编排：S/M 成对六次训练、验证选点、恢复与预算检查（计划 §5 任务 B）。

纪律（计划 §5/§7）：

- 底座 G1 s100 冻结（eval + no_grad + requires_grad=False，只读哈希引用）；
- 每种子 S/M 同初始化（``set_seed`` 后构造头，state_dict 逐位克隆）、同日序、
  同股票抽样（独立 ``random.Random(seed)`` 采样流）；
- 固定 15 epochs × 2000 更新，两臂均不早停；选点只读验证段 10 日 RankIC
  （并列最早、非有限不可选）；
- 16:30 登记 cron 错峰守卫：epoch 边界检查，守卫窗内卸载底座释放 GPU、等待
  窗口结束再继续（不修改/暂停登记任务本身）；
- GPU 实际计算预算 12h（不含守卫等待），超限即停并报告实测瓶颈；
- 中断可同协议续跑（per-epoch checkpoint：模型/优化器/采样 RNG/进度；
  载荷仅张量+Python 原语，``weights_only=True`` 安全加载）。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from mh1_multihorizon.config import (
    ARMS, BATCH, DATA_DIR, EPOCHS, GPU_BUDGET_HOURS, HORIZONS, LR,
    MAIN_HORIZON_IDX, POOL, REGISTRY_GUARD, RUN_ORDER, STEPS_PER_EPOCH,
    TRAIN_LABEL_END, TRAIN_START, VAL_LABEL_END, VAL_START, WD,
    sha256_file, verify_protocol,
)
from mh1_multihorizon.heads import MultiHorizonHead, horizon_loss

CKPT_DIR = DATA_DIR / "ckpt"
BUDGET_PATH = DATA_DIR / "gpu_budget.json"


# ============================================================
# 选点（只读主期限 10 日 RankIC）
# ============================================================


def select_best_epoch(history: list[dict]) -> int:
    """验证段 10 日日均 RankIC 最大的 epoch；并列选最早，非有限不可选。

    :raises ValueError: 全部 epoch 非有限（无法选点）。
    """
    best_epoch, best_val = None, -np.inf
    for row in history:
        v = row.get("val_rankic_main")
        if v is None or not np.isfinite(v):
            continue
        if v > best_val:   # 严格大于 → 并列保留最早
            best_val, best_epoch = v, int(row["epoch"])
    if best_epoch is None:
        raise ValueError("验证选点失败：无任何有限 val_rankic_main")
    return best_epoch


# ============================================================
# checkpoint 范围（只含新头；底座哈希引用）
# ============================================================


def checkpoint_backbone_refs(backbone) -> dict:
    """底座经哈希引用（完整 SHA256 + 维数），不进 checkpoint。"""
    from finetune_suite.train_g1 import G1Config

    g1 = G1Config()
    return {
        "tokenizer_sha256": sha256_file(
            Path(g1.finetuned_tokenizer_path) / "model.safetensors"),
        "predictor_sha256": sha256_file(
            Path(g1.finetuned_predictor_path) / "model.safetensors"),
        "d_model": int(backbone.d_model),
    }


def _state_hash(state: dict) -> str:
    """state_dict 稳定哈希（S/M 同初始化对拍用）。"""
    import hashlib

    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def save_checkpoint(
    head: torch.nn.Module, opt: torch.optim.Optimizer, *, epoch: int,
    history: list[dict], extra: dict, out_dir: Path | str | None,
) -> dict:
    """构造 checkpoint dict；``out_dir`` 给定时落盘（只含新头/优化器/进度）。

    载荷仅张量与 Python 原语（history 数值已转 float、RNG 状态为
    int/float 元组），全程可 ``torch.load(weights_only=True)`` 安全加载。
    """
    ckpt = {
        "state_dict": head.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": int(epoch),
        "history": history,
        **extra,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)
    return ckpt


def load_checkpoint(path: Path | str) -> dict:
    """安全加载 checkpoint（``weights_only=True``，拒绝任意代码执行）。"""
    return torch.load(path, map_location="cpu", weights_only=True)


# ============================================================
# 16:30 登记 cron 错峰守卫（不修改登记任务；守卫窗内释放 GPU）
# ============================================================


def in_registry_guard(now: datetime | None = None) -> bool:
    """当前时刻是否落在 16:30 登记 cron 的守卫窗内（默认 16:25~16:45）。

    该登记任务历史证据使用 GPU（2026-08-20 曾因他进程占显存 OOM，见
    ``finetune_suite/registry/cron.log``），故守卫窗内训练进程必须让出显存。
    """
    t = (now or datetime.now()).time().strftime("%H:%M")
    return REGISTRY_GUARD[0] <= t < REGISTRY_GUARD[1]


def wait_out_guard(backbone, device: str, on_epoch: int) -> None:
    """守卫窗内卸载底座释放 GPU，等待窗口结束再回载（epoch 边界调用）。"""
    if not in_registry_guard():
        return
    logger.warning(f"[guard] epoch {on_epoch} 边界进入 16:30 登记守卫窗 "
                   f"{REGISTRY_GUARD}：卸载底座释放 GPU")
    backbone.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    n_waited = 0
    while in_registry_guard():
        time.sleep(30)
        n_waited += 1
    backbone.to(device)
    logger.info(f"[guard] 守卫窗结束（等待 {n_waited * 0.5:.1f}min），"
                f"底座已回载 {device}，继续训练")


# ============================================================
# GPU 预算（实际计算时长，不含守卫等待）
# ============================================================


def _load_budget() -> dict:
    if BUDGET_PATH.is_file():
        return json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    return {"compute_seconds": 0.0, "runs": {}}


def _add_budget(run: str, seconds: float) -> dict:
    bud = _load_budget()
    bud["compute_seconds"] = bud.get("compute_seconds", 0.0) + seconds
    bud["runs"][run] = bud["runs"].get(run, 0.0) + seconds
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BUDGET_PATH.write_text(json.dumps(bud, indent=2), encoding="utf-8")
    if bud["compute_seconds"] > GPU_BUDGET_HOURS * 3600:
        raise RuntimeError(
            f"GPU 计算预算超限：{bud['compute_seconds'] / 3600:.2f}h > "
            f"{GPU_BUDGET_HOURS}h（按计划停止并报告，不缩样本/换头/改超参）")
    return bud


# ============================================================
# 验证段隐状态缓存 + 选点指标
# ============================================================


def cache_pit_hidden(backbone, pit_days: list, device: str,
                     chunk: int = 128) -> list[dict]:
    """PIT 评估日过底座一次 → CPU 缓存 hidden/labels（冻结底座确定性保证）。"""
    out: list[dict] = []
    with torch.no_grad():
        for i, d in enumerate(pit_days):
            hiddens: list[torch.Tensor] = []
            for s in range(0, len(d.codes), chunk):
                h = backbone.extract(d.x_norm[s:s + chunk].to(device),
                                     d.stamp[s:s + chunk].to(device))
                hiddens.append(h.cpu())
            out.append({"date": d.date, "hidden": torch.cat(hiddens, dim=0),
                        "labels": d.labels, "codes": d.codes})
            if (i + 1) % 20 == 0 or i == 0:
                logger.info(f"  val hidden [{i + 1}/{len(pit_days)}] "
                            f"{d.date.date()} {len(d.codes)} 只")
    return out


def evaluate_val_rankic(model: torch.nn.Module, cached: list[dict],
                        device: str) -> dict:
    """验证段各期限日均 RankIC（选点只读 ``main`` = 10 日；其余描述性）。"""
    from mh1_multihorizon.evaluate import rank_ic_by_code

    per_h: dict[int, list[float]] = {k: [] for k in range(len(HORIZONS))}
    model.eval()
    with torch.no_grad():
        for day in cached:
            scores = model(day["hidden"].to(device)).cpu().numpy()
            for hi in range(len(HORIZONS)):
                ic, _ = rank_ic_by_code(
                    {c: float(s[hi]) for c, s in zip(day["codes"], scores)},
                    {c: float(l) for c, l in zip(day["codes"],
                                                 day["labels"][:, hi])},
                    min_stocks=1)
                if ic is not None and np.isfinite(ic):
                    per_h[hi].append(ic)
    out = {f"h{k}": float(np.mean(v)) if v else float("nan")
           for k, v in zip(HORIZONS, per_h.values())}
    out["main"] = out[f"h{HORIZONS[MAIN_HORIZON_IDX]}"]
    out["n_days"] = len(per_h[MAIN_HORIZON_IDX])
    return out


# ============================================================
# 单臂训练（正式 / smoke）
# ============================================================


def _build_segments():
    """训练段（union pkl）+ 验证段（qlib PIT csi300）——训练/验证不触 W3/W4。"""
    from mh1_multihorizon.data import (
        build_pit_days, load_union_tables, scan_train_days, union_calendar,
    )

    tables = load_union_tables(POOL)
    calendar = union_calendar(tables)
    train_days, train_stats = scan_train_days(
        tables, calendar, start=TRAIN_START, label_end=TRAIN_LABEL_END)
    logger.info(f"训练段：{train_stats.n_days} 日 / {train_stats.n_samples:,} 样本 "
                f"（决策 {train_stats.decision_min}~{train_stats.decision_max}，"
                f"终点上界 {train_stats.label_endpoint_max}；跳过 "
                f"{train_stats.skipped_days_lt_min} 个 <128 只日）")

    from kronos_qlib import QlibProvider

    prov = QlibProvider(POOL, VAL_START, VAL_LABEL_END)
    val_days, val_stats = build_pit_days(
        prov, start=VAL_START, end=VAL_LABEL_END, label_end=VAL_LABEL_END,
        require_complete_labels=True)
    n_val = sum(len(d.codes) for d in val_days)
    logger.info(f"验证段：{val_stats.n_days} 日 / {n_val:,} 样本（决策 "
                f"{val_stats.decision_min}~{val_stats.decision_max}）")
    return tables, calendar, train_days, train_stats, val_days, val_stats


def train_one(
    arm: str, seed: int, *, device: str = "cuda:0", smoke: bool = False,
    smoke_steps: int = 2, smoke_val_days: int = 5,
) -> tuple[MultiHorizonHead, dict]:
    """单臂训练。smoke=True 时每臂仅 ``smoke_steps`` 批、验证段少量日评估。

    smoke 输出独立落盘（``data/smoke_{arm}{seed}.json``），不进入正式统计。
    """
    assert arm in ARMS
    protocol, protocol_sha = verify_protocol()
    tables, calendar, train_days, tstats, val_days, vstats = _build_segments()

    from cross_section_kda.train import set_seed
    from g5_head.backbone_g1 import load_g1_backbone

    backbone = load_g1_backbone(device)
    refs = checkpoint_backbone_refs(backbone)
    for k, v in protocol["g1_weights"].items():
        if k in refs:   # 只比对哈希/维数键（协议另含路径与证据字段）
            assert refs[k] == v, f"底座哈希与协议不一致：{k}（协议 {v} vs 实测）"

    set_seed(seed)
    head = MultiHorizonHead(backbone.d_model).to(device)
    init_hash = _state_hash(head.state_dict())
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WD)

    from mh1_multihorizon.data import PairedDailySampler

    sampler = PairedDailySampler(tables, train_days, calendar, seed=seed,
                                 batch_size=BATCH)
    val_hidden = cache_pit_hidden(backbone, val_days[:smoke_val_days] if smoke
                                  else val_days, device)

    run = f"{arm}{seed}"
    epochs = 1 if smoke else EPOCHS
    steps = smoke_steps if smoke else STEPS_PER_EPOCH
    resume_path = CKPT_DIR / f"run_{run}.pt"
    history: list[dict] = []
    start_epoch = 1
    best_epoch, best_main = None, -np.inf
    best_state: dict | None = None

    if resume_path.is_file() and not smoke:
        ck = load_checkpoint(resume_path)
        assert ck["init_hash"] == init_hash, "续跑初始化哈希不一致（种子被改）"
        assert ck["protocol_sha256"] == protocol_sha, "续跑协议哈希不一致"
        head.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sampler.rng.setstate(ck["sampler_rng_state"])
        history = ck["history"]
        start_epoch = ck["epoch"] + 1
        best_epoch = ck["best_epoch"]
        best_main = ck["best_main"]
        best_state = ck["best_state"]
        logger.info(f"[{run}] 续跑：从 epoch {start_epoch} 起（已存 "
                    f"{len(history)} epoch 历史）")

    logger.info(f"[{run}] 臂={arm} 种子={seed} 可训练 {n_trainable:,} | "
                f"lr={LR} wd={WD} batch={BATCH} steps×{steps} epochs={epochs} "
                f"| FP32 | 底座哈希 {refs['predictor_sha256'][:12]}…")

    compute_s = 0.0
    for epoch in range(start_epoch, epochs + 1):
        wait_out_guard(backbone, device, epoch)   # cron 守卫（等待不计预算）
        t0 = time.time()
        head.train()
        epoch_loss, n_steps = 0.0, 0
        for x, stamp, labels, _info in sampler.updates(steps):
            xb = x.to(device, non_blocking=True)
            sb = stamp.to(device, non_blocking=True)
            yb = labels.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.no_grad():
                hidden = backbone.extract(xb, sb)
            loss = horizon_loss(head(hidden), yb, arm)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        train_loss = epoch_loss / max(n_steps, 1)

        val_ic = evaluate_val_rankic(head, val_hidden, device)
        row = {"epoch": epoch, "train_loss": float(train_loss),
               "val_rankic_main": float(val_ic["main"]),
               **{f"val_rankic_{k}": float(val_ic[k]) for k in
                  ("h1", "h5", "h10", "h20")}}
        history.append(row)
        elapsed = time.time() - t0
        compute_s += elapsed
        improved = np.isfinite(val_ic["main"]) and val_ic["main"] > best_main
        marker = "★" if improved else ""
        logger.info(f"[{run}] epoch {epoch:02d}/{epochs} "
                    f"train_loss={train_loss:.5f} "
                    f"val_main={val_ic['main']:+.4f} "
                    f"(h1={val_ic['h1']:+.4f} h5={val_ic['h5']:+.4f} "
                    f"h20={val_ic['h20']:+.4f}) {elapsed:.0f}s {marker}")
        if improved:
            best_main = float(val_ic["main"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in head.state_dict().items()}
        if not smoke:
            save_checkpoint(
                head, opt, epoch=epoch, history=history, out_dir=resume_path,
                extra={
                    "arm": arm, "seed": int(seed), "init_hash": init_hash,
                    "protocol_sha256": protocol_sha,
                    "sampler_rng_state": sampler.rng.getstate(),
                    "best_epoch": int(best_epoch), "best_main": float(best_main),
                    "best_state": best_state, "done": False,
                    "backbone_refs": refs,
                })
    if not smoke:
        _add_budget(run, compute_s)
        assert best_state is not None, "无有限选点（协议失败，检查验证段）"
        head.load_state_dict(best_state)
        save_checkpoint(
            head, opt, epoch=epochs, history=history,
            out_dir=DATA_DIR / f"run_{run}_best.pt",
            extra={
                "arm": arm, "seed": int(seed), "init_hash": init_hash,
                "protocol_sha256": protocol_sha,
                "best_epoch": int(best_epoch), "best_main": float(best_main),
                "done": True, "backbone_refs": refs,
                "val_stats_days": int(vstats.n_days),
                "train_stats_days": int(tstats.n_days),
            })
        (DATA_DIR / f"run_{run}_history.json").write_text(
            json.dumps({"arm": arm, "seed": seed, "init_hash": init_hash,
                        "n_trainable": n_trainable,
                        "best_epoch": best_epoch, "best_main": best_main,
                        "history": history, "compute_seconds": compute_s,
                        "sampler_skipped_degenerate": sampler.n_skipped_degenerate,
                        "train_stats": tstats.__dict__,
                        "val_stats": {"n_days": vstats.n_days,
                                      "decision_min": vstats.decision_min,
                                      "decision_max": vstats.decision_max}},
                       ensure_ascii=False, indent=2, default=float),
            encoding="utf-8")
    else:
        peak = (torch.cuda.max_memory_allocated() / 2**30
                if device.startswith("cuda") else 0.0)
        per_step = compute_s / max(smoke_steps, 1)
        extrapolate = per_step * STEPS_PER_EPOCH * EPOCHS * len(RUN_ORDER)
        out = {"arm": arm, "seed": seed, "init_hash": init_hash,
               "steps": smoke_steps, "seconds": compute_s,
               "seconds_per_step": per_step, "peak_gb": peak,
               "extrapolated_six_runs_hours": extrapolate / 3600,
               "val_ic": history[0] if history else None}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / f"smoke_{run}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8")
        logger.info(f"[smoke {run}] {compute_s:.1f}s / {smoke_steps} 步 → 六次训练"
                    f"外推 {extrapolate / 3600:.2f}h | 峰值显存 {peak:.2f}GB")
    return head, {"history": history, "best_epoch": best_epoch,
                  "best_main": best_main, "init_hash": init_hash}


def run_all(device: str = "cuda:0", smoke: bool = False) -> dict:
    """按 RUN_ORDER 顺序执行六次训练（正式态幂等：完成者跳过）。"""
    results: dict[str, dict] = {}
    for run in RUN_ORDER:
        arm, seed = run[0], int(run[1:])
        if not smoke and (DATA_DIR / f"run_{run}_best.pt").is_file():
            ck = load_checkpoint(DATA_DIR / f"run_{run}_best.pt")
            if ck.get("done"):
                logger.info(f"[{run}] 已完成，跳过（幂等）")
                results[run] = {"best_epoch": ck["best_epoch"],
                                "best_main": ck["best_main"], "skipped": True}
                continue
        logger.info("=" * 70)
        _, info = train_one(arm, seed, device=device, smoke=smoke)
        results[run] = info
    # S/M 同初始化对拍（同种子 init_hash 必须一致）
    if not smoke:
        for seed in (42, 43, 44):
            h_s = json.loads((DATA_DIR / f"run_S{seed}_history.json")
                             .read_text(encoding="utf-8"))["init_hash"]
            h_m = json.loads((DATA_DIR / f"run_M{seed}_history.json")
                             .read_text(encoding="utf-8"))["init_hash"]
            assert h_s == h_m, f"s{seed} S/M 初始化不一致（配对失败）"
        logger.info("S/M 同种子初始化哈希对拍：3/3 一致")
    return results


__all__ = [
    "select_best_epoch", "checkpoint_backbone_refs", "save_checkpoint",
    "load_checkpoint", "in_registry_guard", "wait_out_guard",
    "cache_pit_hidden", "evaluate_val_rankic", "train_one", "run_all",
]
