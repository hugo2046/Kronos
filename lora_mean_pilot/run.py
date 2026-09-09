"""LoRA-mean 统一 CLI：preflight → pilot → confirm（计划 §6/§7）。

GPU 预算 12h 逐 attempt 累计；16:30 登记 cron 守卫（训练整段避开守卫窗、
推理逐日边界让位且逐日落盘可断点续推）；R 记录每次训练/推理/回测。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
from loguru import logger

from lora_mean_pilot import evaluate as ev
from lora_mean_pilot.config import REGISTRY_GUARD, GPU_BUDGET_HOURS

OUT = ev.OUT_DIR
STATE = OUT / "state.json"
BUDGET = OUT / "gpu_budget.json"
EXPERIMENT = "kronos-lora-mean-20260909"
EXPECT_SHA = {
    "W3": ("9352b40302eb34c714bfebd808481b625a21d2f7f4f668ccca15cda0cfe21d"
           "66"),
    "W4": ("453e2aeae7fe4bee8ce8fa62a9908da0843d5ad8b52777cd3b299a84cc4a3e"
           "36"),
}
SEEDS = (100, 101, 102)


def _in_guard(now=None) -> bool:
    t = (now or datetime.now()).time().strftime("%H:%M")
    return REGISTRY_GUARD[0] <= t < REGISTRY_GUARD[1]


def _budget_add(seconds: float) -> None:
    b = json.loads(BUDGET.read_text(encoding="utf-8")) if BUDGET.is_file() \
        else {"compute_seconds": 0.0, "attempts": {}}
    b["compute_seconds"] += seconds
    BUDGET.parent.mkdir(parents=True, exist_ok=True)
    BUDGET.write_text(json.dumps(b, indent=2), encoding="utf-8")
    if b["compute_seconds"] > GPU_BUDGET_HOURS * 3600:
        raise RuntimeError(f"GPU 预算超限 {b['compute_seconds'] / 3600:.2f}h")


def _mark(stage: str, payload: dict) -> None:
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    st[stage] = {"at": datetime.now().isoformat(timespec="seconds"), **payload}
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2, default=str),
                     encoding="utf-8")


def _require(stage: str) -> dict:
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    if stage not in st:
        raise RuntimeError(f"阶段未完成：{stage}")
    return st[stage]


def _qlib_fns():
    from qlib_mean_comparison import run as mc

    return mc.run_qlib_backtest, mc.compute_curves


def _score_window_resumable(predictor, tokenizer, wname: str, provider,
                            device: str, seed: int, out_path: Path) -> Path:
    """逐日 FULL 推理（mean），逐日落盘可续推；守卫窗让位。

    复用 canonical 构件（build_inference_windows / predict_batch_chunked /
    compute_signal_from_preds），与 g10 evaluate 同链路。
    """
    import torch

    from kronos_qlib import build_inference_windows
    from model import KronosPredictor
    from paper_replication.signal import (
        compute_signal_from_preds, predict_batch_chunked,
    )

    raw_model, raw_tokenizer = predictor, tokenizer   # 守卫卸载用原始模块
    predictor = KronosPredictor(model=predictor, tokenizer=tokenizer,
                                device=device)   # chunked 推理需包装器

    start, end = ev.WINDOW_BOUNDS[wname]
    rebalances = provider.trading_days(start, end)
    if out_path.is_file():                      # 断点续推：跳过已出日
        partial = pd.read_parquet(out_path)
        done = {str(c) for c in partial.columns}
    else:
        partial, done = None, set()
    for d in rebalances:
        ds = str(d.date())
        if ds in done:
            continue
        if _in_guard():                          # 登记让位（推理逐日边界）
            logger.warning(f"[guard] {ds} 进入 16:30 守卫窗，卸载模型")
            raw_model.to("cpu")
            raw_tokenizer.to("cpu")
            torch.cuda.empty_cache()
            while _in_guard():
                time.sleep(30)
            raw_model.to(device)
            raw_tokenizer.to(device)
        torch.manual_seed(seed)                  # 每日重播种（确定性）
        df_list, x_ts, y_ts, codes, stats = build_inference_windows(
            provider, ds, lookback=ev.INFERENCE["lookback"],
            predict_len=ev.INFERENCE["pred_len"], pool="csi300")
        if not df_list:
            continue
        preds = predict_batch_chunked(
            predictor, df_list, x_ts, y_ts, pred_len=ev.INFERENCE["pred_len"],
            T=ev.INFERENCE["T"], top_k=ev.INFERENCE["top_k"],
            top_p=ev.INFERENCE["top_p"],
            sample_count=ev.INFERENCE["sample_count"], chunk_size=32)
        day = {c: compute_signal_from_preds(p["close"],
                float(df_list[i]["close"].iloc[-1]))
               for i, (c, p) in enumerate(zip(codes, preds))}
        col = pd.Series(day, name=ds)
        partial = col.to_frame() if partial is None else partial.join(
            col, how="outer")
        partial.to_parquet(out_path)             # 逐日落盘
    return out_path


def _load_predictor_with_adapter(adapter_path: Path | None, device: str):
    import torch

    from model.kronos import Kronos, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(ev.G1_MODEL["tokenizer"]))
    tokenizer.eval().to(device)
    predictor = Kronos.from_pretrained(str(ev.G1_MODEL["predictor"])).to(device)
    if adapter_path is not None:
        from lora_mean_pilot.adapter import apply_adapter, inject_lora, \
            load_adapter

        ck = load_adapter(adapter_path)
        inject_lora(predictor, seed=int(ck["seed"]))
        apply_adapter(predictor, ck["adapter"])
    predictor.eval()
    return predictor, tokenizer


def cmd_preflight() -> None:
    from lora_mean_pilot.adapter import inject_lora
    from lora_mean_pilot.train import train_lora

    if (OUT / "protocol.json").is_file():
        raise RuntimeError("协议已存在（不可覆盖）")
    # 基线 SHA 锁定（计划预期值逐字核对）
    for w, p in ev.G1_SIGNALS.items():
        got = ev.sha256_file(p)
        assert got == EXPECT_SHA[w], f"{w} G1 信号 SHA 漂移：{got}"
    model_sha = {k: ev.sha256_file(p / "model.safetensors")
                 for k, p in ev.G1_MODEL.items()}
    corpus = Path("/home/user/workspace/Kronos/finetune_suite/data/ashares")
    protocol = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": {"signals_sha256": EXPECT_SHA,
                     "model_sha256": model_sha},
        "lora": {"rank": 8, "alpha": 8, "dropout": 0.0,
                 "targets": "transformer.<i>.self_attn.{q,k,v,out}_proj"},
        "train": {"batch": 50, "epochs": 15, "steps_per_epoch": 2000,
                  "lr": 4e-5, "betas": [0.9, 0.95], "wd": 0.1,
                  "pct_start": 0.03, "div_factor": 10, "clip": 3.0,
                  "fp32": True, "seeds": list(SEEDS),
                  "corpus": str(corpus), "val": "2025H1 400 批固定采样"},
        "inference": ev.INFERENCE, "windows": ev.WINDOW_BOUNDS,
        "qlib": "0b982e8 口径（TopkDropout50/5/hold5、open、0.001/0.0015/5、"
                "0.095、1e8、000300.SH、codes=csi300）",
        "budget_gpu_hours": GPU_BUDGET_HOURS,
    }
    # 真实 smoke（2 步训练）估时
    t0 = time.time()
    train_lora(100, out_dir=OUT / "smoke", device="cuda:0",
               g1_tokenizer_path=ev.G1_MODEL["tokenizer"],
               g1_predictor_path=ev.G1_MODEL["predictor"],
               smoke=True, smoke_steps=2)
    smoke_s = time.time() - t0
    _budget_add(smoke_s)
    est_train_h = smoke_s / 2 * 2000 * 15 / 3600 / 15  # 每 epoch 粗估
    # 推理估时按 g10 实测 ~30s/日 × 260 日/窗
    est_inf_h = 260 * 30 / 3600
    protocol["smoke"] = {"seconds": smoke_s,
                         "est_train_hours_per_seed": est_train_h,
                         "est_inference_hours_per_window": est_inf_h}
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    _mark("preflight", {"smoke_seconds": smoke_s})
    logger.info(f"[preflight] 协议冻结；smoke {smoke_s:.0f}s；训练估 "
                f"{est_train_h:.1f}h/seed；推理估 {est_inf_h:.1f}h/窗")


def _run_seed(seed: int, bt_fn, cv_fn, with_train: bool = True) -> dict:
    """单 seed：训练（可选）→ G1_paired/LoRA 两臂两窗推理+回测。"""
    import torch

    from kronos_qlib import QlibProvider

    hist_path = OUT / f"lora_{seed}" / "history.json"
    if with_train and not hist_path.is_file():
        t0 = time.time()
        from lora_mean_pilot.train import train_lora

        info = train_lora(
            seed, out_dir=OUT / f"lora_{seed}", device="cuda:0",
            g1_tokenizer_path=ev.G1_MODEL["tokenizer"],
            g1_predictor_path=ev.G1_MODEL["predictor"])
        _budget_add(time.time() - t0)
    else:                                          # 训练已完成的幂等跳过
        with_train = False
        h = json.loads(hist_path.read_text(encoding="utf-8"))
        info = {**h, "best_adapter": str(
            OUT / f"lora_{seed}" / f"adapter_epoch_{h['best_epoch']:03d}.pt")}
    results: dict = {}
    sig_dir = OUT / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    provider = QlibProvider("csi300", *ev.WINDOW_BOUNDS["W3"])
    for arm, adapter in (("G1_paired", None), ("LoRA", info["best_adapter"])):
        predictor, tokenizer = _load_predictor_with_adapter(adapter, "cuda:0")
        for w in ev.WINDOW_BOUNDS:
            p = sig_dir / f"{w}_{arm}_{seed}.parquet"
            if not p.is_file() or pd.read_parquet(p).shape[1] < 100:
                t0 = time.time()
                _score_window_resumable(predictor, tokenizer, w, provider,
                                        "cuda:0", seed, p)
                _budget_add(time.time() - t0)
        del predictor, tokenizer
        torch.cuda.empty_cache()
    for arm in ("G1_paired", "LoRA"):
        for w in ev.WINDOW_BOUNDS:
            results[(f"{arm}_{seed}", w)] = ev.backtest_arm(
                sig_dir / f"{w}_{arm}_{seed}.parquet", w, bt_fn, cv_fn,
                tag=f"{arm}_{seed}", out_dir=OUT)
            logger.info(f"[{arm}_{seed}|{w}] curve_end "
                        f"{results[(f'{arm}_{seed}', w)]['curve_end']:+.4%}")
    return {"train": {"best_epoch": info["best_epoch"],
                      "best_ce": info["best_ce"],
                      "n_trainable": info["n_trainable"]},
            "results": results}


def cmd_pilot() -> None:
    _require("preflight")
    from qlib.workflow import R

    from kronos_qlib import QlibProvider

    QlibProvider.init_qlib_once()
    bt_fn, cv_fn = _qlib_fns()
    # 口径门禁：复现 0b982e8 的 G1 report
    gate = ev.reproduce_gate(bt_fn, cv_fn)
    # 主基线（G1_original）沿用 0b982e8 曲线（信号/口径同一）
    results = {("G1_original_mean", w): ev.backtest_arm(
        ev.G1_SIGNALS[w], w, bt_fn, cv_fn, tag="G1_original_mean",
        out_dir=OUT) for w in ev.WINDOW_BOUNDS}
    with R.start(experiment_name=EXPERIMENT,
                 recorder_name=f"pilot_seed100"):
        R.log_params(seed=100, stage="pilot")
        pilot = _run_seed(100, bt_fn, cv_fn)
        results.update(pilot["results"])
        v = ev.paired_verdict(results, [100])
        passed = ev.pilot_gate(v, 100)
        R.log_metrics(**{f"100|{w}|D_original_pp": v[f"100|{w}"]["D_original_pp"]
                         for w in ev.WINDOW_BOUNDS})
        R.log_artifact(str(OUT / "state.json"))
    (OUT / "pilot_results.json").write_text(
        json.dumps({"verdict": v, "passed": passed,
                    "gate_reproduce": gate,
                    "train": pilot["train"],
                    "results": {f"{k[0]}|{k[1]}": val
                                for k, val in results.items()}},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _plot_seed(results, 100)
    _mark("pilot", {"passed": passed})
    logger.info(f"pilot 门禁（两窗 D_original>0 且 D_paired>0）："
                f"{'通过→可 confirm' if passed else '未通过→封存本配置'}")


def cmd_confirm() -> None:
    st = _require("pilot")
    if not st.get("passed"):
        raise RuntimeError("pilot 门禁未通过，confirm 拒绝执行")
    bud = json.loads(BUDGET.read_text(encoding="utf-8"))
    remain = GPU_BUDGET_HOURS * 3600 - bud["compute_seconds"]
    if remain < 2 * (3600 + 4 * 260 * 30):       # 粗估两 seed 训练+推理
        raise RuntimeError(f"剩余预算 {remain / 3600:.2f}h 不足以确认阶段"
                           "——报告'确认阶段待预算'，不缩采样/裁日期")
    from qlib.workflow import R

    from kronos_qlib import QlibProvider

    QlibProvider.init_qlib_once()
    bt_fn, cv_fn = _qlib_fns()
    results = json.loads((OUT / "pilot_results.json")
                         .read_text(encoding="utf-8"))["results"]
    results = {(k.split("|")[0], k.split("|")[1]): v for k, v in results.items()}
    with R.start(experiment_name=EXPERIMENT, recorder_name="confirm_101_102"):
        R.log_params(stage="confirm", seeds=[101, 102])
        for seed in (101, 102):
            r = _run_seed(seed, bt_fn, cv_fn)
            results.update(r["results"])
        v = ev.paired_verdict(results, list(SEEDS))
        cg = ev.confirm_gate(v, list(SEEDS))
        R.log_metrics(repeated_improvement=float(cg["repeated_improvement"]))
    (OUT / "confirm_results.json").write_text(
        json.dumps({"verdict": v, "confirm": cg,
                    "results": {f"{k[0]}|{k[1]}": val
                                for k, val in results.items()}},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for s in SEEDS:
        _plot_seed(results, s)
    _mark("confirm", {"repeated_improvement": cg["repeated_improvement"]})
    logger.info(f"确认判定：{cg}")


def _plot_seed(results: dict, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK HK", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    from qlib_mean_comparison import run as mc

    for w in ev.WINDOW_BOUNDS:
        curves = {}
        for tag in ("G1_original_mean", f"G1_paired_{seed}", f"LoRA_{seed}"):
            d = pd.read_parquet(OUT / f"daily_{w}_{tag}.parquet")
            curves[tag] = d["curve"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        styles = {"G1_original_mean": ("#12456e", "-", "G1 mean（用户基准）"),
                  f"G1_paired_{seed}": ("#7fb3d9", "--",
                                        f"G1 同权重 seed{seed}"),
                  f"LoRA_{seed}": ("#de7c33", "-", f"LoRA seed{seed}")}
        for t, c in curves.items():
            col, ls, lab = styles[t]
            axes[0].plot(c.index, c, color=col, linestyle=ls, label=lab)
        axes[0].set_title(f"{w}：扣费累计（Qlib 原生 open/cumsum，"
                          f"G1 为用户基准）")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(curves[f"LoRA_{seed}"].index,
                     curves[f"LoRA_{seed}"] - curves["G1_original_mean"],
                     color="#a84a10", label="LoRA − 原G1")
        axes[1].plot(curves[f"LoRA_{seed}"].index,
                     curves[f"LoRA_{seed}"] - curves[f"G1_paired_{seed}"],
                     color="#3d7eb0", linestyle="--", label="LoRA − 同seed G1")
        axes[1].axhline(0, color="grey", linewidth=0.8)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_{w}_seed{seed}.png", dpi=150)
        plt.close(fig)
    logger.info(f"[fig] seed{seed} 两窗主图落盘")


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA-mean 配对种子实验 CLI")
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "pilot", "confirm"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logger.add(str(OUT / f"{args.stage}.log"), enqueue=False)
    if args.stage == "preflight":
        cmd_preflight()
    elif args.stage == "pilot":
        cmd_pilot()
    else:
        cmd_confirm()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
