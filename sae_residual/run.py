"""SAE 残差预测头实验 CLI（计划 §8）。

用法::

    python -m sae_residual.run --stage preflight   # 冻结协议 manifest（不可覆盖）
    python -m sae_residual.run --stage smoke       # 2 决策日×64 股实链路估时
    python -m sae_residual.run --stage cache       # 全量 teacher/hidden 缓存
    python -m sae_residual.run --stage pilot       # seed100 两臂训练+同口径评价
    python -m sae_residual.run --stage confirm     # 判据门禁通过后补 101/102
    python -m sae_residual.run --stage report      # 生成 docs 结果报告
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from sae_residual import config as C
from sae_residual import cache as K
from sae_residual import data as D
from sae_residual import evaluate as E
from sae_residual import train as T

REPORT_DOC = C.REPO_ROOT / "docs" / "SAE残差预测头优化G1mean实验结果_20260910.md"
HEADS_DIR = C.CACHE_DIR / "heads"


# ---------------- 预算台账（GPU 12 小时总预算，§1/§7） ----------------

def budget_load() -> dict:
    if C.BUDGET_LEDGER.is_file():
        return json.loads(C.BUDGET_LEDGER.read_text(encoding="utf-8"))
    return {"entries": [], "gpu_budget_s": C.GPU_BUDGET_S}


def budget_log(stage: str, wall_s: float, cuda_s: float = 0.0,
               device: str = "cuda:0", note: str = "") -> dict:
    """追加一条台账；GPU 估用 = GPU 活跃 stage 的 wall 之和（保守上界）。"""
    led = budget_load()
    led["entries"].append({"ts": datetime.now().isoformat(timespec="seconds"),
                           "stage": stage, "wall_s": round(wall_s, 1),
                           "cuda_s": round(cuda_s, 1), "device": device,
                           "note": note})
    led["gpu_used_wall_s"] = round(sum(e["wall_s"] for e in led["entries"]
                                       if e["device"] != "cpu"), 1)
    led["cuda_measured_s"] = round(sum(e["cuda_s"] for e in led["entries"]), 1)
    led["remaining_gpu_budget_s"] = round(
        C.GPU_BUDGET_S - led["gpu_used_wall_s"], 1)
    C.BUDGET_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    C.BUDGET_LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    logger.info(f"[budget] {stage}: wall={wall_s:.1f}s cuda={cuda_s:.1f}s | "
                f"GPU累计wall={led['gpu_used_wall_s']:.0f}s / {C.GPU_BUDGET_S}s "
                f"(余 {led['remaining_gpu_budget_s']:.0f}s)")
    return led


# ---------------- stage: preflight ----------------

def cmd_preflight() -> None:
    from kronos_qlib import QlibProvider
    from sae_residual.cache_io import sha256_file

    mf_path = C.ART_DIR / "preflight_manifest.json"
    if mf_path.exists():
        raise RuntimeError(f"preflight manifest 已存在，不可覆盖：{mf_path}")
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": C.PROTOCOL_VERSION,
        "git_head": _git_head(),
        "baseline": {}, "g1_weights": {}, "sample_rule": {
            "pool": C.POOL, "lookback": C.LOOKBACK, "predict_len": C.PREDICT_LEN,
            "stride": C.STRIDE, "per_day": C.PER_DAY, "data_rng": "default_rng([17, yyyymmdd])",
            "train": [C.TRAIN_START, C.TRAIN_LABEL_END],
            "val": [C.VAL_START, C.VAL_LABEL_END],
            "test_windows": {k: list(v) for k, v in C.TEST_WINDOWS.items()},
        }, "teacher_protocol": {**C.TEACHER,
                                "note": "与原 G1 测试信号生成协议一致（每日 seed42、"
                                        "代码升序、chunk32）；teacher 仅供 head 训练"},
        "head": {**C.HEAD, "loss": C.LOSS, "train": {**C.TRAIN}},
        "budget_gpu_s": C.GPU_BUDGET_S,
    }
    for w, p in C.BASELINE_SIGNALS.items():
        assert p.is_file(), f"基线信号缺失：{p}"
        sha = sha256_file(p)
        assert sha == C.BASELINE_SHA256[w], f"{w} 基线 SHA 不匹配：{sha}"
        manifest["baseline"][w] = {"path": str(p), "sha256": sha}
    for name, p in (("tokenizer", C.G1_TOKENIZER), ("predictor", C.G1_PREDICTOR)):
        assert (p / "model.safetensors").is_file(), f"G1 {name} 权重缺失：{p}"
        manifest["g1_weights"][name] = {"path": str(p),
                                        "sha256": sha256_file(p / "model.safetensors")}
    provider = QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    cal = provider.trading_days()
    import qlib

    manifest["data"] = {
        "qlib_version": qlib.__version__, "qlib_file": qlib.__file__,
        "data_end": C.DATA_END, "forward_cutoff": C.FORWARD_CUTOFF,
        "calendar_first_last": [str(cal.min().date()), str(cal.max().date())],
        "n_train_days": len(D.decision_days(cal, C.TRAIN_START, C.TRAIN_LABEL_END)),
        "n_val_days": len(D.decision_days(cal, C.VAL_START, C.VAL_LABEL_END)),
        "expected_train_samples": len(D.decision_days(cal, C.TRAIN_START, C.TRAIN_LABEL_END)) * C.PER_DAY,
        "expected_val_samples": len(D.decision_days(cal, C.VAL_START, C.VAL_LABEL_END)) * C.PER_DAY,
    }
    for w, p in C.BASELINE_SIGNALS.items():
        wide = E.load_wide(w)
        assert wide.index.max() <= pd.Timestamp(C.FORWARD_CUTOFF)
        import numpy as np

        vals = wide.to_numpy()
        assert np.isfinite(vals[~np.isnan(vals)]).all(), f"{w} 信号含非有限值"
        manifest["baseline"][w].update({
            "n_days": int(wide.shape[0]), "n_cells": int(wide.notna().sum().sum()),
            "date_min": str(wide.index.min().date()),
            "date_max": str(wide.index.max().date())})
    mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    logger.info(f"[preflight] manifest 冻结：{mf_path}")


def _git_head() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


# ---------------- stage: smoke ----------------

def cmd_smoke() -> None:
    from kronos_qlib import QlibProvider

    t0 = time.perf_counter()
    provider = QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    backbone, predictor, identity = K.load_frozen_g1()
    cal = provider.trading_days()
    close_wide = K._bulk_close_wide(provider, "2013-06-01", C.VAL_LABEL_END)
    st_train = K.build_head_split(provider, backbone, predictor, identity,
                                  "train", close_wide, cal, limit_days=2,
                                  subdir="smoke_train")
    st_w3 = K.build_test_split(provider, backbone, identity, "W3", cal,
                               limit_days=1, subdir="smoke_W3")
    wall = time.perf_counter() - t0
    import torch

    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    est = {
        "smoke_wall_s": round(wall, 1),
        "train_days_done": st_train["rebuilt"], "w3_days_done": st_w3["rebuilt"],
        "per_train_day_s": round(st_train["wall_s"] / max(st_train["rebuilt"], 1), 2),
        "per_w3_day_s": round(st_w3["wall_s"] / max(st_w3["rebuilt"], 1), 2),
        "peak_vram_gb": round(vram_gb, 2),
        "n_train_days": 534, "n_val_days": 22, "n_w3_days": 126, "n_w4_days": 134,
        "cuda_s_train": round(st_train["cuda_s"], 1),
        "cuda_s_w3": round(st_w3["cuda_s"], 1),
    }
    est["estimated_full_gpu_wall_s"] = round(
        est["per_train_day_s"] * (534 + 22) + est["per_w3_day_s"] * (126 + 134), 1)
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    (C.ART_DIR / "smoke_report.json").write_text(
        json.dumps(est, ensure_ascii=False, indent=2), encoding="utf-8")
    budget_log("smoke", wall, st_train["cuda_s"] + st_w3["cuda_s"], "cuda:0",
               note=f"train×{st_train['rebuilt']} W3×{st_w3['rebuilt']} "
                    f"est_full={est['estimated_full_gpu_wall_s']}s")
    logger.info(f"[smoke] {json.dumps(est, ensure_ascii=False)}")


# ---------------- stage: cache ----------------

def cmd_cache(scopes: list[str] | None = None) -> None:
    from kronos_qlib import QlibProvider

    scopes = scopes or ["train", "val", "W3", "W4"]
    t0 = time.perf_counter()
    provider = QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    backbone, predictor, identity = K.load_frozen_g1()
    cal = provider.trading_days()
    close_wide = (K._bulk_close_wide(provider, "2013-06-01", C.VAL_LABEL_END)
                  if ("train" in scopes or "val" in scopes) else None)
    all_stats = {}
    for split in scopes:
        t1 = time.perf_counter()
        if split in ("train", "val"):
            st = K.build_head_split(provider, backbone, predictor, identity,
                                    split, close_wide, cal)
        else:
            st = K.build_test_split(provider, backbone, identity, split, cal)
        all_stats[split] = st
        budget_log(f"cache:{split}", time.perf_counter() - t1, st["cuda_s"],
                   "cuda:0",
                   note=f"rebuilt={st['rebuilt']} reused={st['reused']}")
    manifest = K.freeze_cache(C.ART_DIR)
    (C.ART_DIR / "cache_build_stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info(f"[cache] 全段完成 wall={time.perf_counter() - t0:.0f}s；"
                f"manifest 已冻结")


# ---------------- stage: pilot / confirm ----------------

def _run_eval(seeds: list[int], stage_name: str) -> dict:
    """训练指定 seeds 两臂 + 同口径评价 + R 记录（stage=pilot/confirm 共用）。"""
    import os

    from kronos_qlib import QlibProvider
    from sae_residual.cache_io import sha256_file

    QlibProvider.init_qlib_once()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    # 逐 chunk 身份校验：当前 G1 权重必须与缓存构建时一致
    expect = {"protocol": C.PROTOCOL_VERSION,
              "tokenizer_sha": sha256_file(C.G1_TOKENIZER / "model.safetensors"),
              "predictor_sha": sha256_file(C.G1_PREDICTOR / "model.safetensors")}
    train, val, stats = T.prepare_training_data(expect)
    T.save_stats(stats, C.ART_DIR)
    heads = {}
    t0 = time.perf_counter()
    for seed in seeds:
        for arm in C.ARMS:
            heads[(arm, seed)] = T.train_arm(
                train, val, arm, seed, HEADS_DIR, device="cpu",
                epochs=C.TRAIN["epochs"])
    budget_log(f"{stage_name}:train_heads", time.perf_counter() - t0, 0.0, "cpu",
               note=f"seeds={seeds} arms={list(C.ARMS)}（小头 CPU 训练不计 GPU 预算）")
    # 提交用产物：best 头权重 + 指标拷入 art_dir
    for (arm, seed), res in heads.items():
        src = T.head_file(HEADS_DIR, arm, seed, res["best_epoch"])
        shutil.copy(src, C.ART_DIR / src.name)
        for suffix in ("epochs_metrics", "best"):
            src2 = HEADS_DIR / f"{suffix}_{arm}_s{seed}.json"
            if src2.is_file():
                shutil.copy(src2, C.ART_DIR / src2.name)
    gate = E.reproduce_g1_gate(C.ART_DIR)
    t1 = time.perf_counter()
    summary = E.run_comparison(C.ART_DIR, HEADS_DIR, stats, seeds,
                               expect=expect)
    budget_log(f"{stage_name}:backtest", time.perf_counter() - t1, 0.0, "cpu",
               note="Qlib 回测 CPU，不计 GPU 预算")
    # 残差信号落盘（复算凭据，小 parquet 入库；行列对齐基线顺序）
    for seed in seeds:
        for arm in C.ARMS:
            for w in C.TEST_WINDOWS:
                wide, _ = E.infer_test_signal(arm, seed, w, stats, HEADS_DIR,
                                              expect)
                ref = E.load_wide(w)
                wide = wide.reindex(index=ref.index, columns=ref.columns)
                wide.to_parquet(C.ART_DIR / f"signal_{w}_{arm}_s{seed}.parquet")
    verdict = E.judge_confirm_gate(summary["per_seed"])
    out = {"stats": {k: (v if not hasattr(v, "tolist") else None)
                     for k, v in stats.items() if not hasattr(v, "tolist")},
           "sigma_e": stats["sigma_e"], "n_train": stats["n_train"],
           "n_val": stats["n_val"], "heads": {f"{a}_s{s}": {
               "best_epoch": r["best_epoch"],
               "best_val_l_pred": r["best_val_l_pred"]}
               for (a, s), r in heads.items()},
           "reproduction_gate": gate, "verdict": verdict}
    (C.ART_DIR / f"{stage_name}_verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _r_log(stage_name, seeds, summary, stats, heads, verdict)
    return out


def _r_log(stage_name: str, seeds: list[int], summary: dict, stats: dict,
           heads: dict, verdict: dict) -> None:
    from qlib.workflow import R

    with R.start(experiment_name=C.EXPERIMENT_R,
                 recorder_name=f"{stage_name}_s{'_'.join(map(str, seeds))}"):
        R.log_params(protocol=C.PROTOCOL_VERSION, seeds=seeds,
                     sigma_e=stats["sigma_e"], n_train=stats["n_train"],
                     n_val=stats["n_val"], d_in=stats["d_in"],
                     beta_ae=C.LOSS["beta_ae"], beta_sae=C.LOSS["beta_sae"],
                     epochs=C.TRAIN["epochs"])
        for (arm, seed), res in heads.items():
            R.log_metrics(**{f"{arm}_s{seed}_best_epoch": res["best_epoch"],
                             f"{arm}_s{seed}_best_val_l_pred": res["best_val_l_pred"]})
        for seed, p in summary["per_seed"].items():
            for w in C.TEST_WINDOWS:
                # mlflow 指标名禁 |，用 / 分隔
                R.log_metrics(**{
                    f"{w}/s{seed}/D_AG": p["D_AG"][w],
                    f"{w}/s{seed}/D_SG": p["D_SG"][w],
                    f"{w}/s{seed}/D_SA": p["D_SA"][w],
                    f"{w}/s{seed}/G1_curve_end": p["arms"]["G1_mean"][w]["curve_end"],
                })
        R.log_artifact(str(C.ART_DIR / f"{stage_name}_verdict.json"))
        for w in C.TEST_WINDOWS:
            R.log_artifact(str(C.ART_DIR / f"fig_{w}_main.png"))
        R.end_exp(recorder_status="FINISHED")


def cmd_pilot() -> None:
    out = _run_eval([C.PILOT_SEED], "pilot")
    v = out["verdict"]
    gate_pass = v["ae"]["both_windows_count"] >= 1 or v["sae"]["both_windows_count"] >= 1
    led = budget_load()
    summary = json.loads((C.ART_DIR / "comparison_summary.json").read_text(
        encoding="utf-8"))
    logger.info("==== pilot 主结论 ====")
    logger.info(json.dumps(
        {str(s): {"D_AG": p["D_AG"], "D_SG": p["D_SG"],
                  "ae_both": p["ae_beats_g1_both"],
                  "sae_both": p["sae_beats_g1_both"]}
         for s, p in summary["per_seed"].items()}, ensure_ascii=False))
    logger.info(f"确认门禁（任一臂两窗均胜 G1）: "
                f"{'通过 → 可 confirm' if gate_pass else '未通过 → 封存本配置，不补种子'}")
    logger.info(f"预算：GPU wall 已用 {led['gpu_used_wall_s']:.0f}s / "
                f"{C.GPU_BUDGET_S}s")


def cmd_confirm() -> None:
    pilot = json.loads((C.ART_DIR / "pilot_verdict.json").read_text(encoding="utf-8"))
    v = pilot["verdict"]
    gate_pass = (v["ae"]["both_windows_count"] >= 1
                 or v["sae"]["both_windows_count"] >= 1)
    if not gate_pass:
        raise RuntimeError("pilot 门禁未通过（无臂两窗均胜 G1）→ 封存配置，不补种子")
    led = budget_load()
    est_confirm_s = led["gpu_used_wall_s"] * 0.05   # confirm 仅 CPU 训练+回测
    if led["remaining_gpu_budget_s"] < est_confirm_s:
        raise RuntimeError(f"GPU 预算不足：余 {led['remaining_gpu_budget_s']}s")
    _run_eval(list({C.PILOT_SEED, *C.CONFIRM_SEEDS}), "confirm")


# ---------------- stage: report ----------------

def cmd_report() -> None:
    from sae_residual.report import write_report

    write_report(REPORT_DOC, C.ART_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="SAE 残差预测头优化 G1 mean")
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "smoke", "cache", "pilot",
                                 "confirm", "report"])
    parser.add_argument("--scopes", default=None,
                        help="cache 段选择，逗号分隔：train,val,W3,W4")
    args = parser.parse_args()
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(str(C.ART_DIR / f"{args.stage}.log"), enqueue=False)
    if args.stage == "preflight":
        cmd_preflight()
    elif args.stage == "smoke":
        cmd_smoke()
    elif args.stage == "cache":
        cmd_cache(args.scopes.split(",") if args.scopes else None)
    elif args.stage == "pilot":
        cmd_pilot()
    elif args.stage == "confirm":
        cmd_confirm()
    else:
        cmd_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
