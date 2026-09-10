"""PEER1 同日跨股票交互残差头实验 CLI（计划 §8）。

用法::

    python -m peer_residual.run --stage preflight   # 冻结协议 manifest（不可覆盖）
    python -m peer_residual.run --stage smoke       # 3 决策日补 h + 少量更新估时
    python -m peer_residual.run --stage cache       # 全段 PeerSet 补 h 缓存
    python -m peer_residual.run --stage pilot       # seed100 两臂训练+同口径评价
    python -m peer_residual.run --stage confirm     # 判据门禁通过后补 101/102
    python -m peer_residual.run --stage report      # 生成 docs 结果报告

confirm 是条件入口，不能无条件串在上述命令后。
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

import numpy as np
import pandas as pd
from loguru import logger

from peer_residual import config as C
from peer_residual import cache as K
from peer_residual import data as PD
from peer_residual import evaluate as E
from peer_residual import train as T

REPORT_DOC = C.REPO_ROOT / "docs" / "同日跨股票交互残差头实验结果_20260910.md"


# ---------------- 预算台账（GPU 12 小时总预算，§1/§8） ----------------

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


def _git_head() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


# ---------------- stage: preflight ----------------

def cmd_preflight() -> None:
    from kronos_qlib import QlibProvider
    from sae_residual.cache_io import sha256_file

    mf_path = C.ART_DIR / "preflight_manifest.json"
    if mf_path.exists():
        raise RuntimeError(f"preflight manifest 已存在，不可覆盖：{mf_path}")
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    stats = PD.load_frozen_stats()
    from peer_residual.model import param_table

    manifest: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": C.PROTOCOL_VERSION, "run_id": C.RUN_ID,
        "git_head": _git_head(),
        "baseline": {}, "g1_weights": {}, "sae_assets": {},
        "peerset_rule": {
            "pool": C.POOL, "lookback": C.LOOKBACK,
            "definition": "PeerSet(t) = t日csi300 PIT成员 ∩ 截至t具有合格"
                          "90日历史输入（build_inference_windows 口径：窗口"
                          "完整且无停牌）；不使用未来收益/停牌/退池/标签完整性",
            "note": "原SAE 64股/日仅为 LossSet，不是 PeerSet；额外 peer 仅作 "
                    "attention context，不新增交易候选",
        },
        "lossset_rule": {
            "source": "SAE f6c0726 冻结监督缓存原值（train 534日×64、"
                      "val 22日×64）；s_G1/y_mean 不按新 seed 重推",
            "target": "r = (y_mean - s_G1)/σe，不减残差均值；s_final = "
                      "s_G1 + σe·r_hat",
        },
        "head": {**C.HEAD, "param_table": param_table(stats["d_in"]),
                 "train": {**C.TRAIN}},
        "test_windows": {k: list(v) for k, v in C.TEST_WINDOWS.items()},
        "budget_gpu_s": C.GPU_BUDGET_S,
    }
    for w, p in C.BASELINE_SIGNALS.items():
        assert p.is_file(), f"基线信号缺失：{p}"
        sha = sha256_file(p)
        assert sha == C.BASELINE_SHA256[w], f"{w} 基线 SHA 不匹配：{sha}"
        wide = E.load_wide(w)
        assert wide.index.max() <= pd.Timestamp(C.FORWARD_CUTOFF)
        vals = wide.to_numpy()
        assert np.isfinite(vals[~np.isnan(vals)]).all(), f"{w} 信号含非有限值"
        manifest["baseline"][w] = {"path": str(p), "sha256": sha,
                                   "n_days": int(wide.shape[0]),
                                   "n_cells": int(wide.notna().sum().sum()),
                                   "date_min": str(wide.index.min().date()),
                                   "date_max": str(wide.index.max().date())}
    for name, p in (("tokenizer", C.G1_TOKENIZER), ("predictor", C.G1_PREDICTOR)):
        assert (p / "model.safetensors").is_file(), f"G1 {name} 权重缺失：{p}"
        manifest["g1_weights"][name] = {"path": str(p),
                                        "sha256": sha256_file(p / "model.safetensors")}
    for name, p in (("norm_stats", C.SAE_NORM_STATS),
                    ("cache_manifest", C.SAE_CACHE_MANIFEST)):
        assert p.is_file(), f"SAE 资产缺失：{p}"
        manifest["sae_assets"][name] = {"path": str(p), "sha256": sha256_file(p)}
    manifest["sigma_e"] = stats["sigma_e"]
    manifest["d_in"] = stats["d_in"]
    # SAE 缓存段规模（只读核对）
    sae_counts = {}
    for split in ("train", "val", "W3", "W4"):
        dates = PD.split_dates(split)
        sae_counts[split] = {"n_days": len(dates),
                             "first": dates[0], "last": dates[-1]}
    manifest["sae_cache"] = sae_counts
    mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    logger.info(f"[preflight] manifest 冻结：{mf_path}")


# ---------------- stage: smoke ----------------

def cmd_smoke() -> None:
    import torch

    from kronos_qlib import QlibProvider

    t0 = time.perf_counter()
    provider = QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    backbone, _, identity = K.load_frozen_g1()
    cal = provider.trading_days()
    del cal
    # 复用核验（§4 固定抽样：train/val 最早日 + W3/W4 首日各前 3 股）
    verify = K.verify_sample_against_g1(provider, backbone, identity)
    # 最早 2 个训练日期 + 最早 1 个验证日期补 h（smoke 隔离目录）
    st = K.build_peer_split(provider, backbone, identity, "train",
                            limit_days=2, subdir="smoke_train")
    st_v = K.build_peer_split(provider, backbone, identity, "val",
                              limit_days=1, subdir="smoke_val")
    stats = PD.load_frozen_stats()
    days = [PD.load_day("train", d, stats, "smoke_train")
            for d in PD.split_dates("train")[:2]]
    val_days = [PD.load_day("val", d, stats, "smoke_val")
                for d in PD.split_dates("val")[:1]]
    # 少量头更新：batch≤4 日期、~300 股/日，测显存/速度
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = T.build_head(C.PILOT_SEED, stats["d_in"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.TRAIN["lr"],
                            betas=tuple(C.TRAIN["betas"]),
                            weight_decay=C.TRAIN["weight_decay"])
    x, valid, loss, r = PD.collate_batch(days)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t_b = time.perf_counter()
    n_upd = 3
    for _ in range(n_upd):
        T.train_step(model, opt, x.to(device), valid.to(device),
                     loss.to(device), r.to(device), "ON")
    per_update_s = (time.perf_counter() - t_b) / n_upd
    v = T.val_loss_days(model, val_days, "ON", device)
    vram_gb = (torch.cuda.max_memory_allocated() / 1e9
               if torch.cuda.is_available() else 0.0)
    per_train_day_s = st["wall_s"] / max(st["rebuilt"], 1)
    n_batches_per_epoch = int(np.ceil(534 / C.TRAIN["batch_days"]))
    est = {
        "verify": verify,
        "smoke_train_days": st["rebuilt"], "smoke_val_days": st_v["rebuilt"],
        "n_days_loaded": len(days), "n_per_day": [len(d.codes) for d in days],
        "n_extra_per_day": [d.n_extra for d in days],
        "batch_shape": list(x.shape), "device": device,
        "per_update_s": round(per_update_s, 4),
        "peak_vram_gb": round(vram_gb, 3),
        "per_train_day_s": round(per_train_day_s, 2),
        "est_cache_full_gpu_s": round(
            per_train_day_s * (534 + 22) + per_train_day_s * 0.4 * (126 + 134), 0),
        "est_pilot_train_s": round(per_update_s * n_batches_per_epoch
                                   * C.TRAIN["epochs"] * len(C.ARMS), 0),
        "smoke_val_mse_after3upd": v,
    }
    est["est_total_gpu_s"] = round(
        (time.perf_counter() - t0) + est["est_cache_full_gpu_s"]
        + est["est_pilot_train_s"], 0)
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    (C.ART_DIR / "smoke_report.json").write_text(
        json.dumps(est, ensure_ascii=False, indent=2), encoding="utf-8")
    budget_log("smoke", time.perf_counter() - t0, 0.0, "cuda:0",
               note=f"verify+3日补h+{n_upd}次更新 est_total={est['est_total_gpu_s']}s")
    logger.info(f"[smoke] {json.dumps(est, ensure_ascii=False)}")


# ---------------- stage: cache ----------------

def cmd_cache() -> None:
    from kronos_qlib import QlibProvider

    t0 = time.perf_counter()
    provider = QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    backbone, _, identity = K.load_frozen_g1()
    verify = K.verify_sample_against_g1(provider, backbone, identity)
    all_stats = {}
    for split in ("train", "val", "W3", "W4"):
        t1 = time.perf_counter()
        st = K.build_peer_split(provider, backbone, identity, split)
        all_stats[split] = st
        budget_log(f"cache:{split}", time.perf_counter() - t1, st["cuda_s"],
                   "cuda:0", note=f"rebuilt={st['rebuilt']} reused={st['reused']}")
    manifest = K.freeze_peer_cache(C.ART_DIR)
    (C.ART_DIR / "peer_cache_build_stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (C.ART_DIR / "peer_verify_gate.json").write_text(
        json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")
    _r_log_cache(manifest, all_stats)
    logger.info(f"[cache] 全段完成 wall={time.perf_counter() - t0:.0f}s；"
                f"manifest 已冻结")


def _r_log_cache(manifest: dict, all_stats: dict) -> None:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    from qlib.workflow import R

    with R.start(experiment_name=C.EXPERIMENT_R, recorder_name="cache"):
        R.log_params(protocol=C.PROTOCOL_VERSION, run_id=C.RUN_ID,
                     stage="cache")
        for split, st in all_stats.items():
            R.log_metrics(**{f"cache/{split}/days": st["n_days"],
                             f"cache/{split}/rebuilt": st["rebuilt"],
                             f"cache/{split}/extra_h": sum(
                                 d.get("n_extra", 0) for d in st["days"])})
        R.log_artifact(str(C.ART_DIR / "peer_cache_manifest.json"))
        R.end_exp(recorder_status="FINISHED")


# ---------------- stage: pilot / confirm ----------------

def _train_missing(seeds: list[int], stage_name: str) -> dict:
    """训练尚未存在的 (arm, seed)（confirm 复用 pilot 的 s100 权重）。"""
    import torch

    stats = PD.load_frozen_stats()
    weight_shas = PD.g1_weight_shas()
    train_days = PD.load_split_days("train", stats, weight_shas)
    val_days = PD.load_split_days("val", stats, weight_shas)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    heads = {}
    t0 = time.perf_counter()
    for seed in seeds:
        for arm in C.ARMS:
            if (C.HEADS_DIR / f"best_{arm}_s{seed}.json").is_file():
                logger.info(f"[{stage_name}] {arm} s{seed} 已存在，跳过重训")
                heads[(arm, seed)] = json.loads(
                    (C.HEADS_DIR / f"best_{arm}_s{seed}.json"
                     ).read_text(encoding="utf-8"))
                continue
            heads[(arm, seed)] = T.train_arm(
                train_days, val_days, arm, seed, C.HEADS_DIR, device=device,
                epochs=C.TRAIN["epochs"])
    budget_log(f"{stage_name}:train_heads", time.perf_counter() - t0, 0.0,
               device,
               note=f"seeds={seeds} arms={list(C.ARMS)}（小头训练）")
    # OFF/ON 配对门禁：同 seed 初始参数字节 + 同日键摘要 + 同 shuffle
    for seed in seeds:
        h1 = T.state_dict_hash(T.build_head(seed, int(stats["d_in"])))
        h2 = T.state_dict_hash(T.build_head(seed, int(stats["d_in"])))
        assert h1 == h2, f"s{seed} 同 seed 初始哈希不稳定"
        d_off = heads[("OFF", seed)]["day_keys_digest"]
        d_on = heads[("ON", seed)]["day_keys_digest"]
        assert d_off == d_on, f"s{seed} OFF/ON 日键摘要不一致"
        assert heads[("OFF", seed)]["first_epoch_perm"] == \
            heads[("ON", seed)]["first_epoch_perm"], \
            f"s{seed} OFF/ON 首 epoch shuffle 不一致"
    # 提交用产物：best 头权重 + 指标拷入 art_dir
    for (arm, seed), res in heads.items():
        src = T.head_file(C.HEADS_DIR, arm, seed, res["best_epoch"])
        shutil.copy(src, C.ART_DIR / src.name)
        for suffix in ("epochs_metrics", "best"):
            src2 = C.HEADS_DIR / f"{suffix}_{arm}_s{seed}.json"
            if src2.is_file():
                shutil.copy(src2, C.ART_DIR / src2.name)
    return {"stats": stats, "heads": heads, "weight_shas": weight_shas,
            "n_train_days": len(train_days), "n_val_days": len(val_days)}


def _run_eval(seeds: list[int], stage_name: str) -> dict:
    """训练缺失 seeds 两臂 + 同口径评价 + R 记录（pilot/confirm 共用）。"""
    import os

    from kronos_qlib import QlibProvider

    QlibProvider.init_qlib_once()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    ctx = _train_missing(seeds, stage_name)
    stats, heads = ctx["stats"], ctx["heads"]
    gate = E.reproduce_g1_gate(C.ART_DIR)
    t1 = time.perf_counter()
    summary = E.run_comparison(C.ART_DIR, C.HEADS_DIR, stats, seeds,
                               weight_shas=ctx["weight_shas"])
    budget_log(f"{stage_name}:backtest", time.perf_counter() - t1, 0.0, "cpu",
               note="Qlib 回测 CPU，不计 GPU 预算")
    # 残差信号落盘（复算凭据，小 parquet 入库；行列对齐基线顺序）
    for seed in seeds:
        for arm in C.ARMS:
            for w in C.TEST_WINDOWS:
                wide, _ = E.infer_test_signal(arm, seed, w, stats, C.HEADS_DIR,
                                              ctx["weight_shas"])
                ref = E.load_wide(w)
                wide = wide.reindex(index=ref.index, columns=ref.columns)
                wide.to_parquet(C.ART_DIR / f"signal_{w}_{arm}_s{seed}.parquet")
    verdict = E.judge_confirm_gate(summary["per_seed"])
    out = {"sigma_e": stats["sigma_e"], "d_in": stats["d_in"],
           "n_train_days": ctx["n_train_days"], "n_val_days": ctx["n_val_days"],
           "heads": {f"{a}_s{s}": {"best_epoch": r["best_epoch"],
                                   "best_val_l_pred": r["best_val_l_pred"]}
                     for (a, s), r in heads.items()},
           "reproduction_gate": gate, "verdict": verdict}
    (C.ART_DIR / f"{stage_name}_verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _r_log_eval(stage_name, seeds, summary, stats, heads, verdict)
    return out


def _r_log_eval(stage_name: str, seeds: list[int], summary: dict, stats: dict,
                heads: dict, verdict: dict) -> None:
    from qlib.workflow import R

    with R.start(experiment_name=C.EXPERIMENT_R,
                 recorder_name=f"{stage_name}_s{'_'.join(map(str, seeds))}"):
        R.log_params(protocol=C.PROTOCOL_VERSION, seeds=seeds,
                     sigma_e=stats["sigma_e"], d_in=stats["d_in"],
                     epochs=C.TRAIN["epochs"], batch_days=C.TRAIN["batch_days"])
        for (arm, seed), res in heads.items():
            R.log_metrics(**{f"{arm}_s{seed}_best_epoch": res["best_epoch"],
                             f"{arm}_s{seed}_best_val_mse": res["best_val_l_pred"]})
        for seed, p in summary["per_seed"].items():
            for w in C.TEST_WINDOWS:
                R.log_metrics(**{
                    f"{w}/s{seed}/D_OFF": p["D_OFF"][w],
                    f"{w}/s{seed}/D_ON": p["D_ON"][w],
                    f"{w}/s{seed}/D_peer": p["D_peer"][w],
                    f"{w}/s{seed}/G1_curve_end": p["arms"]["G1_mean"][w]["curve_end"],
                })
        R.log_artifact(str(C.ART_DIR / f"{stage_name}_verdict.json"))
        for w in C.TEST_WINDOWS:
            R.log_artifact(str(C.ART_DIR / f"fig_{w}_main.png"))
        R.end_exp(recorder_status="FINISHED")


def cmd_pilot() -> None:
    out = _run_eval([C.PILOT_SEED], "pilot")
    v = out["verdict"]
    gate_pass = (v["off"]["both_windows_count"] >= 1
                 or v["on"]["both_windows_count"] >= 1)
    led = budget_load()
    summary = json.loads((C.ART_DIR / "comparison_summary.json").read_text(
        encoding="utf-8"))
    logger.info("==== pilot 主结论 ====")
    logger.info(json.dumps(
        {str(s): {"D_OFF": p["D_OFF"], "D_ON": p["D_ON"], "D_peer": p["D_peer"],
                  "off_both": p["off_beats_g1_both"],
                  "on_both": p["on_beats_g1_both"]}
         for s, p in summary["per_seed"].items()}, ensure_ascii=False))
    logger.info(f"确认门禁（任一头两窗均胜 G1）: "
                f"{'通过 → 可 confirm' if gate_pass else '未通过 → 封存本配置，不补种子'}")
    logger.info(f"预算：GPU wall 已用 {led['gpu_used_wall_s']:.0f}s / "
                f"{C.GPU_BUDGET_S}s")


def cmd_confirm() -> None:
    pilot = json.loads((C.ART_DIR / "pilot_verdict.json").read_text(encoding="utf-8"))
    v = pilot["verdict"]
    gate_pass = (v["off"]["both_windows_count"] >= 1
                 or v["on"]["both_windows_count"] >= 1)
    if not gate_pass:
        raise RuntimeError("pilot 门禁未通过（无头两窗均胜 G1）→ 封存配置，不补种子")
    led = budget_load()
    # confirm 仅补训 101/102 + CPU 回测；以 pilot 训练耗时上界估 GPU 需求
    est_confirm_s = led["gpu_used_wall_s"] * 0.1
    if led["remaining_gpu_budget_s"] < est_confirm_s:
        raise RuntimeError(f"GPU 预算不足：余 {led['remaining_gpu_budget_s']}s")
    _run_eval(list({C.PILOT_SEED, *C.CONFIRM_SEEDS}), "confirm")


# ---------------- stage: report ----------------

def cmd_report() -> None:
    from peer_residual.report import write_report

    write_report(REPORT_DOC, C.ART_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="PEER1 同日跨股票交互残差头")
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "smoke", "cache", "pilot",
                                 "confirm", "report"])
    args = parser.parse_args()
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(str(C.ART_DIR / f"{args.stage}.log"), enqueue=False)
    if args.stage == "preflight":
        cmd_preflight()
    elif args.stage == "smoke":
        cmd_smoke()
    elif args.stage == "cache":
        cmd_cache()
    elif args.stage == "pilot":
        cmd_pilot()
    elif args.stage == "confirm":
        cmd_confirm()
    else:
        cmd_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
