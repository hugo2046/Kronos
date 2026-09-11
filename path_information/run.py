"""PATH1 实验 CLI（计划 §8）。

用法::

    python -m path_information.run --stage preflight  # 冻结协议 manifest
    python -m path_information.run --stage smoke      # train 前2日×64股估时
    python -m path_information.run --stage cache      # 三段路径+train/val标签
    python -m path_information.run --stage train      # 双臂小头（CPU，seed100）
    python -m path_information.run --stage evaluate   # 冻结分数→开封→诊断判据
    python -m path_information.run --stage report     # docs 结果报告

训练 stage 不读 dev_eval 标签；evaluate 只读已冻结分数，经 spy 锁定后
才加载 dev_eval 标签（计划 §6/§8）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from path_information import config as C
from path_information import data as D
from path_information import evaluate as E
from path_information import paths as PP
from path_information import train as T

GPU_STAGES = {"smoke", "cache"}


# ---------------- 预算台账（GPU 2h / CPU 1h，失败尝试计入） ----------------

def budget_load() -> dict:
    if C.BUDGET_LEDGER.is_file():
        return json.loads(C.BUDGET_LEDGER.read_text(encoding="utf-8"))
    return {"entries": [], "gpu_budget_s": C.GPU_BUDGET_S,
            "cpu_budget_s": C.CPU_BUDGET_S}


def budget_log(stage: str, wall_s: float, device: str, note: str = "") -> dict:
    led = budget_load()
    led["entries"].append({"ts": datetime.now().isoformat(timespec="seconds"),
                           "stage": stage, "wall_s": round(wall_s, 1),
                           "device": device, "note": note})
    led["gpu_used_wall_s"] = round(sum(e["wall_s"] for e in led["entries"]
                                       if e["device"] != "cpu"), 1)
    led["cpu_used_wall_s"] = round(sum(e["wall_s"] for e in led["entries"]
                                       if e["device"] == "cpu"), 1)
    led["remaining_gpu_budget_s"] = round(
        C.GPU_BUDGET_S - led["gpu_used_wall_s"], 1)
    led["remaining_cpu_budget_s"] = round(
        C.CPU_BUDGET_S - led["cpu_used_wall_s"], 1)
    C.BUDGET_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    C.BUDGET_LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    if led["remaining_gpu_budget_s"] < 0 and stage in GPU_STAGES:
        raise RuntimeError(f"GPU 预算超限：{led['gpu_used_wall_s']}s / "
                           f"{C.GPU_BUDGET_S}s —— 停止并回传预算缺口，"
                           f"不删日期/减采样/扩预算")
    if led["remaining_cpu_budget_s"] < 0 and stage == "train":
        raise RuntimeError(f"CPU 预算超限：{led['cpu_used_wall_s']}s / "
                           f"{C.CPU_BUDGET_S}s")
    logger.info(f"[budget] {stage}: wall={wall_s:.1f}s device={device} | "
                f"GPU累计={led['gpu_used_wall_s']}s 余"
                f"{led['remaining_gpu_budget_s']}s | CPU累计="
                f"{led['cpu_used_wall_s']}s 余{led['remaining_cpu_budget_s']}s")
    return led


def get_calendar(provider) -> pd.DatetimeIndex:
    return provider.trading_days()


def _git_head() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


# ---------------- stage: preflight ----------------

def cmd_preflight() -> None:
    from kronos_qlib import QlibProvider

    mf_path = C.ART_DIR / "preflight_manifest.json"
    if mf_path.exists():
        raise RuntimeError(f"preflight manifest 已存在，不可覆盖：{mf_path}")
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    weight_shas = PP.g1_weight_shas()        # 含与计划冻结值的比对
    provider = QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    cal = get_calendar(provider)
    manifest: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": C.PROTOCOL_VERSION,
        "git_head": _git_head(),
        "run_id": C.RUN_ID,
        "baseline": {}, "g1_weights": weight_shas,
        "segments": {}, "sample_rule": {
            "pool": C.POOL, "lookback": C.LOOKBACK,
            "predict_len": C.PREDICT_LEN, "per_day": C.PER_DAY,
            "select": "SHA256('PATH1|17|YYYY-MM-DD|code') 升序前 64，"
                      "最终推理按 code 排序",
            "eligibility": "PIT csi300 ∩ fetch≤t 90日窗口 ∩ 基线G1有限信号；"
                           "不以未来收益/停牌/标签完整性挑选",
        },
        "inference": {**C.INFERENCE, "note": "与原 G1 同参的新采样链；"
                     "新链均值与基线差仅记录不替换"},
        "head": {"params": 753, **C.HEAD, "train": {**C.TRAIN,
                "note": "e0 为合法候选；val 日等权 MSE 最低选点"}},
        "budget": {"gpu_s": C.GPU_BUDGET_S, "cpu_s": C.CPU_BUDGET_S},
        "dedup": {
            "return_samples_scan": "全远端分支 git grep 仅本计划文档命中；"
                                   "SAE/PEER 缓存 schema 仅 h/s_g1/y_mean 无样本轴",
            "prior_art": "旧 mean/max 为固定聚合；N50 改采样数；"
                         "截面zero-shot 平均后用 mean；无采样集合学习头先例",
            "reference_doc": "docs/Kronos结构实验去重与方向复核_20260910.md"},
        "g1_time_evidence": {
            "evidence_level": "配置/目录级（非历史真实时点部署证明）",
            "train_time_range": ["2011-01-01", "2024-12-31"],
            "val_time_range": ["2025-01-01", "2025-06-30"],
            "dataset_end_time": "2025-06-30",
            "source": "finetune_suite/config.py（G1 套件声明）",
            "upstream_pretrain_boundary": "上游 Kronos 预训练语料边界未知，"
                                          "如实披露；本轮 train 段 2025-07 起"
                                          "晚于 G1 选模末 2025-06-30"},
    }
    for w, p in C.BASELINE_SIGNALS.items():
        assert p.is_file(), f"基线信号缺失（主仓库只读引用）：{p}"
        sha = PP.sha256_file(p)
        assert sha == C.BASELINE_SHA256[w], f"{w} 基线 SHA 不匹配：{sha}"
        wide = pd.read_parquet(p)
        if not isinstance(wide.index, pd.DatetimeIndex):
            wide = wide.T
        wide = wide.sort_index()
        assert wide.index.max() <= pd.Timestamp(C.FORWARD_CUTOFF)
        vals = wide.to_numpy()
        assert np.isfinite(vals[~np.isnan(vals)]).all()
        manifest["baseline"][w] = {
            "path": str(p), "sha256": sha, "n_days": int(wide.shape[0]),
            "n_cells": int(wide.notna().sum().sum()),
            "date_min": str(wide.index.min().date()),
            "date_max": str(wide.index.max().date())}
    for split, (start, label_end) in C.SEGMENTS.items():
        days = D.segment_days(cal, start, label_end)
        wname = C.SEGMENT_WINDOW[split]
        wide = pd.read_parquet(C.BASELINE_SIGNALS[wname])
        if not isinstance(wide.index, pd.DatetimeIndex):
            wide = wide.T
        missing = [str(d.date()) for d in days if d not in wide.index]
        assert not missing, f"{split} 决策日不在 {wname} 基线内：{missing[:5]}"
        manifest["segments"][split] = {
            "decision_days": [str(days.min().date()), str(days.max().date())],
            "n_days": int(len(days)), "label_end": label_end,
            "baseline_window": wname,
            "expected_cells": int(len(days)) * C.PER_DAY}
    mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    logger.info(f"[preflight] manifest 冻结：{mf_path} | 段日期规模 "
                f"{ {k: v['n_days'] for k, v in manifest['segments'].items()} }")


# ---------------- stage: smoke ----------------

def cmd_smoke() -> None:
    from kronos_qlib import QlibProvider

    t0 = time.perf_counter()
    provider = QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    cal = get_calendar(provider)
    weight_shas = PP.g1_weight_shas()
    wide = _load_baseline("W3")

    def loader():
        return PP.load_frozen_g1()

    st = PP.build_segment_paths(provider, loader, "train", cal,
                                C.CACHE_DIR / "smoke_train", wide, weight_shas,
                                limit_days=2)
    wall = time.perf_counter() - t0
    import torch

    vram_gb = (torch.cuda.max_memory_allocated() / 1e9
               if torch.cuda.is_available() else 0.0)
    n_days = {s: len(D.segment_days(cal, *bounds))
              for s, bounds in C.SEGMENTS.items()}
    total_days = sum(n_days.values())
    per_day = st["wall_s"] / max(st["rebuilt"], 1)
    est = round(per_day * total_days * 1.5, 1)     # 保守外推 ×1.5（§8）
    out = {"smoke_wall_s": round(wall, 1),
           "days_done": st["rebuilt"],
           "per_day_slower_s": round(max(e["wall_s"] for e in st["excluded"]),
                                     2),
           "per_day_mean_s": round(per_day, 2),
           "peak_vram_gb": round(vram_gb, 2),
           "segment_days": n_days, "total_days": total_days,
           "estimated_full_gpu_wall_s_x1_5": est,
           "gpu_budget_s": C.GPU_BUDGET_S}
    (C.ART_DIR / "smoke_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    budget_log("smoke", wall, "cuda:0",
               note=f"train×{st['rebuilt']} est_full_x1.5={est}s")
    logger.info(f"[smoke] {json.dumps(out, ensure_ascii=False)}")
    if est > C.GPU_BUDGET_S:
        raise RuntimeError(
            f"预算缺口：保守外推 {est}s > GPU 预算 {C.GPU_BUDGET_S}s —— 停止，"
            f"不减采样/删日期/扩大预算")


# ---------------- stage: cache ----------------

def _load_baseline(wname: str) -> pd.DataFrame:
    wide = pd.read_parquet(C.BASELINE_SIGNALS[wname])
    if not isinstance(wide.index, pd.DatetimeIndex):
        wide = wide.T
    return wide.sort_index()


def cmd_cache() -> None:
    from kronos_qlib import QlibProvider

    t0 = time.perf_counter()
    provider = QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    cal = get_calendar(provider)
    weight_shas = PP.g1_weight_shas()
    baselines = {w: _load_baseline(w) for w in ("W3", "W4")}

    def loader():
        return PP.load_frozen_g1()

    all_stats: dict = {}
    for split in C.SEGMENTS:
        led = budget_load()
        if led["remaining_gpu_budget_s"] <= 0:
            raise RuntimeError("GPU 预算耗尽，停止 cache")
        t1 = time.perf_counter()
        wname = C.SEGMENT_WINDOW[split]
        st = PP.build_segment_paths(provider, loader, split, cal,
                                    C.CACHE_DIR / split, baselines[wname],
                                    weight_shas)
        all_stats[f"paths:{split}"] = st
        budget_log(f"cache:paths:{split}", time.perf_counter() - t1, "cuda:0",
                   note=f"rebuilt={st['rebuilt']} reused={st['reused']}")
    # 标签独立阶段（train/val；dev_eval 由 evaluate 开封后读取）
    for split in ("train", "val"):
        t1 = time.perf_counter()
        all_stats[f"labels:{split}"] = PP.build_segment_labels(
            provider, split, cal, C.CACHE_DIR, weight_shas)
        budget_log(f"cache:labels:{split}", time.perf_counter() - t1, "cpu",
                   note=f"valid={all_stats[f'labels:{split}']['n_valid']}")
    cov = {s: all_stats[f"labels:{s}"]["coverage"] for s in ("train", "val")}
    for s, c in cov.items():
        if c < 0.95:
            raise RuntimeError(f"{s} 有效标签覆盖率 {c:.4f} < 95% —— 停止解释，"
                               f"不靠删日继续")
    manifest = freeze_cache_manifest()
    (C.ART_DIR / "cache_build_stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info(f"[cache] 全段完成 wall={time.perf_counter() - t0:.0f}s；"
                f"manifest 冻结（{manifest['n_chunks']} chunks）")


def freeze_cache_manifest() -> dict:
    """路径+标签缓存冻结：逐 chunk SHA + 样本键 parquet + 总规模。"""
    chunks_all = []
    sample_rows = []
    for split in C.SEGMENTS:
        for p in sorted((C.CACHE_DIR / split).glob(f"{split}_*.npz")):
            arr = PP.read_path_chunk(p, {"protocol": C.PROTOCOL_VERSION,
                                         "split": split})
            sample_rows.append(pd.DataFrame({
                "split": split, "date": arr["dates"],
                "instrument": arr["instruments"],
                "x_start": arr["x_start"], "x_end": arr["x_end"],
                "label_start": arr["label_start"],
                "label_end": arr["label_end"],
                "close_t": arr["close_t"], "s_g1": arr["s_g1"],
                "s_new_mean": arr["s_new_mean"]}))
            chunks_all.append({"split": split, "file": p.relative_to(
                C.CACHE_DIR).as_posix(), "sha256": PP.sha256_file(p),
                "n": int(arr["close_paths"].shape[0]),
                "shape": list(arr["close_paths"].shape)})
    keys = pd.concat(sample_rows, ignore_index=True)
    keys_path = C.ART_DIR / "sample_keys.parquet"
    keys.to_parquet(keys_path)
    labels = []
    for split in ("train", "val"):
        p = C.CACHE_DIR / f"labels_{split}.npz"
        labels.append({"split": split, "file": p.relative_to(
            C.CACHE_DIR).as_posix(), "sha256": PP.sha256_file(p)})
    total_bytes = sum((C.CACHE_DIR / c["file"]).stat().st_size
                      for c in chunks_all)
    manifest = {"created_at": datetime.now().isoformat(timespec="seconds"),
                "protocol": C.PROTOCOL_VERSION,
                "n_chunks": len(chunks_all), "chunks": chunks_all,
                "labels": labels,
                "sample_keys_file": keys_path.name,
                "sample_keys_sha256": PP.sha256_file(keys_path),
                "n_cells": int(len(keys)),
                "cache_total_bytes": total_bytes,
                "cache_total_mib": round(total_bytes / 2**20, 2)}
    (C.ART_DIR / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ---------------- stage: train ----------------

def load_segment_days(split: str, need_labels: bool = True) -> list[dict]:
    """路径 chunk + （可选）标签 chunk → 逐日训练数据（shape/s_g1/y/valid）。"""
    chunks = sorted((C.CACHE_DIR / split).glob(f"{split}_*.npz"))
    assert chunks, f"{split} 路径缓存缺失"
    label_path = C.CACHE_DIR / f"labels_{split}.npz"
    y_map: dict[tuple[str, str], float] = {}
    if need_labels:
        assert label_path.is_file(), f"{split} 标签缓存缺失：{label_path}"
        lab = PP.read_path_chunk(label_path, {"protocol": C.PROTOCOL_VERSION,
                                              "split": split, "kind": "labels"})
        for ds, code, y in zip(lab["dates"], lab["instruments"], lab["y_mean"]):
            y_map[(str(ds), str(code))] = float(y)
    days = []
    for p in chunks:
        arr = PP.read_path_chunk(p, {"protocol": C.PROTOCOL_VERSION,
                                     "split": split})
        p_rel = D.to_relative_paths(arr["close_paths"], arr["close_t"])
        shape = D.path_shape(p_rel)
        y = np.array([y_map.get((str(ds), str(c)), np.nan)
                      for ds, c in zip(arr["dates"], arr["instruments"])])
        days.append({"date": str(arr["dates"][0]), "shape": shape,
                     "s_g1": arr["s_g1"], "y": y,
                     "valid": D.valid_label_mask(y)})
    return days


def cmd_train() -> None:
    t0 = time.perf_counter()
    train_days = load_segment_days("train")
    val_days = load_segment_days("val")
    stats = T.fit_stats(train_days)           # 仅 train 有效格；退化即抛错
    payload = {k: v for k, v in stats.items()}
    (C.ART_DIR / "norm_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_sha = PP.sha256_file(C.ART_DIR / "norm_stats.json")
    cache_sha = PP.sha256_file(C.ART_DIR / "cache_manifest.json")
    context = {"stats_sha256": stats_sha, "cache_manifest_sha256": cache_sha,
               "protocol": C.PROTOCOL_VERSION}
    histories = {}
    for arm in C.ARMS:
        histories[arm] = T.train_arm(arm, train_days, val_days, stats,
                                     C.HEADS_DIR, seed=C.SEED,
                                     context=context)
    budget_log("train", time.perf_counter() - t0, "cpu",
               note=f"arms={list(C.ARMS)} seed={C.SEED} "
                    f"best={ {a: histories[a]['best_epoch'] for a in C.ARMS} }")
    out = {"stats": stats, "stats_sha256": stats_sha,
           "heads": {a: {"best_epoch": histories[a]["best_epoch"],
                         "best_val_mse": histories[a]["best_val_mse"],
                         "g1_zero_val_mse": histories[a]["g1_zero_val_mse"]}
                     for a in C.ARMS}}
    (C.ART_DIR / "train_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- stage: evaluate ----------------

def _score_segment(arm: str, split: str, stats: dict) -> dict:
    """载入 best 头，对全部段内格算 ``s_final``（此刻不读任何标签）。"""
    import torch

    from path_information import model as PM

    head, _ = T.load_best(C.HEADS_DIR, arm, C.SEED)
    parts_s, parts_d, parts_c = [], [], []
    with torch.no_grad():
        for p in sorted((C.CACHE_DIR / split).glob(f"{split}_*.npz")):
            arr = PP.read_path_chunk(p, {"protocol": C.PROTOCOL_VERSION,
                                         "split": split})
            p_rel = D.to_relative_paths(arr["close_paths"], arr["close_t"])
            shape = PM.arm_input(torch.from_numpy(
                (D.path_shape(p_rel) / stats["sigma_e"]).astype(np.float32)),
                arm)
            b = torch.from_numpy(
                ((arr["s_g1"] - stats["mu_g1"]) / stats["sigma_g1"]
                 ).astype(np.float32))
            r = head(shape, b).numpy().astype(np.float64)
            parts_s.append(E.final_scores(arr["s_g1"], r, stats))
            parts_d.extend(str(x) for x in arr["dates"])
            parts_c.extend(str(x) for x in arr["instruments"])
    return {"s_final": np.concatenate(parts_s), "dates": np.array(parts_d),
            "instruments": np.array(parts_c)}


def cmd_evaluate() -> None:
    t0 = time.perf_counter()
    stats = json.loads((C.ART_DIR / "norm_stats.json").read_text(encoding="utf-8"))
    # ① 先冻结两头全部段分数与来源 SHA（此刻不加载任何标签）
    files = {}
    for arm in C.ARMS:
        for split in C.SEGMENTS:
            out = _score_segment(arm, split, stats)
            name = f"scores_{arm}_{split}.npz"
            PP.write_path_chunk(C.ART_DIR, name, {
                "arm": np.array(arm), "split": np.array(split),
                "dates": out["dates"], "instruments": out["instruments"],
                "s_final": out["s_final"].astype(np.float64)}, {
                "protocol": C.PROTOCOL_VERSION, "arm": arm, "split": split,
                "kind": "scores", "seed": C.SEED})
            files[f"{arm}:{split}"] = {
                "file": name,
                "sha256": PP.sha256_file(C.ART_DIR / name)}
    (C.ART_DIR / "scores_manifest.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"),
                    "files": files}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    # ② spy 锁定开封：分数校验通过后才加载 dev_eval 标签
    dev_lab = E.unseal(C.ART_DIR)
    assert dev_lab["split"] == "dev_eval"
    # ③ 三臂逐日诊断（val / dev_eval，共同有效标签格）
    verdict: dict = {"segments": {}}
    for split in ("val", "dev_eval"):
        arrays = _segment_arrays(split)
        r_hat = {arm: (arrays[f"scores_{arm}"] - arrays["s_g1"])
                 / stats["sigma_e"] for arm in C.ARMS}
        ic = E.daily_spearman(
            arrays["dates"], arrays["y"], arrays["valid"],
            G1=arrays["s_g1"],
            MEAN=arrays["scores_MEAN"], PATH=arrays["scores_PATH"])
        dates_sorted = sorted(set(arrays["dates"]))
        pairs = E.paired_daily_ic(ic, dates_sorted)
        verdict["segments"][split] = {
            "n_cells": int(len(arrays["dates"])),
            "n_valid_cells": int(arrays["valid"].sum()),
            "n_days": len(dates_sorted),
            "ic_days_valid": {a: int(np.isfinite(v).sum()) for a, v in ic.items()},
            "daily_ic_mean": {a: float(np.nanmean(v)) for a, v in ic.items()},
            "mse": {
                "G1": E.arm_mse(arrays["dates"], arrays["y"], arrays["valid"],
                                arrays["s_g1"], np.zeros(len(arrays["dates"])),
                                stats),
                **{arm: E.arm_mse(arrays["dates"], arrays["y"],
                                  arrays["valid"], arrays["s_g1"],
                                  r_hat[arm], stats) for arm in C.ARMS}},
            "paired": pairs}
    v, dv = verdict["segments"]["val"], verdict["segments"]["dev_eval"]
    verdict["criterion_pass"] = E.criterion(v["paired"], dv["paired"])
    verdict["criterion"] = ("val 与 dev_eval 的 PATH−MEAN 日均配对 IC 差均>0 "
                            "且 dev_eval 的 PATH−G1>0")
    (C.ART_DIR / "evaluate_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    budget_log("evaluate", time.perf_counter() - t0, "cpu",
               note=f"criterion_pass={verdict['criterion_pass']}")
    logger.info(f"[evaluate] 判据 {'触发' if verdict['criterion_pass'] else '未触发'}"
                f"（不满足即封存本配置，不自动扩展）")
    logger.info(json.dumps({s: {"paired": {k: p["mean"] for k, p in
                                           seg["paired"].items()},
                                "daily_ic_mean": seg["daily_ic_mean"]}
                            for s, seg in verdict["segments"].items()},
                           ensure_ascii=False))


def _segment_arrays(split: str) -> dict:
    """段数组装配：dates/codes/y/valid/s_g1 + 两臂冻结分数（顺序一致）。"""
    chunks = sorted((C.CACHE_DIR / split).glob(f"{split}_*.npz"))
    assert chunks, f"{split} 缓存缺失"
    dates, codes, s_g1 = [], [], []
    for p in chunks:
        arr = PP.read_path_chunk(p, {"protocol": C.PROTOCOL_VERSION,
                                     "split": split})
        dates.extend(str(x) for x in arr["dates"])
        codes.extend(str(x) for x in arr["instruments"])
        s_g1.append(arr["s_g1"])
    label_path = C.CACHE_DIR / f"labels_{split}.npz"
    lab = PP.read_path_chunk(label_path, {"protocol": C.PROTOCOL_VERSION,
                                          "split": split, "kind": "labels"})
    y_map = {(str(d), str(c)): float(y) for d, c, y in
             zip(lab["dates"], lab["instruments"], lab["y_mean"])}
    y = np.array([y_map.get((d, c), np.nan) for d, c in zip(dates, codes)])
    scores = {}
    for arm in C.ARMS:
        sc = PP.read_path_chunk(C.ART_DIR / f"scores_{arm}_{split}.npz",
                                {"protocol": C.PROTOCOL_VERSION, "arm": arm,
                                 "split": split, "kind": "scores",
                                 "seed": C.SEED})
        assert [str(x) for x in sc["dates"]] == dates, \
            f"{arm}:{split} 冻结分数与缓存键不对齐"
        assert [str(x) for x in sc["instruments"]] == codes, \
            f"{arm}:{split} 冻结分数与缓存股票键不对齐"
        scores[f"scores_{arm}"] = sc["s_final"]
    return {"dates": np.array(dates), "codes": np.array(codes),
            "s_g1": np.concatenate(s_g1), "y": y,
            "valid": D.valid_label_mask(y), **scores}


# ---------------- stage: report ----------------

def cmd_report() -> None:
    from path_information.report import write_report

    write_report(C.REPORT_DOC, C.ART_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="PATH1 采样路径信息增量实验")
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "smoke", "cache", "train",
                                 "evaluate", "report"])
    args = parser.parse_args()
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(str(C.ART_DIR / f"{args.stage}.log"), enqueue=False)
    {"preflight": cmd_preflight, "smoke": cmd_smoke, "cache": cmd_cache,
     "train": cmd_train, "evaluate": cmd_evaluate,
     "report": cmd_report}[args.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
