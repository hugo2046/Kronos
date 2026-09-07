"""MH1 统一 CLI（计划 §7）：preflight → smoke → train → evaluate。

阶段状态落 ``data/stage_state.json``；preflight 冻结协议（protocol.json +
SHA256），后续阶段一律校验哈希；evaluate 在协议、六份 checkpoint 与训练
完成状态不一致时拒绝执行。

用法（仓库根目录）::

    /home/user/miniconda3/envs/quant/bin/python -m mh1_multihorizon.run --stage preflight
    /home/user/miniconda3/envs/quant/bin/python -m mh1_multihorizon.run --stage smoke
    /home/user/miniconda3/envs/quant/bin/python -m mh1_multihorizon.run --stage train
    /home/user/miniconda3/envs/quant/bin/python -m mh1_multihorizon.run --stage evaluate
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from mh1_multihorizon import config as C

STATE_PATH = C.DATA_DIR / "stage_state.json"


def _read_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _mark_stage(stage: str, payload: dict) -> None:
    st = _read_state()
    st[stage] = {"at": datetime.now().isoformat(timespec="seconds"),
                 **payload}
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(st, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


def _require_stage(*stages: str) -> None:
    st = _read_state()
    missing = [s for s in stages if s not in st]
    if missing:
        raise RuntimeError(f"阶段未完成：{missing}（按 preflight→smoke→train→evaluate 顺序）")


# ============================================================
# preflight：环境与只读依赖检查 + 协议冻结（禁训练、禁 forward）
# ============================================================


def _env_record() -> dict:
    import torch

    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
            "cuda_version": torch.version.cuda,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        }
    import qlib  # noqa: F401

    return {
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "gpu": gpu,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=C.REPO_ROOT).stdout.strip(),
        "git_branch": subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True,
            cwd=C.REPO_ROOT).stdout.strip(),
    }


def _g1_record() -> dict:
    from finetune_suite.train_g1 import G1Config

    g1 = G1Config()
    tok = Path(g1.finetuned_tokenizer_path)
    pred = Path(g1.finetuned_predictor_path)
    for d in (tok, pred):
        assert (d / "model.safetensors").is_file(), f"G1 权重缺失：{d}"
    cfg = json.loads((pred / "config.json").read_text(encoding="utf-8"))
    return {
        "tokenizer_sha256": C.sha256_file(tok / "model.safetensors"),
        "predictor_sha256": C.sha256_file(pred / "model.safetensors"),
        "d_model": int(cfg["d_model"]),
        "tokenizer_path": str(tok),
        "predictor_path": str(pred),
        "seed_evidence": "finetune_suite/config.py:87 seed=100 + "
                         "docs/扩语料微调实验结果_20260815.md（单次 seed=100）",
    }


def _pkl_record() -> dict:
    from h1_readout.corpus_loader import load_corpus_split

    rec: dict = {}
    for split in ("train", "val"):
        data = load_corpus_split(C.POOL, split)
        dates = [df.index for df in data.values()]
        path = str(C.REPO_ROOT / "finetune_suite" / "data" /
                   ("train_data.pkl" if split == "train" else "val_data.pkl"))
        rec[split] = {
            "path": path, "sha256": C.sha256_file(path),
            "n_symbols": len(data),
            "date_min": str(min(d.min() for d in dates).date()),
            "date_max": str(max(d.max() for d in dates).date()),
        }
    return rec


def _segments_record() -> tuple[dict, dict]:
    """真实四段构造摘要（训练 union pkl + 验证/W3/W4 qlib PIT）。"""
    from kronos_qlib import QlibProvider
    from mh1_multihorizon.data import (
        build_pit_days, load_union_tables, scan_train_days, union_calendar,
    )
    from mh1_multihorizon.evaluate import WINDOW_BOUNDS

    tables = load_union_tables(C.POOL)
    calendar = union_calendar(tables)
    train_days, tstats = scan_train_days(
        tables, calendar, start=C.TRAIN_START, label_end=C.TRAIN_LABEL_END)
    assert train_days, "训练段为空（数据异常，停止）"
    assert tstats.label_endpoint_max <= C.TRAIN_LABEL_END

    prov = QlibProvider(C.POOL, C.VAL_START, C.FORWARD_CUTOFF)
    full_cal = prov.trading_days(end=C.FORWARD_CUTOFF)

    def _endpoint_max(decision_max: str) -> str:
        c = int(full_cal.searchsorted(pd.Timestamp(decision_max)))
        return str(full_cal[c + C.PURGE].date())

    manifest: dict = {"train": {
        "per_day_counts": [len(d.codes) for d in train_days],
        "decision_dates": [str(d.date.date()) for d in train_days],
    }}
    seg: dict = {
        "train": {
            "source": "union pkl（G1 同源，H1a 同口径）",
            **{k: v for k, v in tstats.__dict__.items()},
        }
    }
    val_days, vstats = build_pit_days(
        prov, start=C.VAL_START, end=C.VAL_LABEL_END, label_end=C.VAL_LABEL_END,
        require_complete_labels=True)
    assert val_days, "验证段为空"
    seg["val"] = {
        "source": "qlib PIT csi300（H1 早停段同口径）", "n_days": vstats.n_days,
        "decision_min": vstats.decision_min, "decision_max": vstats.decision_max,
        "n_samples": int(sum(len(d.codes) for d in val_days)),
        "label_endpoint_max": _endpoint_max(vstats.decision_max),
        "excluded_missing_label_total": int(sum(
            v["missing_label"] for v in vstats.per_day.values())),
    }
    assert seg["val"]["label_endpoint_max"] <= C.VAL_LABEL_END
    manifest["val"] = {"per_day": vstats.per_day}

    for wname, (start, end) in WINDOW_BOUNDS.items():
        days, wstats = build_pit_days(
            prov, start=start, end=end, label_end=end,
            require_complete_labels=False)
        assert days, f"{wname} 为空"
        seg[wname] = {
            "source": "qlib PIT csi300（H1 打分同口径；+20 终点留段内）",
            "n_days": wstats.n_days, "decision_min": wstats.decision_min,
            "decision_max": wstats.decision_max,
            "n_samples": int(sum(len(d.codes) for d in days)),
            "label_endpoint_max": _endpoint_max(wstats.decision_max),
        }
        assert seg[wname]["label_endpoint_max"] <= end, (
            f"{wname} 标签终点越界：{seg[wname]['label_endpoint_max']} > {end}")
        manifest[wname] = {"per_day": wstats.per_day}
    # forward 封存总门禁：全部段决策日 + 期限终点 ≤ 2026-07-24
    for wname in ("val", "W3", "W4"):
        assert seg[wname]["decision_max"] <= C.FORWARD_CUTOFF
    seg["forward_cutoff"] = C.FORWARD_CUTOFF
    seg["purge"] = C.PURGE
    return seg, manifest


def _cron_record() -> dict:
    try:
        out = subprocess.run(["crontab", "-l"], capture_output=True,
                             text=True).stdout
    except OSError:
        return {"status": "crontab 不可读"}
    entry = [ln for ln in out.splitlines()
             if "run_registry" in ln or "ys-data" in ln]
    return {
        "entries": entry,
        "guard": list(C.REGISTRY_GUARD),
        "note": "16:30 登记任务优先；守卫窗内训练进程释放 GPU（08-20 OOM 教训）",
    }


def cmd_preflight() -> None:
    # 命名冲突检查：本包 data 目录已含他人协议则停下报告
    if C.PROTOCOL_PATH.is_file():
        raise RuntimeError(
            f"已存在协议 {C.PROTOCOL_PATH}——同名实验/旧协议在场，"
            "按计划停下报告，不覆盖")
    env = _env_record()
    g1 = _g1_record()
    pkl = _pkl_record()
    seg, manifest = _segments_record()
    cron = _cron_record()
    assert g1["d_model"] == 832, f"d_model 异常：{g1['d_model']}"

    protocol = {
        "experiment": "MH1",
        "plan": "docs/TimesFM多期限监督实验计划_20260907.md",
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "design": {
            "horizons": list(C.HORIZONS), "main_horizon_idx": C.MAIN_HORIZON_IDX,
            "pool": C.POOL, "lookback": C.LOOKBACK, "clip": C.CLIP,
            "head": "LayerNorm(d)→Linear(d,128)→GELU→Linear(128,4)",
            "batch": C.BATCH, "lr": C.LR, "wd": C.WD,
            "steps_per_epoch": C.STEPS_PER_EPOCH, "epochs": C.EPOCHS,
            "seeds": list(C.SEEDS), "arms": list(C.ARMS),
            "run_order": list(C.RUN_ORDER), "precision": C.PRECISION,
            "min_train_cross": C.MIN_TRAIN_CROSS,
            "ic_min_stocks": C.IC_MIN_STOCKS, "nw_lag": C.NW_LAG,
        },
        "g1_weights": g1,
        "corpus_pkl": pkl,
        "segments": seg,
        "cron": cron,
        "budget_gpu_hours": C.GPU_BUDGET_HOURS,
    }
    path, sha = C.write_protocol(protocol)
    (C.DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    _mark_stage("preflight", {"protocol_sha256": sha})
    logger.info(f"协议冻结：{path}（SHA256 {sha}）")
    logger.info(f"训练段 {seg['train']['n_days']} 日 / "
                f"{seg['train']['n_samples']:,} 样本；验证段 "
                f"{seg['val']['n_days']} 日；W3 {seg['W3']['n_days']} 日；"
                f"W4 {seg['W4']['n_days']} 日")
    logger.info(f"环境：{env['python']} / torch {env['torch']} / "
                f"GPU {env['gpu'].get('name')}")


# ============================================================
# smoke / train / evaluate
# ============================================================


def cmd_smoke() -> None:
    _require_stage("preflight")
    protocol, sha = C.verify_protocol()
    from mh1_multihorizon.train import train_one

    for arm in C.ARMS:   # S42 + M42 各 2 批
        train_one(arm, 42, device="cuda:0", smoke=True)
    s42 = json.loads((C.DATA_DIR / "smoke_S42.json").read_text(encoding="utf-8"))
    m42 = json.loads((C.DATA_DIR / "smoke_M42.json").read_text(encoding="utf-8"))
    assert s42["init_hash"] == m42["init_hash"], "smoke S/M 初始化不一致"
    est = max(s42["extrapolated_six_runs_hours"], m42["extrapolated_six_runs_hours"])
    if est > C.GPU_BUDGET_HOURS:
        raise RuntimeError(
            f"smoke 外推六次训练 {est:.2f}h 超出 GPU 预算 {C.GPU_BUDGET_HOURS}h"
            "——按计划停止并报告实测瓶颈，不缩样本/换头/改超参")
    _mark_stage("smoke", {
        "protocol_sha256": sha,
        "extrapolated_hours": est,
        "peak_gb": max(s42["peak_gb"], m42["peak_gb"]),
        "init_hash_match": True})
    logger.info(f"smoke 通过：六次训练外推 {est:.2f}h ≤ 预算 "
                f"{C.GPU_BUDGET_HOURS}h；峰值显存 "
                f"{max(s42['peak_gb'], m42['peak_gb']):.2f}GB")


def cmd_train() -> None:
    _require_stage("preflight", "smoke")
    from mh1_multihorizon.train import run_all

    results = run_all(device="cuda:0")
    bud = json.loads((C.DATA_DIR / "gpu_budget.json").read_text(encoding="utf-8"))
    _mark_stage("train", {
        "runs": {k: {"best_epoch": v.get("best_epoch"),
                     "best_main": v.get("best_main")} for k, v in results.items()},
        "compute_hours": round(bud["compute_seconds"] / 3600, 3)})
    logger.info(f"六次训练完成：GPU 计算 {bud['compute_seconds'] / 3600:.2f}h")


def cmd_evaluate() -> None:
    _require_stage("preflight", "smoke", "train")
    protocol, sha = C.verify_protocol()
    protocol["_sha256"] = sha
    from mh1_multihorizon.evaluate import (
        build_windows, compute_daily_paired_ic, engine_attachment, judge,
        load_models, reference_ic, score_and_write_signals, timing_benchmark,
    )
    from g5_head.backbone_g1 import load_g1_backbone
    from kronos_qlib import QlibProvider

    models = load_models(protocol)
    backbone = load_g1_backbone("cuda:0")
    prov = QlibProvider(C.POOL, C.W3_START, C.FORWARD_CUTOFF)
    windows = build_windows(prov)
    frames = score_and_write_signals(models, backbone, windows, "cuda:0", sha)

    paired = compute_daily_paired_ic(frames, windows)
    calendar = prov.trading_days(end=C.FORWARD_CUTOFF)
    verdict = judge(paired, calendar)
    verdict["criteria"]["4_gates_protocol_tests_training"] = True  # 由门禁保证
    verdict["reference"] = reference_ic(windows)

    logger.info("==== MH1 主表（一次开封） ====")
    logger.info(json.dumps(verdict["windows"], ensure_ascii=False, indent=2,
                           default=str))
    logger.info(f"判据： {verdict['criteria']}")
    verdict["criteria"]["pass_all"] = all(
        v for k, v in verdict["criteria"].items() if k != "pass_all")
    paired_out = {
        w: {
            kind: {
                "n_common_days": len(vals),
                "mean": float(np.mean(
                    [v for per_seed in vals.values() for v in per_seed.values()]))
                if vals else None,
            } for kind in ("delta", "m_main", "s_main")
            for vals in [pw[kind]]
        } for w, pw in paired["per_window"].items()
    }
    (C.DATA_DIR / "judge_summary.json").write_text(
        json.dumps({"protocol_sha256": sha, "judge": verdict,
                    "paired_digest": paired_out},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    timing = timing_benchmark(models, backbone, prov, "cuda:0",
                              day=windows["W4"][len(windows["W4"]) // 2])
    (C.DATA_DIR / "timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info(f"计时：{json.dumps({k: v for k, v in timing.items() if k != 'g1_generative_ref'}, ensure_ascii=False)}")

    engine = engine_attachment(frames, windows)
    (C.DATA_DIR / "engine_v2_attachment.json").write_text(
        json.dumps(engine, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info("引擎 v2 附表落盘 engine_v2_attachment.json（描述性，不进判据）")
    _mark_stage("evaluate", {"protocol_sha256": sha,
                             "pass_all": verdict["criteria"]["pass_all"]})


def main() -> int:
    parser = argparse.ArgumentParser(description="MH1 多期限监督实验统一 CLI")
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "smoke", "train", "evaluate"])
    args = parser.parse_args()
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        cmd_preflight()
    elif args.stage == "smoke":
        cmd_smoke()
    elif args.stage == "train":
        cmd_train()
    else:
        cmd_evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
