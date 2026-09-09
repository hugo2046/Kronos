"""G10H-pilot 统一 CLI：preflight → smoke → train → evaluate（计划 §4/§6）。

阶段门禁：smoke/train/evaluate 均校验协议哈希；evaluate 要求两臂完成 +
FULL 复现门禁通过。输出全部落 ``data/<run_id>/``，逐阶段状态落
``stage_state.json``；GPU 实算预算 12h（训练+推理合计）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from g10_head_pilot import config as C
from g10_head_pilot.config import RUN_DIR, WINDOW_BOUNDS

STATE = RUN_DIR / "stage_state.json"
BUDGET = RUN_DIR / "gpu_budget.json"


def _mark(stage: str, payload: dict) -> None:
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    st[stage] = {"at": datetime.now().isoformat(timespec="seconds"), **payload}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2, default=str),
                     encoding="utf-8")


def _require(*stages: str) -> None:
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    missing = [s for s in stages if s not in st]
    if missing:
        raise RuntimeError(f"阶段未完成：{missing}（preflight→smoke→train→evaluate）")


def _budget_add(seconds: float) -> None:
    b = json.loads(BUDGET.read_text(encoding="utf-8")) if BUDGET.is_file() \
        else {"compute_seconds": 0.0}
    b["compute_seconds"] += seconds
    BUDGET.write_text(json.dumps(b, indent=2), encoding="utf-8")
    if b["compute_seconds"] > C.GPU_BUDGET_HOURS * 3600:
        raise RuntimeError(f"GPU 预算超限 {b['compute_seconds'] / 3600:.2f}h > "
                           f"{C.GPU_BUDGET_HOURS}h（停止并报告，不缩协议）")


# ============================================================
# preflight
# ============================================================


def cmd_preflight() -> None:
    if C.PROTOCOL_PATH.is_file():
        raise RuntimeError("协议已存在（不可覆盖）：如需重跑请新 run_id")
    import torch

    env = {
        "interpreter": sys.executable, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True, cwd=C.REPO_ROOT).stdout.strip(),
    }
    # 资产：G1 tokenizer / ashares 语料 / 官方起点（本地缓存） / FULL G1 参照
    assert (C.G1_TOKENIZER / "model.safetensors").is_file(), "G1 tokenizer 缺失"
    for split in ("train_data.pkl", "val_data.pkl"):
        assert (C.ASHARES_DATA / split).is_file(), f"语料缺失：{split}"
    for w, p in C.G1_FULL_SIGNALS.items():
        assert p.is_file(), f"FULL G1 参照缺失：{w}"
    from huggingface_hub import snapshot_download

    base_dir = snapshot_download(C.OFFICIAL_PREDICTOR)   # 本地缓存命中，不下载
    from model.kronos import Kronos, KronosTokenizer

    base = Kronos.from_pretrained(C.OFFICIAL_PREDICTOR)
    tok = KronosTokenizer.from_pretrained(str(C.G1_TOKENIZER))
    trainable_probe = sorted(n for n, _ in base.named_parameters())
    a1_names = sorted(C.A1_ALLOWED & set(trainable_probe))

    from kronos_qlib import QlibProvider

    cal_info = {}
    for wname, (start, end) in WINDOW_BOUNDS.items():
        p = QlibProvider("csi300", start, end)
        days = p.trading_days(start, end)
        assert days[-1] <= pd.Timestamp(C.FORWARD_CUTOFF)
        cal_info[wname] = {"n_days": len(days),
                           "first": str(days[0].date()),
                           "last": str(days[-1].date())}

    protocol = {
        "experiment": "G10H-pilot", "frozen_at": datetime.now().isoformat(),
        "env": env,
        "design": {
            "arms": list(C.ARMS), "seed": C.SEED, "epochs": C.EPOCHS,
            "batch": C.BATCH, "n_train_iter": C.N_TRAIN_ITER,
            "n_val_iter": C.N_VAL_ITER, "lr": C.LR,
            "adam": {"betas": list(C.ADAM_BETAS), "wd": C.ADAM_WD},
            "onecycle": {"pct_start": C.PCT_START, "div_factor": C.DIV_FACTOR},
            "grad_clip": C.GRAD_CLIP, "fp32": True,
            "a1_allowed": sorted(C.A1_ALLOWED),
            "a1_param_count": sum(
                dict(base.named_parameters())[n].numel() for n in a1_names),
            "a0_param_count": sum(p.numel() for p in base.parameters()),
            "predictor_start": C.OFFICIAL_PREDICTOR,
            "tokenizer_sha256": C.sha256_file(C.G1_TOKENIZER / "model.safetensors"),
            "official_base_dir": str(base_dir),
            "d_model": base.d_model,
        },
        "inference": C.INFERENCE,
        "windows": {w: {"bounds": list(b), **cal_info[w]}
                    for w, b in WINDOW_BOUNDS.items()},
        "forward_cutoff": C.FORWARD_CUTOFF,
        "g1_full_signals_sha256": {w: C.sha256_file(p)
                                   for w, p in C.G1_FULL_SIGNALS.items()},
        "corpus": {"ashares": str(C.ASHARES_DATA),
                   "sha256": {s: C.sha256_file(C.ASHARES_DATA / s)
                              for s in ("train_data.pkl", "val_data.pkl")},
                   "note": "原 G1 同源全 A 历史语料（非逐日 csi300 PIT，披露）"},
        "guard": list(C.REGISTRY_GUARD),
        "budget_gpu_hours": C.GPU_BUDGET_HOURS,
    }
    sha = C.write_protocol(protocol)
    _mark("preflight", {"protocol_sha256": sha})
    logger.info(f"协议冻结 {C.PROTOCOL_PATH}（SHA {sha[:16]}…）；"
                f"A1 参数 {protocol['design']['a1_param_count']:,} / "
                f"A0 {protocol['design']['a0_param_count']:,}；"
                f"窗口 {cal_info}")


# ============================================================
# smoke
# ============================================================


def cmd_smoke() -> None:
    _require("preflight")
    C.load_protocol()
    from g10_head_pilot.train import train_arm

    t0 = time.time()
    digests = {}
    inits = {}
    for arm in C.ARMS:
        out = train_arm(arm, smoke=True)
        digests[arm] = out["history"][0]["batch_digest"]
        inits[arm] = out["init_hash"]
    elapsed = time.time() - t0
    assert inits["A0"] == inits["A1"], "两臂起点权重哈希不等（配对失败）"
    assert digests["A0"] == digests["A1"], "smoke 批次摘要不一致（配对失败）"
    per_epoch = elapsed / (2 * 1)         # 每臂 smoke=1 epoch(2步)+验证2批
    full_train_est = per_epoch * C.EPOCHS * len(C.ARMS)
    # 推理估时：smoke 阶段不做整日推理（成本高），用计划公式外推 + 报告上限
    _budget_add(elapsed)
    _mark("smoke", {"seconds": elapsed, "train_est_hours": full_train_est / 3600,
                    "init_match": True, "digest_match": True})
    logger.info(f"smoke 通过：两臂起点/批次摘要一致；训练外推 "
                f"{full_train_est / 3600:.2f}h（不含推理）")


# ============================================================
# train
# ============================================================


def cmd_train() -> None:
    _require("preflight", "smoke")
    _, proto_sha = C.load_protocol()
    from g10_head_pilot.train import train_arm

    for arm in C.ARMS:
        t0 = time.time()
        out = train_arm(arm)
        _budget_add(time.time() - t0)
        if out.get("skipped"):
            continue
        assert out["init_hash"], "缺起点哈希"
    # 配对审计：两臂起点哈希 + 逐 epoch 批次摘要一致
    h = {a: json.loads((RUN_DIR / a / "history.json").read_text(encoding="utf-8"))
         for a in C.ARMS}
    assert h["A0"]["init_hash"] == h["A1"]["init_hash"], "起点不一致"
    for e0, e1 in zip(h["A0"]["history"], h["A1"]["history"]):
        assert e0["batch_digest"] == e1["batch_digest"], (
            f"epoch {e0['epoch']} 批次摘要不一致（配对失败）")
    logger.info("两臂配对审计通过：起点哈希 + 15 epoch 批次摘要全一致")
    _mark("train", {"protocol_sha256": proto_sha,
                    "best": {a: {"epoch": h[a]["best_epoch"],
                                 "ce": h[a]["best_ce"]} for a in C.ARMS}})


# ============================================================
# evaluate
# ============================================================


def cmd_evaluate() -> None:
    _require("preflight", "smoke", "train")
    _, proto_sha = C.load_protocol()
    import torch

    from g10_head_pilot import evaluate as ev
    from g10_head_pilot.train import set_seed_100, wait_out_guard

    # FULL 复现门禁（先于任何新收益计算）
    gate = ev.verify_full_gate()
    h = {a: json.loads((RUN_DIR / a / "history.json").read_text(encoding="utf-8"))
         for a in C.ARMS}

    from kronos_qlib import QlibProvider
    from model.kronos import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(C.G1_TOKENIZER))
    tokenizer.eval().to("cuda:0")

    states: dict[tuple[str, str], int] = {}
    for arm in C.ARMS:
        best = int(h[arm]["best_epoch"])
        states[(arm, "e15")] = 15
        states[(arm, "bestCE")] = best

    # 逐状态加载权重并推理两窗（同权重同窗只算一次；bestCE==e15 记别名）
    sig_dir = RUN_DIR / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    ic_stats: dict = {}
    trade_stats: dict = {}
    daily_diff_e15: dict[str, list[float]] = {}
    daily_pos: dict[str, list[int]] = {}
    t_all = time.time()
    from g10_head_pilot.runtime_state import (
        cache_reusable, capture_rng, load_versioned_checkpoint,
        restore_rng, signal_meta_for, write_signal_meta,
    )

    tok_sha = C.sha256_file(C.G1_TOKENIZER / "model.safetensors")
    rng_chain_path = sig_dir / "rng_chain.pt"
    for arm in C.ARMS:
        ck = load_versioned_checkpoint(RUN_DIR / arm / "final.pt")
        predictor = Kronos.from_pretrained(C.OFFICIAL_PREDICTOR).to("cuda:0")
        predictor.load_state_dict(ck["predictor"])
        predictor.eval()
        pred_wrap = KronosPredictor(model=predictor, tokenizer=tokenizer,
                                    device="cuda:0")
        for state, epoch in (("e15", 15), ("bestCE", states[(arm, "bestCE")])):
            # 实际载入权重的身份（内容哈希）——cache 元数据由此构造，不手写
            ck_file = (RUN_DIR / arm / "final.pt" if state == "e15"
                       or epoch == 15 else RUN_DIR / arm / "best.pt")
            if state == "bestCE" and epoch != 15:
                predictor.load_state_dict(
                    load_versioned_checkpoint(RUN_DIR / arm / "best.pt")
                    ["predictor"])
                predictor.eval()
            ck_sha = C.sha256_file(ck_file)

            def _identity(w):
                return {"runtime_schema": "g10h-runtime-v2",
                        "checkpoint_sha256": ck_sha,
                        "checkpoint_epoch": int(epoch),
                        "tokenizer_sha256": tok_sha,
                        "protocol_sha256": proto_sha,
                        "inference": C.INFERENCE, "window": w}

            alias = ""
            if state == "bestCE" and epoch == 15:
                alias = "（别名 e15，同权重不重复推理）"
            tag = f"{arm}_{state}"
            if alias:
                # bestCE==e15：同权重只推理一次，落别名副本（含元数据）
                for wname in WINDOW_BOUNDS:
                    src = sig_dir / f"{wname}_{arm}_e15.parquet"
                    dst = sig_dir / f"{wname}_{tag}.parquet"
                    dst.write_bytes(src.read_bytes())
                    mfile = Path(str(src) + ".meta.json")
                    if mfile.is_file():
                        Path(str(dst) + ".meta.json").write_text(
                            mfile.read_text(encoding="utf-8"), encoding="utf-8")
                logger.info(f"[{tag}] 别名 e15，信号复制 {alias}")
            elif all(cache_reusable(sig_dir / f"{w}_{tag}.parquet",
                                    _identity(w)) for w in WINDOW_BOUNDS):
                logger.info(f"[{tag}] cache 身份全匹配（checkpoint/协议/推理"
                            "参数/输出 SHA），跳过推理")
            else:
                set_seed_100()      # 首次运行语义：每 checkpoint 一次 seed100
                provider = QlibProvider("csi300", *WINDOW_BOUNDS["W3"])
                chain = (torch.load(rng_chain_path, weights_only=True)
                         if rng_chain_path.is_file() else {})
                chain_key = tag
                if chain_key in chain:
                    restore_rng(chain[chain_key])   # 恢复上次结束流再续推
                for wname in WINDOW_BOUNDS:
                    spath = sig_dir / f"{wname}_{tag}.parquet"
                    if cache_reusable(spath, _identity(wname)):
                        # W3 命中 cache：必须恢复该窗结束 RNG 链才许推 W4
                        if f"{chain_key}_{wname}_end" in chain:
                            restore_rng(chain[f"{chain_key}_{wname}_end"])
                        continue
                    wait_out_guard([predictor, tokenizer], f"[{tag}] {wname}")
                    t0 = time.time()
                    wide = ev.score_full_window(pred_wrap, provider, wname)
                    wide.to_parquet(spath)
                    write_signal_meta(
                        spath, signal_meta_for(ck_sha, epoch, tok_sha,
                                               proto_sha, C.INFERENCE,
                                               wname, spath))
                    _budget_add(time.time() - t0)
                    chain[f"{chain_key}_{wname}_end"] = capture_rng()
                    torch.save(chain, rng_chain_path)
                    logger.info(f"[{tag}] {wname} 信号落盘 "
                                f"({wide.shape[1]} 日，含身份元数据)")
            # IC + 交易（CPU）；臂信号 parquet 为"股票×日期"，引擎前转置
            provider = QlibProvider("csi300", WINDOW_BOUNDS["W3"][0], C.FORWARD_CUTOFF)
            for wname in WINDOW_BOUNDS:
                wide = pd.read_parquet(sig_dir / f"{wname}_{tag}.parquet")
                labels = ev.window_labels(provider, wname)
                series = ev.daily_ic_series(wide, labels)
                ic_stats[(arm, state, wname)] = float(np.mean(series)) \
                    if series else float("nan")
                t = time.time()
                trade_stats[(arm, state, wname)] = ev.trade_arm_window(
                    wide.T, wname, tag=f"{arm}_{state}")
                _budget_add(time.time() - t)   # CPU 计时（预算含合计）
        del predictor, pred_wrap
        torch.cuda.empty_cache()

    # P 判据的逐日差：A1_e15 − A0_e15（共同有效日期，code 对齐已在 IC 内）
    provider = QlibProvider("csi300", WINDOW_BOUNDS["W3"][0], C.FORWARD_CUTOFF)
    for wname in WINDOW_BOUNDS:
        labels = ev.window_labels(provider, wname)
        w1 = pd.read_parquet(sig_dir / f"{wname}_A1_e15.parquet")
        w0 = pd.read_parquet(sig_dir / f"{wname}_A0_e15.parquet")
        cal = provider.trading_days(*WINDOW_BOUNDS[wname])
        pos_map = {str(d.date()): i for i, d in enumerate(cal)}
        diffs, pos = [], []
        for ds in sorted(set(w1.columns) & set(w0.columns) & set(labels)):
            ic1, _ = ev.rank_ic(dict(w1[ds].dropna().items()), labels[ds])
            ic0, _ = ev.rank_ic(dict(w0[ds].dropna().items()), labels[ds])
            if ic1 is not None and ic0 is not None:
                diffs.append(ic1 - ic0)
                pos.append(pos_map[ds])
        daily_diff_e15[wname] = diffs
        daily_pos[wname] = pos
    from mh1_multihorizon.evaluate import segmented_hac

    comb = segmented_hac([(w, daily_diff_e15[w], daily_pos[w])
                          for w in WINDOW_BOUNDS], lag=C.NW_LAG)
    p_pre = {
        "W3": float(np.mean(daily_diff_e15["W3"])) if daily_diff_e15["W3"] else float("nan"),
        "W4": float(np.mean(daily_diff_e15["W4"])) if daily_diff_e15["W4"] else float("nan"),
        "combined_t": comb["t"], "combined_judgable": comb["judgable"],
        "n_days": {w: len(daily_diff_e15[w]) for w in WINDOW_BOUNDS},
    }
    # G1 FULL 参照（IC + 交易）——注意 G1 parquet 为"日期索引/股票列"，
    # 需转置成与臂信号一致的"日期列"形态再算逐日 IC
    g1_ref = {}
    for wname in WINDOW_BOUNDS:
        raw = pd.read_parquet(C.G1_FULL_SIGNALS[wname])
        raw.index = pd.DatetimeIndex(raw.index)
        raw.columns = [str(c) for c in raw.columns]
        labels = ev.window_labels(provider, wname)
        series = ev.daily_ic_series(raw.T, labels)
        g1_ref[wname] = {
            "ic_mean": float(np.mean(series)) if series else float("nan"),
            "ic_nw_t": ev.nw_t(series),
            "trade": ev.trade_arm_window(raw, wname, tag="G1_full"),
        }

    stats = {"ic": {f"{a}|{s}|{w}": ic_stats[(a, s, w)]
                    for (a, s, w) in ic_stats},
             "trade": {f"{a}|{s}|{w}": trade_stats[(a, s, w)]
                       for (a, s, w) in trade_stats},
             "ic_diff_e15": p_pre, "g1_full_ref": g1_ref, "gate": gate}
    crit = ev.judge({"ic": {(a, s, w): ic_stats[(a, s, w)]
                            for (a, s, w) in ic_stats},
                     "trade": {(a, s, w): trade_stats[(a, s, w)]
                               for (a, s, w) in trade_stats},
                     "ic_diff_e15": p_pre})
    out = {"protocol_sha256": proto_sha,
           "created_at": datetime.now().isoformat(timespec="seconds"),
           "states": {f"{a}|{s}": states[(a, s)] for (a, s) in states},
           "val_ce": {a: {"best_epoch": h[a]["best_epoch"],
                          "best_ce": h[a]["best_ce"],
                          "epochs_run": len(h[a]["history"])} for a in C.ARMS},
           "criteria": crit, **stats,
           "gpu_seconds": json.loads(BUDGET.read_text(encoding="utf-8"))
           ["compute_seconds"] if BUDGET.is_file() else 0}
    (RUN_DIR / "evaluation_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    _mark("evaluate", {"protocol_sha256": proto_sha,
                       "criteria": {k: v for k, v in crit.items()
                                    if isinstance(v, bool)}})
    logger.info("==== G10H-pilot 判据（一次开封） ====")
    logger.info(json.dumps({k: v for k, v in crit.items()}, ensure_ascii=False,
                           default=str, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="G10H-pilot CLI")
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "smoke", "train", "evaluate"])
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(str(RUN_DIR / "logs" / f"{args.stage}.log"), enqueue=False)
    t0 = time.time()
    if args.stage == "preflight":
        cmd_preflight()
    elif args.stage == "smoke":
        cmd_smoke()
    elif args.stage == "train":
        cmd_train()
    else:
        cmd_evaluate()
    logger.info(f"[{args.stage}] 阶段完成，用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
