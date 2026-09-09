"""LoRA-mean 评价：FULL 逐日生成推理 + 固定 Qlib 口径回测（计划 §5/§6）。

复用 ``qlib_mean_comparison``（0b982e8 口径）的回测/曲线函数与
``g10_head_pilot.evaluate`` 的 FULL 推理函数；先复现 0b982e8 的 G1 report
（≤1e-10）作为口径门禁，再评新臂。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts" / "experiments"
REF_DIR = ART / "mean_comparison_20260909"
OUT_DIR = ART / "lora_mean_20260909"
FORWARD_CUTOFF = "2026-07-24"
WINDOW_BOUNDS = {"W3": ("2025-07-01", "2025-12-31"),
                 "W4": ("2026-01-01", "2026-07-24")}
INFERENCE = {"lookback": 90, "pred_len": 10, "sample_count": 20, "T": 1.0,
             "top_p": 0.9, "top_k": 0, "clip": 5, "max_context": 512}
G1_SIGNALS = {
    "W3": Path("/home/user/workspace/Kronos/g5_head/data/"
               "daily_signals_2025h2_G1_mean.parquet"),
    "W4": Path("/home/user/workspace/Kronos/finetune_suite/data/g1/"
               "daily_signals_backtest_G1_mean.parquet"),
}
G1_MODEL = {
    "tokenizer": Path("/home/user/workspace/Kronos/finetune_suite/outputs/"
                      "models/finetune_tokenizer_g1/checkpoints/best_model"),
    "predictor": Path("/home/user/workspace/Kronos/finetune_suite/outputs/"
                      "models/finetune_predictor_g1/checkpoints/best_model"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def reproduce_gate(backtest_fn, curves_fn) -> dict:
    """复现 0b982e8 的 G1 report（逐日 return/cost/bench ≤1e-10）。"""
    errs = {}
    for w, (start, end) in WINDOW_BOUNDS.items():
        raw = pd.read_parquet(G1_SIGNALS[w])
        raw.index = pd.DatetimeIndex(raw.index)
        wide = raw.T                                   # 股票×日期 统一形态
        s = wide.stack()
        s.index.names = ["instrument", "datetime"]     # 层序=（股票，日期）
        sig = s.reorder_levels(["instrument", "datetime"]).sort_index()
        report, _ = backtest_fn(sig, start, end)
        ref = pd.read_parquet(REF_DIR / f"report_{w}_G1_mean.parquet")
        for col in ("return", "cost", "bench"):
            e = float((report[col] - ref[col].reindex(report.index))
                      .abs().max())
            assert e <= 1e-10, f"{w} {col} 复现失败 {e:.2e}（查口径后重试）"
            errs[f"{w}|{col}"] = e
        assert report.index.equals(ref.index), f"{w} 报告日期不一致"
    logger.info(f"[gate] 0b982e8 G1 report 复现 max|Δ| "
                f"{max(errs.values()):.1e} OK")
    return errs


def score_arm(predictor, tokenizer, wname: str, provider, device: str,
              seed: int, out_path: Path) -> Path:
    """单臂单窗 FULL 生成推理（mean 变体；加载后重新播种推理流）。"""
    import torch

    from g10_head_pilot.evaluate import score_full_window

    torch.manual_seed(seed)
    from model import KronosPredictor

    wrap = KronosPredictor(model=predictor, tokenizer=tokenizer, device=device)
    wide = score_full_window(wrap, provider, wname)
    wide.to_parquet(out_path)
    del wrap
    torch.cuda.empty_cache()
    return out_path


def backtest_arm(signal_path: Path, wname: str, backtest_fn, curves_fn,
                 tag: str, out_dir: Path) -> dict:
    """信号 parquet（股票×日期）→ Qlib 回测 → 曲线/指标落盘。"""
    wide = pd.read_parquet(signal_path)
    if isinstance(wide.index, pd.DatetimeIndex):     # 日期×股票 → 股票×日期
        wide = wide.T
    if not isinstance(wide.columns, pd.DatetimeIndex):
        wide.columns = pd.DatetimeIndex(pd.to_datetime(wide.columns))
    s = wide.stack()
    s.index.names = ["instrument", "datetime"]
    sig = s.sort_index()
    report, meta = backtest_fn(sig, *WINDOW_BOUNDS[wname])
    c = curves_fn(report)
    report.to_parquet(out_dir / f"report_{wname}_{tag}.parquet")
    pd.DataFrame({"net_daily": c["net_daily"], "curve": c["curve"],
                  "excess_curve": c["excess_curve"],
                  "compound_nav": c["compound_nav"]}
                 ).to_parquet(out_dir / f"daily_{wname}_{tag}.parquet")
    return {"tag": tag, "window": wname,
            "curve_end": float(c["curve"].iloc[-1]),
            "compound_cum": c["compound_cum"],
            "max_drawdown_compound": c["max_drawdown_compound"],
            "cost_sum": c["cost_sum"],
            "turnover_daily_mean": c["turnover_daily_mean"],
            "n_days": len(c["curve"]), "meta": meta}


def paired_verdict(results: dict, seeds: list[int]) -> dict:
    """D_original / D_paired（百分点差）与阶段判定。"""
    v: dict = {}
    for s in seeds:
        for w in WINDOW_BOUNDS:
            c_lora = results[(f"LoRA_{s}", w)]["curve_end"]
            c_orig = results[("G1_original_mean", w)]["curve_end"]
            c_pair = results[(f"G1_paired_{s}", w)]["curve_end"]
            v[f"{s}|{w}"] = {
                "D_original_pp": round((c_lora - c_orig) * 100, 2),
                "D_paired_pp": round((c_lora - c_pair) * 100, 2),
                "win_original": c_lora > c_orig, "win_paired": c_lora > c_pair,
            }
    return v


def pilot_gate(v: dict, seed: int) -> bool:
    return all(v[f"{seed}|{w}"]["win_original"]
               and v[f"{seed}|{w}"]["win_paired"] for w in WINDOW_BOUNDS)


def confirm_gate(v: dict, seeds: list[int]) -> dict:
    """重复改善：≥2 seed 两窗双胜 且 两窗各 D_original/D_paired 中位数>0。"""
    wins = {s: all(v[f"{s}|{w}"]["win_original"]
                   and v[f"{s}|{w}"]["win_paired"] for w in WINDOW_BOUNDS)
            for s in seeds}
    med = {}
    for w in WINDOW_BOUNDS:
        for kind in ("D_original_pp", "D_paired_pp"):
            med[f"{w}|{kind}"] = float(np.median(
                [v[f"{s}|{w}"][kind] for s in seeds]))
    repeated = (sum(wins.values()) >= 2
                and all(x > 0 for x in med.values()))
    return {"seed_wins": wins, "medians": med, "repeated_improvement": repeated}
