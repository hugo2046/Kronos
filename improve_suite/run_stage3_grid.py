"""阶段 3：L/H/T 网格（计划 §5）。

三个冻结配置（禁止追加或插值）：

    - C1：L=8  / H=5 / T=1.0（用户 8/5 实盘经验先验）
    - C2：L=30 / H=5 / T=1.0（论文合成评估器日频设定，C1↔canonical 中点）
    - C3：L=90 / H=10 / T=0.6（论文点预测低温建议）

每配置：N=20、seed=42、csi300、mean 聚合、两窗、逐路径落盘。组合口径不变
（k=50/n=5/min_hold=5/15bp）。H=5 窗口边界仍与 canonical 同（oos 末日 2026-07-24）。

判据（冻结）：某配置胜出 ⟺ paper AER(等权) ≥ canonical mean +4.47% − 2pp **且**
oos1 AER(等权) > 0。全败则记录"推理期窗口/温度不解释样本外塌方"。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_stage3_grid --infer C1
    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_stage3_grid --engine
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import DATA_DIR as BL_DATA_DIR
from baseline_suite.common import BaselineConfig, ensure_dirs as bl_ensure_dirs
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from improve_suite.common import DATA_DIR, ImproveConfig
from improve_suite.path_store import read_paths, write_paths
from improve_suite.run_canonical_paths import _build_provider, _load_predictor, _stack_day_long

# 跑前冻结的三配置
GRID_CONFIGS: dict[str, dict] = {
    "C1": {"lookback": 8, "predict_len": 5, "T": 1.0},
    "C2": {"lookback": 30, "predict_len": 5, "T": 1.0},
    "C3": {"lookback": 90, "predict_len": 10, "T": 0.6},
}
CANONICAL_PAPER_EW = 0.0447  # §0：canonical mean paper AER(等权)


def infer_config(name: str, window: str, checkpoint_every: int = 10) -> Path:
    """单配置单窗逐路径推理 + mean 信号落盘。"""
    import torch
    from improve_suite.path_inference import predict_batch_paths
    from kronos_qlib import QlibProvider, build_inference_windows

    spec = GRID_CONFIGS[name]
    cfg = ImproveConfig.load(window=window, **spec)
    label = cfg.canonical_label()
    out_path = DATA_DIR / f"paths_{window}_{label}.parquet"
    sig_path = DATA_DIR / f"daily_signals_{window}_{label}_mean.parquet"

    cal = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
    rebalances = cal.trading_days(cfg.backtest_start, cfg.backtest_end)

    done_dates: set[pd.Timestamp] = set()
    existing_blocks: list[pd.DataFrame] = []
    sig_rows: dict[pd.Timestamp, dict] = {}
    if out_path.exists():
        prev = read_paths(out_path)
        done_dates = set(pd.to_datetime(prev["date"].unique()))
        existing_blocks = [prev]
        logger.info(f"[{name}/{window}] 断点续跑：已有 {len(done_dates)} 日")
    pending = [d for d in rebalances if d not in done_dates]
    logger.info(f"[{name}/{window}] {label}：{len(pending)} 日待跑（N={cfg.sample_count}）")

    provider = _build_provider(cfg)
    predictor = _load_predictor(cfg)

    # 续跑时只在起点读一次既有 mean 信号（避免 checkpoint 重读导致日期重复）
    prev_sig_init = None
    if sig_path.exists() and existing_blocks:
        prev_sig_init = pd.read_parquet(sig_path)
        prev_sig_init = prev_sig_init[~prev_sig_init.index.duplicated(keep="last")]

    new_blocks: list[pd.DataFrame] = []
    for i, d in enumerate(pending):
        ds = d.strftime("%Y-%m-%d")
        df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows(
            provider, ds, lookback=cfg.lookback, predict_len=cfg.predict_len, pool=cfg.pool,
        )
        if len(df_list) == 0:
            continue
        last_closes = [float(df["close"].iloc[-1]) for df in df_list]
        torch.manual_seed(cfg.seed)
        preds, paths_close = predict_batch_paths(
            predictor, df_list, x_ts_list, y_ts_list,
            pred_len=cfg.predict_len, T=cfg.T, top_k=cfg.sample_top_k,
            top_p=cfg.top_p, sample_count=cfg.sample_count,
        )
        block = _stack_day_long(d, codes, paths_close)
        new_blocks.append(block)
        sig_rows[d] = {
            c: float(np.mean(preds[j][cfg.signal_field].values) / last_closes[j] - 1.0)
            for j, c in enumerate(codes)
        }

        if (i + 1) % checkpoint_every == 0 or i == len(pending) - 1:
            full = pd.concat(existing_blocks + new_blocks, ignore_index=True)
            write_paths(full, out_path)
            sig_wide = pd.DataFrame.from_dict(sig_rows, orient="index").sort_index()
            # 续跑时合并已有 mean 信号（prev_sig_init 只在起点读一次，防重复）
            if prev_sig_init is not None:
                sig_wide = pd.concat([prev_sig_init, sig_wide])
            sig_wide = sig_wide[~sig_wide.index.duplicated(keep="last")].sort_index()
            sig_wide.to_parquet(sig_path)
            n_done = len(done_dates) + i + 1
            logger.info(f"[{name}/{window}] checkpoint [{n_done}/{len(rebalances)}] {ds}")
    logger.info(f"[{name}/{window}] 落盘完成：{out_path.name} + {sig_path.name}")
    return sig_path


def backtest_config(name: str) -> dict:
    """单配置两窗过引擎（双基准）。"""
    spec = GRID_CONFIGS[name]
    out = {}
    for window in ("paper", "oos"):
        cfg = ImproveConfig.load(window=window, **spec)
        label = cfg.canonical_label()
        sig_path = DATA_DIR / f"daily_signals_{window}_{label}_mean.parquet"
        if not sig_path.exists():
            raise FileNotFoundError(f"{name}/{window} mean 信号缺失：{sig_path}（先跑 --infer）")
        sig = pd.read_parquet(sig_path)

        bc = BaselineConfig.load(window=window)
        from kronos_qlib import QlibProvider

        provider = QlibProvider(bc.pool, bc.backtest_start, bc.backtest_end)
        rebalances = provider.trading_days(bc.backtest_start, bc.backtest_end)
        sig = sig.reindex(rebalances)
        all_cols = sorted(sig.columns)
        px, trd = build_px_tradeable(provider, bc, rebalances, all_cols)
        bench_idx, bench_ew, _ = build_dual_benchmarks(provider, bc, px, trd)
        pi, pe, dr, _, _ = run_group(sig, px, trd, bench_idx, bench_ew, cfg=bc, name=f"{name}_{window}")
        out[window] = {"perf_idx": pi, "perf_ew": pe, "daily_ret": dr}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 3 L/H/T 网格")
    parser.add_argument("--infer", choices=list(GRID_CONFIGS), help="只跑某配置推理")
    parser.add_argument("--engine", action="store_true", help="跑引擎 + 判读（推理已全完成）")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.infer:
        for window in ("paper", "oos"):
            infer_config(args.infer, window)
        return

    if args.engine:
        bl_ensure_dirs()
        all_res = {}
        for name in GRID_CONFIGS:
            logger.info(f"==== {name} {GRID_CONFIGS[name]} ====")
            all_res[name] = backtest_config(name)

        # 判据（冻结）：paper AER(ew) ≥ 4.47%-2pp 且 oos AER(ew) > 0
        threshold = CANONICAL_PAPER_EW - 0.02
        verdicts = {}
        any_winner = False
        for name, res in all_res.items():
            p_ew = res["paper"]["perf_ew"].aer
            o_ew = res["oos"]["perf_ew"].aer
            wins = (p_ew >= threshold) and (o_ew > 0)
            any_winner = any_winner or wins
            verdicts[name] = {
                "paper_ew": p_ew, "oos_ew": o_ew,
                "paper_threshold": threshold, "wins": bool(wins),
            }
        summary = (
            "胜出配置待前向确认" if any_winner
            else "推理期窗口/温度不解释样本外塌方（全败，如实记录）"
        )
        logger.info(f"==== 阶段 3 判读：{summary} ====")
        for n, v in verdicts.items():
            logger.info(f"  {n}: paper {v['paper_ew']:+.2%} / oos {v['oos_ew']:+.2%} → wins={v['wins']}")

        out = {
            "stage": 3, "configs": GRID_CONFIGS,
            "canonical_paper_ew": CANONICAL_PAPER_EW,
            "paper_threshold": threshold,
            "results": {
                name: {w: {"perf_idx": r[w]["perf_idx"].to_dict(), "perf_ew": r[w]["perf_ew"].to_dict()}
                       for w in ("paper", "oos")}
                for name, r in all_res.items()
            },
            "verdicts": verdicts, "summary": summary,
        }
        out_path = DATA_DIR / "stage3_grid_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"结果落盘 {out_path}")


if __name__ == "__main__":
    main()
