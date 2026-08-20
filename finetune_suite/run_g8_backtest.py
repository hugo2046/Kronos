"""G8 统一回测全表落盘（计划 §1/§3.3，20260820 G8+E1 计划）——不判读。

三种子（100/101/102）× 四变体（last/mean/max/min）backtest 窗引擎全表 +
G1 冻结对照片 + predictor 逐 epoch val 表与 best epoch（机制证据，计划 §3.3
"best 是否仍=1 是数据饥饿假说的追加机制证据，如实记录"）。

判据纪律：本脚本**只落盘不判读**——G8-1~G8-4 统一在 3.4 一次开封。

口径（与 run_g2_judge 逐字一致）：
    - 引擎 canonical（top-k/drop-n/min_hold/15bp，paper_replication.engine 只读）；
    - 列宇宙 = G8 三种子 ∪ F0 ∪ M backtest 信号并集（= M 的 338 列）；
    - rebalances = 第 4 轮 M backtest 索引；双基准 = 000300 指数 / 同池等权。

落盘（finetune_suite/data/g8/）：
    g8_backtest_results.json（全表 + epoch 表 + G1 对照，无判据字段）

用法::

    /home/user/miniconda3/envs/quant/bin/python -m finetune_suite.run_g8_backtest
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from finetune_suite.run_g2_judge import parse_epoch_table

PKG_DIR = Path(__file__).resolve().parent
G8_DIR = PKG_DIR / "data" / "g8"
G2_JUDGE = PKG_DIR / "data" / "g2" / "g2_judge_results.json"
ROUND4_DATA = PKG_DIR / "data"

SEEDS = (100, 101, 102)


def arm_tag(seed: int) -> str:
    return f"G8S{seed}"


def load_signals() -> dict[str, pd.DataFrame]:
    """G8 三种子 × 四变体（backtest）+ 第 4 轮 F0 四变体/M（只读，列宇宙用）。"""
    signals: dict[str, pd.DataFrame] = {}
    for s in SEEDS:
        for v in VARIANTS:
            signals[f"s{s}_{v}"] = pd.read_parquet(
                G8_DIR / f"s{s}" / f"daily_signals_backtest_{arm_tag(s)}_{v}.parquet"
            )
    for v in VARIANTS:
        signals[f"F0_{v}"] = pd.read_parquet(ROUND4_DATA / f"daily_signals_backtest_F0_{v}.parquet")
    signals["M"] = pd.read_parquet(ROUND4_DATA / "daily_signals_backtest_M.parquet")
    return signals


def main() -> None:
    from kronos_qlib import QlibProvider

    cfg = replace(
        BaselineConfig.load(window="oos"),
        window="g8_backtest",
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
    )
    signals = load_signals()
    all_cols = sorted(set().union(*[set(df.columns) for df in signals.values()]))
    rebalances = pd.DatetimeIndex(signals["M"].index)

    provider = QlibProvider(cfg.pool, BACKTEST_START, BACKTEST_END)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    results: dict[str, dict] = {}
    for s in SEEDS:
        for v in VARIANTS:
            tag = f"s{s}_{v}"
            pi, pe, _, _, _ = run_group(
                signals[tag], px, trd, bench_idx, bench_ew, cfg=cfg, name=f"g8/{tag}"
            )
            results[tag] = {"perf_idx": pi.to_dict(), "perf_ew": pe.to_dict()}

    # —— 机制证据：三种子 predictor 逐 epoch val 表 + best epoch（如实记录）——
    epoch_tables: dict[str, dict] = {}
    for s in SEEDS:
        table, best = parse_epoch_table(G8_DIR / f"train_predictor_g8_s{s}_console.txt")
        epoch_tables[f"s{s}"] = {
            "best_epoch": best,
            "val_losses": table["val_loss"].tolist(),
        }

    # —— G1 冻结对照（G8-3 配对差的只读输入；不重算）——
    g2j = json.loads(G2_JUDGE.read_text(encoding="utf-8"))
    g1_refs = {
        "s100": g2j["full_table"]["s100@backtest"],
        "s101": g2j["full_table"]["s101@backtest"],
        "s102": g2j["full_table"]["s102@backtest"],
    }

    # —— 中位数汇总（机械聚合，无判据字段；判读留 3.4）——
    g8_mean_ew = {s: results[f"s{s}_mean"]["perf_ew"]["aer"] for s in SEEDS}
    g8_mean_idx = {s: results[f"s{s}_mean"]["perf_idx"]["aer"] for s in SEEDS}
    med = {
        "g8_mean_aer_ew_median": float(np.median(list(g8_mean_ew.values()))),
        "g8_mean_aer_idx_median": float(np.median(list(g8_mean_idx.values()))),
        "g1_mean_aer_ew_median": float(np.median(
            [g1_refs[f"s{s}"]["mean"]["aer_ew"] for s in SEEDS])),
        "median_to_median_ew_diff": None,  # 3.4 开封时计算并判读
    }
    med["median_to_median_ew_diff"] = (
        med["g8_mean_aer_ew_median"] - med["g1_mean_aer_ew_median"]
    )

    out = {
        "experiment": "g8_backtest",
        "date": "2026-08-20",
        "window": "backtest",
        "period": [BACKTEST_START, BACKTEST_END],
        "benchmarks": {
            "index": "000300.SH csi300 市值加权",
            "equal_weight": "同池等权",
            "beta_gap": beta_gap,
            "universe_cols": len(all_cols),
        },
        "engine": {"top_k": cfg.top_k, "drop_n": cfg.drop_n,
                   "min_hold": cfg.min_hold, "cost_bps": cfg.cost_bps},
        "groups": results,
        "epoch_tables": epoch_tables,
        "g1_frozen_refs": g1_refs,
        "medians": med,
        "note": "只落盘不判读；G8-1~G8-4 于 3.4 统一一次开封",
    }
    out_path = G8_DIR / "g8_backtest_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"G8 backtest 全表落盘 {out_path}")

    # —— 阶段输出：全表打印（落盘后）——
    print("=== G8 三种子 backtest 全表（四变体双基准，已落盘不判读）===")
    rows = []
    for s in SEEDS:
        for v in VARIANTS:
            r = results[f"s{s}_{v}"]
            rows.append({
                "seed": f"s{s}", "variant": v,
                "aer_ew": r["perf_ew"]["aer"], "aer_idx": r["perf_idx"]["aer"],
                "ir_ew": r["perf_ew"]["ir"], "turnover": r["perf_ew"]["daily_turnover"],
            })
    df = pd.DataFrame(rows)
    show = df.copy()
    for c in ("aer_ew", "aer_idx", "turnover"):
        show[c] = show[c].map(lambda x: f"{x:+.2%}")
    show["ir_ew"] = show["ir_ew"].map(lambda x: f"{x:+.2f}")
    print(show.to_string(index=False))
    print("=== 机制证据：predictor 逐 epoch val loss 与 best epoch ===")
    for s in SEEDS:
        e = epoch_tables[f"s{s}"]
        vals = ", ".join(f"{x:.4f}" for x in e["val_losses"])
        print(f"  s{s}: best_epoch={e['best_epoch']}  val=[{vals}]")
    print(f"落盘：{out_path}")


if __name__ == "__main__":
    main()
