"""G2 增补臂补充节判读（跑前增补 37aba7d）——**不触碰 S1~S4 冻结判定**。

增补臂均为补充诊断，单独成节：

- **D-seed+**：predictor seed=103/104（共享 G1 tokenizer）backtest 窗四变体
  引擎 → 与核心三种子合成 **5 种子面板（100~104）**，报 mean AER 全距与
  正号比例（补充证据，不进判据）；
- **D-tok**：tokenizer+predictor 全管线 seed=101 backtest 窗引擎 →
  ``D-tok − G2S101`` 同种子差值 = **tokenizer 种子效应**（补计划 §3 声明的洞）。

产出：``finetune_suite/data/g2/g2_supp_results.json`` + 控制台表。
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START

PKG_DIR = Path(__file__).resolve().parent
G2_DIR = PKG_DIR / "data" / "g2"
ROUND4_DATA = PKG_DIR / "data"

SUPP_ARMS = ("G2S103", "G2S104", "DTOK")


def main() -> None:
    from kronos_qlib import QlibProvider

    cfg = replace(
        BaselineConfig.load(window="oos"),
        backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
    )
    signals: dict[str, pd.DataFrame] = {}
    for arm in SUPP_ARMS:
        sub = "dtok" if arm == "DTOK" else f"s{arm[3:]}"  # G2S103 → s103
        for v in VARIANTS:
            signals[f"{arm}_{v}"] = pd.read_parquet(
                G2_DIR / sub / f"daily_signals_backtest_{arm}_{v}.parquet"
            )

    # —— 基准池与核心判读（run_g2_judge）严格同构：并集含 F0×4+M 的列，
    #    否则同池等权基准随增补臂列集漂移，5 种子面板不可比 ——
    ref_cols: set = set()
    for v in VARIANTS:
        ref_cols |= set(
            pd.read_parquet(ROUND4_DATA / f"daily_signals_backtest_F0_{v}.parquet").columns
        )
    ref_cols |= set(pd.read_parquet(ROUND4_DATA / "daily_signals_backtest_M.parquet").columns)

    provider = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
    all_cols = sorted(set().union(*[set(df.columns) for df in signals.values()]) | ref_cols)
    rebalances = pd.DatetimeIndex(signals[f"{SUPP_ARMS[0]}_mean"].index)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    results: dict[str, dict] = {}
    for tag, wide in signals.items():
        pi, pe, _, _, _ = run_group(wide, px, trd, bench_idx, bench_ew, cfg=cfg, name=tag)
        results[tag] = {"aer_ew": pe.aer, "aer_idx": pi.aer}

    # —— 5 种子面板（s100/s101 取第 5 轮+G2 核心冻结数字，只读拼表）——
    g1 = json.loads((PKG_DIR / "data" / "g1" / "g1_backtest_results.json").read_text(encoding="utf-8"))
    g2 = json.loads((G2_DIR / "g2_judge_results.json").read_text(encoding="utf-8"))
    panel = {
        "s100": g1["groups"]["G1_mean"]["perf_ew"]["aer"],
        "s101": g2["perf"]["backtest"]["s101"]["aer_ew"],
        "s102": g2["perf"]["backtest"]["s102"]["aer_ew"],
        "s103": results["G2S103_mean"]["aer_ew"],
        "s104": results["G2S104_mean"]["aer_ew"],
    }
    panel_idx = {
        "s100": g1["groups"]["G1_mean"]["perf_idx"]["aer"],
        "s101": g2["perf"]["backtest"]["s101"]["aer_idx"],
        "s102": g2["perf"]["backtest"]["s102"]["aer_idx"],
        "s103": results["G2S103_mean"]["aer_idx"],
        "s104": results["G2S104_mean"]["aer_idx"],
    }
    five = {
        "aer_ew_by_seed": panel,
        "aer_idx_by_seed": panel_idx,
        "range_ew": max(panel.values()) - min(panel.values()),
        "min_ew": min(panel.values()),
        "max_ew": max(panel.values()),
        "n_positive_ew": sum(1 for x in panel.values() if x > 0),
        "note": "5 种子面板（增补 D-seed+）：补充证据，不进 S1~S4 判据",
    }

    # —— D-tok 同种子差值（tokenizer 种子效应）——
    dtok = {
        "dtok_mean_ew": results["DTOK_mean"]["aer_ew"],
        "dtok_mean_idx": results["DTOK_mean"]["aer_idx"],
        "shared_tok_s101_mean_ew": panel["s101"],
        "tokenizer_seed_effect_ew": results["DTOK_mean"]["aer_ew"] - panel["s101"],
        "by_variant_ew": {
            v: results[f"DTOK_{v}"]["aer_ew"] - g2["full_table"][f"s101@backtest"][v]["aer_ew"]
            for v in VARIANTS
        },
        "note": "D-tok(全管线 seed101) − G2S101(共享G1 tokenizer, seed101)：差值=tokenizer 种子效应",
    }

    out = {
        "window": "backtest",
        "period": [BACKTEST_START, BACKTEST_END],
        "beta_gap": beta_gap,
        "supp_arms_perf": results,
        "five_seed_panel": five,
        "dtok_vs_shared": dtok,
    }
    out_path = G2_DIR / "g2_supp_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    print("=== G2 增补臂补充节（不进 S1~S4 判据）===")
    print(f"[5 种子面板 backtest mean AER(等权)] " +
          ", ".join(f"{s}={v:+.2%}" for s, v in panel.items()))
    print(f"全距 {five['min_ew']:+.2%} ~ {five['max_ew']:+.2%}"
          f"（极差 {five['range_ew']:.2%}），正号 {five['n_positive_ew']}/5")
    print(f"[D-tok tokenizer 种子效应] DTOK_mean {dtok['dtok_mean_ew']:+.2%} − "
          f"G2S101_mean {panel['s101']:+.2%} = {dtok['tokenizer_seed_effect_ew']:+.2%}")
    print(f"补充节落盘：{out_path}")


if __name__ == "__main__":
    main()
