"""G9 引擎全表（计划 §4.4，20260821 G9 计划）——**落盘不判读，一次开封**。

臂（计划 §1 冻结）× 双基准（000300.SH 市值加权 + 同池等权）同引擎：
k=50/n=5/min_hold=5/单边 15bp（paper_replication/config.yaml 逐字）：

- backtest 窗（2026-01-01~2026-07-24）：G9E0/E1/E5/E10/E15 × 4 变体 +
  F0×4 / M 只读参照；
- 2025H2 窗（2025-07-01~2025-12-31）：G9E1/E15 × 4 变体 + F0×4 / M
  （g0 子集只读参照）。

**纪律 §5**：数字落盘后不判读不外泄，统一 ``run_g9_judge.py`` 一次开封——
本脚本：loguru console sink 移除，绩效数字只写封存日志
``g9_ckpt/data/g9_backtest_sealed.log`` 与结果 JSON（judge 之前不读不贴）；
终端只打印结构信息；结果 JSON 不含任何判定字段。

产出：g9_ckpt/data/g9_backtest_results.json（K0~K4 + E0 判读的冻结输入）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

# —— 纪律 §5：移除 console sink，数字只落封存日志 ——
logger.remove()

PKG_DIR = Path(__file__).resolve().parent
G9_DATA_DIR = PKG_DIR / "data"
ROUND4_DATA = PKG_DIR.parent / "finetune_suite" / "data"
G0_DIR = ROUND4_DATA / "g0"

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable

WINDOW_DEFS = {
    "backtest": ("2026-01-01", "2026-07-24"),
    "2025h2": ("2025-07-01", "2025-12-31"),
}
BACKTEST_ARMS = ["G9E0", "G9E1", "G9E5", "G9E10", "G9E15"]
H2_ARMS = ["G9E1", "G9E15"]


def _load_window_signals(window: str) -> dict[str, pd.DataFrame]:
    """载入一窗的全部宽表：参照（F0×4/M 只读）+ G9 臂×4 变体。"""
    signals: dict[str, pd.DataFrame] = {}
    if window == "backtest":
        for v in VARIANTS:
            signals[f"F0_{v}"] = pd.read_parquet(ROUND4_DATA / f"daily_signals_backtest_F0_{v}.parquet")
        signals["M"] = pd.read_parquet(ROUND4_DATA / "daily_signals_backtest_M.parquet")
        arms = BACKTEST_ARMS
        sub_of = {a: a[2:].lower() for a in BACKTEST_ARMS}
    else:
        for v in VARIANTS:
            signals[f"F0_{v}"] = pd.read_parquet(G0_DIR / f"daily_signals_2025h2_F0_{v}.parquet")
        signals["M"] = pd.read_parquet(G0_DIR / "daily_signals_2025h2_M.parquet")
        arms = H2_ARMS
        sub_of = {a: a[2:].lower() for a in H2_ARMS}
    for arm in arms:
        for v in VARIANTS:
            signals[f"{arm}_{v}"] = pd.read_parquet(
                G9_DATA_DIR / sub_of[arm] / f"daily_signals_{window}_{arm}_{v}.parquet"
            )
    return signals


def _order(window: str) -> list[str]:
    """表序照惯例：参照在前、G9 臂按 epoch 升序、每臂 mean 最后醒目。"""
    arms = BACKTEST_ARMS if window == "backtest" else H2_ARMS
    head = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")]
    tail = [f"{a}_{v}" for a in arms for v in ("min", "max", "last", "mean")]
    return head + tail


def main() -> None:
    sealed_log = G9_DATA_DIR / "g9_backtest_sealed.log"
    logger.add(str(sealed_log), enqueue=False)

    results: dict[str, dict] = {}
    n_days_total = 0
    for window, (start, end) in WINDOW_DEFS.items():
        from dataclasses import replace

        cfg = replace(
            BaselineConfig.load(window="oos"),
            window=f"g9_{window}", backtest_start=start, backtest_end=end,
        )
        signals = _load_window_signals(window)
        order = _order(window)

        from kronos_qlib import QlibProvider

        provider = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
        all_cols = sorted(set().union(*[set(s.columns) for s in signals.values()]))
        rebalances = pd.DatetimeIndex(signals["M"].index)
        px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
        bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

        for tag in order:
            pi, pe, _, _, _ = run_group(
                signals[tag], px, trd, bench_idx, bench_ew, cfg=cfg, name=tag
            )
            results[f"{window}:{tag}"] = {
                "perf_idx": pi.to_dict(), "perf_ew": pe.to_dict(),
            }
        n_days_total += len(rebalances)

    out = {
        "windows": {w: list(bd) for w, bd in WINDOW_DEFS.items()},
        "benchmarks": {"index": "000300.SH csi300 市值加权", "equal_weight": "同池等权"},
        "engine": {"top_k": cfg.top_k, "drop_n": cfg.drop_n,
                   "min_hold": cfg.min_hold, "cost_bps": cfg.cost_bps},
        "groups": results,
        "note": (
            "纪律 §5：G9 数字落盘后不判读不外泄，统一 run_g9_judge.py 一次开封；"
            "K0~K4 + E0 以本表为冻结输入判定"
        ),
    }
    out_path = G9_DATA_DIR / "g9_backtest_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))

    # —— 终端只出结构信息（无绩效数字）——
    print(f"G9 引擎出表完成：{len(results)} 组 × 双基准，{n_days_total} 日（两窗合计）")
    for window in WINDOW_DEFS:
        print(f"窗口 {window}：{len(_order(window))} 组，组序 {_order(window)}")
    print(f"结果落盘：{out_path}")
    print(f"封存日志：{sealed_log}（judge 之前不读不贴）")


if __name__ == "__main__":
    main()
