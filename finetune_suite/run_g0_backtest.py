"""阶段 1.2：G0 三臂双基准引擎出表（计划 §3，20260815）——**落盘不判读**。

臂（冻结）：G0×4 变体（= 第 4 轮 F1 权重 @ 2025H2 推理）、F0×4 变体
（zero-shot oos 子集）、M（10 日动量）。同引擎同双基准：k=50/n=5/min_hold=5/
单边 15bp，000300.SH + 同池等权。

**纪律 §8**：G0 数字落盘后不判读不外泄，统一阶段 3 封盘判读——本脚本：

- loguru 默认 console sink 移除，绩效数字只写入封存日志
  ``data/g0/g0_backtest_sealed.log`` 与结果 JSON（阶段 3 之前不读不贴）；
- 终端只打印结构信息（组名、天数、落盘路径）；
- 结果 JSON **不含任何判定字段**（无 verdict / criterion）。

产出：finetune_suite/data/g0/g0_backtest_results.json（判据 4 的冻结输入）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

# —— 纪律 §8：移除 console sink，数字只落封存日志 ——
logger.remove()

PKG_DIR = Path(__file__).resolve().parent
G0_DIR = PKG_DIR / "data" / "g0"

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from finetune_suite.run_g0_signals import G0_END, G0_START, build_g0_config


def load_signals() -> dict[str, pd.DataFrame]:
    """载入 9 张宽表：G0×4 + F0×4 + M（全部 2025H2 窗）。"""
    signals: dict[str, pd.DataFrame] = {}
    for v in VARIANTS:
        signals[f"G0_{v}"] = pd.read_parquet(G0_DIR / f"daily_signals_2025h2_G0_{v}.parquet")
    for v in VARIANTS:
        signals[f"F0_{v}"] = pd.read_parquet(G0_DIR / f"daily_signals_2025h2_F0_{v}.parquet")
    signals["M"] = pd.read_parquet(G0_DIR / "daily_signals_2025h2_M.parquet")
    return signals


def main() -> None:
    sealed_log = G0_DIR / "g0_backtest_sealed.log"
    logger.add(str(sealed_log), enqueue=False)

    cfg = build_g0_config()
    signals = load_signals()

    # 表序照第 4 轮惯例：参照/对照在前，mean 最后醒目
    order = ["M", "F0_min", "F0_max", "F0_last", "F0_mean",
             "G0_min", "G0_max", "G0_last", "G0_mean"]

    from kronos_qlib import QlibProvider

    provider = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
    all_cols = sorted(set().union(*[set(s.columns) for s in signals.values()]))
    rebalances = pd.DatetimeIndex(signals["M"].index)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    results: dict[str, dict] = {}
    for tag in order:
        pi, pe, _, _, _ = run_group(
            signals[tag], px, trd, bench_idx, bench_ew, cfg=cfg, name=tag
        )
        results[tag] = {"perf_idx": pi.to_dict(), "perf_ew": pe.to_dict()}

    out = {
        "window": "g0_2025h2",
        "period": [G0_START, G0_END],
        "benchmarks": {
            "index": "000300.SH csi300 市值加权",
            "equal_weight": "同池等权",
            "beta_gap": beta_gap,
        },
        "engine": {"top_k": cfg.top_k, "drop_n": cfg.drop_n,
                   "min_hold": cfg.min_hold, "cost_bps": cfg.cost_bps},
        "groups": results,
        "note": (
            "纪律 §8：G0 数字落盘后不判读不外泄，统一阶段 3 封盘；"
            "判据 4（G0 稳定性）在阶段 3 以本表为冻结输入判定"
        ),
    }
    out_path = G0_DIR / "g0_backtest_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    # —— 终端只出结构信息（无绩效数字）——
    print(f"G0 引擎出表完成：{len(order)} 组 × 双基准，{len(rebalances)} 交易日")
    print(f"窗口：{G0_START}~{G0_END} | 组序：{order}")
    print(f"结果落盘：{out_path}")
    print(f"封存日志：{sealed_log}（阶段 3 之前不读不贴）")


if __name__ == "__main__":
    main()
