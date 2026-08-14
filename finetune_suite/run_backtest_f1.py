"""阶段 3.2/3.4：三臂双基准引擎出表 + 预注册判定（计划 §5，修订2，跑前冻结）。

臂（全部冻结）：F1×4 变体（微调权重）、F0×4 变体（zero-shot oos 子集）、M（10 日动量）。
同引擎同双基准：k=50/n=5/min_hold=5/单边 15bp，000300.SH + 同池等权。

预注册判据（只定义在 mean 上，判读一次封盘）：
    1. 主判据（存活）：F1 mean AER(等权) > 0 且 AER(指数) > 0；
    2. 改善判据（方向）：F1 mean AER(等权) ≥ F0 mean AER(等权) + 10pp；
    3. M 同窗结果仅作裁判参照，不设通过/失败；
    4. 失败判据：1、2 均未过 → 本土化微调路线关闭。
四变体纪律：last/max/min 为预注册记录族，全表呈现不参与判据；某变体 AER(等权)
高出 mean > 3pp 仅记探索性观察。

产出：finetune_suite/data/backtest_results.json + figures/fig_backtest_variants.png。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.plot import plot_dual
from baseline_suite.signal import build_px_tradeable
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START, build_f1_config

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"


def load_signals() -> dict[str, pd.DataFrame]:
    """载入 9 张宽表：F1×4 + F0×4 + M。"""
    signals: dict[str, pd.DataFrame] = {}
    for prefix, arm in (("F1", "F1"), ("F0", "F0")):
        for v in VARIANTS:
            p = DATA_DIR / f"daily_signals_backtest_{arm}_{v}.parquet"
            signals[f"{prefix}_{v}"] = pd.read_parquet(p)
    signals["M"] = pd.read_parquet(DATA_DIR / "daily_signals_backtest_M.parquet")
    return signals


def judge(f1_mean_ew: dict, f1_mean_idx: dict, f0_mean_ew: dict) -> dict:
    """预注册判据 1~4（只看 mean，判读一次封盘）。"""
    c1 = f1_mean_ew["aer"] > 0 and f1_mean_idx["aer"] > 0
    c2 = f1_mean_ew["aer"] >= f0_mean_ew["aer"] + 0.10
    if c1:
        v1 = "本土化微调信号在半污染窗存活，待前向确认"
    else:
        v1 = (
            f"mean AER(等权)={f1_mean_ew['aer']:+.2%} ≤ 0 或 AER(指数)"
            f"={f1_mean_idx['aer']:+.2%} ≤ 0 → 存活不成立"
        )
    if c2:
        v2 = "微调相对零样本有实质改善（≥ +10pp）"
    else:
        v2 = (
            f"F1−F0 AER(等权)={f1_mean_ew['aer'] - f0_mean_ew['aer']:+.2%} < +10pp "
            "→ 实质改善不成立（塌方变浅与否见差值符号）"
        )
    verdict = (
        "本土化微调（base 规格、本协议 15 epochs）不解释也不缓解 oos 塌方，微调路线关闭"
        if not (c1 or c2) else "部分判据通过（见各条），按计划 §7 触发条件后续"
    )
    return {
        "criterion_1_survival": {"passed": bool(c1), "note": v1},
        "criterion_2_improvement": {"passed": bool(c2), "note": v2},
        "criterion_3_M_reference_only": "M 为 regime 依赖的免费基线，仅参照不判定",
        "criterion_4_failure": {
            "triggered": bool(not (c1 or c2)),
            "note": verdict,
        },
        "verdict": verdict,
    }


def main() -> None:
    cfg = build_f1_config()
    signals = load_signals()

    # 判据只看 mean，但表序按 第1轮惯例：参照/对照在前，mean 最后醒目
    order = ["M", "F0_min", "F0_max", "F0_last", "F0_mean",
             "F1_min", "F1_max", "F1_last", "F1_mean"]

    from kronos_qlib import QlibProvider

    provider = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
    all_cols = sorted(set().union(*[set(s.columns) for s in signals.values()]))
    rebalances = pd.DatetimeIndex(signals["M"].index)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    results: dict[str, dict] = {}
    daily_rets: dict[str, pd.Series] = {}
    for tag in order:
        pi, pe, dr, _, _ = run_group(
            signals[tag], px, trd, bench_idx, bench_ew, cfg=cfg, name=tag
        )
        results[tag] = {"perf_idx": pi.to_dict(), "perf_ew": pe.to_dict()}
        daily_rets[tag] = dr

    # —— 四变体纪律（修订2）：非 mean 变体强于 mean > 3pp 仅记探索性观察 ——
    exploratory = {}
    for v in ("last", "max", "min"):
        gap = results[f"F1_{v}"]["perf_ew"]["aer"] - results["F1_mean"]["perf_ew"]["aer"]
        if gap > 0.03:
            exploratory[f"F1_{v}"] = (
                f"较 mean 高 {gap:+.2%}（> 3pp）→ 仅探索性观察，留待前向独立验证"
            )
    for v in ("last", "max", "min"):
        gap = results[f"F0_{v}"]["perf_ew"]["aer"] - results["F0_mean"]["perf_ew"]["aer"]
        if gap > 0.03:
            exploratory[f"F0_{v}"] = (
                f"较 mean 高 {gap:+.2%}（> 3pp）→ 仅探索性观察，留待前向独立验证"
            )

    verdict = judge(
        results["F1_mean"]["perf_ew"], results["F1_mean"]["perf_idx"],
        results["F0_mean"]["perf_ew"],
    )

    # —— 净值图（判据表只含 mean 行；全变体曲线仅作记录族呈现）——
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_dual(
        daily_rets, bench_idx,
        title=f"F1 vs F0 vs M — backtest window {BACKTEST_START}~{BACKTEST_END} (sealed)",
        out_path=FIG_DIR / "fig_backtest_variants.png",
    )

    out = {
        "window": "backtest",
        "period": [BACKTEST_START, BACKTEST_END],
        "benchmarks": {
            "index": "000300.SH csi300 市值加权",
            "equal_weight": "同池等权",
            "beta_gap": beta_gap,
        },
        "engine": {"top_k": cfg.top_k, "drop_n": cfg.drop_n,
                   "min_hold": cfg.min_hold, "cost_bps": cfg.cost_bps},
        "groups": results,
        "four_variant_exploratory": exploratory,
        "verdict": verdict,
    }
    out_path = DATA_DIR / "backtest_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"backtest 结果落盘 {out_path}")
    logger.info(f"=== 预注册判定：{verdict['verdict']} ===")


if __name__ == "__main__":
    main()
