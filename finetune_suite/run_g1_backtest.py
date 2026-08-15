"""阶段 3：统一封盘判读（计划 §5，20260815）——判据 1~5 一次判读。

**预注册判据（跑前冻结，只定义在 mean）**：

1. 机制（数据饥饿）：G1 predictor best epoch ≥ 2 **或** val loss 非单调恶化
   （存在 epoch≥2 的 val 改善）——扩语料是否后移过拟合墙；
2. 改善（vs Kronos 基线链）：G1_mean backtest AER(等权) ≥ F1_mean(-4.24%)+3pp
   （即 ≥ -1.24%）；
3. 存活（vs 市场基准）：G1_mean AER(等权) > 0 且 AER(指数) > 0；
4. G0 稳定性：F1_mean@2025H2 AER(等权) ≥ F0_mean 同窗 +5pp；
5. 路线关闭：判据 1、2 均失败。

四变体纪律：last/max/min 为记录族全表呈现，不参与判据；M 同窗仅参照。

产出：
    finetune_suite/data/g1/g1_backtest_results.json（含判据判定）
    finetune_suite/figures/fig_g1_vs_market.png（净值：vs 市场基准节）
    finetune_suite/figures/fig_g1_vs_kronos.png（净值：vs Kronos 基线节）
    finetune_suite/figures/fig_g1_beat_benchmark.png（击败基准条形图）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 本机有 Noto Sans CJK SC（图表含中文标签）；无 CJK 环境回退 DejaVu（英文轴标）
matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START, build_f1_config

PKG_DIR = Path(__file__).resolve().parent
G1_DIR = PKG_DIR / "data" / "g1"
G0_DIR = PKG_DIR / "data" / "g0"
ROUND4_DATA = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"

# 第 4 轮封盘冻结基线（计划 §0/§5，判据 2 的参照；引擎复跑后做一致性对拍）
F1_MEAN_AER_EW_FROZEN = -0.0424
F1_MEAN_AER_IDX_FROZEN = -0.0778
F0_MEAN_AER_EW_FROZEN = -0.1723
M_AER_EW_FROZEN = 0.3602

# 家族配色（计划 §5：G1 蓝系 / F1 青系 / F0 灰系 / M 橙 / CSI300 黑虚线）
FAMILY_COLORS = {
    "G1": {"mean": "#08306b", "min": "#2171b5", "max": "#4292c6", "last": "#6baed6"},
    "F1": {"mean": "#006666", "min": "#2e9191", "max": "#5fb3b3", "last": "#8fd0d0"},
    "F0": {"mean": "#4d4d4d", "min": "#8a8a8a", "max": "#ababab", "last": "#cccccc"},
    "M": {"-": "#ff7f0e"},
}
CSI300_STYLE = dict(color="black", linestyle="--", linewidth=1.4)
EW_STYLE = dict(color="black", linestyle=":", linewidth=1.2)


# ============================================================================
# 判据逻辑（纯函数，tests/test_finetune_ashares_stage3.py 单元覆盖）
# ============================================================================
def judge_criterion1_mechanism(epoch_table: pd.DataFrame, best_epoch: int) -> tuple[bool, str]:
    """判据 1：best epoch ≥ 2，或 val 存在 epoch≥2 的严格改善（非单调恶化）。"""
    if best_epoch >= 2:
        return True, f"best epoch = {best_epoch} ≥ 2 → 过拟合墙后移"
    val = epoch_table["val_loss"].tolist()
    improves = [
        int(e) for e in epoch_table["epoch"].tolist()
        if e >= 2 and val[e - 1] < val[e - 2]
    ]
    if improves:
        return True, (
            f"best epoch = {best_epoch}，但 val 存在 epoch≥2 改善（epoch "
            f"{improves}）→ 非单调恶化，墙有后移迹象"
        )
    return False, (
        f"best epoch = {best_epoch} 且 val 自 epoch 2 起单调恶化（无改善）"
        "→ 过拟合墙未后移"
    )


def judge_criteria(
    g1_mean_aer_ew: float,
    g1_mean_aer_idx: float,
    g0_mean_aer_ew: float,
    g0_f0_mean_aer_ew: float,
    c1_passed: bool,
) -> dict:
    """判据 2~5（判据 1 由 :func:`judge_criterion1_mechanism` 单独给出）。"""
    c2 = g1_mean_aer_ew >= F1_MEAN_AER_EW_FROZEN + 0.03
    c3 = g1_mean_aer_ew > 0 and g1_mean_aer_idx > 0
    c4 = g0_mean_aer_ew >= g0_f0_mean_aer_ew + 0.05
    c5 = (not c1_passed) and (not c2)
    return {
        "criterion_2_improvement": {
            "g1_mean_aer_ew": g1_mean_aer_ew,
            "threshold": F1_MEAN_AER_EW_FROZEN + 0.03,
            "passed": bool(c2),
            "note": (
                f"G1_mean AER(等权)={g1_mean_aer_ew:+.2%} "
                f"≥ -1.24%（F1_mean -4.24% + 3pp）→ 更多语料带来更多修复"
                if c2 else
                f"G1_mean AER(等权)={g1_mean_aer_ew:+.2%} < -1.24% "
                "→ 改善未达门槛"
            ),
        },
        "criterion_3_survival": {
            "g1_mean_aer_ew": g1_mean_aer_ew,
            "g1_mean_aer_idx": g1_mean_aer_idx,
            "passed": bool(c3),
            "note": (
                "G1_mean 双基准 AER 均为正 → 首个存活 Kronos 系信号"
                "（单种子，待前向确认）" if c3 else
                f"G1_mean AER(等权)={g1_mean_aer_ew:+.2%} / "
                f"AER(指数)={g1_mean_aer_idx:+.2%} 未同时 > 0 → 存活不成立"
            ),
        },
        "criterion_4_g0_stability": {
            "g0_mean_aer_ew": g0_mean_aer_ew,
            "g0_f0_mean_aer_ew": g0_f0_mean_aer_ew,
            "passed": bool(c4),
            "note": (
                f"F1_mean@2025H2 较 F0_mean 同窗 +{g0_mean_aer_ew - g0_f0_mean_aer_ew:.2%} "
                "≥ +5pp → 第 4 轮改善跨时段复现" if c4 else
                f"F1_mean@2025H2 较 F0_mean 同窗 +{g0_mean_aer_ew - g0_f0_mean_aer_ew:.2%} "
                "< +5pp → 改善未跨时段复现"
            ),
        },
        "criterion_5_route_closed": {
            "triggered": bool(c5),
            "note": (
                "判据 1、2 均失败 → 数据饥饿证伪，扩语料路线关闭，"
                "微调线仅剩 forward 结算" if c5 else
                "判据 1/2 至少一项通过 → 路线未关闭（按 §7 触发条件后续）"
            ),
        },
    }


# ============================================================================
# 机制证据：解析 G1 predictor 逐 epoch 表（判据 1 的冻结输入）
# ============================================================================
def parse_epoch_table(console_path: Path) -> tuple[pd.DataFrame, int]:
    """从训练控制台日志解析逐 epoch val loss 与 best epoch。

    匹配 ``--- Epoch N/15 Summary ---`` 之后的 ``Validation Loss: X`` 与
    ``Best model saved ...`` 行；best epoch = 最后一次 Best model saved 的 epoch。
    """
    text = console_path.read_text(encoding="utf-8", errors="replace")
    epochs, vals, bests = [], [], []
    cur = None
    for line in text.splitlines():
        m = re.search(r"--- Epoch (\d+)/\d+ Summary ---", line)
        if m:
            cur = int(m.group(1))
            continue
        m = re.search(r"Validation Loss: ([0-9.]+)", line)
        if m and cur is not None and (not epochs or epochs[-1] != cur):
            epochs.append(cur)
            vals.append(float(m.group(1)))
            continue
        if "Best model saved" in line and cur is not None:
            bests.append(cur)
    if not epochs:
        raise ValueError(f"{console_path} 未解析到 epoch 表")
    return pd.DataFrame({"epoch": epochs, "val_loss": vals}), (max(bests) if bests else None)


# ============================================================================
# 图表（两节式 + 条形图）
# ============================================================================
def _nav(ret: pd.Series) -> pd.Series:
    return (1 + ret.fillna(0)).cumprod() - 1


def _style(tag: str) -> dict:
    if tag == "M":
        return dict(color=FAMILY_COLORS["M"]["-"], linewidth=1.6)
    arm, variant = tag.split("_", 1)
    lw = 2.4 if variant == "mean" else 1.3
    return dict(color=FAMILY_COLORS[arm][variant], linewidth=lw)


def plot_vs_market(
    daily_rets: dict[str, pd.Series], bench_idx: pd.Series, bench_ew: pd.Series
) -> None:
    """节一『vs 市场基准』：净值 + 相对等权基准累计超额（判据 3 的 0 线）。"""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for tag, r in daily_rets.items():
        axes[0].plot(_nav(r).index, _nav(r).values, label=tag, **_style(tag))
    axes[0].plot(_nav(bench_idx).index, _nav(bench_idx).values, label="CSI300(指数基准)", **CSI300_STYLE)
    axes[0].plot(_nav(bench_ew).index, _nav(bench_ew).values, label="同池等权基准", **EW_STYLE)
    axes[0].set_ylabel("Cumulative return (with cost)")
    axes[0].set_title(
        f"vs 市场基准（判据3，0线=双基准）— backtest {BACKTEST_START}~{BACKTEST_END}"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)

    for tag, r in daily_rets.items():
        common = r.index.intersection(bench_ew.index)
        ex = (r.loc[common] - bench_ew.loc[common]).fillna(0)
        cum = (1 + ex).cumprod() - 1
        axes[1].plot(cum.index, cum.values, label=tag, **_style(tag))
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Cumulative excess vs 同池等权 (with cost)")
    axes[1].set_xlabel("Decision date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g1_vs_market.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vs_kronos(daily_rets: dict[str, pd.Series], bench_idx: pd.Series) -> None:
    """节二『vs Kronos 基线』：净值 + 相对 F1_mean 的累计超额（判据 2 的基线）。"""
    f1_mean = daily_rets["F1_mean"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for tag, r in daily_rets.items():
        axes[0].plot(_nav(r).index, _nav(r).values, label=tag, **_style(tag))
    axes[0].plot(_nav(bench_idx).index, _nav(bench_idx).values, label="CSI300", **CSI300_STYLE)
    axes[0].set_ylabel("Cumulative return (with cost)")
    axes[0].set_title(
        f"vs Kronos 基线（判据2/4，基线=F1/F0 组合）— backtest {BACKTEST_START}~{BACKTEST_END}"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)

    for tag, r in daily_rets.items():
        common = r.index.intersection(f1_mean.index)
        ex = (r.loc[common] - f1_mean.loc[common]).fillna(0)
        cum = (1 + ex).cumprod() - 1
        axes[1].plot(cum.index, cum.values, label=f"{tag} − F1_mean", **_style(tag))
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Cumulative excess vs F1_mean (with cost)")
    axes[1].set_xlabel("Decision date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g1_vs_kronos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_beat_benchmark(results: dict[str, dict]) -> None:
    """击败基准条形图：13 组 AER(等权)，0 线（市场）与 F1_mean/-1.24%（Kronos）。"""
    order = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")] + \
            [f"F1_{v}" for v in ("min", "max", "last", "mean")] + \
            [f"G1_{v}" for v in ("min", "max", "last", "mean")]
    aers = [results[t]["perf_ew"]["aer"] for t in order]
    colors = [FAMILY_COLORS["M"]["-"] if t == "M" else FAMILY_COLORS[t.split("_")[0]][t.split("_", 1)[1]] for t in order]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(order, aers, color=colors)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="0（市场基准）")
    ax.axhline(F1_MEAN_AER_EW_FROZEN, color="#006666", linestyle=":", linewidth=1.4,
               label=f"F1_mean {F1_MEAN_AER_EW_FROZEN:+.2%}（Kronos 基线）")
    ax.axhline(F1_MEAN_AER_EW_FROZEN + 0.03, color="#08306b", linestyle="-.",
               linewidth=1.4, label="判据2门槛 -1.24%")
    for i, a in enumerate(aers):
        ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}",
                ha="center", fontsize=8)
    ax.set_ylabel("AER (等权基准, with cost)")
    ax.set_title(f"击败基准条形图 — backtest {BACKTEST_START}~{BACKTEST_END}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g1_beat_benchmark.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 主流程
# ============================================================================
def load_signals() -> dict[str, pd.DataFrame]:
    """13 张宽表：G1×4（本轮）+ F1×4 / F0×4 / M（第 4 轮 backtest，只读复用）。"""
    signals: dict[str, pd.DataFrame] = {}
    for v in VARIANTS:
        signals[f"G1_{v}"] = pd.read_parquet(G1_DIR / f"daily_signals_backtest_G1_{v}.parquet")
    for arm in ("F1", "F0"):
        for v in VARIANTS:
            signals[f"{arm}_{v}"] = pd.read_parquet(
                ROUND4_DATA / f"daily_signals_backtest_{arm}_{v}.parquet"
            )
    signals["M"] = pd.read_parquet(ROUND4_DATA / "daily_signals_backtest_M.parquet")
    return signals


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_f1_config()  # 同第 4 轮 backtest 窗/池/引擎口径（权重字段不用于引擎）
    signals = load_signals()

    order = ["M", "F0_min", "F0_max", "F0_last", "F0_mean",
             "F1_min", "F1_max", "F1_last", "F1_mean",
             "G1_min", "G1_max", "G1_last", "G1_mean"]

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

    # —— 引擎确定性对拍：F1/F0/M 与第 4 轮封盘数字一致（同窗可比的前提）——
    r4 = json.loads((ROUND4_DATA / "backtest_results.json").read_text(encoding="utf-8"))
    for tag, frozen in (
        ("F1_mean", F1_MEAN_AER_EW_FROZEN), ("F0_mean", F0_MEAN_AER_EW_FROZEN),
        ("M", M_AER_EW_FROZEN),
    ):
        rerun = results[tag]["perf_ew"]["aer"]
        assert abs(rerun - frozen) < 5e-4, (
            f"{tag} 引擎复跑 {rerun:+.4%} 与第 4 轮冻结 {frozen:+.4%} 不一致"
        )
        logger.info(f"对拍 {tag}: 复跑 {rerun:+.4%} vs 冻结 {frozen:+.4%} 一致")

    # —— 机制证据（判据 1）：G1 predictor 逐 epoch 表 ——
    epoch_table, best_epoch = parse_epoch_table(
        G1_DIR / "train_predictor_g1_console.txt"
    )
    c1_passed, c1_note = judge_criterion1_mechanism(epoch_table, best_epoch)

    # —— G0 封存数字开封（判据 4 冻结输入）——
    g0 = json.loads((G0_DIR / "g0_backtest_results.json").read_text(encoding="utf-8"))
    verdict = judge_criteria(
        g1_mean_aer_ew=results["G1_mean"]["perf_ew"]["aer"],
        g1_mean_aer_idx=results["G1_mean"]["perf_idx"]["aer"],
        g0_mean_aer_ew=g0["groups"]["G0_mean"]["perf_ew"]["aer"],
        g0_f0_mean_aer_ew=g0["groups"]["F0_mean"]["perf_ew"]["aer"],
        c1_passed=c1_passed,
    )
    verdict = {
        "criterion_1_mechanism": {
            "best_epoch": best_epoch,
            "val_losses": epoch_table["val_loss"].tolist(),
            "passed": bool(c1_passed),
            "note": c1_note,
        },
        **verdict,
    }

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
        "g0_sealed": {
            "period": g0["period"],
            "groups": {k: v for k, v in g0["groups"].items() if k.endswith("mean") or k == "M"},
        },
        "verdict": verdict,
    }
    out_path = G1_DIR / "g1_backtest_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"G1 backtest+判读落盘 {out_path}")

    # —— 两节式净值图 + 条形图 ——
    plot_vs_market(daily_rets, bench_idx, bench_ew)
    plot_vs_kronos(daily_rets, bench_idx)
    plot_beat_benchmark(results)

    print("=== 预注册判据 1~5 封盘判读 ===")
    for k, v in verdict.items():
        head = v.get("passed", v.get("triggered"))
        print(f"[{k}] {'通过' if head else '未通过/未触发'}：{v['note']}")
    print(f"机制证据：G1 predictor best_epoch={best_epoch}，逐 epoch val="
          f"{epoch_table['val_loss'].tolist()}")


if __name__ == "__main__":
    main()
