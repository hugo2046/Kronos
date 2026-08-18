"""G4 阶段 2：统一封盘判读（G4 计划 §4，20260817）——判据 J1~J5 一次封盘。

**预注册判据（跑前冻结，只定义在 mean）**：

| # | 判据 | 冻结定义 |
|---|---|---|
| J1 | 存活 | 中位种子 backtest AER(等权) > 0 且 AER(指数) > 0 |
| J2 | 增量（核心） | 中位 AER(等权) ≥ +16.33%（在位者 +14.33%+2pp）且 AER(指数) ≥ +12.66% |
| J3 | 中性带 | J2 未过但中位 ∈ (+12.33%, +16.33%) |
| J4 | 跨窗 | J2 通过者 2025H2 AER(等权) > 0 |
| J5 | 否定 | 中位 ≤ +12.33%（G1−2pp）→ 特征路线关闭，不做特征搜索 |
| 注册 | J2∧J4 → G4 中位种子信号注册进 G3 登记表新列 |

冻结措辞约束：J2 阈值基于在位者 s100 单点值（G1 种子分布 +10.74~+29.06%）——
G4 中位落入该分布内部时，判读必须并列呈现两族种子分布（3 vs 5），不得以
单点差宣称优劣；三种子中位对三种子中位对比表**必列**。

产出：g4_features/data/g4_stage2_results.json + g4_features/figures/ 三图。
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from finetune_suite.run_g2_signals import WINDOW_DEFS

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"
R4 = PKG_DIR.parent / "finetune_suite" / "data"
G5_DATA = PKG_DIR.parent / "g5_head" / "data"

# —— 预注册判据阈值（计划 §4 表，跑前冻结）——
INCUMBENT_AER_EW = 0.1433
INCUMBENT_AER_IDX = 0.1066
J2_THRESH_EW = 0.1633
J2_THRESH_IDX = 0.1266
J5_THRESH_EW = 0.1233  # = G1 − 2pp
J3_LO, J3_HI = 0.1233, 0.1633

SEEDS = (100, 101, 102)
G4_ARMS = {s: f"G4S{s}" for s in SEEDS}
G1_ARMS = {100: "G1", 101: "G2S101", 102: "G2S102"}  # G1 族三种子（同宇宙复跑对照）

WINDOW_BOUNDS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": WINDOW_DEFS["2025h2"],
}

# 在位者信号（参照行 + 对拍锚：backtest 冻结 +14.33%）
INCUMBENT_PARQUET = {
    "backtest": R4 / "g1" / "daily_signals_backtest_G1_mean.parquet",
    "2025h2": G5_DATA / "daily_signals_2025h2_G1_mean.parquet",
}
M_PARQUET = {
    "backtest": R4 / "daily_signals_backtest_M.parquet",
    "2025h2": R4 / "g0" / "daily_signals_2025h2_M.parquet",
}

# 引擎宇宙（px/等权基准列集）——与冻结数字的原始跑法逐字一致（g5 同款）：
# backtest = 第 4 轮 G1 回测的 13 信号并集；2025h2 = G0 回测的 9 信号并集。
UNIVERSE_PARQUETS = {
    "backtest": [R4 / "g1" / f"daily_signals_backtest_G1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "daily_signals_backtest_M.parquet"],
    "2025h2": [R4 / "g0" / f"daily_signals_2025h2_G0_{v}.parquet" for v in VARIANTS]
    + [R4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "g0" / "daily_signals_2025h2_M.parquet"],
}

# G1 族两窗信号 parquet（同宇宙引擎复跑对照，非引用旧数字）
G1_SIGNAL_PARQUETS = {
    "backtest": {
        100: R4 / "g1" / "daily_signals_backtest_G1_{v}.parquet",
        101: R4 / "g2" / "s101" / "daily_signals_backtest_G2S101_{v}.parquet",
        102: R4 / "g2" / "s102" / "daily_signals_backtest_G2S102_{v}.parquet",
    },
    "2025h2": {
        100: G5_DATA / "daily_signals_2025h2_G1_{v}.parquet",
        101: R4 / "g2" / "s101" / "daily_signals_2025h2_G2S101_{v}.parquet",
        102: R4 / "g2" / "s102" / "daily_signals_2025h2_G2S102_{v}.parquet",
    },
}

# G1 族五种子 backtest 冻结分布（种子诊断+增补封盘：g2_judge/g2_supp；
# 范围 +10.74%~+29.06% 与计划 §4 措辞一致；两族并列呈现用，不参与判据）
G1_FIVE_SEED_BACKTEST_EW = {
    "s100": 0.1433, "s101": 0.2906, "s102": 0.1867, "s103": 0.2036, "s104": 0.1074,
}

# 族配色（G4 红系 / G1 蓝系 / M 橙 / CSI300 黑虚线）
FAMILY_COLORS = {
    "G4": {"s100": "#a50f15", "s101": "#ef3b2c", "s102": "#fb6a4a"},
    "G1": {"s100": "#08306b", "s101": "#2171b5", "s102": "#6baed6"},
    "M": "#ff7f0e",
}
CSI300_STYLE = dict(color="black", linestyle="--", linewidth=1.4)
EW_STYLE = dict(color="black", linestyle=":", linewidth=1.2)


# ============================================================
# 引擎（冻结宇宙）
# ============================================================
def engine_window(signals: dict[str, pd.DataFrame], window: str,
                  universe_cols: list[str]) -> tuple[dict, dict]:
    from kronos_qlib import QlibProvider

    start, end = WINDOW_BOUNDS[window]
    cfg = replace(BaselineConfig.load(window="oos"),
                  window=f"g4_{window}", backtest_start=start, backtest_end=end)
    provider = QlibProvider(cfg.pool, start, end)
    rebalances = pd.DatetimeIndex(signals[list(signals)[0]].index)
    px, trd = build_px_tradeable(provider, cfg, rebalances, universe_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    results, rets = {}, {}
    for name, sig in signals.items():
        pi, pe, dr, _, _ = run_group(sig, px, trd, bench_idx, bench_ew, cfg=cfg, name=name)
        results[name] = {"perf_idx": pi.to_dict(), "perf_ew": pe.to_dict()}
        rets[name] = dr
    return results, {"beta_gap": beta_gap, "rets": rets,
                     "bench_idx": bench_idx, "bench_ew": bench_ew}


# ============================================================
# 判据（纯函数）
# ============================================================
def median_seed(perf_by_seed: dict[int, float]) -> int:
    """按 backtest AER(等权) 排序取中位种子（3 种子 → 第 2 名）。"""
    ranked = sorted(perf_by_seed, key=lambda s: perf_by_seed[s])
    return ranked[1]


def judge(g4: dict, g1: dict) -> dict:
    """J1~J5 + 注册触发（阈值冻结于模块常量；一次判读）。

    :param g4: {seed: {window: {"perf_ew": {...}, "perf_idx": {...}}}}
    :param g1: 同构（G1 族三种子同宇宙复跑，中位对中位对比表用）
    """
    bt_ew = {s: g4[s]["backtest"]["perf_ew"]["aer"] for s in SEEDS}
    bt_idx = {s: g4[s]["backtest"]["perf_idx"]["aer"] for s in SEEDS}
    med = median_seed(bt_ew)
    med_ew, med_idx = bt_ew[med], bt_idx[med]
    h2_ew_med = g4[med]["2025h2"]["perf_ew"]["aer"]

    j1 = med_ew > 0 and med_idx > 0
    j2 = med_ew >= J2_THRESH_EW and med_idx >= J2_THRESH_IDX
    j3 = (not j2) and (J3_LO < med_ew < J3_HI)
    j5 = med_ew <= J5_THRESH_EW
    j4 = j2 and (h2_ew_med > 0)
    register = j2 and j4

    # G1 族三种子同宇宙中位（必列对比表；判据锚仍为冻结单点 +14.33%）
    g1_bt_ew = {s: g1[s]["backtest"]["perf_ew"]["aer"] for s in SEEDS}
    g1_med = median_seed(g1_bt_ew)

    in_g1_range = min(G1_FIVE_SEED_BACKTEST_EW.values()) <= med_ew <= max(
        G1_FIVE_SEED_BACKTEST_EW.values()
    )
    return {
        "median_seed": f"s{med}",
        "median_backtest": {
            "aer_ew": med_ew, "aer_idx": med_idx,
            "by_seed_ew": {f"s{s}": bt_ew[s] for s in SEEDS},
            "by_seed_idx": {f"s{s}": bt_idx[s] for s in SEEDS},
        },
        "g1_three_seed_median": {
            "median_seed": f"s{g1_med}",
            "by_seed_ew": {f"s{s}": g1_bt_ew[s] for s in SEEDS},
            "median_aer_ew": g1_bt_ew[g1_med],
        },
        "J1_survival": {"passed": bool(j1), "note": (
            f"中位 s{med} AER(等权)={med_ew:+.2%} / AER(指数)={med_idx:+.2%} "
            + ("双正 → 存活" if j1 else "未同时 > 0 → 存活不成立"))},
        "J2_increment": {"passed": bool(j2), "note": (
            f"中位 AER(等权)={med_ew:+.2%} vs 门槛 {J2_THRESH_EW:+.2%}，"
            f"AER(指数)={med_idx:+.2%} vs 门槛 {J2_THRESH_IDX:+.2%} "
            + ("→ 市场上下文有增量，待前向确认" if j2 else "→ 未达增量门槛"))},
        "J3_neutral_band": {"triggered": bool(j3), "note": (
            f"中位 {med_ew:+.2%} ∈ ({J3_LO:+.2%}, {J3_HI:+.2%}) → 特征中性"
            if j3 else f"中位 {med_ew:+.2%} 不在中性带") if not j2 else "J2 已过，中性带不适用"},
        "J4_cross_window": {"passed": bool(j4), "applicable": bool(j2), "note": (
            f"中位 s{med} 2025H2 AER(等权)={h2_ew_med:+.2%} "
            + ("→ 跨窗存活" if j4 else "→ 未跨窗") if j2 else "J2 未过，J4 不适用")},
        "J5_rejection": {"triggered": bool(j5), "note": (
            f"中位 {med_ew:+.2%} ≤ {J5_THRESH_EW:+.2%}（G1−2pp）→ 市场上下文损害或"
            "无增量，特征路线关闭，不做特征搜索" if j5 else
            f"中位 {med_ew:+.2%} > {J5_THRESH_EW:+.2%} → 否定线未触发")},
        "registration": {"triggered": bool(register), "note": (
            "J2∧J4 通过 → G4 中位种子信号注册进 G3 登记表新列" if register else
            "未触发注册（J2∧J4 至少一项未过）")},
        "wording_constraint": {
            "median_in_g1_five_seed_range": bool(in_g1_range),
            "g1_five_seed_backtest_ew": G1_FIVE_SEED_BACKTEST_EW,
            "note": (
                "G4 中位落入 G1 五种子分布内部 → 两族分布并列呈现，"
                "不得以单点差宣称优劣" if in_g1_range else
                "G4 中位在 G1 五种子分布之外，单点对比可直言方向"),
        },
    }


# ============================================================
# 图（两节式 + 条形图）
# ============================================================
def _nav(ret: pd.Series) -> pd.Series:
    return (1 + ret.fillna(0)).cumprod() - 1


def plot_two_panel(daily_rets: dict[str, pd.Series], bench_idx, bench_ew,
                   g1_med_tag: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for tag, r in daily_rets.items():
        arm, s = tag.split("_", 1)
        axes[0].plot(_nav(r).index, _nav(r).values, label=tag,
                     color=FAMILY_COLORS[arm][s],
                     linewidth=2.4 if arm == "G4" else 1.3)
    axes[0].plot(_nav(bench_idx).index, _nav(bench_idx).values,
                 label="CSI300(指数基准)", **CSI300_STYLE)
    axes[0].plot(_nav(bench_ew).index, _nav(bench_ew).values,
                 label="同池等权基准", **EW_STYLE)
    axes[0].set_ylabel("Cumulative return (with cost)")
    axes[0].set_title(f"G4 vs G1 三种子 mean — backtest {BACKTEST_START}~{BACKTEST_END}（冻结宇宙）")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)

    g1_med_ret = daily_rets[g1_med_tag]
    for tag, r in daily_rets.items():
        arm, s = tag.split("_", 1)
        common = r.index.intersection(g1_med_ret.index)
        ex = (r.loc[common] - g1_med_ret.loc[common]).fillna(0)
        cum = (1 + ex).cumprod() - 1
        axes[1].plot(cum.index, cum.values, label=f"{tag} − {g1_med_tag}",
                     color=FAMILY_COLORS[arm][s],
                     linewidth=2.4 if arm == "G4" else 1.3)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Cumulative excess vs G1 中位 (with cost)")
    axes[1].set_xlabel("Decision date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g4_nav_backtest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(full_table: dict, judgment: dict) -> None:
    order = (
        [f"G4_s{s}_{v}" for s in SEEDS for v in ("min", "max", "last", "mean")]
        + [f"G1_s{s}_{v}" for s in SEEDS for v in ("min", "max", "last", "mean")]
    )
    aers, colors, labels = [], [], []
    for key in order:
        arm, s, v = key.split("_")
        aers.append(full_table[f"{s}@backtest"][v]["perf_ew"]["aer"] if arm == "G4"
                    else full_table[f"G1_{s}@backtest"][v]["perf_ew"]["aer"])
        colors.append(FAMILY_COLORS["G4" if arm == "G4" else "G1"][s])
        labels.append(key)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, aers, color=colors)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="0（市场基准）")
    ax.axhline(INCUMBENT_AER_EW, color="#08306b", linestyle=":", linewidth=1.4,
               label=f"在位者 s100 {INCUMBENT_AER_EW:+.2%}")
    ax.axhline(J2_THRESH_EW, color="#a50f15", linestyle="-.", linewidth=1.4,
               label=f"J2 增量线 {J2_THRESH_EW:+.2%}")
    ax.axhline(J5_THRESH_EW, color="#969696", linestyle="-.", linewidth=1.2,
               label=f"J5 否定线 {J5_THRESH_EW:+.2%}")
    for i, a in enumerate(aers):
        ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}", ha="center", fontsize=7)
    ax.set_ylabel("AER (等权基准, with cost)")
    ax.set_title("G4/G1 三种子四变体 AER(等权) — backtest（冻结宇宙）")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g4_aer_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    g4_signals, g1_signals, inc_signals = {}, {}, {}
    for window in ("backtest", "2025h2"):
        for s in SEEDS:
            p = DATA_DIR / f"s{s}" / f"daily_signals_{window}_{G4_ARMS[s]}_{{v}}.parquet"
            missing = [v for v in VARIANTS if not p.format(v=v).exists()]
            assert not missing, f"[{window}] G4 s{s} 信号缺失 {missing}——禁止评估（落盘前置纪律）"
            g4_signals[(window, s)] = {
                v: pd.read_parquet(p.format(v=v)) for v in VARIANTS
            }
        # 在位者（对拍锚）+ G1 族三种子（同宇宙复跑对照）
        inc_signals[window] = {"G1_mean_ref": pd.read_parquet(INCUMBENT_PARQUET[window])}
        for s in SEEDS:
            tpl = G1_SIGNAL_PARQUETS[window][s]
            missing = [v for v in VARIANTS if not Path(str(tpl).format(v=v)).exists()]
            assert not missing, f"[{window}] G1 族 s{s} 参照缺失 {missing}"
            g1_signals[(window, s)] = {
                v: pd.read_parquet(str(tpl).format(v=v)) for v in VARIANTS
            }

    full_table: dict = {}
    g4_perf: dict = {s: {} for s in SEEDS}
    g1_perf: dict = {s: {} for s in SEEDS}
    daily_rets_bt: dict[str, pd.Series] = {}

    for window in ("backtest", "2025h2"):
        universe_cols = sorted(set().union(*[
            set(pd.read_parquet(p).columns) for p in UNIVERSE_PARQUETS[window]]))
        logger.info(f"[{window}] 引擎宇宙：{len(universe_cols)} 列（冻结口径轮次并集）")

        # —— G4 三种子四变体 ——
        for s in SEEDS:
            signals = {f"{G4_ARMS[s]}_{v}": df
                       for v, df in g4_signals[(window, s)].items()}
            results, ctx = engine_window(signals, window, universe_cols)
            for v in VARIANTS:
                full_table.setdefault(f"{s}@{window}", {})[v] = {
                    "perf_ew": results[f"{G4_ARMS[s]}_{v}"]["perf_ew"],
                    "perf_idx": results[f"{G4_ARMS[s]}_{v}"]["perf_idx"],
                }
            g4_perf[s][window] = {
                "perf_ew": results[f"{G4_ARMS[s]}_mean"]["perf_ew"],
                "perf_idx": results[f"{G4_ARMS[s]}_mean"]["perf_idx"],
            }
            if window == "backtest":
                daily_rets_bt[f"G4_s{s}"] = ctx["rets"][f"{G4_ARMS[s]}_mean"]

        # —— 在位者对拍（backtest：引擎复跑须与冻结 +14.33% 一致）——
        results_ref, ctx = engine_window(
            {"G1_mean_ref": inc_signals[window]["G1_mean_ref"]}, window, universe_cols)
        rerun = results_ref["G1_mean_ref"]["perf_ew"]["aer"]
        full_table.setdefault(f"incumbent@{window}", {})["mean"] = {
            "perf_ew": results_ref["G1_mean_ref"]["perf_ew"],
            "perf_idx": results_ref["G1_mean_ref"]["perf_idx"],
        }
        if window == "backtest":
            assert abs(rerun - INCUMBENT_AER_EW) < 5e-4, (
                f"在位者引擎复跑 {rerun:+.4%} 与冻结 {INCUMBENT_AER_EW:+.4%} 不一致")
            logger.info(f"对拍在位者 G1_mean：复跑 {rerun:+.4%} vs 冻结 {INCUMBENT_AER_EW:+.4%} 一致")
            daily_rets_bt["bench_idx"] = ctx["bench_idx"]
            daily_rets_bt["bench_ew"] = ctx["bench_ew"]

        # —— G1 族三种子四变体（同宇宙引擎复跑对照，两窗）——
        for s in SEEDS:
            signals = {f"{G1_ARMS[s]}_{v}": df
                       for v, df in g1_signals[(window, s)].items()}
            results, ctx1 = engine_window(signals, window, universe_cols)
            for v in VARIANTS:
                full_table.setdefault(f"G1_{s}@{window}", {})[v] = {
                    "perf_ew": results[f"{G1_ARMS[s]}_{v}"]["perf_ew"],
                    "perf_idx": results[f"{G1_ARMS[s]}_{v}"]["perf_idx"],
                }
            g1_perf[s][window] = {
                "perf_ew": results[f"{G1_ARMS[s]}_mean"]["perf_ew"],
                "perf_idx": results[f"{G1_ARMS[s]}_mean"]["perf_idx"],
            }
            if window == "backtest":
                daily_rets_bt[f"G1_s{s}"] = ctx1["rets"][f"{G1_ARMS[s]}_mean"]

    judgment = judge(g4_perf, g1_perf)

    out = {
        "window_bounds": {k: list(v) for k, v in WINDOW_BOUNDS.items()},
        "benchmarks": {"index": "000300.SH 市值加权", "equal_weight": "同池等权（冻结宇宙）"},
        "criteria_thresholds": {
            "incumbent_aer_ew": INCUMBENT_AER_EW, "j2_ew": J2_THRESH_EW,
            "j2_idx": J2_THRESH_IDX, "j5_ew": J5_THRESH_EW,
        },
        "full_table": full_table,
        "verdict": judgment,
    }
    out_path = DATA_DIR / "g4_stage2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"G4 封盘判读落盘 {out_path}")

    plot_two_panel(
        {k: v for k, v in daily_rets_bt.items() if not k.startswith("bench")},
        daily_rets_bt["bench_idx"], daily_rets_bt["bench_ew"],
        g1_med_tag=f"G1_s{judgment['g1_three_seed_median']['median_seed'][1:]}",
    )
    plot_bar(full_table, judgment)

    print("=== 预注册判据 J1~J5 一次封盘 ===")
    for k in ("J1_survival", "J2_increment", "J3_neutral_band",
              "J4_cross_window", "J5_rejection", "registration"):
        v = judgment[k]
        head = v.get("passed", v.get("triggered"))
        print(f"[{k}] {'通过/触发' if head else '未通过/未触发'}：{v['note']}")
    print(f"中位种子：{judgment['median_seed']}（backtest AER 等权 "
          f"{judgment['median_backtest']['aer_ew']:+.2%} / 指数 "
          f"{judgment['median_backtest']['aer_idx']:+.2%}）")
    print(f"三种子中位对中位：G4 s{judgment['median_seed']} "
          f"{judgment['median_backtest']['aer_ew']:+.2%} vs G1 "
          f"{judgment['g1_three_seed_median']['median_seed']} "
          f"{judgment['g1_three_seed_median']['median_aer_ew']:+.2%}")
    print(f"措辞约束：{judgment['wording_constraint']['note']}")


if __name__ == "__main__":
    main()
