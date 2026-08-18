"""G7 阶段 3.3/3.4：引擎全表 + K1~K4 一次封盘判读（G7 计划 §2/§3，20260818）。

**预注册判据（跑前冻结，只定义在 mean）**：

| # | 判据 | 冻结定义 |
|---|---|---|
| K1 增量 | 中位种子 backtest AER(等权) ≥ +20.67%（在位者中位 +18.67%+2pp）且 AER(指数) ≥ +16.64% → "短窗有增量，待前向确认"，注册进 G3 |
| K2 中性带 | 中位 ∈ (+16.67%, +20.67%) → "短窗中性"，记录不注册 |
| K3 否定 | 中位 ≤ +16.67%（在位者−2pp）→ "短窗在本土化底座上亦无增量，L/H 议题两代底座终审关闭"，不做窗口搜索 |
| K4 跨窗 | K1 通过者 2025H2 AER(等权) > 0 方可注册 |

冻结措辞约束：中位对中位并列呈现两族（W85 / G1）三种子分布。
在位者对拍锚：G1 族同宇宙引擎复跑 s100 backtest mean = +14.33%（第 4 轮冻结）、
中位 s102 = +18.67%（等权）/ +14.64%（指数）（G4 stage2 复跑冻结）。

产出：g7_shortwindow/data/g7_results.json + g7_shortwindow/figures/ 两图。
纪律：6 组信号全部落盘后才允许运行本脚本（main 入口断言）。
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

# —— 预注册判据阈值（G7 计划 §2 表，跑前冻结）——
INCUMBENT_MEDIAN_EW = 0.1867  # 在位者中位 s102 backtest AER(等权)
INCUMBENT_MEDIAN_IDX = 0.1464  # 在位者中位 s102 backtest AER(指数)
INCUMBENT_S100_EW = 0.1433  # 第 4 轮冻结（对拍锚）
K1_THRESH_EW = 0.2067  # = 在位者中位 +2pp
K1_THRESH_IDX = 0.1664  # = 在位者中位(指数) +2pp
K3_THRESH_EW = 0.1667  # = 在位者中位 −2pp
K2_LO, K2_HI = 0.1667, 0.2067

SEEDS = (100, 101, 102)
W85_ARMS = {s: f"W85S{s}" for s in SEEDS}
G1_ARMS = {100: "G1", 101: "G2S101", 102: "G2S102"}  # G1 族三种子（在位者对拍行）

WINDOW_BOUNDS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": WINDOW_DEFS["2025h2"],
}

# 引擎宇宙（px/等权基准列集）——与 G4 封盘同款冻结口径（在位者冻结数字的原始跑法）：
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

# 在位者 G1 族两窗信号 parquet（同宇宙引擎复跑对拍行）
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

# 族配色（W85 红系 / G1 蓝系 / CSI300 黑虚线）——与 G4 封盘配色连续
FAMILY_COLORS = {
    "W85": {"s100": "#a50f15", "s101": "#ef3b2c", "s102": "#fb6a4a"},
    "G1": {"s100": "#08306b", "s101": "#2171b5", "s102": "#6baed6"},
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
                  window=f"g7_{window}", backtest_start=start, backtest_end=end)
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


def judge(w85: dict, g1: dict) -> dict:
    """K1~K4 + 注册触发（阈值冻结于模块常量；一次判读）。

    :param w85: {seed: {window: {"perf_ew": {...}, "perf_idx": {...}}}}（W85 三种子）
    :param g1: 同构（G1 族三种子同宇宙复跑对拍行，中位对中位必列）
    """
    bt_ew = {s: w85[s]["backtest"]["perf_ew"]["aer"] for s in SEEDS}
    bt_idx = {s: w85[s]["backtest"]["perf_idx"]["aer"] for s in SEEDS}
    med = median_seed(bt_ew)
    med_ew, med_idx = bt_ew[med], bt_idx[med]
    h2_ew_med = w85[med]["2025h2"]["perf_ew"]["aer"]

    k1 = med_ew >= K1_THRESH_EW and med_idx >= K1_THRESH_IDX
    k2 = (not k1) and (K2_LO < med_ew < K2_HI)
    k3 = med_ew <= K3_THRESH_EW
    k4 = k1 and (h2_ew_med > 0)
    register = k1 and k4

    # G1 族三种子同宇宙中位（措辞约束：中位对中位，两族三种子分布并列呈现）
    g1_bt_ew = {s: g1[s]["backtest"]["perf_ew"]["aer"] for s in SEEDS}
    g1_med = median_seed(g1_bt_ew)
    spread = med_ew - g1_bt_ew[g1_med]

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
        "median_vs_median_spread": spread,
        "K1_increment": {"passed": bool(k1), "note": (
            f"中位 s{med} AER(等权)={med_ew:+.2%} vs 门槛 {K1_THRESH_EW:+.2%}，"
            f"AER(指数)={med_idx:+.2%} vs 门槛 {K1_THRESH_IDX:+.2%} "
            + ("→ 短窗有增量，待前向确认" if k1 else "→ 未达增量门槛"))},
        "K2_neutral_band": {"triggered": bool(k2), "note": (
            f"中位 {med_ew:+.2%} ∈ ({K2_LO:+.2%}, {K2_HI:+.2%}) → 短窗中性，记录不注册"
            if k2 else f"中位 {med_ew:+.2%} 不在中性带") if not k1 else "K1 已过，中性带不适用"},
        "K3_rejection": {"triggered": bool(k3), "note": (
            f"中位 {med_ew:+.2%} ≤ {K3_THRESH_EW:+.2%}（在位者−2pp）→ 短窗在本土化底座上"
            "亦无增量，L/H 议题两代底座终审关闭，不做窗口搜索" if k3 else
            f"中位 {med_ew:+.2%} > {K3_THRESH_EW:+.2%} → 否定线未触发")},
        "K4_cross_window": {"passed": bool(k4), "applicable": bool(k1), "note": (
            f"中位 s{med} 2025H2 AER(等权)={h2_ew_med:+.2%} "
            + ("→ 跨窗存活，可注册" if k4 else "→ 未跨窗") if k1 else "K1 未过，K4 不适用")},
        "registration": {"triggered": bool(register), "note": (
            "K1∧K4 通过 → W85 中位种子信号注册进 G3 登记表" if register else
            "未触发注册（K1∧K4 至少一项未过）")},
        "wording_constraint": {
            "g1_five_seed_backtest_ew": {
                "s100": 0.1433, "s101": 0.2906, "s102": 0.1867,
                "s103": 0.2036, "s104": 0.1074,
            },
            "note": "措辞约束（冻结）：中位对中位并列呈现两族三种子分布，"
                    "不得以单点差宣称优劣",
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
                     linewidth=2.4 if arm == "W85" else 1.3)
    axes[0].plot(_nav(bench_idx).index, _nav(bench_idx).values,
                 label="CSI300(指数基准)", **CSI300_STYLE)
    axes[0].plot(_nav(bench_ew).index, _nav(bench_ew).values,
                 label="同池等权基准", **EW_STYLE)
    axes[0].set_ylabel("Cumulative return (with cost)")
    axes[0].set_title(f"W85 vs G1 三种子 mean — backtest {BACKTEST_START}~{BACKTEST_END}（冻结宇宙）")
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
                     linewidth=2.4 if arm == "W85" else 1.3)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Cumulative excess vs G1 中位 (with cost)")
    axes[1].set_xlabel("Decision date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g7_nav_backtest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(full_table: dict) -> None:
    order = (
        [f"W85_s{s}_{v}" for s in SEEDS for v in ("min", "max", "last", "mean")]
        + [f"G1_s{s}_{v}" for s in SEEDS for v in ("min", "max", "last", "mean")]
    )
    aers, colors, labels = [], [], []
    for key in order:
        arm, s, v = key.split("_")
        tab_key = (f"W85_{s}" if arm == "W85" else f"G1_{s}") + "@backtest"
        aers.append(full_table[tab_key][v]["perf_ew"]["aer"])
        colors.append(FAMILY_COLORS["W85" if arm == "W85" else "G1"][s])
        labels.append(key)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, aers, color=colors)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="0（市场基准）")
    ax.axhline(INCUMBENT_MEDIAN_EW, color="#08306b", linestyle=":", linewidth=1.4,
               label=f"在位者中位 s102 {INCUMBENT_MEDIAN_EW:+.2%}")
    ax.axhline(K1_THRESH_EW, color="#a50f15", linestyle="-.", linewidth=1.4,
               label=f"K1 增量线 {K1_THRESH_EW:+.2%}")
    ax.axhline(K3_THRESH_EW, color="#969696", linestyle="-.", linewidth=1.2,
               label=f"K3 否定线 {K3_THRESH_EW:+.2%}")
    for i, a in enumerate(aers):
        ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}", ha="center", fontsize=7)
    ax.set_ylabel("AER (等权基准, with cost)")
    ax.set_title("W85/G1 三种子四变体 AER(等权) — backtest（冻结宇宙）")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g7_aer_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    w85_signals, g1_signals = {}, {}
    for window in ("backtest", "2025h2"):
        for s in SEEDS:
            p = DATA_DIR / f"s{s}" / f"daily_signals_{window}_{W85_ARMS[s]}_{{v}}.parquet"
            missing = [v for v in VARIANTS if not Path(str(p).format(v=v)).exists()]
            assert not missing, f"[{window}] W85 s{s} 信号缺失 {missing}——禁止评估（落盘前置纪律）"
            w85_signals[(window, s)] = {
                v: pd.read_parquet(str(p).format(v=v)) for v in VARIANTS
            }
            tpl = G1_SIGNAL_PARQUETS[window][s]
            missing = [v for v in VARIANTS if not Path(str(tpl).format(v=v)).exists()]
            assert not missing, f"[{window}] G1 族 s{s} 参照缺失 {missing}"
            g1_signals[(window, s)] = {
                v: pd.read_parquet(str(tpl).format(v=v)) for v in VARIANTS
            }

    full_table: dict = {}
    w85_perf: dict = {s: {} for s in SEEDS}
    g1_perf: dict = {s: {} for s in SEEDS}
    daily_rets_bt: dict[str, pd.Series] = {}

    for window in ("backtest", "2025h2"):
        universe_cols = sorted(set().union(*[
            set(pd.read_parquet(p).columns) for p in UNIVERSE_PARQUETS[window]]))
        logger.info(f"[{window}] 引擎宇宙：{len(universe_cols)} 列（冻结口径轮次并集）")

        # —— W85 三种子四变体 ——
        for s in SEEDS:
            signals = {f"{W85_ARMS[s]}_{v}": df
                       for v, df in w85_signals[(window, s)].items()}
            results, ctx = engine_window(signals, window, universe_cols)
            for v in VARIANTS:
                full_table.setdefault(f"W85_s{s}@{window}", {})[v] = {
                    "perf_ew": results[f"{W85_ARMS[s]}_{v}"]["perf_ew"],
                    "perf_idx": results[f"{W85_ARMS[s]}_{v}"]["perf_idx"],
                }
            w85_perf[s][window] = {
                "perf_ew": results[f"{W85_ARMS[s]}_mean"]["perf_ew"],
                "perf_idx": results[f"{W85_ARMS[s]}_mean"]["perf_idx"],
            }
            if window == "backtest":
                daily_rets_bt[f"W85_s{s}"] = ctx["rets"][f"{W85_ARMS[s]}_mean"]

        # —— 在位者对拍行：G1 族三种子四变体（同宇宙引擎复跑）——
        for s in SEEDS:
            signals = {f"{G1_ARMS[s]}_{v}": df
                       for v, df in g1_signals[(window, s)].items()}
            results, ctx1 = engine_window(signals, window, universe_cols)
            for v in VARIANTS:
                full_table.setdefault(f"G1_s{s}@{window}", {})[v] = {
                    "perf_ew": results[f"{G1_ARMS[s]}_{v}"]["perf_ew"],
                    "perf_idx": results[f"{G1_ARMS[s]}_{v}"]["perf_idx"],
                }
            g1_perf[s][window] = {
                "perf_ew": results[f"{G1_ARMS[s]}_mean"]["perf_ew"],
                "perf_idx": results[f"{G1_ARMS[s]}_mean"]["perf_idx"],
            }
            if window == "backtest":
                daily_rets_bt[f"G1_s{s}"] = ctx1["rets"][f"{G1_ARMS[s]}_mean"]
                if s == 102:
                    daily_rets_bt["bench_idx"] = ctx1["bench_idx"]
                    daily_rets_bt["bench_ew"] = ctx1["bench_ew"]

    # —— 在位者对拍锚（backtest mean：s100=+14.33% 第 4 轮冻结；中位 s102=+18.67%）——
    rerun_s100 = full_table["G1_s100@backtest"]["mean"]["perf_ew"]["aer"]
    rerun_s102 = full_table["G1_s102@backtest"]["mean"]["perf_ew"]["aer"]
    assert abs(rerun_s100 - INCUMBENT_S100_EW) < 5e-4, (
        f"在位者 s100 引擎复跑 {rerun_s100:+.4%} 与冻结 {INCUMBENT_S100_EW:+.4%} 不一致")
    assert abs(rerun_s102 - INCUMBENT_MEDIAN_EW) < 5e-4, (
        f"在位者中位 s102 引擎复跑 {rerun_s102:+.4%} 与冻结 {INCUMBENT_MEDIAN_EW:+.4%} 不一致")
    logger.info(f"对拍在位者：s100 复跑 {rerun_s100:+.4%} vs 冻结 {INCUMBENT_S100_EW:+.4%}；"
                f"中位 s102 复跑 {rerun_s102:+.4%} vs 冻结 {INCUMBENT_MEDIAN_EW:+.4%} 一致")

    judgment = judge(w85_perf, g1_perf)

    out = {
        "window_bounds": {k: list(v) for k, v in WINDOW_BOUNDS.items()},
        "benchmarks": {"index": "000300.SH 市值加权", "equal_weight": "同池等权（冻结宇宙）"},
        "criteria_thresholds": {
            "incumbent_median_ew": INCUMBENT_MEDIAN_EW,
            "incumbent_median_idx": INCUMBENT_MEDIAN_IDX,
            "k1_ew": K1_THRESH_EW, "k1_idx": K1_THRESH_IDX,
            "k3_ew": K3_THRESH_EW,
        },
        "full_table": full_table,
        "verdict": judgment,
    }
    out_path = DATA_DIR / "g7_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"G7 封盘判读落盘 {out_path}")

    plot_two_panel(
        {k: v for k, v in daily_rets_bt.items() if not k.startswith("bench")},
        daily_rets_bt["bench_idx"], daily_rets_bt["bench_ew"],
        g1_med_tag=f"G1_s{judgment['g1_three_seed_median']['median_seed'][1:]}",
    )
    plot_bar(full_table)

    print("=== 预注册判据 K1~K4 一次封盘 ===")
    for k in ("K1_increment", "K2_neutral_band", "K3_rejection",
              "K4_cross_window", "registration"):
        v = judgment[k]
        head = v.get("passed", v.get("triggered"))
        print(f"[{k}] {'通过/触发' if head else '未通过/未触发'}：{v['note']}")
    print(f"中位种子：{judgment['median_seed']}（backtest AER 等权 "
          f"{judgment['median_backtest']['aer_ew']:+.2%} / 指数 "
          f"{judgment['median_backtest']['aer_idx']:+.2%}）")
    print(f"三种子中位对中位：W85 {judgment['median_seed']} "
          f"{judgment['median_backtest']['aer_ew']:+.2%} vs G1 "
          f"{judgment['g1_three_seed_median']['median_seed']} "
          f"{judgment['g1_three_seed_median']['median_aer_ew']:+.2%}"
          f"（价差 {judgment['median_vs_median_spread']:+.2%}）")


if __name__ == "__main__":
    main()
