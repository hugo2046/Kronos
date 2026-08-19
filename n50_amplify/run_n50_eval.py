"""N50 阶段 3.3/3.4：引擎全表 + N1~N3 一次封盘判读（N50 计划 §2/§3，20260819）。

**预注册判据（跑前冻结，只定义在 mean）**：

| # | 判据 | 冻结定义 |
|---|---|---|
| N1 增量 | 中位种子 backtest AER(等权) ≥ +20.67%（在位者中位 +18.67%+2pp）且 AER(指数) ≥ +16.64% → "N 放大有增量，待前向确认"；注册行 = 以新增臂 G1N50 进 G3 登记表（新列 n50_sXXX_*，不改既有列/规则），登记推理成本 2.5× 由 cron 承受；因 2025H2 未跑（预算裁定），注册前置条件 = 下一自然月登记数据自证 |
| N2 中性 | 中位 ∈ (+16.67%, +20.67%) → "N=20 已充分，边际收益耗尽"，记录不注册不改任何东西（先验最可能结局） |
| N3 反常 | 中位 ≤ +16.67% → "更多采样反而更差 = 统计涨落警示"，如实记录并在结果文档讨论对 N=20 在位数字稳定性的含义，不注册 |

冻结措辞约束：中位对中位并列两族三种子分布；**逐种子配对差（N50−N20 同种子）
必列**——同权重同窗同 RNG 家族的最干净配对对照。
在位者对拍锚：G1 族同宇宙引擎复跑 s100 backtest mean = +14.33%（第 4 轮冻结）、
中位 s102 = +18.67%（等权）/ +14.64%（指数）（G4 stage2 复跑冻结）。

产出：n50_amplify/data/n50_results.json + n50_amplify/figures/ 两图。
纪律：3 种子信号全部落盘后才允许运行本脚本（main 入口断言）；
本脚本随 3.2 提交但**未运行**（阈值冻结先于任何 N50 数字存在）。
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

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"
R4 = PKG_DIR.parent / "finetune_suite" / "data"

# —— 预注册判据阈值（N50 计划 §2 表，跑前冻结）——
INCUMBENT_MEDIAN_EW = 0.1867  # 在位者中位 s102 backtest AER(等权)
INCUMBENT_MEDIAN_IDX = 0.1464  # 在位者中位 s102 backtest AER(指数)
INCUMBENT_S100_EW = 0.1433  # 第 4 轮冻结（对拍锚）
N1_THRESH_EW = 0.2067  # = 在位者中位 +2pp
N1_THRESH_IDX = 0.1664  # = 在位者中位(指数) +2pp
N3_THRESH_EW = 0.1667  # = 在位者中位 −2pp
N2_LO, N2_HI = 0.1667, 0.2067

SEEDS = (100, 101, 102)
N50_ARMS = {s: f"G1N50S{s}" for s in SEEDS}
G1_ARMS = {100: "G1", 101: "G2S101", 102: "G2S102"}  # G1 族三种子（在位者 N20 对拍行）

WINDOW_BOUNDS = {"backtest": (BACKTEST_START, BACKTEST_END)}

# 引擎宇宙（px/等权基准列集）——与 G4/G7 封盘同款冻结口径（在位者冻结数字的原始跑法）：
# backtest = 第 4 轮 G1 回测的 13 信号并集。
UNIVERSE_PARQUETS = {
    "backtest": [R4 / "g1" / f"daily_signals_backtest_G1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "daily_signals_backtest_M.parquet"],
}

# 在位者 G1 族 N20 信号 parquet（同宇宙引擎复跑对拍行 = 逐种子配对差的 N20 侧）
G1_SIGNAL_PARQUETS = {
    100: R4 / "g1" / "daily_signals_backtest_G1_{v}.parquet",
    101: R4 / "g2" / "s101" / "daily_signals_backtest_G2S101_{v}.parquet",
    102: R4 / "g2" / "s102" / "daily_signals_backtest_G2S102_{v}.parquet",
}

# 族配色（N50 红系 / G1 蓝系 / CSI300 黑虚线）——与 G4/G7 封盘配色连续
FAMILY_COLORS = {
    "G1N50": {"s100": "#a50f15", "s101": "#ef3b2c", "s102": "#fb6a4a"},
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
                  window=f"n50_{window}", backtest_start=start, backtest_end=end)
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


def judge(n50: dict, g1: dict) -> dict:
    """N1~N3 + 注册触发（阈值冻结于模块常量；一次判读）。

    :param n50: {seed: {"perf_ew": {...}, "perf_idx": {...}}}（N50 三种子 backtest）
    :param g1: 同构（G1 族三种子 N20 同宇宙复跑对拍行，中位对中位必列）
    """
    bt_ew = {s: n50[s]["perf_ew"]["aer"] for s in SEEDS}
    bt_idx = {s: n50[s]["perf_idx"]["aer"] for s in SEEDS}
    med = median_seed(bt_ew)
    med_ew, med_idx = bt_ew[med], bt_idx[med]

    n1 = med_ew >= N1_THRESH_EW and med_idx >= N1_THRESH_IDX
    n2 = (not n1) and (N2_LO < med_ew < N2_HI)
    n3 = med_ew <= N3_THRESH_EW
    register = n1  # 注册仅在 N1 触发（新列，不改既有列/规则）

    # G1 族三种子同宇宙中位（措辞约束：中位对中位，两族三种子分布并列呈现）
    g1_bt_ew = {s: g1[s]["perf_ew"]["aer"] for s in SEEDS}
    g1_med = median_seed(g1_bt_ew)
    spread = med_ew - g1_bt_ew[g1_med]

    # 逐种子配对差（N50−N20 同种子）——冻结措辞约束必列
    paired = {f"s{s}": bt_ew[s] - g1_bt_ew[s] for s in SEEDS}

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
        "paired_diff_n50_minus_n20_ew": paired,
        "N1_increment": {"passed": bool(n1), "note": (
            f"中位 s{med} AER(等权)={med_ew:+.2%} vs 门槛 {N1_THRESH_EW:+.2%}，"
            f"AER(指数)={med_idx:+.2%} vs 门槛 {N1_THRESH_IDX:+.2%} "
            + ("→ N 放大有增量，待前向确认" if n1 else "→ 未达增量门槛"))},
        "N2_neutral_band": {"triggered": bool(n2), "note": (
            f"中位 {med_ew:+.2%} ∈ ({N2_LO:+.2%}, {N2_HI:+.2%}) → "
            "N=20 已充分，边际收益耗尽，记录不注册不改任何东西（先验最可能结局）"
            if n2 else f"中位 {med_ew:+.2%} 不在中性带") if not n1 else "N1 已过，中性带不适用"},
        "N3_anomaly": {"triggered": bool(n3), "note": (
            f"中位 {med_ew:+.2%} ≤ {N3_THRESH_EW:+.2%}（在位者−2pp）→ "
            "更多采样反而更差 = 统计涨落警示，如实记录并在结果文档讨论"
            "对 N=20 在位数字稳定性的含义，不注册" if n3 else
            f"中位 {med_ew:+.2%} > {N3_THRESH_EW:+.2%} → 反常线未触发")},
        "registration": {"triggered": bool(register), "note": (
            "N1 通过 → 以新增臂 G1N50 进 G3 登记表（新列 n50_sXXX_*，不改既有列/规则；"
            "登记推理成本 2.5× 由 cron 承受；因 2025H2 未跑，注册前置条件 = "
            "下一自然月登记数据自证）" if register else
            "未触发注册（N1 未过）——不改 run_registry.py，不改既有登记列与 C1/C2/C3 规则")},
        "wording_constraint": {
            "g1_five_seed_backtest_ew": {
                "s100": 0.1433, "s101": 0.2906, "s102": 0.1867,
                "s103": 0.2036, "s104": 0.1074,
            },
            "note": "措辞约束（冻结）：中位对中位并列呈现两族三种子分布，"
                    "逐种子配对差（N50−N20 同种子）必列，不得以单点差宣称优劣",
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
                     linewidth=2.4 if arm == "G1N50" else 1.3)
    axes[0].plot(_nav(bench_idx).index, _nav(bench_idx).values,
                 label="CSI300(指数基准)", **CSI300_STYLE)
    axes[0].plot(_nav(bench_ew).index, _nav(bench_ew).values,
                 label="同池等权基准", **EW_STYLE)
    axes[0].set_ylabel("Cumulative return (with cost)")
    axes[0].set_title(
        f"G1N50 vs G1(N20) 三种子 mean — backtest {BACKTEST_START}~{BACKTEST_END}（冻结宇宙）"
    )
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
                     linewidth=2.4 if arm == "G1N50" else 1.3)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Cumulative excess vs G1 中位 (with cost)")
    axes[1].set_xlabel("Decision date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_n50_nav_backtest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(full_table: dict) -> None:
    order = (
        [f"G1N50_s{s}_{v}" for s in SEEDS for v in ("min", "max", "last", "mean")]
        + [f"G1_s{s}_{v}" for s in SEEDS for v in ("min", "max", "last", "mean")]
    )
    aers, colors, labels = [], [], []
    for key in order:
        arm, s, v = key.split("_")
        tab_key = (f"G1N50_{s}" if arm == "G1N50" else f"G1_{s}") + "@backtest"
        aers.append(full_table[tab_key][v]["perf_ew"]["aer"])
        colors.append(FAMILY_COLORS["G1N50" if arm == "G1N50" else "G1"][s])
        labels.append(key)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, aers, color=colors)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="0（市场基准）")
    ax.axhline(INCUMBENT_MEDIAN_EW, color="#08306b", linestyle=":", linewidth=1.4,
               label=f"在位者中位 s102 {INCUMBENT_MEDIAN_EW:+.2%}")
    ax.axhline(N1_THRESH_EW, color="#a50f15", linestyle="-.", linewidth=1.4,
               label=f"N1 增量线 {N1_THRESH_EW:+.2%}")
    ax.axhline(N3_THRESH_EW, color="#969696", linestyle="-.", linewidth=1.2,
               label=f"N3 反常线 {N3_THRESH_EW:+.2%}")
    for i, a in enumerate(aers):
        ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}", ha="center", fontsize=7)
    ax.set_ylabel("AER (等权基准, with cost)")
    ax.set_title("G1N50/G1(N20) 三种子四变体 AER(等权) — backtest（冻结宇宙）")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_n50_aer_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n50_signals, g1_signals = {}, {}
    for s in SEEDS:
        p = DATA_DIR / f"s{s}" / f"daily_signals_backtest_{N50_ARMS[s]}_{{v}}.parquet"
        missing = [v for v in VARIANTS if not Path(str(p).format(v=v)).exists()]
        assert not missing, f"N50 s{s} 信号缺失 {missing}——禁止评估（落盘前置纪律）"
        n50_signals[s] = {v: pd.read_parquet(str(p).format(v=v)) for v in VARIANTS}
        tpl = G1_SIGNAL_PARQUETS[s]
        missing = [v for v in VARIANTS if not Path(str(tpl).format(v=v)).exists()]
        assert not missing, f"G1 族 s{s} 参照缺失 {missing}"
        g1_signals[s] = {v: pd.read_parquet(str(tpl).format(v=v)) for v in VARIANTS}

    universe_cols = sorted(set().union(*[
        set(pd.read_parquet(p).columns) for p in UNIVERSE_PARQUETS["backtest"]]))
    logger.info(f"[backtest] 引擎宇宙：{len(universe_cols)} 列（冻结口径轮次并集）")

    full_table: dict = {}
    n50_perf: dict = {s: {} for s in SEEDS}
    g1_perf: dict = {s: {} for s in SEEDS}
    daily_rets_bt: dict[str, pd.Series] = {}

    # —— N50 三种子四变体 ——
    for s in SEEDS:
        signals = {f"{N50_ARMS[s]}_{v}": df for v, df in n50_signals[s].items()}
        results, ctx = engine_window(signals, "backtest", universe_cols)
        for v in VARIANTS:
            full_table.setdefault(f"G1N50_s{s}@backtest", {})[v] = {
                "perf_ew": results[f"{N50_ARMS[s]}_{v}"]["perf_ew"],
                "perf_idx": results[f"{N50_ARMS[s]}_{v}"]["perf_idx"],
            }
        n50_perf[s] = {
            "perf_ew": results[f"{N50_ARMS[s]}_mean"]["perf_ew"],
            "perf_idx": results[f"{N50_ARMS[s]}_mean"]["perf_idx"],
        }
        daily_rets_bt[f"G1N50_s{s}"] = ctx["rets"][f"{N50_ARMS[s]}_mean"]

    # —— 在位者对拍行：G1 族三种子 N20 四变体（同宇宙引擎复跑）——
    for s in SEEDS:
        signals = {f"{G1_ARMS[s]}_{v}": df for v, df in g1_signals[s].items()}
        results, ctx1 = engine_window(signals, "backtest", universe_cols)
        for v in VARIANTS:
            full_table.setdefault(f"G1_s{s}@backtest", {})[v] = {
                "perf_ew": results[f"{G1_ARMS[s]}_{v}"]["perf_ew"],
                "perf_idx": results[f"{G1_ARMS[s]}_{v}"]["perf_idx"],
            }
        g1_perf[s] = {
            "perf_ew": results[f"{G1_ARMS[s]}_mean"]["perf_ew"],
            "perf_idx": results[f"{G1_ARMS[s]}_mean"]["perf_idx"],
        }
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

    judgment = judge(n50_perf, g1_perf)

    out = {
        "window_bounds": {k: list(v) for k, v in WINDOW_BOUNDS.items()},
        "benchmarks": {"index": "000300.SH 市值加权", "equal_weight": "同池等权（冻结宇宙）"},
        "criteria_thresholds": {
            "incumbent_median_ew": INCUMBENT_MEDIAN_EW,
            "incumbent_median_idx": INCUMBENT_MEDIAN_IDX,
            "n1_ew": N1_THRESH_EW, "n1_idx": N1_THRESH_IDX,
            "n3_ew": N3_THRESH_EW,
        },
        "full_table": full_table,
        "verdict": judgment,
    }
    out_path = DATA_DIR / "n50_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"N50 封盘判读落盘 {out_path}")

    plot_two_panel(
        {k: v for k, v in daily_rets_bt.items() if not k.startswith("bench")},
        daily_rets_bt["bench_idx"], daily_rets_bt["bench_ew"],
        g1_med_tag=f"G1_s{judgment['g1_three_seed_median']['median_seed'][1:]}",
    )
    plot_bar(full_table)

    print("=== 预注册判据 N1~N3 一次封盘 ===")
    for k in ("N1_increment", "N2_neutral_band", "N3_anomaly", "registration"):
        v = judgment[k]
        head = v.get("passed", v.get("triggered"))
        print(f"[{k}] {'通过/触发' if head else '未通过/未触发'}：{v['note']}")
    print(f"中位种子：{judgment['median_seed']}（backtest AER 等权 "
          f"{judgment['median_backtest']['aer_ew']:+.2%} / 指数 "
          f"{judgment['median_backtest']['aer_idx']:+.2%}）")
    print(f"三种子中位对中位：G1N50 {judgment['median_seed']} "
          f"{judgment['median_backtest']['aer_ew']:+.2%} vs G1(N20) "
          f"{judgment['g1_three_seed_median']['median_seed']} "
          f"{judgment['g1_three_seed_median']['median_aer_ew']:+.2%}"
          f"（价差 {judgment['median_vs_median_spread']:+.2%}）")
    print("逐种子配对差 AER(等权) N50−N20（冻结措辞约束必列）：")
    for k, v in judgment["paired_diff_n50_minus_n20_ew"].items():
        print(f"  {k}: {v:+.2%}")


if __name__ == "__main__":
    main()
