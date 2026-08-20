"""C4 阶段 3.3/3.4：引擎全表 + T1~T5 一次封盘判读（C4 计划 §2/§3，20260820）。

**预注册判据（跑前冻结，只定义在 c4 变体 vs G1 mean）**：

| # | 判据 | 冻结定义 |
|---|---|---|
| T1 存活 | C4 中位种子全期 AER(等权) > 0 且 AER(指数) > 0 |
| T2 配对呈现 | 同种子配对差（C4−G1）全期与分窗必列；差距在 ±26pp 噪声底内一律措辞"不可判"，仅方向性描述 |
| T3 显著改善 | C4 中位 − G1 中位（全期 AER 等权）> 26pp → "时间维处理显著增值"（预期极难达到） |
| T4 换手红利 | C4 族中位日均换手 < G1 族中位的 2/3 → "平滑降换手成立"（独立判读） |
| T5 否定 | C4 中位全期 AER(等权) ≤ 0 → "三值化+平滑损害信号，该形态关闭"，不做阈值/半衰期搜索 |
| 注册 | T1 且 (T3 或 T4 至少一项) → C4 注册进 G3 登记表（新列，规则文档另发） |

冻结执行口径（跑前定案，结果文档 §执行披露 照列）：
    - 评估调仓日 = 合并 260 日信号剔除预热前 30 日（=230 日：2025h2 分窗 96 +
      backtest 分窗 134；分窗为全期连续运行的日期切片，持仓跨界结转）；
    - G1 对拍行 = 同种子原始 mean 信号、同评估窗/同引擎/同基准（同日同池配对）；
    - 中位种子按各自族全期 AER(等权) 排序取中；换手 T4 用族中位（3 取中）日均换手；
    - 引擎宇宙 = 两窗冻结宇宙（backtest 13 信号并集与 2025h2 9 信号并集，实测
      同 338 列）的并集；引擎参数 canonical（csi300/k=50/n=5/min_hold=5/单边
      15bp）零改动——直接调 paper_replication.engine 冻结原语；
    - 在位者对拍锚：backtest 单窗 G1 mean 同宇宙复跑 s100=+14.33%/中位
      s102=+18.67%（第 4 轮/G4 stage2 冻结）。

产出：c4_temporal/data/c4_results.json + c4_temporal/figures/ 两图。
纪律：三种子 C4 信号全部落盘后才允许运行本脚本（main 入口断言）；
判读一次封盘；T5 触发不做任何阈值/半衰期搜索。
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
from baseline_suite.pipeline import build_dual_benchmarks
from baseline_suite.signal import build_px_tradeable
from c4_temporal.transform import (
    MERGED_WINDOW,
    WARMUP_DAYS,
    eval_rebalance_dates,
    load_g1_mean_merged,
)
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from paper_replication.engine import (
    EngineConfig,
    TradeLog,
    attach_benchmark,
    compute_perf,
    run_portfolio,
)

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"
R4 = PKG_DIR.parent / "finetune_suite" / "data"

# —— 预注册判据阈值（C4 计划 §2 表，跑前冻结）——
NOISE_FLOOR = 0.26  # ±26pp 噪声底（134 日窗 AER 标准误，§0 冻结）
TURNOVER_RATIO = 2 / 3  # T4：C4 族中位换手 < G1 族中位的 2/3
INCUMBENT_S100_EW = 0.1433  # 第 4 轮冻结（对拍锚）
INCUMBENT_MEDIAN_EW = 0.1867  # 中位 s102（G4 stage2 复跑冻结）

SEEDS = (100, 101, 102)
C4_ARMS = {s: f"C4S{s}" for s in SEEDS}
G1_ARMS = {100: "G1", 101: "G2S101", 102: "G2S102"}  # G1 族三种子（对拍行）

WINDOW_BOUNDS = {
    "merged": MERGED_WINDOW,
    "backtest": (BACKTEST_START, BACKTEST_END),  # 仅对拍锚复跑用
}
SUBWINDOW_SPLIT = "2026-01-01"  # 分窗切点（backtest 窗首日）

# 引擎宇宙（px/等权基准列集）——与 G4/G7/N50 封盘同款冻结口径：
# backtest = 第 4 轮 13 信号并集；2025h2 = G0 回测 9 信号并集（实测两集同 338
# 列）；merged 取二者并集。
UNIVERSE_PARQUETS = {
    "backtest": [R4 / "g1" / f"daily_signals_backtest_G1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "daily_signals_backtest_M.parquet"],
    "2025h2": [R4 / "g0" / f"daily_signals_2025h2_G0_{v}.parquet" for v in VARIANTS]
    + [R4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "g0" / "daily_signals_2025h2_M.parquet"],
}

# G1 族三种子 backtest 单窗 mean parquet（对拍锚信号源；只读）
G1_ANCHOR_PARQUETS = {
    100: R4 / "g1" / "daily_signals_backtest_G1_mean.parquet",
    101: R4 / "g2" / "s101" / "daily_signals_backtest_G2S101_mean.parquet",
    102: R4 / "g2" / "s102" / "daily_signals_backtest_G2S102_mean.parquet",
}

# 族配色（C4 红系 / G1 蓝系 / CSI300 黑虚线 / 等权 黑点线）——与 G4/G7/N50 连续
FAMILY_COLORS = {
    "C4": {"s100": "#a50f15", "s101": "#ef3b2c", "s102": "#fb6a4a"},
    "G1": {"s100": "#08306b", "s101": "#2171b5", "s102": "#6baed6"},
}
CSI300_STYLE = dict(color="black", linestyle="--", linewidth=1.4)
EW_STYLE = dict(color="black", linestyle=":", linewidth=1.2)


# ============================================================
# 引擎（冻结宇宙；直接调 paper_replication.engine 冻结原语，引擎零改动）
# ============================================================
def engine_window(
    signals: dict[str, pd.DataFrame], window: str, bounds: tuple[str, str],
    universe_cols: list[str],
) -> tuple[dict, dict]:
    """单窗引擎：全部信号共用同 px/双基准（同日同池可比）。

    与 ``baseline_suite.pipeline.run_group`` 同构（同 reindex/同 attach_benchmark/
    同 compute_perf），额外保留 daily_ret 与 trades 供分窗切片复算。

    :param bounds: 引擎窗 ``(start, end)``——必须与信号调仓日一致（引擎第 0 日
        建满 top-k；若首行信号全 NaN 将永不建仓，断言拦截）。
    :return: ``(results, ctx)``——results[name] 含 perf_idx/perf_ew（PerfStats）、
        daily_ret、trades；ctx 含 beta_gap/两基准/各臂 daily_ret。
    """
    from kronos_qlib import QlibProvider

    start, end = bounds
    cfg = replace(
        BaselineConfig.load(window="oos"),
        window=f"c4_{window}", backtest_start=start, backtest_end=end,
    )
    provider = QlibProvider(cfg.pool, start, end)
    rebalances = pd.DatetimeIndex(signals[list(signals)[0]].index)
    px, trd = build_px_tradeable(provider, cfg, rebalances, universe_cols)
    assert px.index.equals(rebalances), (
        f"px 索引（{px.index[0].date()}~{px.index[-1].date()}，{len(px)} 日）与信号"
        f"调仓日不一致——引擎首行须为有效信号行（建仓日）"
    )
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    ec = EngineConfig(top_k=cfg.top_k, drop_n=cfg.drop_n, min_hold=cfg.min_hold,
                      cost_bps=cfg.cost_bps)
    results: dict = {}
    for name, sig in signals.items():
        sig_r = sig.reindex(index=px.index, columns=px.columns)
        daily_ret, _, trades = run_portfolio(sig_r, px, trd, cfg=ec)
        perf_idx = compute_perf(attach_benchmark(daily_ret, bench_idx), trades, name=name)
        perf_ew = compute_perf(attach_benchmark(daily_ret, bench_ew), trades, name=name)
        logger.info(
            f"[{name}] AER(指数)={perf_idx.aer:+.2%} IR={perf_idx.ir:+.3f} | "
            f"AER(等权)={perf_ew.aer:+.2%} IR={perf_ew.ir:+.3f} | "
            f"MDD={perf_idx.max_drawdown:.2%} 日均换手={perf_idx.daily_turnover:.2%} "
            f"(n={perf_idx.n_days})"
        )
        results[name] = {"perf_idx": perf_idx, "perf_ew": perf_ew,
                         "daily_ret": daily_ret, "trades": trades}
    return results, {"beta_gap": beta_gap,
                     "bench_idx": bench_idx, "bench_ew": bench_ew}


def perf_slice(
    daily_ret: pd.Series, bench: pd.Series, trades: TradeLog,
    mask_dates: pd.DatetimeIndex, name: str,
):
    """全期连续运行的日期切片绩效（持仓跨界结转；切片只作用于超额序列与换手日志）。"""
    excess = attach_benchmark(daily_ret, bench).reindex(mask_dates)
    tl = TradeLog()
    keep = set(mask_dates)
    tl.rows = [r for r in trades.rows if r["date"] in keep]
    return compute_perf(excess, tl, name=name)


# ============================================================
# 判据（纯函数）
# ============================================================
def median_seed(perf_by_seed: dict[int, float]) -> int:
    """按全期 AER(等权) 排序取中位种子（3 种子 → 第 2 名）。"""
    ranked = sorted(perf_by_seed, key=lambda s: perf_by_seed[s])
    return ranked[1]


def judge(c4: dict, g1: dict) -> dict:
    """T1~T5 + 注册触发（阈值冻结于模块常量；一次判读）。

    :param c4: {seed: {window: {"perf_ew": dict, "perf_idx": dict}}}（C4 三种子）
    :param g1: 同构（G1 族三种子对拍行）
    """
    full_ew = {s: c4[s]["full"]["perf_ew"]["aer"] for s in SEEDS}
    full_idx = {s: c4[s]["full"]["perf_idx"]["aer"] for s in SEEDS}
    med = median_seed(full_ew)
    med_ew, med_idx = full_ew[med], full_idx[med]

    g1_full_ew = {s: g1[s]["full"]["perf_ew"]["aer"] for s in SEEDS}
    g1_med = median_seed(g1_full_ew)
    spread = med_ew - g1_full_ew[g1_med]

    t1 = med_ew > 0 and med_idx > 0
    t3 = spread > NOISE_FLOOR
    t5 = med_ew <= 0
    to_c4 = sorted(c4[s]["full"]["perf_idx"]["daily_turnover"] for s in SEEDS)[1]
    to_g1 = sorted(g1[s]["full"]["perf_idx"]["daily_turnover"] for s in SEEDS)[1]
    t4 = to_c4 < TURNOVER_RATIO * to_g1
    register = t1 and (t3 or t4)

    # T2：同种子配对差全期与分窗必列 + ±26pp 不可判措辞（族中位配对差 = 3 取中）
    paired, wording = {}, {}
    for w in ("full", "2025h2", "backtest"):
        paired[w] = {
            f"s{s}": c4[s][w]["perf_ew"]["aer"] - g1[s][w]["perf_ew"]["aer"] for s in SEEDS
        }
        fam_spread = sorted(paired[w].values())[1]
        wording[w] = (
            f"族中位配对差 {fam_spread:+.2%}，|Δ| ≤ 26pp 噪声底 → 不可判，仅方向性描述"
            if abs(fam_spread) <= NOISE_FLOOR else
            f"族中位配对差 {fam_spread:+.2%} 超噪声底（仍按 §2 配对措辞呈现）"
        )

    return {
        "median_seed": f"s{med}",
        "median_full": {
            "aer_ew": med_ew, "aer_idx": med_idx,
            "by_seed_ew": {f"s{s}": full_ew[s] for s in SEEDS},
            "by_seed_idx": {f"s{s}": full_idx[s] for s in SEEDS},
        },
        "g1_three_seed_median": {
            "median_seed": f"s{g1_med}",
            "by_seed_ew": {f"s{s}": g1_full_ew[s] for s in SEEDS},
            "median_aer_ew": g1_full_ew[g1_med],
        },
        "median_vs_median_spread": spread,
        "T1_survival": {"passed": bool(t1), "note": (
            f"中位 s{med} 全期 AER(等权)={med_ew:+.2%} / AER(指数)={med_idx:+.2%} → "
            + ("双正存活" if t1 else "未双正 → 存活失败"))},
        "T2_paired_presentation": {
            "paired_diff_c4_minus_g1_ew": paired,
            "noise_floor_wording": wording,
            "note": "同种子配对差（C4−G1）全期与分窗必列；±26pp 内一律'不可判'",
        },
        "T3_significant_gain": {"passed": bool(t3), "note": (
            f"中位对中位价差 {spread:+.2%} vs 噪声底 +{NOISE_FLOOR:.0%} → "
            + ("时间维处理显著增值（预期极难达到）" if t3 else "未超噪声底"))},
        "T4_turnover_dividend": {"passed": bool(t4), "note": (
            f"C4 族中位日均换手 {to_c4:.2%} vs G1 族中位 {to_g1:.2%} 的 2/3 "
            f"（{TURNOVER_RATIO * to_g1:.2%}）→ "
            + ("平滑降换手成立（独立判读）" if t4 else "降换手未达 2/3 线"))},
        "T5_rejection": {"triggered": bool(t5), "note": (
            f"中位 s{med} 全期 AER(等权)={med_ew:+.2%} → "
            + ("≤ 0：三值化+平滑损害信号，时间维处理该形态关闭，不做阈值/半衰期搜索"
               if t5 else "> 0：否定线未触发"))},
        "registration": {"triggered": bool(register), "note": (
            "T1 ∧ (T3∨T4) → C4 注册进 G3 登记表（新列，规则文档另发）" if register else
            "未触发注册——不改 run_registry.py，不改既有登记列与 C1/C2/C3 规则")},
        "turnover": {
            "c4_by_seed": {f"s{s}": c4[s]["full"]["perf_idx"]["daily_turnover"] for s in SEEDS},
            "g1_by_seed": {f"s{s}": g1[s]["full"]["perf_idx"]["daily_turnover"] for s in SEEDS},
            "c4_family_median": to_c4, "g1_family_median": to_g1,
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
                     linewidth=2.4 if arm == "C4" else 1.3)
    axes[0].plot(_nav(bench_idx).index, _nav(bench_idx).values,
                 label="CSI300(指数基准)", **CSI300_STYLE)
    axes[0].plot(_nav(bench_ew).index, _nav(bench_ew).values,
                 label="同池等权基准", **EW_STYLE)
    axes[0].set_ylabel("Cumulative return (with cost)")
    axes[0].set_title(
        f"C4 vs G1(直接用信号) 三种子 — 合并评估窗（预热 30 日已烧，冻结宇宙）"
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
                     linewidth=2.4 if arm == "C4" else 1.3)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Cumulative excess vs G1 中位 (with cost)")
    axes[1].set_xlabel("Decision date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_c4_nav_merged.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(full_table: dict) -> None:
    windows = ("full", "2025h2", "backtest")
    labels, aers, colors = [], [], []
    for w in windows:
        for i, (fam, s) in enumerate((f, s) for f in ("C4", "G1") for s in SEEDS):
            aers.append(full_table[f"{fam}_s{s}@{w}"]["perf_ew"]["aer"])
            colors.append(FAMILY_COLORS[fam][f"s{s}"])
            labels.append((f"[{w}]\n" if i == 0 else "") + f"{fam}_s{s}")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(aers)), aers, color=colors)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="0（市场基准）")
    for i, a in enumerate(aers):
        ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}",
                ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(aers)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("AER (等权基准, with cost)")
    ax.set_title("C4/G1 三种子 AER(等权) — 全期/2025H2/backtest 分窗（合并连续运行切片）")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_c4_aer_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # —— 落盘前置断言（评估解锁条件）——
    for s in SEEDS:
        p = DATA_DIR / f"s{s}" / f"daily_signals_merged_{C4_ARMS[s]}_c4.parquet"
        assert p.exists(), f"C4 s{s} 信号缺失 {p}——禁止评估（落盘前置纪律）"

    universe_cols = sorted(set().union(*[
        set(pd.read_parquet(p).columns)
        for w in ("backtest", "2025h2") for p in UNIVERSE_PARQUETS[w]
    ]))
    logger.info(f"[merged] 引擎宇宙：{len(universe_cols)} 列（两窗冻结口径并集）")

    # —— 合并窗（评估调仓日 = 预热后 230 日）：C4×3 + G1 对拍×3 同池同日 ——
    signals: dict[str, pd.DataFrame] = {}
    for s in SEEDS:
        c4_full = pd.read_parquet(
            DATA_DIR / f"s{s}" / f"daily_signals_merged_{C4_ARMS[s]}_c4.parquet")
        signals[f"{C4_ARMS[s]}_c4"] = c4_full.loc[eval_rebalance_dates(c4_full.index)]
        g1_merged = load_g1_mean_merged(s)
        signals[f"{G1_ARMS[s]}_mean"] = g1_merged.loc[eval_rebalance_dates(g1_merged.index)]

    # 引擎窗 = 评估窗（首日后第 30 个交易日起~2026-07-24）——预热 30 日完全不进
    # 引擎（引擎第 0 日建满 top-k，预热行进入将导致永不建仓）。
    eval_dates_all = eval_rebalance_dates(c4_full.index)
    results, ctx = engine_window(
        signals, "merged",
        (str(eval_dates_all[0].date()), MERGED_WINDOW[1]), universe_cols,
    )

    eval_dates = pd.DatetimeIndex(signals[f"{C4_ARMS[100]}_c4"].index)
    masks = {
        "full": eval_dates,
        "2025h2": eval_dates[eval_dates < SUBWINDOW_SPLIT],
        "backtest": eval_dates[eval_dates >= SUBWINDOW_SPLIT],
    }
    logger.info(
        f"评估窗 {len(masks['full'])} 日（2025h2 {len(masks['2025h2'])} + "
        f"backtest {len(masks['backtest'])}，预热 {WARMUP_DAYS} 日已烧，"
        f"评估首日 {masks['full'][0].date()}）"
    )

    bench_idx, bench_ew = ctx["bench_idx"], ctx["bench_ew"]
    c4_perf: dict = {s: {} for s in SEEDS}
    g1_perf: dict = {s: {} for s in SEEDS}
    full_table: dict = {}
    for s in SEEDS:
        for fam, tag, store in (("C4", f"{C4_ARMS[s]}_c4", c4_perf),
                                ("G1", f"{G1_ARMS[s]}_mean", g1_perf)):
            r = results[tag]
            for w, m in masks.items():
                store[s][w] = {
                    "perf_ew": perf_slice(r["daily_ret"], bench_ew, r["trades"], m,
                                          f"{fam}_s{s}@{w}").to_dict(),
                    "perf_idx": perf_slice(r["daily_ret"], bench_idx, r["trades"], m,
                                           f"{fam}_s{s}@{w}").to_dict(),
                }
                full_table[f"{fam}_s{s}@{w}"] = store[s][w]

    # —— 对拍锚：backtest 单窗 G1 mean 同宇宙复跑（引擎/数据漂移门禁）——
    anchor_signals = {
        f"{G1_ARMS[s]}_mean": pd.read_parquet(G1_ANCHOR_PARQUETS[s]) for s in SEEDS
    }
    anchor_res, _ = engine_window(
        anchor_signals, "backtest", WINDOW_BOUNDS["backtest"], universe_cols
    )
    rerun_s100 = anchor_res[f"{G1_ARMS[100]}_mean"]["perf_ew"].aer
    rerun_s102 = anchor_res[f"{G1_ARMS[102]}_mean"]["perf_ew"].aer
    assert abs(rerun_s100 - INCUMBENT_S100_EW) < 5e-4, (
        f"对拍锚 s100 复跑 {rerun_s100:+.4%} 与冻结 {INCUMBENT_S100_EW:+.4%} 不一致")
    assert abs(rerun_s102 - INCUMBENT_MEDIAN_EW) < 5e-4, (
        f"对拍锚中位 s102 复跑 {rerun_s102:+.4%} 与冻结 {INCUMBENT_MEDIAN_EW:+.4%} 不一致")
    logger.info(f"对拍锚：s100 复跑 {rerun_s100:+.4%} vs 冻结 {INCUMBENT_S100_EW:+.4%}；"
                f"s102 复跑 {rerun_s102:+.4%} vs 冻结 {INCUMBENT_MEDIAN_EW:+.4%} 一致")

    judgment = judge(c4_perf, g1_perf)

    out = {
        "window_bounds": {"merged": list(MERGED_WINDOW)},
        "eval_window": {"start": str(masks["full"][0].date()),
                        "end": str(masks["full"][-1].date()),
                        "n_days": len(masks["full"]),
                        "warmup_burned": WARMUP_DAYS,
                        "subwindow_split": SUBWINDOW_SPLIT,
                        "sub_days": {w: len(m) for w, m in masks.items()}},
        "benchmarks": {"index": "000300.SH 市值加权", "equal_weight": "同池等权（冻结宇宙）"},
        "criteria_thresholds": {"noise_floor": NOISE_FLOOR,
                                "turnover_ratio": TURNOVER_RATIO},
        "anchor_rerun": {"s100": rerun_s100, "s102": rerun_s102},
        "full_table": full_table,
        "verdict": judgment,
    }
    out_path = DATA_DIR / "c4_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"C4 封盘判读落盘 {out_path}")

    daily_rets: dict[str, pd.Series] = {}
    for s in SEEDS:
        daily_rets[f"C4_s{s}"] = results[f"{C4_ARMS[s]}_c4"]["daily_ret"]
        daily_rets[f"G1_s{s}"] = results[f"{G1_ARMS[s]}_mean"]["daily_ret"]
    plot_two_panel(
        daily_rets, bench_idx, bench_ew,
        g1_med_tag=f"G1_s{judgment['g1_three_seed_median']['median_seed'][1:]}",
    )
    plot_bar(full_table)

    print("=== 预注册判据 T1~T5 一次封盘 ===")
    for k in ("T1_survival", "T3_significant_gain", "T4_turnover_dividend",
              "T5_rejection", "registration"):
        v = judgment[k]
        head = v.get("passed", v.get("triggered"))
        print(f"[{k}] {'通过/触发' if head else '未通过/未触发'}：{v['note']}")
    print(f"中位种子：{judgment['median_seed']}（全期 AER 等权 "
          f"{judgment['median_full']['aer_ew']:+.2%} / 指数 "
          f"{judgment['median_full']['aer_idx']:+.2%}）")
    print(f"三种子中位对中位：C4 {judgment['median_seed']} "
          f"{judgment['median_full']['aer_ew']:+.2%} vs G1 "
          f"{judgment['g1_three_seed_median']['median_seed']} "
          f"{judgment['g1_three_seed_median']['median_aer_ew']:+.2%}"
          f"（价差 {judgment['median_vs_median_spread']:+.2%}）")
    print("同种子配对差 AER(等权) C4−G1（T2 必列）：")
    for w, d in judgment["T2_paired_presentation"]["paired_diff_c4_minus_g1_ew"].items():
        print(f"  [{w}] " + "  ".join(f"{k}: {v:+.2%}" for k, v in d.items()))
    print(f"日均换手（全期）：C4 族中位 {judgment['turnover']['c4_family_median']:.2%} vs "
          f"G1 族中位 {judgment['turnover']['g1_family_median']:.2%}")


if __name__ == "__main__":
    main()
