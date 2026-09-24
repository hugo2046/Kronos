"""TAS1 组合层对比（engine_v2 完整口径，披露式——阶段 A 已停止，不称可交易胜出）。

产出（README ``backtest_result_example.png`` 同形式——策略对基准的累计收益曲线）：

- ``fig_tas_backtest_w1.png`` / ``fig_tas_backtest_w2.png``：各臂组合累计
  收益 vs 双基准（CSI300 指数 + 同池等权）；
- ``backtest_perf.json``：v2 口径 AER/IR/MDD/换手（相对双基准）。

口径（计划 §1.1 / §6.2 条件5 同源）：long-only top_k=50、drop_n=5、
min_hold=5、单边 15bp；engine_v2 六项修正全开（t 信号 t+1 收盘成交、
涨跌停禁成交、双边成本、年化 252）；tradeable = ``tradestatuscode == -1``
∧ px 非缺失；池 = 各臂信号列并集（universe 对齐）。

用法：``python -m tas_state.backtest``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paper_replication.benchmark import probe_index_benchmark
from paper_replication.engine_v2 import (
    EngineConfigV2,
    attach_benchmark_v2,
    build_limit_masks,
    build_pool_equal_weight_benchmark_v2,
    compute_perf_v2,
    run_portfolio_v2,
)
from tas_state.config import TASConfig

FIG_DIR = REPO_ROOT / "tas_state" / "figures"
DATA_DIR = REPO_ROOT / "tas_state" / "data"
SIG_ROOT = DATA_DIR / "signals" / "s42"

# 臂 → 显示名（曲线颜色在 draw 中按序分配）
ARM_ORDER = ["B0", "M", "T-PRE", "T-POST", "C0", "T-SHUFFLE", "B1"]
ARM_LABEL = {
    "B0": "B0 (F0_mean, canonical)",
    "M": "M (10d momentum)",
    "T-PRE": "T-PRE (main candidate)",
    "T-POST": "T-POST",
    "C0": "C0 (STATIC)",
    "T-SHUFFLE": "T-SHUFFLE",
    "B1": "B1 (F0, N=40)",
}
WINDOW_TITLE = {"W1": "W1: 2025-07-01 ~ 2025-12-31",
                "W2": "W2: 2026-01-01 ~ 2026-07-24"}


def load_arm_signals(cfg: TASConfig, window: str) -> dict[str, pd.DataFrame]:
    """读入七臂信号（TAS 五臂本地 + B0/M 只读 parquet）。"""
    b0_paths = {"W1": cfg.baseline_signal_paths["F0_W3"],
                "W2": cfg.baseline_signal_paths["F0_W4"]}
    m_paths = {"W1": cfg.baseline_signal_paths["M_W3"],
               "W2": cfg.baseline_signal_paths["M_W4"]}
    arms: dict[str, pd.DataFrame] = {}
    for a in ("T-PRE", "T-POST", "C0", "T-SHUFFLE", "B1"):
        arms[a] = pd.read_parquet(
            SIG_ROOT / a / f"daily_signals_{window}_{a}.parquet"
        )
    arms["B0"] = pd.read_parquet(REPO_ROOT / b0_paths[window])
    arms["M"] = pd.read_parquet(REPO_ROOT / m_paths[window])
    return arms


def fetch_window_data(provider, codes: list[str], start: str, end: str):
    """px / tradeable / 涨跌停掩码（replay_v2.fetch_window 同口径）。"""
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = start
        provider._end_date = end
        provider.instruments_ = list(codes)
        df = provider.fetch(
            ["$close", "$tradestatuscode", "$up_down_limit_status"], freq="day"
        )
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig
    px = df["close"].unstack("instrument").sort_index()
    tsc = (df["tradestatuscode"].unstack("instrument").sort_index()
           .reindex_like(px))
    uls = (df["up_down_limit_status"].unstack("instrument").sort_index()
           .reindex_like(px))
    if (uls != 0).sum().sum() == 0:
        raise RuntimeError("up_down_limit_status 全零——涨跌停修正不可用")
    trd = (tsc == -1).fillna(False) & px.notna()
    return px, trd, uls


def run_window_backtest(cfg: TASConfig, window: str) -> tuple[dict, dict]:
    """单窗七臂 engine_v2 回测：返回 {臂: 日收益} 与绩效 dict。"""
    from kronos_qlib import QlibProvider

    bounds = {"W1": cfg.eval_window_1, "W2": cfg.eval_window_2}[window]
    arms = load_arm_signals(cfg, window)
    universe = sorted(set().union(*[set(s.columns) for s in arms.values()]))

    # 行情只拉窗口末 + 20 交易日缓冲（末笔决策的成交/结算），不拉到
    # data_end——否则信号 reindex 会把组合收益序列拖入下一评估窗
    from kronos_qlib import QlibProvider as _QP

    _probe = _QP("csi300", bounds[0], bounds[1])
    _cal = _probe.trading_days(bounds[0], bounds[1])
    fetch_end = _probe.trading_days(bounds[0], cfg.data_end)[
        min(len(_probe.trading_days(bounds[0], cfg.data_end)) - 1,
            len(_cal) + 20)
    ]
    fetch_end = f"{fetch_end:%Y-%m-%d}"

    provider = QlibProvider("csi300", bounds[0], fetch_end)
    px, trd, uls = fetch_window_data(provider, universe, bounds[0], fetch_end)
    bench_idx = probe_index_benchmark(provider, bounds[0], fetch_end)
    win_dates = _cal  # 窗口内交易日

    ecfg = EngineConfigV2(
        top_k=cfg.top_k, drop_n=cfg.drop_n, min_hold=cfg.min_hold,
        cost_bps=cfg.cost_bps,
    )
    buy_b, sell_b = build_limit_masks(uls_wide=uls)

    daily_rets: dict[str, pd.Series] = {}
    perfs: dict[str, dict] = {}
    for a in ARM_ORDER:
        # 只对齐列；行索引保持窗口内信号日（缓冲日不产生决策/收益）
        sig = arms[a].reindex(index=win_dates, columns=px.columns)
        ret, trades = run_portfolio_v2(
            sig, px, trd, cfg=ecfg, buy_blocked=buy_b, sell_blocked=sell_b
        )
        daily_rets[a] = ret
        bench_ew = build_pool_equal_weight_benchmark_v2(px, trd, sig)
        p_idx = compute_perf_v2(
            attach_benchmark_v2(ret, bench_idx), trades,
            name=f"{a}|idx", fix_annualization_252=True,
        )
        p_ew = compute_perf_v2(
            attach_benchmark_v2(ret, bench_ew), trades,
            name=f"{a}|ew", fix_annualization_252=True,
        )
        perfs[a] = {
            "aer_vs_index": p_idx.aer, "ir_vs_index": p_idx.ir,
            "mdd_vs_index": p_idx.max_drawdown,
            "aer_vs_equal_weight": p_ew.aer, "ir_vs_equal_weight": p_ew.ir,
            "daily_turnover": p_idx.daily_turnover,
        }
        logger.info(
            f"[{window}] {a}: AER(idx)={p_idx.aer:+.2%} AER(ew)={p_ew.aer:+.2%} "
            f"IR(idx)={p_idx.ir:+.2f} 换手={p_idx.daily_turnover:.3f}"
        )
    daily_rets["__bench_idx"] = bench_idx
    return daily_rets, perfs


def draw(
    window: str,
    daily_rets: dict[str, pd.Series],
    perfs: dict[str, dict],
    bench_idx: pd.Series,
) -> Path:
    """README 参照形式：策略 vs 基准的累计收益曲线（含回撤次轴面板）。"""
    from tas_state.config import TASConfig as _C  # noqa: F401

    sig_root = SIG_ROOT
    cfg = TASConfig.load()
    bounds = {"W1": cfg.eval_window_1, "W2": cfg.eval_window_2}[window]

    colors = {
        "B0": "#222222", "M": "#888888", "T-PRE": "#d62728",
        "T-POST": "#1f77b4", "C0": "#2ca02c", "T-SHUFFLE": "#9467bd",
        "B1": "#ff7f0e",
    }
    styles = {a: ("-" if a in ("B0", "M", "T-PRE") else "--") for a in ARM_ORDER}

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    # —— 主面板：累计收益 ——
    for a in ARM_ORDER:
        r = daily_rets[a].dropna()
        cum = (1 + r).cumprod() - 1
        lw = 2.0 if a in ("B0", "T-PRE", "M") else 1.3
        ax.plot(cum.index, cum.values * 100, color=colors[a],
                linestyle=styles[a], lw=lw, label=ARM_LABEL[a], alpha=0.95)
    cum_idx = ((1 + bench_idx).cumprod() - 1).reindex(
        daily_rets["T-PRE"].index).dropna()
    ax.plot(cum_idx.index, cum_idx.values * 100, color="#000000",
            linestyle=":", lw=2.0, label="CSI300 index (buy & hold)")
    # 同池等权：用 T-PRE 池口径（逐臂差异极小）
    from kronos_qlib import QlibProvider  # noqa: F401（已在外层拉过）

    # —— 次面板：T-PRE 与 B0 回撤 ——
    for a, c in (("T-PRE", "#d62728"), ("B0", "#222222")):
        nav = (1 + daily_rets[a].dropna()).cumprod()
        dd = (nav / nav.cummax() - 1) * 100
        ax2.fill_between(dd.index, dd.values, 0, color=c, alpha=0.35,
                         label=f"{a} drawdown")
    ax2.axhline(0, color="#999999", lw=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(alpha=0.25)

    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(
        f"Portfolio cumulative returns vs benchmarks — {WINDOW_TITLE[window]}\n"
        "(long-only top-50/drop-5/hold-5, 15bp per side, engine_v2, "
        "disclosure-only)", fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    out = FIG_DIR / f"fig_tas_backtest_{window.lower()}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{window}] 回测图 → {out}")
    return out


def main() -> int:
    cfg = TASConfig.load()
    report: dict = {"config_sha256": cfg.sha256(),
                    "note": "披露式组合对照（阶段A已停止，不构成可交易胜出宣称）",
                    "windows": {}}
    for window in ("W1", "W2"):
        daily_rets, perfs = run_window_backtest(cfg, window)
        bench_idx = daily_rets.pop("__bench_idx")
        draw(window, daily_rets, perfs, bench_idx)
        report["windows"][window] = perfs
    out = DATA_DIR / "backtest_perf.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    logger.info(f"组合绩效 JSON → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
