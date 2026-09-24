"""TAS1 结题对比图（docs/状态前置重读模型实验结果_20260924.md 配图）。

两图（dpi=200，`tas_state/figures/`）：

1. ``fig_tas_cum_ic.png``：两窗累计 RankIC（逐日 IC cumsum）——TAS 各臂
   vs 基准（B0=F0_mean、M=动量）；
2. ``fig_tas_roll_ic.png``：两窗 20 日滚动均值 RankIC（局部结构）；
3. ``fig_tas_delta_b0.png``：T-PRE 相对 B0 的逐日配对 ΔIC 与 30 日滚动均值
   （阶段 A 判据的可视化）。

用法：``python -m tas_state.plot``（需 DDB 与已落盘的五臂信号）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tas_state.config import TASConfig
from tas_state.evaluate import daily_rank_ic, fetch_window_forward_returns

FIG_DIR = REPO_ROOT / "tas_state" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIG_ROOT = REPO_ROOT / "tas_state" / "data" / "signals" / "s42"
IC_CACHE = REPO_ROOT / "tas_state" / "data" / "daily_ic_all_arms.parquet"

# 臂 → (标签, 颜色, 线型)：基准黑实线，TAS 族彩色，M 灰虚线
STYLE = {
    "B0": ("B0 (F0_mean, canonical)", "#222222", "-"),
    "M": ("M (10d momentum)", "#888888", "--"),
    "T-PRE": ("T-PRE (main candidate)", "#d62728", "-"),
    "T-POST": ("T-POST", "#1f77b4", "-"),
    "C0": ("C0 (STATIC)", "#2ca02c", "-"),
    "T-SHUFFLE": ("T-SHUFFLE", "#9467bd", ":"),
    "B1": ("B1 (F0, N=40)", "#ff7f0e", "-."),
}
WINDOW_TITLE = {"W1": "W1: 2025-07-01 ~ 2025-12-31",
                "W2": "W2: 2026-01-01 ~ 2026-07-24"}


def compute_daily_ic(cfg: TASConfig) -> pd.DataFrame:
    """全部臂两窗逐日 RankIC（列 = MultiIndex(window, arm)），缓存 parquet。"""
    if IC_CACHE.exists():
        return pd.read_parquet(IC_CACHE)

    arms = {a: SIG_ROOT / a for a in
            ("T-PRE", "T-POST", "C0", "T-SHUFFLE", "B1")}
    b0_paths = {"W1": cfg.baseline_signal_paths["F0_W3"],
                "W2": cfg.baseline_signal_paths["F0_W4"]}
    m_paths = {"W1": cfg.baseline_signal_paths["M_W3"],
               "W2": cfg.baseline_signal_paths["M_W4"]}

    cols: dict[tuple[str, str], pd.Series] = {}
    for w in ("W1", "W2"):
        dates, fwd = fetch_window_forward_returns(cfg, w)
        wide = {
            "B0": pd.read_parquet(REPO_ROOT / b0_paths[w]).reindex(dates),
            "M": pd.read_parquet(REPO_ROOT / m_paths[w]).reindex(dates),
        }
        for a, d in arms.items():
            wide[a] = pd.read_parquet(
                d / f"daily_signals_{w}_{a}.parquet"
            ).reindex(dates)
        for a, sig in wide.items():
            ic = daily_rank_ic(sig, fwd, dates)
            cols[(w, a)] = ic
            logger.info(f"{w} {a}: mean IC {ic.mean():+.4f}")
    out = pd.DataFrame(cols)
    out.columns = out.columns.map(lambda t: f"{t[0]}|{t[1]}")
    out.to_parquet(IC_CACHE)
    return out


def _get(df: pd.DataFrame, w: str, arm: str) -> pd.Series:
    return df[f"{w}|{arm}"]


def main() -> None:
    cfg = TASConfig.load()
    ic = compute_daily_ic(cfg)
    arms_order = list(STYLE)

    # —— 图1：累计 RankIC（两窗分面；图例置图外底部防遮挡）——
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, w in zip(axes, ("W1", "W2")):
        for a in arms_order:
            s = _get(ic, w, a)
            label, c, ls = STYLE[a]
            lw = 1.8 if a in ("B0", "T-PRE", "M") else 1.1
            ax.plot(s.index, s.cumsum(), color=c, linestyle=ls, lw=lw,
                    label=label, alpha=0.95)
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.set_title(WINDOW_TITLE[w], fontsize=11)
        ax.set_xlabel("Decision date")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Cumulative daily RankIC (k=10)")
    fig.suptitle("TAS1 vs baselines — cumulative RankIC (seed42, mean agg, N=20)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    p = FIG_DIR / "fig_tas_cum_ic.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图1 → {p}")

    # —— 图2：20 日滚动均值 IC（图例置图外底部防遮挡）——
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, w in zip(axes, ("W1", "W2")):
        for a in arms_order:
            s = _get(ic, w, a)
            label, c, ls = STYLE[a]
            lw = 1.8 if a in ("B0", "T-PRE", "M") else 1.1
            ax.plot(s.index, s.rolling(20).mean(), color=c, linestyle=ls,
                    lw=lw, label=label, alpha=0.95)
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.set_title(WINDOW_TITLE[w], fontsize=11)
        ax.set_xlabel("Decision date")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("20-day rolling mean RankIC")
    fig.suptitle("TAS1 vs baselines — rolling RankIC (window=20d)", fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    p = FIG_DIR / "fig_tas_roll_ic.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图2 → {p}")

    # —— 图3：T-PRE 相对 B0 配对 ΔIC（判据可视化）——
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, w in zip(axes, ("W1", "W2")):
        d = _get(ic, w, "T-PRE") - _get(ic, w, "B0")
        ax.bar(d.index, d.values, width=1.6, color="#d62728", alpha=0.35,
               label="daily ΔIC")
        ax.plot(d.index, d.rolling(30).mean(), color="#111111", lw=1.8,
                label="30-day rolling mean")
        ax.axhline(0, color="#666666", lw=0.9)
        mean = d.mean()
        ax.axhline(mean, color="#1f77b4", lw=1.2, ls="--",
                   label=f"window mean = {mean:+.4f}")
        ax.set_title(WINDOW_TITLE[w], fontsize=11)
        ax.set_xlabel("Decision date")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("ΔIC = T-PRE − B0 (paired)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Stage-A criterion — paired ΔIC of T-PRE vs B0 "
                 "(needs >0 in BOTH windows)", fontsize=12)
    fig.tight_layout()
    p = FIG_DIR / "fig_tas_delta_b0.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图3 → {p}")


if __name__ == "__main__":
    main()
