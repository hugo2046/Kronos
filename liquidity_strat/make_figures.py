"""生成图文报告所需图表（落盘 docs/imgs/）。

四张图，对应报告核心叙事：
1. nav_lo5_exst.png — lo5 累计净值（K/M/R/P + 本档等权基准）：绝对净值是 beta
2. excess_nav_K_by_bucket.png — 四档 K 对档等权超额累计净值：非单调、lo5 为负
3. quantile_lo5.png — lo5 信号层 10 分位收益（K/M/R，period_10）：池内分层区分力
4. bar_spread_excess_by_bucket.png — 四档 spread 年化 + K 超额年化：单调性失败

净值图需重跑 qlib_bt 捕获日收益（backtest._run_qlib_bt 已返回 returns）；
分位图与柱状图直接读 analysis_signal_layer.json / analysis_portfolio_layer.json。
图注用英文（规避 CJK 字体风险），报告正文中文。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from liquidity_strat.common import (
    DATA_DIR,
    HIGH_LIQ_BUCKET,
    SIGNAL_KRONOS,
    SIGNAL_MOM,
    SIGNAL_PLACEHOLDER,
    SIGNAL_REV,
    ST_TRACK_MAIN,
    LiquidityConfig,
    init_analyzer_auth,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "docs" / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {"figure.dpi": 130, "savefig.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3}
)
_COLORS = {"K": "#d62728", "M": "#1f77b4", "R": "#2ca02c", "P": "#9467bd", "bench": "#7f7f7f"}
_LABEL = {"K": "Kronos", "M": "Momentum", "R": "Reversal", "P": "Placeholder"}
_BUCKET_LABEL = {
    "lo5": "lo5 [0,5%]",
    "lo5_10": "lo5_10 [5,10%]",
    "mid45_55": "mid45_55 [45,55%]",
    HIGH_LIQ_BUCKET: "csi300",
}


def _cumnav(returns: pd.Series) -> pd.Series:
    r = returns.fillna(0.0)
    return (1.0 + r).cumprod()


def _bt_returns(bucket: str, sig: str, cfg: LiquidityConfig, provider, strat) -> pd.Series:
    """重跑一次 qlib_bt，返回扣费日收益序列（csi300 直接用 baseline mean 信号）。"""
    from liquidity_strat.backtest import (
        UNION_FILES,
        _dynamic_factor_wide,
        _load_csi300_signal,
        _run_qlib_bt,
        _to_long_signal,
        _topk_for_bucket,
    )

    if bucket == HIGH_LIQ_BUCKET:
        wide = _load_csi300_signal(cfg.window_start, cfg.window_end)
        codes = sorted(wide.columns.tolist())
        topk = 34
    else:
        u = pd.read_parquet(UNION_FILES[sig])
        u.index = pd.to_datetime(u.index)
        wide = _dynamic_factor_wide(u, strat, bucket, ST_TRACK_MAIN)
        codes = sorted(strat[(strat.bucket == bucket) & (strat.st_track == ST_TRACK_MAIN)]["code"].unique())
        topk = _topk_for_bucket(strat, bucket, ST_TRACK_MAIN)
    long_sig = _to_long_signal(wide)
    bt = _run_qlib_bt(long_sig, cfg.window_start, cfg.window_end, topk, codes)
    return bt["returns"]


def _bucket_eq(bucket: str, cfg: LiquidityConfig, provider, strat) -> pd.Series:
    from liquidity_strat.backtest import (
        _csi300_pseudo_strat,
        _load_csi300_signal,
        bucket_equal_weight_returns,
    )

    if bucket == HIGH_LIQ_BUCKET:
        csi = _load_csi300_signal(cfg.window_start, cfg.window_end)
        sdf = _csi300_pseudo_strat(csi)
    else:
        sdf = strat
    return bucket_equal_weight_returns(sdf, bucket, ST_TRACK_MAIN, provider, cfg.window_start, cfg.window_end)


def fig1_nav_lo5(cfg, provider, strat) -> Path:
    """lo5 累计净值：K/M/R/P + 本档等权基准。"""
    bench = _bucket_eq("lo5", cfg, provider, strat)
    fig, ax = plt.subplots(figsize=(9, 5))
    bench_nav = _cumnav(bench)
    ax.plot(bench_nav.index, bench_nav.values, color=_COLORS["bench"], lw=2.2, label="bucket equal-weight (beta)", ls="--")
    for sig in ["K", "M", "R", "P"]:
        ret = _bt_returns("lo5", sig, cfg, provider, strat)
        nav = _cumnav(ret)
        ax.plot(nav.index, nav.values, color=_COLORS[sig], lw=1.4, alpha=0.85, label=f"{_LABEL[sig]} (AER={(1+ret).prod()**(244/len(ret))-1:+.0%})")
    ax.set_title("lo5 [0,5%] cumulative NAV — absolute returns ride micro-cap beta (2024-07 ~ 2026-07)")
    ax.set_ylabel("cumulative NAV (start=1)")
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(1.0, color="k", lw=0.5)
    out = IMG_DIR / "nav_lo5_exst.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    logger.info(f"图1 落盘 {out}")
    return out


def fig2_excess_nav_K(cfg, provider, strat) -> Path:
    """四档 K 对档等权超额累计净值。"""
    buckets = ["lo5", "lo5_10", "mid45_55", HIGH_LIQ_BUCKET]
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in buckets:
        ret = _bt_returns(b, "K", cfg, provider, strat)
        bench = _bucket_eq(b, cfg, provider, strat)
        common = ret.index.intersection(bench.index)
        excess = ret.reindex(common).fillna(0) - bench.reindex(common).fillna(0)
        nav = _cumnav(excess)
        ax.plot(nav.index, nav.values, lw=1.7, label=f"{_BUCKET_LABEL[b]} (final={nav.iloc[-1]-1:+.1%})")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_title("Kronos excess NAV vs bucket equal-weight (beta-stripped) — non-monotonic across liquidity")
    ax.set_ylabel("excess cumulative NAV (start=1)")
    ax.legend(loc="upper left", fontsize=8)
    out = IMG_DIR / "excess_nav_K_by_bucket.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    logger.info(f"图2 落盘 {out}")
    return out


def fig3_quantile_lo5() -> Path:
    """lo5 信号层 10 分位平均收益（period_10）K/M/R。"""
    sl = json.loads((DATA_DIR / "analysis_signal_layer.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9, 5))
    qs = list(range(1, 11))
    width = 0.27
    for i, sig in enumerate(["K", "M", "R"]):
        rec = sl.get(f"lo5|exst|{sig}", {})
        qr = rec.get("period_10", {}).get("quantile_returns", {})
        vals = [qr.get(str(q), np.nan) * 100 for q in qs]
        ax.bar([q + (i - 1) * width for q in qs], vals, width, label=_LABEL[sig], color=_COLORS[sig], alpha=0.85)
    ax.set_xticks(qs)
    ax.set_xlabel("factor quantile (1=lowest signal ... 10=highest)")
    ax.set_ylabel("mean 10-day forward return (%)")
    ax.set_title("lo5 in-pool quantile returns (period=10) — stratification power (cf. post Fig.3)")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5)
    out = IMG_DIR / "quantile_lo5.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    logger.info(f"图3 落盘 {out}")
    return out


def fig4_bar_spread_excess() -> Path:
    """四档 spread 年化（K/M/R）+ K 组合层超额年化。"""
    sl = json.loads((DATA_DIR / "analysis_signal_layer.json").read_text(encoding="utf-8"))
    pl = json.loads((DATA_DIR / "analysis_portfolio_layer.json").read_text(encoding="utf-8"))
    buckets = ["lo5", "lo5_10", "mid45_55", HIGH_LIQ_BUCKET]
    bl = [_BUCKET_LABEL[b] for b in buckets]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    width = 0.27
    for i, sig in enumerate(["K", "M", "R"]):
        vals = []
        for b in buckets:
            rec = sl.get(f"{b}|exst|{sig}", {})
            vals.append(rec.get("period_10", {}).get("spread_annualized", np.nan) * 100)
        ax1.bar([x + (i - 1) * width for x in range(len(buckets))], vals, width, label=_LABEL[sig], color=_COLORS[sig], alpha=0.85)
    ax1.set_ylabel("signal-layer Q10-Q1 spread (annualized %)")
    ax1.set_title("Signal-layer spread (left) & portfolio excess vs bucket-eq (right) by liquidity bucket")
    ax1.axhline(0, color="k", lw=0.5); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    # K 组合层超额
    kex = []
    for b in buckets:
        rec = pl.get(f"{b}|exst|K", {})
        kex.append(rec.get("excess_vs_bucket_eq", {}).get("annual_return", np.nan) * 100)
    ax2.bar(range(len(buckets)), kex, width=0.5, color=_COLORS["K"], alpha=0.85, label="Kronos excess")
    ax2.set_xticks(range(len(buckets))); ax2.set_xticklabels(bl)
    ax2.set_ylabel("portfolio excess AER (%)")
    ax2.axhline(0, color="k", lw=0.5); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    out = IMG_DIR / "bar_spread_excess_by_bucket.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    logger.info(f"图4 落盘 {out}")
    return out


def main() -> None:
    init_analyzer_auth()
    from kronos_qlib import QlibProvider

    cfg = LiquidityConfig.load()
    provider = QlibProvider(cfg.pool, cfg.window_start, "2026-08-07")
    strat = pd.read_parquet(DATA_DIR / "strat_membership.parquet")
    strat["date"] = pd.to_datetime(strat["date"])

    fig3_quantile_lo5()
    fig4_bar_spread_excess()
    fig1_nav_lo5(cfg, provider, strat)
    fig2_excess_nav_K(cfg, provider, strat)
    logger.info(f"全部图落盘 {IMG_DIR}")


if __name__ == "__main__":
    main()
