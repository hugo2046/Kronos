"""MH1 逐日超额净值（计划外追加，20260907 用户指令）。

六 MH1 臂 + G1_mean 参照过 ``paper_replication`` 引擎（经 ``baseline_suite``
包装，配置与 ``engine_v2_attachment.json`` 完全一致：同宇宙、同双基准、同
top50/drop5/min_hold5/15bp/delay=1）。G1_mean 参照限定到与 MH1 相同的
调仓网格（信号只留 W3/W4 决策日），与六臂同节奏交易，可公平叠加。

产物：

- ``data/nav_{W3,W4}.parquet``：逐日超额净值（1+excess 累乘，双基准列）；
- ``docs/images/mh1_nav_{idx,ew}.png``：双基准各一张，W3/W4 分图，族配色
  （S 系蓝、M 系橙红、G1_mean 黑虚线）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m mh1_multihorizon.nav``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from loguru import logger

from mh1_multihorizon.config import (
    DATA_DIR, HORIZONS, MAIN_HORIZON_IDX, REPO_ROOT, W3_END, W3_START,
    W4_END, W4_START, verify_protocol,
)
from mh1_multihorizon.evaluate import WINDOW_BOUNDS

RUNS = ("S42", "M42", "S43", "M43", "S44", "M44")
H10 = f"h{HORIZONS[MAIN_HORIZON_IDX]}"
G1_MEAN_PARQUET = {
    "W3": REPO_ROOT / "g5_head" / "data" / "daily_signals_2025h2_G1_mean.parquet",
    "W4": REPO_ROOT / "finetune_suite" / "data" / "g1" /
          "daily_signals_backtest_G1_mean.parquet",
}
IMG_DIR = REPO_ROOT / "docs" / "images"

# 族配色：S 系蓝（种子加深）、M 系橙红、G1_mean 黑虚线
STYLE = {
    "S42": ("#7fb3d9", "-"), "S43": ("#3d7eb0", "-"), "S44": ("#12456e", "-"),
    "M42": ("#f2b279", "-"), "M43": ("#de7c33", "-"), "M44": ("#a84a10", "-"),
    "G1_mean": ("#111111", "--"),
}


def _engine_context(wname: str, rebalances: pd.DatetimeIndex):
    """与 evaluate.engine_attachment 完全一致的引擎上下文（配置/宇宙/基准）。"""
    from dataclasses import replace

    from baseline_suite.common import VARIANTS, BaselineConfig
    from baseline_suite.pipeline import build_dual_benchmarks
    from baseline_suite.signal import build_px_tradeable
    from kronos_qlib import QlibProvider

    start, end = WINDOW_BOUNDS[wname]
    r4 = REPO_ROOT / "finetune_suite" / "data"
    universe = {
        "W4": [r4 / "g1" / f"daily_signals_backtest_G1_{v}.parquet" for v in VARIANTS]
        + [r4 / f"daily_signals_backtest_F1_{v}.parquet" for v in VARIANTS]
        + [r4 / f"daily_signals_backtest_F0_{v}.parquet" for v in VARIANTS]
        + [r4 / "daily_signals_backtest_M.parquet"],
        "W3": [r4 / "g0" / f"daily_signals_2025h2_G0_{v}.parquet" for v in VARIANTS]
        + [r4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet" for v in VARIANTS]
        + [r4 / "g0" / "daily_signals_2025h2_M.parquet"],
    }[wname]
    cfg = replace(BaselineConfig.load(window="oos"),
                  window=f"mh1_{wname}",
                  backtest_start=start, backtest_end=end)
    universe_cols = sorted(set().union(*[
        set(pd.read_parquet(p).columns) for p in universe]))
    provider = QlibProvider(cfg.pool, start, end)
    px, trd = build_px_tradeable(provider, cfg, rebalances, universe_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)
    return cfg, px, trd, bench_idx, bench_ew, beta_gap, len(universe_cols)


def run_nav() -> dict[str, pd.DataFrame]:
    """六臂 + G1_mean 双基准逐日超额净值 → nav_{W3,W4}.parquet。"""
    from baseline_suite.pipeline import run_group

    protocol, sha = verify_protocol()
    out: dict[str, pd.DataFrame] = {}
    attach = json.loads((DATA_DIR / "engine_v2_attachment.json")
                        .read_text(encoding="utf-8"))
    for wname in WINDOW_BOUNDS:
        df = pd.read_parquet(DATA_DIR / f"signals_{wname}.parquet")
        grid = pd.DatetimeIndex(sorted(pd.to_datetime(df["date"].unique())))
        cfg, px, trd, bench_idx, bench_ew, beta_gap, n_uni = _engine_context(
            wname, grid)
        # 一致性对拍：与已落盘附表同宇宙/同 beta_gap（配置与附表一致的门禁）
        assert n_uni == attach["meta"][wname]["n_universe"], "宇宙规模漂移"
        assert abs(beta_gap - attach["meta"][wname]["beta_gap"]) < 1e-12, (
            f"{wname} beta_gap 与附表不一致")

        nav_cols: dict[str, pd.Series] = {}
        for run in RUNS:
            arm, seed = run[0], int(run[1:])
            sub = df[(df["arm"] == arm) & (df["seed"] == seed)]
            sig = sub.pivot(index="date", columns="instrument", values=H10)
            sig.index = pd.DatetimeIndex(sig.index)
            _pi, _pe, _dr, ei, ee = run_group(
                sig, px, trd, bench_idx, bench_ew, cfg=cfg, name=f"{run}@{wname}")
            nav_cols[f"{run}_idx"] = (1.0 + ei).cumprod()
            nav_cols[f"{run}_ew"] = (1.0 + ee).cumprod()
        # G1_mean 参照：限定到与 MH1 相同的调仓网格（同节奏交易）
        g1 = pd.read_parquet(G1_MEAN_PARQUET[wname])
        g1_grid = g1.reindex(grid)
        _pi, _pe, _dr, ei, ee = run_group(
            g1_grid, px, trd, bench_idx, bench_ew, cfg=cfg,
            name=f"G1_mean@{wname}(same-grid)")
        nav_cols["G1_mean_idx"] = (1.0 + ei).cumprod()
        nav_cols["G1_mean_ew"] = (1.0 + ee).cumprod()

        nav = pd.DataFrame(nav_cols)
        nav.index.name = "date"
        path = DATA_DIR / f"nav_{wname}.parquet"
        nav.to_parquet(path)
        logger.info(f"[{wname}] 净值落盘 {path.name}：{nav.shape[0]} 日 × "
                    f"{nav.shape[1]} 列（末日 idx M42={nav['M42_idx'].iloc[-1]:.3f}）")
        out[wname] = nav
    return out


def plot(navs: dict[str, pd.DataFrame]) -> list[Path]:
    """双基准各一张（W3/W4 分图），族配色。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK HK", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    titles = {"W3": f"W3（2025-07~2025-12，{navs['W3'].shape[0]} 个决策日）",
              "W4": f"W4（2026-01~2026-07，{navs['W4'].shape[0]} 个决策日）"}
    paths: list[Path] = []
    for bench, label in (("idx", "相对 000300 指数基准"), ("ew", "相对同池等权基准")):
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.2), sharex=False)
        for ax, wname in zip(axes, ("W3", "W4")):
            nav = navs[wname]
            for name in (*RUNS, "G1_mean"):
                color, ls = STYLE[name]
                ax.plot(nav.index, nav[f"{name}_{bench}"], color=color,
                        linestyle=ls, linewidth=1.8 if name == "G1_mean" else 1.2,
                        label=name.replace("_", " "))
            ax.axhline(1.0, color="grey", linewidth=0.6, alpha=0.7)
            ax.set_title(titles[wname], fontsize=10)
            ax.set_ylabel("累计超额净值", fontsize=9)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7.5, ncol=4, loc="best")
        fig.suptitle(f"MH1 六臂 + G1_mean 参照 · 逐日超额净值 · {label}"
                     "（G1_mean 限定同调仓网格）", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = IMG_DIR / f"mh1_nav_{bench}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)
        logger.info(f"图落盘 {out}")
    return paths


def main() -> int:
    navs = run_nav()
    plot(navs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
