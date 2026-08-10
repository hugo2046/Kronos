"""图1 式双图（计划 §1 / §5.2）。

复刻 ``origin/kronos-ddb:finetune/qlib_test.py`` 的双图布局：

    - 上图：累计收益（含成本）—— 四变体 + 对照 + csi300 指数虚线；
    - 下图：累计超额收益（含成本，相对 csi300 指数）。

口径修正（§1）：横轴为决策日，纵轴为累计净值（``(1+r).cumprod()-1``）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 四变体固定配色（计划 §1 顺序：last/mean/max/min）
VARIANT_COLORS = {
    "last": "#1f77b4",
    "mean": "#d62728",  # canonical 主线——红，醒目
    "max": "#2ca02c",
    "min": "#9467bd",
}
# 对照配色
CTRL_COLORS = {"M": "#ff7f0e", "R": "#17becf", "P": "#7f7f7f"}
# KDA 配色（虚线）
KDA_COLORS = {"B1": "#8c564b", "B2": "#e377c2", "B3": "#bcbd22"}
CSI300_COLOR = "black"


def plot_dual(
    daily_rets: dict[str, pd.Series],
    bench_idx_ret: pd.Series,
    *,
    title: str,
    out_path: Path,
    beta_basis: str = "csi300",
) -> None:
    """画图1 式双图。

    :param daily_rets: ``{label: 逐日净收益 Series（index=date）}``，
        顺序即图例顺序。已扣成本。
    :param bench_idx_ret: csi300 指数逐日收益（上图虚线 + 下图超额基准）。
    :param title: 总标题。
    :param out_path: png 输出路径。
    :param beta_basis: 下图超额基准说明（标题注脚）。
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # —— 上图：累计收益（含成本）——
    cum_bench = (1 + bench_idx_ret).cumprod() - 1
    for label, r in daily_rets.items():
        common = r.index.intersection(bench_idx_ret.index)
        cum = (1 + r.loc[common]).cumprod() - 1
        color, ls = _style_for(label)
        axes[0].plot(cum.index, cum.values, label=label, color=color, linestyle=ls, linewidth=1.6)
    axes[0].plot(
        cum_bench.index, cum_bench.values,
        label="CSI300", color=CSI300_COLOR, linestyle="--", linewidth=1.4,
    )
    axes[0].set_ylabel("Cumulative return (with cost)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)

    # —— 下图：累计超额收益（相对 csi300 指数）——
    for label, r in daily_rets.items():
        common = r.index.intersection(bench_idx_ret.index)
        excess = r.loc[common] - bench_idx_ret.loc[common]
        cum_ex = (1 + excess).cumprod() - 1
        color, ls = _style_for(label)
        axes[1].plot(cum_ex.index, cum_ex.values, label=label, color=color, linestyle=ls, linewidth=1.6)
    axes[1].axhline(0, color=CSI300_COLOR, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel(f"Cumulative excess vs {beta_basis} (with cost)")
    axes[1].set_xlabel("Decision date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _style_for(label: str) -> tuple[str, str]:
    """返回 (color, linestyle)。四变体实线、对照实线、KDA 虚线。"""
    if label in VARIANT_COLORS:
        return VARIANT_COLORS[label], "-"
    if label in CTRL_COLORS:
        return CTRL_COLORS[label], "-"
    if label in KDA_COLORS:
        return KDA_COLORS[label], "--"
    # 未知标签：默认灰实线
    return "#333333", "-"
