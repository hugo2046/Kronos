"""阶段 1 入口：规则式状态切换 R1/R1'/R2 过引擎 + 判读（计划 §3）。

预注册规则族（跑前冻结，禁止追加）：

    - R1：指数收盘 > MA200 → M 动量，否则 canonical mean；
    - R1'：同门控反向（True → mean，False → M）；
    - R2：恒 True → 纯 M（对照）；
    - R3：路径离散度门控（阶段 2 后回填，本脚本不含）。

三段编排：paper（设计窗）/ oos（验证窗）/ full（paper+oos 拼接连续净值），
每段出双基准表（指数 / 同池等权）。判据 1~3 跑前冻结（§3）。

纪律：先跑全部候选（R1/R1'/R2 + 两单臂），再统一判读——不在中途看 oos1 调参。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.run_stage1
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import DATA_DIR as BL_DATA_DIR
from baseline_suite.common import BaselineConfig, ensure_dirs as bl_ensure_dirs
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from improve_suite.common import DATA_DIR, FIG_DIR
from improve_suite.regime_switch import build_switch_signal, gate_ma200

REPO_ROOT = Path(__file__).resolve().parents[1]

FULL_START, FULL_END = "2024-07-01", "2026-07-24"


def _load_signals(window: str) -> dict[str, pd.DataFrame]:
    """载入某窗 mean + M 信号宽表（既有 baseline_suite 落盘，复用不算新推理）。"""
    out = {}
    for tag in ("mean", "M"):
        path = BL_DATA_DIR / f"daily_signals_{window}_{tag}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"信号缺失：{path}（先跑 baseline_suite run_signals）")
        out[tag] = pd.read_parquet(path)
    return out


def _concat_full() -> dict[str, pd.DataFrame]:
    """拼接 paper + oos 两窗信号为全期宽表（并集日期 / 并集列）。"""
    paper = _load_signals("paper")
    oos = _load_signals("oos")
    full: dict[str, pd.DataFrame] = {}
    for tag in ("mean", "M"):
        a, b = paper[tag], oos[tag]
        dates = a.index.union(b.index)
        cols = a.columns.union(b.columns)
        full[tag] = a.reindex(index=dates, columns=cols).combine_first(
            b.reindex(index=dates, columns=cols)
        )
    return full


def _seg_cfg(window: str) -> BaselineConfig:
    """构造段落配置：paper/oos 直载，full 用 replace 覆盖起止。"""
    if window in ("paper", "oos"):
        return BaselineConfig.load(window=window)
    # full：paper 配置 + 全期起止
    return replace(
        BaselineConfig.load(window="paper"),
        backtest_start=FULL_START,
        backtest_end=FULL_END,
        window="full",
    )


def backtest_segment(
    signals: dict[str, pd.DataFrame],
    provider,
    window: str,
) -> tuple[dict, pd.Series]:
    """单段：建 R1/R1'/R2 + mean/M，过引擎出双基准表。

    :returns: ``(results, bench_idx_ret)``——
        ``results`` = ``{name: {perf_idx, perf_ew, daily_ret}}``；
        ``bench_idx_ret`` = 指数日收益（画图用）。
    """
    cfg = _seg_cfg(window)
    start, end = cfg.backtest_start, cfg.backtest_end
    from kronos_qlib import QlibProvider

    rebalances = QlibProvider(cfg.pool, start, end).trading_days(start, end)
    # 信号对齐到交易日
    for tag in signals:
        signals[tag] = signals[tag].reindex(rebalances)

    all_cols = sorted(set().union(*[set(s.columns) for s in signals.values()]))
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    # MA200 门控（该段决策日）
    gate = gate_ma200(provider, rebalances)

    # R1/R1'/R2（规则冻结）
    R1 = build_switch_signal(signals["M"], signals["mean"], gate)   # True→M, False→mean
    R1p = build_switch_signal(signals["mean"], signals["M"], gate)  # True→mean, False→M
    R2 = signals["M"].copy()                                         # 恒 M

    order = {"R1": R1, "R1'": R1p, "R2": R2, "mean": signals["mean"], "M": signals["M"]}
    results: dict[str, dict] = {}
    for name, sig in order.items():
        pi, pe, dr, _, _ = run_group(sig, px, trd, bench_idx, bench_ew, cfg=cfg, name=name)
        results[name] = {"perf_idx": pi, "perf_ew": pe, "daily_ret": dr}
    results["_beta_gap"] = beta_gap
    return results, bench_idx


def judge_stage1(
    paper_res: dict, oos_res: dict, full_res: dict
) -> dict:
    """判据 1~3（跑前冻结，§3）。

    全部候选已跑完后统一判读（不在中途看 oos1 调参）。

    1. 主判据：R1 paper 与 oos 两子窗 AER(等权) 均 > 0，且全期 AER(等权) >
       max(全期纯 mean, 全期纯 M) → 状态假说获支持，阶段 5 解锁；
    2. R1 仅全期占优但某子窗 ≤0 → 弱支持，阶段 5 仅文献调研；
    3. R1 不及 R1' 或不及两个单臂 → 可观测规则层证伪，阶段 5 取消；
    """
    def ew(res, name):
        return res[name]["perf_ew"].aer

    r1_p, r1_o, r1_f = ew(paper_res, "R1"), ew(oos_res, "R1"), ew(full_res, "R1")
    r1p_f = ew(full_res, "R1'")
    mean_f, m_f = ew(full_res, "mean"), ew(full_res, "M")  # M = R2

    both_subwin_pos = (r1_p > 0) and (r1_o > 0)
    full_beats_both = r1_f > max(mean_f, m_f)
    worse_than_r1p = r1_f < r1p_f
    worse_than_both_arms = (r1_f < mean_f) and (r1_f < m_f)

    criterion_main = both_subwin_pos and full_beats_both
    # 证伪：不及 R1' 或不及两个单臂（全期）
    falsified = worse_than_r1p or worse_than_both_arms

    if criterion_main and not falsified:
        verdict = "状态假说获支持，待前向确认（阶段 5 解锁立项）"
        stage5 = "unlock"
    elif falsified:
        verdict = "状态假说在可观测规则层证伪（阶段 5 取消，不做规则搜索）"
        stage5 = "cancel"
    elif full_beats_both and not both_subwin_pos:
        verdict = "弱支持：R1 仅全期占优但某子窗 AER(等权)≤0（阶段 5 仅允许文献调研）"
        stage5 = "literature_only"
    else:
        verdict = "R1 未达任何判据门槛（如实记录，阶段 5 按证伪处理）"
        stage5 = "cancel"

    return {
        "numbers": {
            "R1_paper_ew": r1_p, "R1_oos_ew": r1_o, "R1_full_ew": r1_f,
            "R1'_full_ew": r1p_f, "mean_full_ew": mean_f, "M_full_ew": m_f,
            "both_subwin_pos": bool(both_subwin_pos),
            "full_beats_both": bool(full_beats_both),
            "worse_than_R1'": bool(worse_than_r1p),
            "worse_than_both_arms": bool(worse_than_both_arms),
        },
        "criterion_main_passed": bool(criterion_main and not falsified),
        "falsified": bool(falsified),
        "stage5": stage5,
        "verdict": verdict,
    }


def _plot_full_nav(full_res: dict, bench_idx: pd.Series, out_path: Path) -> None:
    """全期净值图：R1 vs 纯 mean vs 纯 M（累计，含成本）。"""
    fig, ax = plt.subplots(figsize=(13, 6))
    cum_bench = (1 + bench_idx).cumprod() - 1
    styles = {
        "R1": ("#d62728", "-", 2.0),
        "mean": ("#1f77b4", "--", 1.5),
        "M": ("#ff7f0e", "--", 1.5),
        "R1'": ("#2ca02c", ":", 1.5),
    }
    for label in ("R1", "R1'", "mean", "M"):
        r = full_res[label]["daily_ret"]
        common = r.index.intersection(bench_idx.index)
        cum = (1 + r.loc[common]).cumprod() - 1
        c, ls, lw = styles[label]
        ax.plot(cum.index, cum.values, label=label, color=c, linestyle=ls, linewidth=lw)
    ax.plot(cum_bench.index, cum_bench.values, label="CSI300", color="black", linestyle="-.", linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title(f"Stage1 regime switch — full-period NAV ({FULL_START}~{FULL_END})")
    ax.set_ylabel("Cumulative return (with cost)")
    ax.set_xlabel("Decision date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"全期净值图落盘 {out_path}")


def _serialize(res: dict) -> dict:
    """把 PerfStats 对象序列化为 dict（JSON 落盘）。"""
    out = {}
    for k, v in res.items():
        if k.startswith("_"):
            out[k.lstrip("_")] = v
            continue
        out[k] = {
            "perf_idx": v["perf_idx"].to_dict() if hasattr(v["perf_idx"], "to_dict") else v["perf_idx"],
            "perf_ew": v["perf_ew"].to_dict() if hasattr(v["perf_ew"], "to_dict") else v["perf_ew"],
        }
    return out


def main() -> None:
    bl_ensure_dirs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from kronos_qlib import QlibProvider

    logger.info("==== 阶段 1：规则式状态切换 R1/R1'/R2（计划 §3）====")

    paper_sig = _load_signals("paper")
    oos_sig = _load_signals("oos")
    full_sig = _concat_full()

    provider = QlibProvider("csi300", FULL_START, FULL_END)

    logger.info("---- 全期段（full）----")
    full_res, bench_full = backtest_segment(full_sig, provider, "full")
    logger.info("---- 论文窗段（paper）----")
    paper_res, _ = backtest_segment(paper_sig, provider, "paper")
    logger.info("---- 样本外段（oos）----")
    oos_res, _ = backtest_segment(oos_sig, provider, "oos")

    # 判读（全部候选跑完后统一）
    verdict = judge_stage1(paper_res, oos_res, full_res)
    logger.info(f"==== 阶段 1 判读：{verdict['verdict']} ====")
    logger.info(f"判据数字：{verdict['numbers']}")

    # 画图 + 落盘
    _plot_full_nav(full_res, bench_full, FIG_DIR / "fig_stage1_regime_nav.png")

    out = {
        "stage": 1,
        "rules": {
            "R1": "gate(MA200) ? M : mean",
            "R1'": "gate(MA200) ? mean : M",
            "R2": "always M",
            "R3": "(backfill after stage 2)",
        },
        "windows": {"paper": ["2024-07-01", "2025-06-30"], "oos": ["2025-07-01", "2026-07-24"]},
        "benchmarks": {"index": "000300.SH", "equal_weight": "同池等权"},
        "results": {
            "paper": _serialize(paper_res),
            "oos": _serialize(oos_res),
            "full": _serialize(full_res),
        },
        "verdict": verdict,
    }
    out_path = DATA_DIR / "stage1_regime_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"阶段 1 结果落盘 {out_path}")


if __name__ == "__main__":
    main()
