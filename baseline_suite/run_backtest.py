"""阶段 1-3 回测编排 + 预注册判读 + 落盘（计划 §2-4）。

三段编排：

    - ``paper``：阶段1（四变体 + M/R/P 双基准表 + 图1式双图）+ 阶段2（KDA 三臂 long-only）；
    - ``oos``：阶段3（样本外四变体 + M/R/P + KDA，预注册判定，封盘）。

用法（解释器 /home/user/miniconda3/envs/quant/bin/python）::

    python -m baseline_suite.run_backtest paper    # 阶段1+2
    python -m baseline_suite.run_backtest oos      # 阶段3（判读）

前置：信号已落盘（run_signals variants/baselines；KDA checkpoint 已封盘）。
产出 ``data/results_<window>.json`` + ``figures/fig_<window>.png``。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import (
    DATA_DIR,
    FIG_DIR,
    VARIANTS,
    BaselineConfig,
    ensure_dirs,
)
from baseline_suite.pipeline import (
    build_dual_benchmarks,
    judge_oos,
    run_group,
    run_kda_arms,
)
from baseline_suite.signal import build_px_tradeable


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_signal(tag: str, cfg: BaselineConfig) -> pd.DataFrame:
    """载入信号宽表（四变体 / M/R/P）。"""
    path = DATA_DIR / f"daily_signals_{cfg.window}_{tag}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"信号缺失：{path}")
    df = pd.read_parquet(path)
    logger.info(f"载入 {tag} 信号 [{cfg.window}]：{df.shape[0]} 日 × {df.shape[1]} 列")
    return df


def _backtest_suite(
    cfg: BaselineConfig,
    signals: dict[str, pd.DataFrame],
    *,
    device: str,
    include_kda: bool,
    provider,
) -> dict:
    """对一组信号（四变体 + M/R/P [+KDA]）跑引擎 + 双基准绩效。

    :returns: ``{tag: {perf_idx, perf_ew, daily_ret}}`` + benchmarks / beta_gap。
    """
    # 取数：价格/可交易 + 双基准（列取并集）
    all_cols = sorted(set().union(*[set(s.columns) for s in signals.values()]))
    rebalances = pd.DatetimeIndex(signals["M"].index)
    px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    results: dict[str, dict] = {}
    # 顺序：P（门禁）→ M/R → 四变体（mean 在最后醒目）
    order = ["P", "M", "R", "min", "max", "last", "mean"]
    if include_kda:
        order += ["B1", "B2", "B3"]

    daily_rets: dict[str, pd.Series] = {}
    for tag in order:
        if tag not in signals:
            continue
        pi, pe, dr, _, _ = run_group(
            signals[tag], px, trd, bench_idx, bench_ew, cfg=cfg, name=tag
        )
        results[tag] = {
            "perf_idx": pi.to_dict(),
            "perf_ew": pe.to_dict(),
        }
        daily_rets[tag] = dr

    return {
        "results": results,
        "daily_rets": daily_rets,
        "bench_idx_ret": bench_idx,
        "bench_ew_ret": bench_ew,
        "beta_gap": beta_gap,
    }


def cmd_paper(cfg: BaselineConfig, *, device: str) -> dict:
    """阶段 1（四变体 baseline）+ 阶段 2（KDA long-only 重评）。

    纪律：KDA 仅记录，无翻案门槛；若某臂超过 mean，标注"待样本外验证"非"有效"。
    """
    from kronos_qlib import QlibProvider

    # —— 载入信号 ——
    signals: dict[str, pd.DataFrame] = {}
    for v in VARIANTS:
        signals[v] = _load_signal(v, cfg)
    for tag in ("M", "R", "P"):
        signals[tag] = _load_signal(tag, cfg)

    provider = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)

    # —— KDA 三臂 long-only 重评（§3）——
    rebalances = pd.DatetimeIndex(signals["M"].index)
    kda_signals = run_kda_arms(provider, cfg, rebalances, device=device)
    signals.update(kda_signals)

    suite = _backtest_suite(cfg, signals, device=device, include_kda=True, provider=provider)

    # —— 画图：四变体 + M/R/P（KDA 单独一张）——
    from baseline_suite.plot import plot_dual

    main_rets = {t: suite["daily_rets"][t] for t in ("last", "mean", "max", "min", "M", "R", "P")}
    plot_dual(
        main_rets, suite["bench_idx_ret"],
        title=f"Canonical Baseline 四变体 + 对照（论文窗口 {cfg.backtest_start}~{cfg.backtest_end}）",
        out_path=FIG_DIR / "fig_paper_variants.png",
    )
    kda_rets = {t: suite["daily_rets"][t] for t in ("mean", "B1", "B2", "B3")}
    plot_dual(
        kda_rets, suite["bench_idx_ret"],
        title=f"KDA 三臂 long-only 重评 vs mean（论文窗口 {cfg.backtest_start}~{cfg.backtest_end}）",
        out_path=FIG_DIR / "fig_paper_kda.png",
    )

    # —— 落盘 JSON ——
    out = {
        "window": cfg.window,
        "period": [cfg.backtest_start, cfg.backtest_end],
        "benchmarks": {
            "index": "000300.SH csi300 市值加权",
            "equal_weight": "同池等权",
            "beta_gap": suite["beta_gap"],
        },
        "groups": suite["results"],
    }
    # KDA 预注册立场注记
    mean_ew = suite["results"]["mean"]["perf_ew"]["aer"]
    for arm in ("B1", "B2", "B3"):
        arm_ew = suite["results"][arm]["perf_ew"]["aer"]
        if arm_ew > mean_ew:
            out.setdefault("kda_notes", {})[arm] = (
                f"{arm} AER(等权)={arm_ew:+.2%} 超过 mean {mean_ew:+.2%}——待样本外验证，不判有效"
            )
    out_path = DATA_DIR / "results_paper.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"论文窗口结果落盘 {out_path}")
    return out


def _dict_to_perf(d: dict):
    """dict → PerfStats（judge 需要 PerfStats 对象）。"""
    from paper_replication.engine import PerfStats

    return PerfStats(**d)


def cmd_oos(cfg: BaselineConfig, *, device: str, paper_results: dict) -> dict:
    """阶段 3：样本外预注册判定（封盘）。"""
    from kronos_qlib import QlibProvider

    # 运行前断言：窗口末日 + 结算余量 ≤ 数据末日
    cfg.assert_oos_within_data()

    # —— 载入信号 ——
    signals: dict[str, pd.DataFrame] = {}
    for v in VARIANTS:
        signals[v] = _load_signal(v, cfg)
    for tag in ("M", "R", "P"):
        signals[tag] = _load_signal(tag, cfg)

    provider = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)

    # KDA 同窗推理（成本极低，一并跑，§3 记录立场）
    rebalances = pd.DatetimeIndex(signals["M"].index)
    kda_signals = run_kda_arms(provider, cfg, rebalances, device=device)
    signals.update(kda_signals)

    suite = _backtest_suite(cfg, signals, device=device, include_kda=True, provider=provider)

    # —— 预注册判定（§4，跑前冻结）——
    mean_pi = _dict_to_perf(suite["results"]["mean"]["perf_idx"])
    mean_pe = _dict_to_perf(suite["results"]["mean"]["perf_ew"])
    min_pe = _dict_to_perf(suite["results"]["min"]["perf_ew"])
    p_pe = _dict_to_perf(suite["results"]["P"]["perf_ew"])

    # 论文窗口的 min/mean 等权绩效（判据 4 两段对比）
    paper_min_pe = _dict_to_perf(paper_results["groups"]["min"]["perf_ew"])
    paper_mean_pe = _dict_to_perf(paper_results["groups"]["mean"]["perf_ew"])

    verdict = judge_oos(
        mean_perf_ew=mean_pe,
        mean_perf_idx=mean_pi,
        min_perf_ew=min_pe,
        placeholder_perf_ew=p_pe,
        paper_min_perf_ew=paper_min_pe,
        paper_mean_perf_ew=paper_mean_pe,
    )

    # —— 画图 ——
    from baseline_suite.plot import plot_dual

    main_rets = {t: suite["daily_rets"][t] for t in ("last", "mean", "max", "min", "M", "R", "P")}
    plot_dual(
        main_rets, suite["bench_idx_ret"],
        title=f"样本外四变体 + 对照（{cfg.backtest_start}~{cfg.backtest_end}，封盘）",
        out_path=FIG_DIR / "fig_oos_variants.png",
    )
    kda_rets = {t: suite["daily_rets"][t] for t in ("mean", "B1", "B2", "B3")}
    plot_dual(
        kda_rets, suite["bench_idx_ret"],
        title=f"样本外 KDA 三臂 vs mean（{cfg.backtest_start}~{cfg.backtest_end}）",
        out_path=FIG_DIR / "fig_oos_kda.png",
    )

    out = {
        "window": cfg.window,
        "period": [cfg.backtest_start, cfg.backtest_end],
        "benchmarks": {
            "index": "000300.SH csi300 市值加权",
            "equal_weight": "同池等权",
            "beta_gap": suite["beta_gap"],
        },
        "groups": suite["results"],
        "verdict": verdict,
    }
    out_path = DATA_DIR / "results_oos.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"样本外结果落盘 {out_path}")
    logger.info(f"=== 样本外判定：{verdict['verdict']} ===")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="baseline 四变体 + KDA + 样本外回测")
    parser.add_argument(
        "window", choices=["paper", "oos"], help="paper=阶段1+2 | oos=阶段3"
    )
    parser.add_argument("--device", default=None, help="覆盖配置 device（如 cuda:0）")
    args = parser.parse_args()

    cfg = BaselineConfig.load(window=args.window)
    if args.device:
        from dataclasses import replace
        cfg = replace(cfg, device=args.device)
    ensure_dirs()

    if args.window == "paper":
        cmd_paper(cfg, device=cfg.device)
    else:
        # 样本外判定需要论文窗口绩效（判据 4 两段对比）
        paper_path = DATA_DIR / "results_paper.json"
        if not paper_path.exists():
            raise FileNotFoundError(f"样本外判读需论文窗口结果：{paper_path}（先跑 paper）")
        with open(paper_path, encoding="utf-8") as f:
            paper_results = json.load(f)
        cmd_oos(cfg, device=cfg.device, paper_results=paper_results)


if __name__ == "__main__":
    main()
