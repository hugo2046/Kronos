"""G5 阶段 2：评估与统一封盘判读（计划 §3，20260817）。

前置纪律：7 个 checkpoint（H-kda×3 + H-mamba×3 + H-lin×1）全部定型后才运行
（脚本入口断言文件存在）；此前不看任何评估数字。forward（2026-07-25 后）零接触。

流程：
    1. 两窗（backtest 2026-01-01~2026-07-24 / 2025H2 2025-07-01~2025-12-31）逐日
       构样本 → G1 底座单次前向取隐状态（无 AR 采样）→ 7 头 decode 打分；
    2. canonical 引擎（csi300, k=50/n=5/min_hold=5/15bp）双基准全表
       （7 模型 + 在位者 G1_mean 参照；在位者 backtest 数字与冻结值对拍）；
    3. 正交化附表（§1 预承诺）：各头信号对 M 逐日截面 OLS 残差过引擎，仅报告；
    4. 预注册判据 J1~J5 一次封盘判读（阈值跑前冻结于本文件常量）。

中位种子定义（防单种子运气/挑最好）：头内三种子按 backtest AER(等权) 排序取中位
（与第 3 轮 mamba_head._judge 同口径）。

落盘：g5_head/data/g5_stage2_results.json + g5_head/figures/*.png。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m g5_head.run_g5_eval
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from finetune_suite.run_g2_signals import WINDOW_DEFS
from g5_head.run_g5_head import G5_SEEDS, HLIN_SEED

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"
R4 = PKG_DIR.parent / "finetune_suite" / "data"

# —— 预注册判据阈值（计划 §3 表，跑前冻结）——
INCUMBENT_AER_EW = 0.1433   # 在位者 G1 原版 AR 出口（s100 backtest AER 等权）
INCUMBENT_AER_IDX = 0.1066  # 在位者 AER(指数)
J2_THRESH_EW = 0.1633       # 在位者 +2pp
J2_THRESH_IDX = 0.1266      # 在位者(指数) +2pp
J3_MARGIN = 0.02            # > H-lin 同窗 + 2pp

WINDOW_BOUNDS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": WINDOW_DEFS["2025h2"],
}

# 在位者信号（参照行 + 判据锚）
INCUMBENT_PARQUET = {
    "backtest": R4 / "g1" / "daily_signals_backtest_G1_mean.parquet",
    "2025h2": DATA_DIR / "daily_signals_2025h2_G1_mean.parquet",
}
M_PARQUET = {
    "backtest": R4 / "daily_signals_backtest_M.parquet",
    "2025h2": R4 / "g0" / "daily_signals_2025h2_M.parquet",
}

# 引擎宇宙（px/等权基准列集）——与冻结数字的原始跑法逐字一致：
# backtest = 第 4 轮 G1 回测的 13 信号并集（run_g1_backtest.load_signals）；
# 2025h2 = G0 回测的 9 信号并集（run_g0_backtest）。头信号 reindex 到该宇宙
# （覆盖外 NaN=不可买，引擎行为不变），等权基准成分与冻结口径逐位可比
# （在位者对拍断言的门禁前提）。
UNIVERSE_PARQUETS = {
    "backtest": [R4 / "g1" / f"daily_signals_backtest_G1_{v}.parquet" for v in ("last", "mean", "max", "min")]
    + [R4 / f"daily_signals_backtest_F1_{v}.parquet" for v in ("last", "mean", "max", "min")]
    + [R4 / f"daily_signals_backtest_F0_{v}.parquet" for v in ("last", "mean", "max", "min")]
    + [R4 / "daily_signals_backtest_M.parquet"],
    "2025h2": [R4 / "g0" / f"daily_signals_2025h2_G0_{v}.parquet" for v in ("last", "mean", "max", "min")]
    + [R4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet" for v in ("last", "mean", "max", "min")]
    + [R4 / "g0" / "daily_signals_2025h2_M.parquet"],
}

ARMS = ("H-kda", "H-mamba")
MODEL_NAMES = [f"{a}_s{s}" for a in ARMS for s in G5_SEEDS] + [f"H-lin_s{HLIN_SEED}"]

# 族配色（两节式呈现）
FAMILY_COLORS = {
    "G1": "#08306b",
    "H-kda": {"s42": "#2171b5", "s43": "#4292c6", "s44": "#6baed6"},
    "H-mamba": {"s42": "#006d2c", "s43": "#31a354", "s44": "#74c476"},
    "H-lin": {"s42": "#4d4d4d"},
    "M": "#ff7f0e",
}


# ============================================================
# 1. 打分（单前向，无 AR 采样）
# ============================================================


def _load_models(backbone, device: str) -> dict[str, object]:
    """载入 7 个 checkpoint（先断言全部定型——判读前置纪律）。"""
    from g5_head.run_g5_head import _make_head

    missing = [n for n in MODEL_NAMES if not (DATA_DIR / f"{n}_best.pt").exists()]
    assert not missing, f"7 checkpoint 未全部定型，禁止评估：缺 {missing}"

    models = {}
    for name in MODEL_NAMES:
        arm, seed = name.rsplit("_s", 1)
        m = _make_head(arm, backbone).to(device)
        ckpt = torch_load(DATA_DIR / f"{name}_best.pt")
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models[name] = m
        logger.info(f"载入 {name} ← {name}_best.pt（es_RankIC={ckpt['info']['best_es_rankic']:+.4f}）")
    return models


def torch_load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def score_window(models: dict, backbone, window: str, device: str) -> dict[str, pd.DataFrame]:
    """单窗逐日打分：底座隐状态一次前向，7 头 decode（信号落盘 g5_head/data/）。"""
    import torch

    from cross_section_kda.data import build_daily_samples
    from g5_head.heads import decode_score
    from kronos_qlib import QlibProvider

    start, end = WINDOW_BOUNDS[window]
    provider = QlibProvider("csi300", start, end)
    rebalances = provider.trading_days(start, end)

    rows: dict[str, list[dict]] = {n: [] for n in models}
    for i, d in enumerate(rebalances):
        ds = d.strftime("%Y-%m-%d")
        b = build_daily_samples(provider, date=ds, pool="csi300")
        if b is None:
            logger.warning(f"{ds}: 无可用样本")
            for n in models:
                rows[n].append({})
            continue
        with torch.no_grad():
            hidden = backbone.extract(b.x_norm.to(device), b.stamp.to(device))
            for n, m in models.items():
                score = decode_score(m, hidden).cpu().numpy()
                rows[n].append({c: float(s) for c, s in zip(b.codes, score)})
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  score [{window}] [{i + 1}/{len(rebalances)}] {ds}: {len(b.codes)} 只")

    wide = {n: pd.DataFrame(rows[n], index=rebalances) for n in models}
    for n, df in wide.items():
        df.to_parquet(DATA_DIR / f"daily_signals_{window}_{n}.parquet")
        logger.info(f"[{window}] {n} 信号落盘（{df.shape[0]} 日，"
                    f"平均 {df.notna().sum(axis=1).mean():.0f} 只/日）")
    return wide


# ============================================================
# 2. 引擎全表 + 3. 正交化附表
# ============================================================


def orthogonalize_to_m(sig: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 OLS 残差：sig − (α + β·M)（公共非 NaN 列，<3 只该日记 NaN）。"""
    out = {}
    for d in sig.index.intersection(mkt.index):
        a, m = sig.loc[d], mkt.loc[d]
        common = a.notna() & m.notna()
        if int(common.sum()) < 3:
            out[d] = pd.Series(np.nan, index=a.index)
            continue
        x = m[common].values.astype(float)
        y = a[common].values.astype(float)
        beta, alpha = np.polyfit(x, y, 1)
        res = a.copy()
        res[~common] = np.nan
        res[common] = y - (alpha + beta * x)
        out[d] = res
    return pd.DataFrame(out).T.sort_index()


def engine_window(signals: dict[str, pd.DataFrame], window: str,
                  universe_cols: list[str] | None = None) -> tuple[dict, dict]:
    """单窗引擎全表：{name: {perf_idx, perf_ew, daily_ret}}（双基准）。

    :param universe_cols: px/等权基准的列宇宙（None 则取信号并集）。传冻结口径
        的轮次并集保证与在位者冻结数字可比（见 UNIVERSE_PARQUETS）。
    """
    from kronos_qlib import QlibProvider

    start, end = WINDOW_BOUNDS[window]
    cfg = replace(BaselineConfig.load(window="oos"),
                  window=f"g5_{window}", backtest_start=start, backtest_end=end)
    provider = QlibProvider(cfg.pool, start, end)
    if universe_cols is None:
        universe_cols = sorted(set().union(*[set(s.columns) for s in signals.values()]))
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
# 4. 预注册判据 J1~J5（一次封盘）
# ============================================================


def median_seed(arm: str, results: dict) -> str:
    """头内三种子按 backtest AER(等权) 排序取中位（防单种子运气/挑最好）。"""
    seeds = list(G5_SEEDS)
    ranked = sorted(seeds, key=lambda s: results[f"{arm}_s{s}"]["backtest"]["perf_ew"]["aer"])
    return f"{arm}_s{ranked[1]}"


def judge(results: dict) -> dict:
    """J1~J5 + 注册触发（阈值冻结于模块常量；一次判读）。"""
    med = {arm: median_seed(arm, results) for arm in ARMS}
    hlin_bt_ew = results[f"H-lin_s{HLIN_SEED}"]["backtest"]["perf_ew"]["aer"]

    per_head = {}
    for arm in ARMS:
        name = med[arm]
        bt_ew = results[name]["backtest"]["perf_ew"]["aer"]
        bt_idx = results[name]["backtest"]["perf_idx"]["aer"]
        h2_ew = results[name]["2025h2"]["perf_ew"]["aer"]
        j1 = bt_ew > 0 and bt_idx > 0
        j2 = bt_ew >= J2_THRESH_EW and bt_idx >= J2_THRESH_IDX
        j3 = bt_ew > hlin_bt_ew + J3_MARGIN if j2 else None
        j4 = h2_ew > 0 if j2 else None
        per_head[arm] = {
            "median_model": name, "backtest_aer_ew": bt_ew, "backtest_aer_idx": bt_idx,
            "2025h2_aer_ew": h2_ew,
            "J1_survive": bool(j1), "J2_proposition": bool(j2),
            "J3_mechanism": None if j3 is None else bool(j3),
            "J4_cross_window": None if j4 is None else bool(j4),
            "register": bool(j2 and j3 and j4),
        }

    all_below_incumbent = all(
        results[med[arm]]["backtest"]["perf_ew"]["aer"] < INCUMBENT_AER_EW for arm in ARMS
    )
    j5 = all_below_incumbent

    # §5.3：落在在位者种子分布重叠带（+14~+16%）的注明
    overlap_zone = {
        arm: bool(INCUMBENT_AER_EW <= results[med[arm]]["backtest"]["perf_ew"]["aer"] < J2_THRESH_EW)
        for arm in ARMS
    }

    return {
        "median_seeds": med, "hlin_backtest_aer_ew": hlin_bt_ew,
        "per_head": per_head,
        "J5_proposition_rejected": bool(j5),
        "incumbent_overlap_zone": overlap_zone,
        "thresholds": {
            "incumbent_aer_ew": INCUMBENT_AER_EW, "incumbent_aer_idx": INCUMBENT_AER_IDX,
            "J2_ew": J2_THRESH_EW, "J2_idx": J2_THRESH_IDX, "J3_margin": J3_MARGIN,
        },
    }


# ============================================================
# 图（两节式 + 条形图）
# ============================================================


def _model_style(name: str) -> dict:
    arm, seed = name.rsplit("_s", 1)
    if arm == "H-lin":
        return dict(color=FAMILY_COLORS["H-lin"]["s42"], linewidth=1.3)
    return dict(color=FAMILY_COLORS[arm][f"s{seed}"],
                linewidth=2.4 if name.endswith("s42") else 1.3)


def plot_results(results: dict, ctx: dict, judgment: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    med_models = [judgment["median_seeds"][a] for a in ARMS] + [f"H-lin_s{HLIN_SEED}"]
    start, end = WINDOW_BOUNDS["backtest"]

    # 节一：backtest 净值（中位种子 + H-lin + 在位者）vs 双基准
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    nav = lambda r: (1 + r.fillna(0)).cumprod() - 1  # noqa: E731
    for name in med_models:
        r = ctx["backtest"]["rets"][name]
        axes[0].plot(nav(r).index, nav(r).values, label=name, **_model_style(name))
    inc = ctx["backtest"]["rets"]["G1_mean"]
    axes[0].plot(nav(inc).index, nav(inc).values, label="G1_mean（在位者）",
                 color=FAMILY_COLORS["G1"], linewidth=2.0)
    axes[0].plot(nav(ctx["backtest"]["bench_idx"]).index, nav(ctx["backtest"]["bench_idx"]).values,
                 label="CSI300(指数基准)", color="black", linestyle="--", linewidth=1.4)
    axes[0].plot(nav(ctx["backtest"]["bench_ew"]).index, nav(ctx["backtest"]["bench_ew"]).values,
                 label="同池等权基准", color="black", linestyle=":", linewidth=1.2)
    axes[0].set_title(f"G5 换头 vs 在位者 — backtest {start}~{end}（中位种子）")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    for name in med_models:
        r, b = ctx["backtest"]["rets"][name], ctx["backtest"]["bench_ew"]
        common = r.index.intersection(b.index)
        cum = ((1 + (r.loc[common] - b.loc[common]).fillna(0)).cumprod() - 1)
        axes[1].plot(cum.index, cum.values, label=name, **_model_style(name))
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Cumulative excess vs 同池等权")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g5_nav_backtest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 条形图：backtest AER(等权) 全模型 + 阈值线
    order = MODEL_NAMES + ["G1_mean"]
    aers = [results[n]["backtest"]["perf_ew"]["aer"] for n in order]
    colors = [(FAMILY_COLORS["G1"] if n == "G1_mean"
               else FAMILY_COLORS["H-lin"]["s42"] if n.startswith("H-lin")
               else FAMILY_COLORS[n.rsplit("_s", 1)[0]][f"s{n[-2:]}"]) for n in order]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(order, aers, color=colors)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(INCUMBENT_AER_EW, color="#08306b", linestyle=":", linewidth=1.4,
               label=f"在位者 {INCUMBENT_AER_EW:+.2%}")
    ax.axhline(J2_THRESH_EW, color="#006d2c", linestyle="-.", linewidth=1.4,
               label=f"J2 门槛 {J2_THRESH_EW:+.2%}")
    for i, a in enumerate(aers):
        ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}", ha="center", fontsize=8)
    ax.set_ylabel("AER (等权基准, with cost)")
    ax.set_title(f"G5 七模型 backtest AER(等权) — {start}~{end}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_g5_aer_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# main
# ============================================================


def main() -> None:
    import torch

    device = "cuda:0"
    from g5_head.backbone_g1 import load_g1_backbone

    backbone = load_g1_backbone(device)
    models = _load_models(backbone, device)

    results: dict[str, dict] = {n: {"backtest": {}, "2025h2": {}} for n in models}
    ctx: dict = {}
    ortho_results: dict[str, dict] = {n: {} for n in models}

    for window in ("backtest", "2025h2"):
        logger.info("=" * 70)
        logger.info(f"窗口 {window}：{WINDOW_BOUNDS[window]}")
        wide = score_window(models, backbone, window, device)

        # 在位者参照行（G1_mean parquet 过同一引擎）
        inc = pd.read_parquet(INCUMBENT_PARQUET[window])
        first = next(iter(wide.values()))
        assert inc.index.equals(first.index), (
            f"[{window}] 在位者索引与模型打分网格不一致："
            f"{inc.index.min()}~{inc.index.max()} vs {first.index.min()}~{first.index.max()}")
        signals = {**wide, "G1_mean": inc}

        # 引擎宇宙 = 冻结口径轮次并集（在位者可比性门禁）
        universe_cols = sorted(set().union(*[
            set(pd.read_parquet(p).columns) for p in UNIVERSE_PARQUETS[window]]))
        logger.info(f"[{window}] 引擎宇宙：{len(universe_cols)} 列（冻结口径轮次并集）")

        res, meta = engine_window(signals, window, universe_cols=universe_cols)
        ctx[window] = meta
        for name, r in res.items():
            if name == "G1_mean":
                results.setdefault("G1_mean", {})[window] = r
            else:
                results[name][window] = r

        # 在位者对拍（backtest 窗与封盘冻结值一致 → 评估链路无误）
        if window == "backtest":
            rerun = res["G1_mean"]["perf_ew"]["aer"]
            assert abs(rerun - INCUMBENT_AER_EW) < 5e-4, (
                f"在位者引擎复跑 {rerun:+.4%} 与冻结 {INCUMBENT_AER_EW:+.4%} 不一致")
            logger.info(f"对拍在位者 G1_mean：复跑 {rerun:+.4%} vs 冻结 {INCUMBENT_AER_EW:+.4%} 一致")

        # 正交化附表（§1 预承诺，仅报告不进判据）
        mkt = pd.read_parquet(M_PARQUET[window])
        ortho_signals = {n: orthogonalize_to_m(w, mkt) for n, w in wide.items()}
        ores, _ = engine_window(ortho_signals, window, universe_cols=universe_cols)
        for name, r in ores.items():
            ortho_results[name][window] = r
        logger.info(f"[{window}] 正交化附表完成（{len(ores)} 模型）")

    judgment = judge(results)

    out = {
        "experiment": "g5_stage2_eval",
        "date": "2026-08-17",
        "windows": {w: list(b) for w, b in WINDOW_BOUNDS.items()},
        "engine": {"pool": "csi300", "top_k": 50, "drop_n": 5, "min_hold": 5, "cost_bps": 15.0,
                   "benchmarks": ["000300.SH 指数", "同池等权"],
                   "beta_gap": {w: ctx[w]["beta_gap"] for w in WINDOW_BOUNDS}},
        "results": results,
        "orthogonalized_vs_M": ortho_results,
        "judgment": judgment,
    }
    with open(DATA_DIR / "g5_stage2_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    logger.info(f"阶段 2 全表 + 判读落盘 → {DATA_DIR / 'g5_stage2_results.json'}")

    plot_results(results, ctx, judgment)

    print("\n==== G5 阶段 2 预注册判据 J1~J5 封盘判读 ====")
    for arm in ARMS:
        h = judgment["per_head"][arm]
        print(f"[{arm}] 中位种子={h['median_model']} | backtest AER(等权)={h['backtest_aer_ew']:+.2%} "
              f"AER(指数)={h['backtest_aer_idx']:+.2%} | 2025H2 AER(等权)={h['2025h2_aer_ew']:+.2%}")
        print(f"        J1={h['J1_survive']} J2={h['J2_proposition']} "
              f"J3={h['J3_mechanism']} J4={h['J4_cross_window']} 注册={h['register']}")
    print(f"[H-lin] backtest AER(等权)={judgment['hlin_backtest_aer_ew']:+.2%}（机制锚点）")
    print(f"[J5 命题否定] {judgment['J5_proposition_rejected']}")
    print(f"[在位者重叠带] {judgment['incumbent_overlap_zone']}")
    reg = [a for a in ARMS if judgment["per_head"][a]["register"]]
    print(f"[注册触发] {reg if reg else '无'}")


if __name__ == "__main__":
    main()
