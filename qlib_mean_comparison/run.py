"""G1 mean vs G10 A1 mean 同口径 Qlib 原生对照（20260909 计划）。

同一套 ``qlib.backtest`` + ``TopkDropoutStrategy(topk=50, n_drop=5,
hold_thresh=5)`` + ``SimulatorExecutor(day)``，open 成交、买 0.001/卖
0.0015/最低 5、涨跌停阈值 0.095、资金 1 亿、基准 000300.SH；三臂同日期、
同池（三臂共同有限信号股票日集合）、同费用。主口径为
``net_daily=(return-cost)`` 的 **cumsum** 曲线（与截图脚本一致），窗口末
``delta_end = curve_A1[-1] − curve_G1[-1]``；复利口径只入附表。零训练、零
推理、零 GPU、forward（≥2026-07-25）零读取。旧信号生成链保持
``legacy_unverified`` 标记。

用法::

    python -m qlib_mean_comparison.run --stage preflight
    python -m qlib_mean_comparison.run --stage backtest
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

SOURCE_ROOT = Path("/home/user/workspace/Kronos")
G10_RUN = Path("/home/user/workspace/Kronos-g10-pilot/g10_head_pilot/data/"
               "g10h_20260908_1114")
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "mean_comparison_20260909"
FORWARD_CUTOFF = "2026-07-24"
WINDOW_BOUNDS = {"W3": ("2025-07-01", "2025-12-31"),
                 "W4": ("2026-01-01", "2026-07-24")}
ARMS = ("G1_mean", "A1_mean", "A0_bestCE_mean")
SIGNAL_PATHS = {
    "G1_mean": {"W3": SOURCE_ROOT / "g5_head" / "data" /
                "daily_signals_2025h2_G1_mean.parquet",
                "W4": SOURCE_ROOT / "finetune_suite" / "data" / "g1" /
                "daily_signals_backtest_G1_mean.parquet"},
    "A1_mean": {w: G10_RUN / "signals" / f"{w}_A1_e15.parquet"
                for w in WINDOW_BOUNDS},          # bestCE=e15 同权重别名，单线
    "A0_bestCE_mean": {w: G10_RUN / "signals" / f"{w}_A0_bestCE.parquet"
                       for w in WINDOW_BOUNDS},
}
BACKTEST_KWARGS = dict(
    account=100_000_000, benchmark="000300.SH", deal_price="open",
    open_cost=0.001, close_cost=0.0015, min_cost=5, limit_threshold=0.095,
    topk=50, n_drop=5, hold_thresh=5,
)
EXPERIMENT = "kronos-mean-comparison-20260909"
LEGACY_NOTE = "legacy_unverified"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_wide(arm: str, wname: str) -> pd.DataFrame:
    """统一成 date×instrument 宽表（DatetimeIndex；原值不动）。"""
    p = SIGNAL_PATHS[arm][wname]
    if not p.is_file():
        raise FileNotFoundError(f"信号缺失：{p}")
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.T                                   # 股票×日期 → 日期×股票
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "datetime"
    return df.sort_index()


def common_set(wides: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """三臂共同有限信号股票日掩码（只看信号，不看标签）。"""
    # 逐臂 notna 相交（只看信号，不看标签/未来）
    mask = None
    for w in wides.values():
        m = w.notna()
        mask = m if mask is None else (mask & m)
    stats = {
        "n_days": int(mask.any(axis=1).sum()),
        "n_common_cells": int(mask.sum().sum()),
        "per_arm_coverage": {a: int(w.notna().sum().sum())
                             for a, w in wides.items()},
    }
    return mask, stats


def prepare_signal_series(wide: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """共同掩码下的信号 Series（instrument, datetime MultiIndex，无平移）。

    qlib-ddb 的 TopkDropoutStrategy 内部以 ``get_step_time(shift=1)`` 读上一
    交易日信号 → t 日信号在 t+1 开盘执行；本函数**不再额外平移**（防双重
    shift，合成测试锁定索引恒等）。
    """
    sig = wide.where(mask)
    s = sig.stack()
    s.index.names = ["datetime", "instrument"]
    return s.swaplevel().sort_index()


def run_qlib_backtest(signal: pd.Series, start: str, end: str) -> pd.DataFrame:
    """单臂窗 Qlib 原生回测 → 逐日 report（return/bench/cost/turnover）。"""
    import qlib
    from qlib.backtest import backtest, executor
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.utils.time import Freq

    from kronos_qlib import QlibProvider

    QlibProvider.init_qlib_once()          # DDB 后端（凭据自 .env，不打印）
    strategy = TopkDropoutStrategy(
        topk=BACKTEST_KWARGS["topk"], n_drop=BACKTEST_KWARGS["n_drop"],
        hold_thresh=BACKTEST_KWARGS["hold_thresh"], signal=signal)
    executor_config = {"time_per_step": "day",
                       "generate_portfolio_metrics": True}
    sig_params = inspect.signature(executor.SimulatorExecutor).parameters
    delay_supported = "delay_execution" in sig_params
    if delay_supported:
        executor_config["delay_execution"] = True
    backtest_config = {
        "start_time": start, "end_time": end,
        "account": BACKTEST_KWARGS["account"],
        "benchmark": BACKTEST_KWARGS["benchmark"],
        "exchange_kwargs": {
            "freq": "day", "limit_threshold": BACKTEST_KWARGS["limit_threshold"],
            "deal_price": BACKTEST_KWARGS["deal_price"],
            "open_cost": BACKTEST_KWARGS["open_cost"],
            "close_cost": BACKTEST_KWARGS["close_cost"],
            "min_cost": BACKTEST_KWARGS["min_cost"],
            "codes": "csi300",   # DDB instruments 表无 "all"，显式预载
        },
        "executor": executor.SimulatorExecutor(**executor_config),
    }
    portfolio_metric_dict, _ = backtest(strategy=strategy, **backtest_config)
    freq = "{0}{1}".format(*Freq.parse("day"))
    report, _ = portfolio_metric_dict.get(freq)
    return report[["return", "cost", "bench", "turnover"]].copy(), {
        "delay_execution_supported": delay_supported,
        "qlib_file": qlib.__file__,
    }


def compute_curves(report: pd.DataFrame) -> dict:
    """主口径 cumsum + 附表复利口径（明确分开，不拼接）。"""
    net = report["return"] - report["cost"]
    return {
        "report": report, "net_daily": net,
        "curve": net.cumsum(),
        "excess_curve": (net - report["bench"]).cumsum(),
        "compound_nav": (1 + net).cumprod(),
        "compound_cum": float((1 + net).prod() - 1),
        "max_drawdown_compound": float(
            ((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min()),
        "cost_sum": float(report["cost"].sum()),
        "turnover_daily_mean": float(report["turnover"].mean()),
    }


def cmd_preflight() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "window_bounds": WINDOW_BOUNDS, "arms": list(ARMS),
        "backtest_kwargs": BACKTEST_KWARGS,
        "signal_paths": {a: {w: str(p) for w, p in m.items()}
                         for a, m in SIGNAL_PATHS.items()},
        "signal_sha256": {}, "per_window": {},
        "signal_generation_link": LEGACY_NOTE,
        "formula_main": "net_daily=return-cost; curve=net_daily.cumsum(); "
                        "delta_end=curve_A1[-1]-curve_G1[-1]",
    }
    for arm in ARMS:
        for w in WINDOW_BOUNDS:
            p = SIGNAL_PATHS[arm][w]
            assert p.is_file(), f"信号缺失：{p}"
            manifest["signal_sha256"][f"{arm}|{w}"] = sha256_file(p)
    for w, (start, end) in WINDOW_BOUNDS.items():
        wides = {a: load_wide(a, w) for a in ARMS}
        mask, stats = common_set(wides)
        for a, wide in wides.items():
            assert not wide.stack().index.duplicated().any(), f"{a} 重复键"
            assert wide.index.max() <= pd.Timestamp(FORWARD_CUTOFF)
            assert np.isfinite(wide.to_numpy()[~np.isnan(wide.to_numpy())]).all()
        per_day = pd.DataFrame({
            "n_common": mask.sum(axis=1),
            **{f"cov_{a}": w.notna().sum(axis=1)
               for a, w in wides.items()}})
        low_days = per_day[per_day["n_common"] < BACKTEST_KWARGS["topk"]]
        manifest["per_window"][w] = {
            **stats, "n_days_low_common": int(len(low_days)),
            "low_common_dates": [str(d.date()) for d in low_days.index],
        }
        per_day.to_parquet(OUT_DIR / f"common_coverage_{w}.parquet")
        # 共同集合 SHA（不看结果即可冻结）
        manifest[f"common_set_sha256_{w}"] = hashlib.sha256(
            mask.astype(np.uint8).to_numpy().tobytes()).hexdigest()
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info(f"[preflight] manifest 冻结：{len(manifest['signal_sha256'])} "
                f"信号哈希；覆盖 {json.dumps({w: manifest['per_window'][w]['n_common_cells'] for w in WINDOW_BOUNDS})}")


def cmd_backtest() -> None:
    from kronos_qlib import QlibProvider
    from qlib.workflow import R

    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    QlibProvider.init_qlib_once()          # R 记录前先初始化 DDB 后端

    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    for key, sha in manifest["signal_sha256"].items():
        arm, w = key.split("|")
        assert sha256_file(SIGNAL_PATHS[arm][w]) == sha, f"输入被改动：{key}"

    R.start(experiment_name=EXPERIMENT, recorder_name="mean_comparison_summary")
    R.log_params(**{k: v for k, v in BACKTEST_KWARGS.items()},
                 windows=json.dumps(WINDOW_BOUNDS))
    results: dict = {}
    meta: dict = {}
    for w, (start, end) in WINDOW_BOUNDS.items():
        wides = {a: load_wide(a, w) for a in ARMS}
        mask, _ = common_set(wides)
        for arm in ARMS:
            sig = prepare_signal_series(wides[arm], mask)
            report, bt_meta = run_qlib_backtest(sig, start, end)
            curves = compute_curves(report)
            report.to_parquet(OUT_DIR / f"report_{w}_{arm}.parquet")
            pd.DataFrame({"net_daily": curves["net_daily"],
                          "curve": curves["curve"],
                          "excess_curve": curves["excess_curve"],
                          "compound_nav": curves["compound_nav"]}
                         ).to_parquet(OUT_DIR / f"daily_{w}_{arm}.parquet")
            results[(w, arm)] = curves
            meta[(w, arm)] = bt_meta
            R.log_metrics(**{
                f"{w}|{arm}|curve_end": float(curves["curve"].iloc[-1]),
                f"{w}|{arm}|compound_cum": curves["compound_cum"],
                f"{w}|{arm}|cost_sum": curves["cost_sum"],
            })
            logger.info(f"[{w}|{arm}] curve_end "
                        f"{curves['curve'].iloc[-1]:+.4%} | 复利 "
                        f"{curves['compound_cum']:+.2%} | costΣ "
                        f"{curves['cost_sum']:.4f}")
        # 同窗基准逐值一致断言（日期一致）
        benches = [results[(w, a)]["report"]["bench"] for a in ARMS]
        for b in benches[1:]:
            assert benches[0].equals(b), f"{w} 基准/报告日期不一致"
    # 主结论：A1 − G1（cumsum 窗口末差，百分点）
    verdict = {}
    for w in WINDOW_BOUNDS:
        c_g1 = float(results[(w, "G1_mean")]["curve"].iloc[-1])
        c_a1 = float(results[(w, "A1_mean")]["curve"].iloc[-1])
        c_a0 = float(results[(w, "A0_bestCE_mean")]["curve"].iloc[-1])
        verdict[w] = {
            "curve_end_G1": c_g1, "curve_end_A1": c_a1,
            "delta_end_A1_minus_G1": c_a1 - c_g1,
            "delta_end_A1_minus_G1_pp": round((c_a1 - c_g1) * 100, 2),
            "A1_exceeds_G1": bool(c_a1 > c_g1),
            "delta_end_A0_minus_G1": c_a0 - c_g1,
            "days_A1_above_G1": int((results[(w, "A1_mean")]["curve"]
                                     > results[(w, "G1_mean")]["curve"]).sum()),
            "n_days": len(results[(w, "G1_mean")]["curve"]),
        }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": EXPERIMENT, "verdict": verdict,
        "backtest_meta": {f"{w}|{a}": m for (w, a), m in meta.items()},
        "manifest": manifest,
        "metrics_table": {
            f"{w}|{a}": {
                "curve_end": float(results[(w, a)]["curve"].iloc[-1]),
                "compound_cum": results[(w, a)]["compound_cum"],
                "max_drawdown_compound": results[(w, a)]["max_drawdown_compound"],
                "cost_sum": results[(w, a)]["cost_sum"],
                "turnover_daily_mean": results[(w, a)]["turnover_daily_mean"],
            } for w in WINDOW_BOUNDS for a in ARMS},
    }
    (OUT_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    R.log_artifact(str(OUT_DIR / "comparison_summary.json"))
    _plot(results)
    R.save_objects(figure_main_W3=OUT_DIR / "fig_W3_main.png",
                   figure_main_W4=OUT_DIR / "fig_W4_main.png")
    R.end_exp(recorder_status="FINISHED")  # FINISHED=回测结束，不代表优化通过
    logger.info("==== 主结论（cumsum 口径，A1−G1 窗口末百分点差） ====")
    logger.info(json.dumps({w: verdict[w]["delta_end_A1_minus_G1_pp"]
                            for w in WINDOW_BOUNDS}))


def _plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK HK", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    style = {"G1_mean": ("#12456e", "-"), "A1_mean": ("#de7c33", "-"),
             "A0_bestCE_mean": ("#777777", ":")}
    for w in WINDOW_BOUNDS:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for arm in ("G1_mean", "A1_mean"):
            c, ls = style[arm]
            axes[0].plot(results[(w, arm)]["curve"].index,
                         results[(w, arm)]["curve"], color=c, linestyle=ls,
                         label=arm.replace("_", " "))
        axes[0].plot(results[(w, "G1_mean")]["report"].index,
                     results[(w, "G1_mean")]["report"]["bench"].cumsum(),
                     color="black", linestyle="--", label="沪深300")
        axes[0].set_title(f"{w}：扣费累计收益（Qlib 原生，open 成交，cumsum 口径，项目 G1 mean 基线）")
        axes[0].set_ylabel("累计日收益（相加）")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        d = (results[(w, "A1_mean")]["curve"]
             - results[(w, "G1_mean")]["curve"])
        axes[1].plot(d.index, d, color="#de7c33")
        axes[1].axhline(0, color="grey", linewidth=0.8)
        axes[1].set_ylabel("A1 − G1 差额")
        axes[1].set_title(f"A1−G1 逐日差额（窗口末 {d.iloc[-1] * 100:+.2f}pp）")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fig_{w}_main.png", dpi=150)
        plt.close(fig)
        # 参考图（含 A0）
        fig2, ax = plt.subplots(figsize=(10, 5))
        for arm in ARMS:
            c, ls = style[arm]
            ax.plot(results[(w, arm)]["curve"].index,
                    results[(w, arm)]["curve"], color=c, linestyle=ls,
                    label=arm.replace("_", " ") + ("（全参数重训参考）"
                                                   if arm.startswith("A0") else ""))
        ax.set_title(f"{w}：三臂扣费累计（A0_bestCE 仅参考）")
        ax.legend()
        ax.grid(alpha=0.3)
        fig2.savefig(OUT_DIR / f"fig_{w}_arms.png", dpi=150)
        plt.close(fig2)


def main() -> int:
    parser = argparse.ArgumentParser(description="G1 vs A1 mean 同口径 Qlib 对照")
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "backtest"])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(str(OUT_DIR / f"{args.stage}.log"), enqueue=False)
    if args.stage == "preflight":
        cmd_preflight()
    else:
        cmd_backtest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
