"""G1 max 与 C1 固定候选的同口径 Qlib 核验（20260910 计划）。

仅使用历史缓存信号（≤2026-07-24），零训练、零推理、零 GPU、forward 零
读取。三候选定义在跑前冻结（计划 §2）：

- ``G1_mean``：原 G1 s100 mean 缓存原值（唯一主基线）；
- ``G1_max``：同一 s100 生成链的 max 缓存（20 采样路径先平均 close，再
  对 10 个预测时间步取 max，除 close_t 减 1——见
  ``baseline_suite/signal.py:compute_variants_from_preds``，非未来真实
  最高价、非挑最好路径）；
- ``C1_mean_ensemble``： ``(s100+s101+s102)/3``，平均同日同股**信号**
  （skipna=False，缺任一 seed 即无信号），不是平均策略收益。

主口径：``net_daily=return-cost`` 的 ``cumsum``；对每窗报告
``D_max=curve_max[-1]-curve_mean[-1]``、``D_C1=curve_C1[-1]-curve_mean[-1]``
（百分点）。股票池固定为**原 G1 有效股票日集合**（不缩池）；候选缺格只
标记不裁剪。先逐值复现 ``mean_comparison_20260909`` 的 G1 基线 report
（≤1e-10、日期一致）再跑候选；基准 DDB 标识以能逐值复现旧 report.bench
者为准（探针实证 ``000300.SH`` 可复现、``SH000300`` 报日历错误——与旧
manifest 记录不一致处如实留档）。旧信号生成链证据不全，保持
``legacy_unverified``。结果仅作历史候选核验，不认证“无过拟合赢家”。

用法::

    python -m g1_fixed_candidates.run --stage preflight
    python -m g1_fixed_candidates.run --stage backtest
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qlib.workflow.recorder import Recorder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "g1_fixed_candidates_20260910"
BASELINE_DIR = REPO_ROOT / "artifacts" / "experiments" / "mean_comparison_20260909"
FORWARD_CUTOFF = "2026-07-24"
WINDOW_BOUNDS = {"W3": ("2025-07-01", "2025-12-31"),
                 "W4": ("2026-01-01", "2026-07-24")}
EXPECTED_DAYS = {"W3": 126, "W4": 134}
INPUT_ARMS = ("G1_mean", "G1_max", "G2S101_mean", "G2S102_mean")
ARMS_MAIN = ("G1_mean", "G1_max", "C1_mean_ensemble")
ARMS_APPENDIX = ("G2S101_mean", "G2S102_mean")   # 仅附表：C1 构造一致性与风险
C1_SEED_ARMS = ("G1_mean", "G2S101_mean", "G2S102_mean")
SIGNAL_PATHS = {
    "G1_mean": {"W3": REPO_ROOT / "g5_head" / "data" /
                "daily_signals_2025h2_G1_mean.parquet",
                "W4": REPO_ROOT / "finetune_suite" / "data" / "g1" /
                "daily_signals_backtest_G1_mean.parquet"},
    "G1_max": {"W3": REPO_ROOT / "g5_head" / "data" /
               "daily_signals_2025h2_G1_max.parquet",
               "W4": REPO_ROOT / "finetune_suite" / "data" / "g1" /
               "daily_signals_backtest_G1_max.parquet"},
    "G2S101_mean": {"W3": REPO_ROOT / "finetune_suite" / "data" / "g2" / "s101" /
                    "daily_signals_2025h2_G2S101_mean.parquet",
                    "W4": REPO_ROOT / "finetune_suite" / "data" / "g2" / "s101" /
                    "daily_signals_backtest_G2S101_mean.parquet"},
    "G2S102_mean": {"W3": REPO_ROOT / "finetune_suite" / "data" / "g2" / "s102" /
                    "daily_signals_2025h2_G2S102_mean.parquet",
                    "W4": REPO_ROOT / "finetune_suite" / "data" / "g2" / "s102" /
                    "daily_signals_backtest_G2S102_mean.parquet"},
}
BACKTEST_KWARGS = dict(
    account=100_000_000, benchmark="000300.SH", deal_price="open",
    open_cost=0.001, close_cost=0.0015, min_cost=5, limit_threshold=0.095,
    topk=50, n_drop=5, hold_thresh=5,
)
MAX_GE_MEAN_TOL = 1e-6
REPRO_TOL = 1e-10
EXPERIMENT = "kronos-g1-fixed-candidates-20260910"
LEGACY_NOTE = ("legacy_unverified：信号为历史缓存，生成链脚本在库"
               "（g5_head/gen_g1_2025h2.py、finetune_suite/run_g1_backtest.py、"
               "run_g2_supp.py、baseline_suite/signal.py:compute_variants_from_"
               "preds），但推理日志/推理 seed 未归档，无法重建链式证据；"
               "训练 seed：G1=tokenizer+predictor s100 链、G2S101/S102=共享 G1 "
               "tokenizer 的 predictor s101/s102（run_g2_supp.py 头注）")
# 旧 manifest 的 benchmark 记为 SH000300，与提交代码 000300.SH 不一致；
# 20260910 探针实证：000300.SH 逐值复现旧 report.bench，SH000300 报
# “calendar not exists”。以复现者为准，差异如实留档（计划 §4）。
BENCHMARK_PROVENANCE = {
    "chosen": "000300.SH",
    "old_manifest_record": "SH000300",
    "probe": "3 日微回测：000300.SH bench=[0,0.000234,0.006183] 与旧 "
             "report 逐值一致；SH000300 抛 ValueError calendar not exists",
}


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


def g1_pool(wides: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """主对照池 = 原 G1 s100 mean 有效股票日集合（不缩池）。"""
    return wides["G1_mean"].notna()


def build_c1(seed_wides: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """C1 = 三 seed 同日同股**信号**等权平均（skipna=False）。

    逐 (datetime, code) 键对齐（pandas 按索引/列对齐，非位置配对）；
    任一 seed 缺格 → 该格 NaN（不缩池、不补 0、不改两 seed 均值）。
    平均对象是信号宽表，不是策略收益/NAV 曲线。
    """
    missing = [a for a in C1_SEED_ARMS if a not in seed_wides]
    if missing:
        raise ValueError(f"C1 缺 seed 输入：{missing}")
    a, b, c = (seed_wides[k] for k in C1_SEED_ARMS)
    return (a + b + c) / 3                          # NaN 传播 = skipna=False


def require_main_ready(c1: pd.DataFrame, mask: pd.DataFrame) -> None:
    """主比较门禁：C1 必须在全部基线格上有有限信号，否则拒绝主比较。"""
    miss = int((mask & c1.isna()).sum().sum())
    if miss:
        raise ValueError(f"C1 在基线池上缺 {miss} 格（缺 seed 拒绝主比较，"
                         "不改两 seed 均值/不补 0）")


def max_ge_mean_gate(mean: pd.DataFrame, mx: pd.DataFrame,
                     tol: float = MAX_GE_MEAN_TOL) -> dict:
    """文件一致性门禁：同一 s100 有效格上 G1_max ≥ G1_mean（容差 tol）。

    仅校验文件一致性，不保证策略收益 max ≥ mean。
    """
    diff = (mx - mean).where(mean.notna())
    violations = int((diff < -tol).sum().sum())
    return {"min_diff": float(diff.min().min()),
            "max_diff": float(diff.max().max()),
            "violations": violations, "tolerance": tol}


def gate_ok(stats: dict) -> bool:
    return stats["violations"] == 0


def assert_full_window(wide: pd.DataFrame, bounds: tuple[str, str],
                       expected_days: int) -> None:
    """完整决策窗口：天数=预期且不裁尾日（禁止去掉末尾 20 日）。"""
    start, end = bounds
    assert len(wide) == expected_days, \
        f"窗口天数 {len(wide)} ≠ 预期 {expected_days}（怀疑裁尾/缺日）"
    assert wide.index.min() >= pd.Timestamp(start), "窗口起点早于边界"
    assert wide.index.max() <= pd.Timestamp(end), "窗口终点晚于边界"


def prepare_signal_series(wide: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """池掩码下的信号 Series（instrument, datetime MultiIndex，无平移）。

    qlib-ddb 的 TopkDropoutStrategy 内部以 ``get_step_time(shift=1)`` 读上一
    交易日信号 → t 日信号在 t+1 开盘执行；本函数**不再额外平移**（防双重
    shift，合成测试锁定索引恒等）。缺格保持 NaN：候选该日可选集自然变小，
    池不缩（计划 §3）。
    """
    sig = wide.where(mask)
    s = sig.stack()
    s.index.names = ["datetime", "instrument"]
    return s.swaplevel().sort_index()


def run_qlib_backtest(signal: pd.Series, start: str, end: str) -> tuple:
    """单臂窗 Qlib 原生回测 → 逐日 report（return/bench/cost/turnover）。

    与 ``qlib_mean_comparison.run``（0b982e8 已核验）同一路径；每臂窗
    独立 strategy/executor/account。
    """
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
        "days_with_trade": int((report["turnover"] > 0).sum()),
    }


def verify_baseline_reproduction(new_report: pd.DataFrame,
                                 base_report: pd.DataFrame,
                                 tol: float = REPRO_TOL) -> None:
    """基线复现门禁：日期逐值一致 + return/cost/bench/turnover ≤ tol。"""
    assert new_report.index.equals(base_report.index), \
        "基线复现日期不一致（拒绝比较）"
    for col in ("return", "cost", "bench", "turnover"):
        diff = float((new_report[col] - base_report[col]).abs().max())
        assert diff <= tol, f"基线复现 {col} 偏差 {diff:.3e} > {tol:.0e}"


def _fingerprint(wide: pd.DataFrame) -> dict:
    return {
        "shape": list(wide.shape), "dtype": str(wide.dtypes.iloc[0]),
        "dates": [str(wide.index.min().date()), str(wide.index.max().date())],
        "n_days": int(len(wide)), "n_codes": int(wide.shape[1]),
        "n_cells_notna": int(wide.notna().sum().sum()),
        "all_finite": bool(np.isfinite(
            wide.to_numpy()[~np.isnan(wide.to_numpy())]).all()),
        "dup_keys": bool(wide.index.duplicated().any()
                         or wide.columns.duplicated().any()),
    }


def cmd_preflight() -> None:
    inputs_dir = OUT_DIR / "inputs_original"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "window_bounds": WINDOW_BOUNDS, "expected_days": EXPECTED_DAYS,
        "arms_main": list(ARMS_MAIN), "arms_appendix": list(ARMS_APPENDIX),
        "c1_seed_arms": list(C1_SEED_ARMS),
        "backtest_kwargs": BACKTEST_KWARGS,
        "benchmark_provenance": BENCHMARK_PROVENANCE,
        "formula_main": "net_daily=return-cost; curve=net_daily.cumsum(); "
                        "D_max=curve_max[-1]-curve_mean[-1]; "
                        "D_C1=curve_C1[-1]-curve_mean[-1]",
        "signal_generation_link": LEGACY_NOTE,
        "inputs": {}, "per_window": {},
    }
    for arm in INPUT_ARMS:
        for w in WINDOW_BOUNDS:
            src = SIGNAL_PATHS[arm][w]
            assert src.is_file(), f"信号缺失：{src}"
            dst = inputs_dir / f"{arm}_{w}.parquet"
            shutil.copy2(src, dst)                 # 原字节另存（非软链）
            sha_src, sha_dst = sha256_file(src), sha256_file(dst)
            assert sha_src == sha_dst, f"导出字节不一致：{arm}|{w}"
            wide = load_wide(arm, w)
            manifest["inputs"][f"{arm}|{w}"] = {
                "original_path": str(src), "source_sha256": sha_src,
                "export_path": str(dst.relative_to(REPO_ROOT)),
                "export_sha256": sha_dst,
                "size_bytes": src.stat().st_size,
                **_fingerprint(wide),
                "evidence": LEGACY_NOTE,
            }
    for w, (start, end) in WINDOW_BOUNDS.items():
        wides = {a: load_wide(a, w) for a in INPUT_ARMS}
        for a, wide in wides.items():
            assert_full_window(wide, WINDOW_BOUNDS[w], EXPECTED_DAYS[w])
            assert wide.index.max() <= pd.Timestamp(FORWARD_CUTOFF)
            assert not _fingerprint(wide)["dup_keys"], f"{a} 重复键"
            assert _fingerprint(wide)["all_finite"], f"{a} 非有限值"
        g1m = wides["G1_mean"]
        assert all(a_w.index.equals(g1m.index) and a_w.columns.equals(g1m.columns)
                   for a_w in wides.values()), f"{w} 各臂网格不一致"
        mask = g1_pool(wides)
        gate = max_ge_mean_gate(g1m, wides["G1_max"])
        assert gate_ok(gate), f"{w} max≥mean 门禁失败：{gate}"
        c1 = build_c1(wides)
        c1_missing = int((mask & c1.isna()).sum().sum())
        require_main_ready(c1, mask)
        c1.to_parquet(OUT_DIR / f"c1_signal_{w}.parquet")
        mask.astype(np.uint8).rename_axis("datetime").to_parquet(
            OUT_DIR / f"g1_pool_mask_{w}.parquet")
        pd.DataFrame({"datetime": g1m.index}).set_index("datetime").to_parquet(
            OUT_DIR / f"calendar_{w}.parquet")
        manifest["per_window"][w] = {
            "n_days": EXPECTED_DAYS[w],
            "n_pool_cells": int(mask.sum().sum()),
            "max_ge_mean_gate": gate,
            "c1_missing_on_pool": c1_missing,
            "per_arm_vs_pool": {
                a: {"extra_vs_pool": int((wide.notna() & ~mask).sum().sum()),
                    "missing_on_pool": int((mask & wide.isna()).sum().sum())}
                for a, wide in wides.items()},
            "pool_set_sha256": hashlib.sha256(
                mask.astype(np.uint8).to_numpy().tobytes()).hexdigest(),
        }
        logger.info(f"[preflight|{w}] 池 {int(mask.sum().sum())} 格；max−mean "
                    f"min {gate['min_diff']:.6f} 违反 {gate['violations']}；"
                    f"C1 缺格 {c1_missing}")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info(f"[preflight] manifest 冻结：{len(manifest['inputs'])} 输入已"
                f"复制入 {inputs_dir.relative_to(REPO_ROOT)} 并记录双向 SHA")


def save_figure_artifacts(recorder: Recorder, output_dir: Path) -> None:
    """按文件上传四张对照图，避免把Path对象pickle为图片产物。

    :param recorder: 当前Qlib实验记录。
    :param output_dir: 已生成对照图的目录。
    :returns: 无返回值。
    """
    for window in WINDOW_BOUNDS:
        for kind in ("main", "appendix"):
            path = output_dir / f"fig_{window}_{kind}.png"
            recorder.save_objects(local_path=str(path), artifact_path="figures")


def cmd_backtest() -> None:
    from kronos_qlib import QlibProvider
    from qlib.workflow import R

    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    QlibProvider.init_qlib_once()

    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    for key, rec in manifest["inputs"].items():
        assert sha256_file(Path(rec["original_path"])) == rec["source_sha256"], \
            f"输入被改动：{key}"

    with R.start(experiment_name=EXPERIMENT,
                 recorder_name="g1_fixed_candidates_summary"):
        rec = R.get_recorder()
        rid = rec.id                      # MLflowRecorder.id = mlflow run_id
        exp_id = rec.experiment_id
        R.log_params(**{k: v for k, v in BACKTEST_KWARGS.items()},
                     windows=json.dumps(WINDOW_BOUNDS),
                     arms=",".join(ARMS_MAIN),
                     c1_formula="(G1_s100+G2S101+G2S102)/3, skipna=False")
        R.set_tags(stage="backtest", training="0", inference="0",
                   gpu="0", forward_read="0",
                   claim="historical_candidate_check_only")
        results: dict = {}
        meta: dict = {}
        repro: dict = {}
        for w, (start, end) in WINDOW_BOUNDS.items():
            wides = {a: load_wide(a, w) for a in INPUT_ARMS}
            mask = g1_pool(wides)
            gate = max_ge_mean_gate(wides["G1_mean"], wides["G1_max"])
            assert gate_ok(gate), f"{w} max≥mean 门禁失败"
            c1 = build_c1(wides)
            require_main_ready(c1, mask)
            series = {**{a: prepare_signal_series(wides[a], mask)
                         for a in INPUT_ARMS},
                      "C1_mean_ensemble": prepare_signal_series(c1, mask)}
            for arm in ("G1_mean",) + ARMS_MAIN[1:] + ARMS_APPENDIX:
                report, bt_meta = run_qlib_backtest(series[arm], start, end)
                if arm == "G1_mean":               # 基线复现硬门禁
                    base = pd.read_parquet(
                        BASELINE_DIR / f"report_{w}_G1_mean.parquet")
                    verify_baseline_reproduction(report, base, REPRO_TOL)
                    repro[w] = {"tol": REPRO_TOL, "passed": True,
                                "n_days": len(report)}
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
                    f"{w}.{arm}.curve_end": float(curves["curve"].iloc[-1]),
                    f"{w}.{arm}.compound_cum": curves["compound_cum"],
                    f"{w}.{arm}.cost_sum": curves["cost_sum"],
                    f"{w}.{arm}.turnover_mean": curves["turnover_daily_mean"],
                })
                logger.info(f"[{w}|{arm}] curve_end "
                            f"{curves['curve'].iloc[-1]:+.4%} | costΣ "
                            f"{curves['cost_sum']:.4f} | 成交日 "
                            f"{curves['days_with_trade']}/{len(report)}")
            benches = [results[(w, a)]["report"]["bench"]
                       for a in ARMS_MAIN + ARMS_APPENDIX]
            for b in benches[1:]:
                assert benches[0].equals(b), f"{w} 基准/报告日期不一致"
        verdict = {}
        for w in WINDOW_BOUNDS:
            c_mean = results[(w, "G1_mean")]["curve"]
            c_max = results[(w, "G1_max")]["curve"]
            c_c1 = results[(w, "C1_mean_ensemble")]["curve"]
            d_max = float(c_max.iloc[-1] - c_mean.iloc[-1])
            d_c1 = float(c_c1.iloc[-1] - c_mean.iloc[-1])
            verdict[w] = {
                "curve_end_G1_mean": float(c_mean.iloc[-1]),
                "curve_end_G1_max": float(c_max.iloc[-1]),
                "curve_end_C1": float(c_c1.iloc[-1]),
                "D_max_pp": round(d_max * 100, 2),
                "D_C1_pp": round(d_c1 * 100, 2),
                "days_max_above": int((c_max > c_mean).sum()),
                "days_c1_above": int((c_c1 > c_mean).sum()),
                "n_days": len(c_mean),
                "appendix_curve_ends": {
                    a: float(results[(w, a)]["curve"].iloc[-1])
                    for a in ARMS_APPENDIX},
            }
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": EXPERIMENT, "recorder_id": rid,
            "verdict": verdict,
            "baseline_reproduction": repro,
            "backtest_meta": {f"{w}|{a}": m for (w, a), m in meta.items()},
            "metrics_table": {
                f"{w}|{a}": {
                    "curve_end": float(results[(w, a)]["curve"].iloc[-1]),
                    "compound_cum": results[(w, a)]["compound_cum"],
                    "max_drawdown_compound":
                        results[(w, a)]["max_drawdown_compound"],
                    "cost_sum": results[(w, a)]["cost_sum"],
                    "turnover_daily_mean":
                        results[(w, a)]["turnover_daily_mean"],
                    "days_with_trade": results[(w, a)]["days_with_trade"],
                } for w in WINDOW_BOUNDS for a in ARMS_MAIN + ARMS_APPENDIX},
            "manifest": manifest,
            "disclaimer": "历史候选核验（W3/W4 已参与候选选择），不构成独立"
                          "验证，不排除过拟合；不宣称已排除过拟合。",
        }
        (OUT_DIR / "comparison_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        _plot(results)
        R.log_artifact(str(OUT_DIR / "comparison_summary.json"))
        R.log_artifact(str(OUT_DIR / "manifest.json"))
        save_figure_artifacts(rec, OUT_DIR)
        (OUT_DIR / "R_identity.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "experiment_id": exp_id,
                        "recorder_id": rid},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("==== 主结论（cumsum 口径，相对 G1 mean 窗口末百分点差） ====")
    logger.info(json.dumps({w: {"D_max_pp": verdict[w]["D_max_pp"],
                                "D_C1_pp": verdict[w]["D_C1_pp"]}
                            for w in WINDOW_BOUNDS}, ensure_ascii=False))


def _plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK HK", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    style = {"G1_mean": ("#12456e", "-"), "G1_max": ("#de7c33", "-"),
             "C1_mean_ensemble": ("#3d9970", "-"),
             "G2S101_mean": ("#999999", ":"), "G2S102_mean": ("#bbbbbb", ":")}
    for w in WINDOW_BOUNDS:
        curves = {a: results[(w, a)]["curve"] for a in ARMS_MAIN}
        bench = results[(w, "G1_mean")]["report"]["bench"].cumsum()
        # 主图：三候选 + 市场参考 + 下半幅 D_max/D_C1（不只画收益高的一条）
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for arm in ARMS_MAIN:
            c, ls = style[arm]
            axes[0].plot(curves[arm].index, curves[arm], color=c,
                         linestyle=ls, label=arm.replace("_", " "))
        axes[0].plot(bench.index, bench, color="black", linestyle="--",
                     label="沪深300")
        axes[0].set_title(f"{w}：扣费累计收益（Qlib 原生，open 成交，cumsum "
                          "口径，G1 mean 基线）")
        axes[0].set_ylabel("累计日收益（相加）")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        d_max = curves["G1_max"] - curves["G1_mean"]
        d_c1 = curves["C1_mean_ensemble"] - curves["G1_mean"]
        axes[1].plot(d_max.index, d_max, color="#de7c33", label="G1 max − G1 mean")
        axes[1].plot(d_c1.index, d_c1, color="#3d9970", label="C1 − G1 mean")
        axes[1].axhline(0, color="grey", linewidth=0.8)
        axes[1].set_ylabel("相对 G1 mean 差额")
        axes[1].set_title(f"候选−基线逐日差额（窗口末 D_max {d_max.iloc[-1]*100:+.2f}pp"
                          f"｜D_C1 {d_c1.iloc[-1]*100:+.2f}pp）")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fig_{w}_main.png", dpi=150)
        plt.close(fig)
        # 附图：s101/s102 单体仅作 C1 构造一致性与风险解释（非候选）
        fig2, ax = plt.subplots(figsize=(10, 5))
        for arm in ARMS_MAIN + ARMS_APPENDIX:
            c, ls = style[arm]
            ax.plot(results[(w, arm)]["curve"].index,
                    results[(w, arm)]["curve"], color=c, linestyle=ls,
                    label=arm.replace("_", " ")
                    + ("（附表，非候选）" if arm in ARMS_APPENDIX else ""))
        ax.set_title(f"{w}：三候选与 C1 成分 seed（s101/s102 仅附表）")
        ax.legend()
        ax.grid(alpha=0.3)
        fig2.savefig(OUT_DIR / f"fig_{w}_appendix.png", dpi=150)
        plt.close(fig2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="G1 max 与 C1 固定候选同口径 Qlib 核验")
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
