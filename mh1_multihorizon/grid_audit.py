"""G1 信号日期对照（FULL vs CROPPED）· 两臂两窗固定对照（20260908 计划）。

在完全相同的 engine_v2 条件下，解释 G1 完整逐日信号（FULL）与 MH1 裁剪
日期信号（CROPPED）之间的收益差异：同一 G1 文件、同一冻结列宇宙、同一行情
/涨跌停/停牌掩码、同一 v2 配置，**唯一区别是提供信号的日期**。零训练、零
推理；读取上界 ≤ 2026-07-24。

CROPPED 复现锚：逐日净收益必须与 ``22c6677`` 纠偏轮的 G1_mean（同网格参照）
逐位一致（≤1e-12），不一致先查原因、不进收益解释。

用法::

    python -m mh1_multihorizon.grid_audit --stage preflight
    python -m mh1_multihorizon.grid_audit --stage replay
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from mh1_multihorizon.config import sha256_file
from mh1_multihorizon.replay_corrected import (
    FORWARD_CUTOFF, G1_MEAN_PARQUET, RUNS, SOURCE_DATA, WINDOW_BOUNDS,
    decompose, fetch_market, frozen_universe, replay_one, save_transactions,
)

GRID_DIR = Path(__file__).resolve().parent / "data" / "grid_audit_20260908"
ARMS = ("FULL", "CROPPED")


# ============================================================
# 核心构造（计划 §4 数据操作等价）
# ============================================================


def build_arms(
    raw_g1: pd.DataFrame, calendar: pd.DatetimeIndex,
    columns: list[str], mh1_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FULL = 原始逐日信号按完整日历/冻结列宇宙重索引；CROPPED 仅保留
    MH1 日期（其余置 NaN，不 ffill；原始缺失保持缺失）。"""
    full = raw_g1.reindex(index=calendar, columns=columns)
    cropped = full.copy()
    cropped.loc[~cropped.index.isin(mh1_dates), :] = float("nan")
    return full, cropped


def report_extra_dates(
    calendar: pd.DatetimeIndex, sig_dates: pd.DatetimeIndex,
    mh1_dates: pd.DatetimeIndex,
) -> dict:
    """FULL 比 CROPPED 多出的信号日期清单与位置（连续尾部 vs 内部缺口）。

    尾部定义：最后一个保留日之后的全部缺失信号日；其余为内部缺口。
    """
    dropped = sorted(set(sig_dates) - set(mh1_dates))
    last_kept = max(mh1_dates)
    tail = [d for d in dropped if d > last_kept]
    internal = [d for d in dropped if d <= last_kept]
    return {
        "n_extra_dates": len(dropped),
        "extra_dates": [str(d.date()) for d in dropped],
        "n_tail_dates": len(tail),
        "tail_dates": [str(d.date()) for d in tail],
        "n_internal_dates": len(internal),
        "internal_dates": [str(d.date()) for d in internal],
    }


def common_mask_ew(
    px_wide: pd.DataFrame, tradeable: pd.DataFrame,
    full: pd.DataFrame, cropped: pd.DataFrame,
) -> pd.Series:
    """共同掩码等权基准：``tradeable & full.notna() & cropped.notna()``。

    尾部无共同信号的日子不产生基准值（不填零、不声称覆盖完整窗口）。
    """
    rets = px_wide.pct_change(fill_method=None)
    mask = (tradeable.reindex_like(rets).fillna(False)
            & full.reindex_like(rets).notna().fillna(False)
            & cropped.reindex_like(rets).notna().fillna(False))
    denom = (mask & rets.notna()).sum(axis=1).replace(0, np.nan)
    return (rets.where(mask).sum(axis=1) / denom).dropna()


def verify_manifest_hashes(manifest: dict) -> None:
    for path, expect in manifest["inputs"].items():
        got = sha256_file(Path(path))
        if got != expect:
            raise RuntimeError(f"输入被改动：{path}")


# ============================================================
# 评价（计划 §5 主表）
# ============================================================


def _cum(series: pd.Series) -> float:
    v = series.dropna()
    return float((1.0 + v).prod() - 1.0) if len(v) else float("nan")


def _aer252(series: pd.Series) -> float:
    v = series.dropna()
    if len(v) < 2:
        return float("nan")
    nav = (1.0 + v).prod()
    return float(nav ** (252.0 / len(v)) - 1.0)


def _mdd(series: pd.Series) -> float:
    v = series.dropna()
    if not len(v):
        return float("nan")
    nav = (1.0 + v).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def evaluate_arm(
    name: str, daily: pd.DataFrame, trades, bench_idx: pd.Series,
    bench_ew_arm: pd.Series, bench_ew_common: pd.Series,
) -> dict:
    """单臂主表行：自身毛/净、指数毛/净超额（累计+AER）、真实回撤、成本、
    换手、交易日数；双等权口径（各臂掩码 vs 共同掩码）分别给有效日期数。"""
    ex_idx_g = daily["gross_return"] - bench_idx
    ex_idx_n = daily["net_return"] - bench_idx
    ex_ew_g = daily["gross_return"] - bench_ew_arm
    ex_ew_n = daily["net_return"] - bench_ew_arm
    cm_idx = daily.index.intersection(bench_ew_common.index)
    ex_cm_g = daily["gross_return"].loc[cm_idx] - bench_ew_common.loc[cm_idx]
    ex_cm_n = daily["net_return"].loc[cm_idx] - bench_ew_common.loc[cm_idx]
    tf = trades.to_frame()
    return {
        "arm": name,
        "self_gross_cum": _cum(daily["gross_return"]),
        "self_net_cum": _cum(daily["net_return"]),
        "self_net_nav_end": float(daily["net_nav"].iloc[-1]),
        "self_gross_nav_end": float(daily["gross_nav"].iloc[-1]),
        "strategy_max_drawdown_net": _mdd(daily["net_return"]),
        "idx_excess_gross_cum": _cum(ex_idx_g),
        "idx_excess_net_cum": _cum(ex_idx_n),
        "idx_excess_gross_aer": _aer252(ex_idx_g),
        "idx_excess_net_aer": _aer252(ex_idx_n),
        "ew_own_mask_excess_net_cum": _cum(ex_ew_n),
        "ew_own_mask_n_days": int(len(ex_ew_n.dropna())),
        "ew_common_excess_gross_cum": _cum(ex_cm_g),
        "ew_common_excess_net_cum": _cum(ex_cm_n),
        "ew_common_n_days": int(len(cm_idx)),
        "cost_sum": float(daily["cost"].sum()),
        "turnover_double_sum": float(daily["turnover_double"].sum()),
        "n_trade_days": int((daily["cost"] > 0).sum()),
        "n_cal_days": int(len(daily)),
    }


# ============================================================
# 差异切分（计划 §5：首次实际输入信号不同的日期为切点）
# ============================================================


def split_analysis(
    full: pd.DataFrame, cropped: pd.DataFrame,
    daily_f: pd.DataFrame, daily_c: pd.DataFrame,
) -> dict:
    diverge = min(d for d in full.index
                  if full.loc[d].notna().any() and not cropped.loc[d].notna().any())
    pre = daily_f.index < diverge
    post = ~pre
    pre_net_err = float((daily_f["net_return"][pre]
                         - daily_c["net_return"][pre]).abs().max())
    return {
        "diverge_date": str(diverge.date()),
        "pre_days": int(pre.sum()),
        "pre_max_abs_net_err": pre_net_err,
        "post_days": int(post.sum()),
        "post_gross_diff_sum": float(
            (daily_f["gross_return"][post] - daily_c["gross_return"][post]).sum()),
        "post_net_diff_sum": float(
            (daily_f["net_return"][post] - daily_c["net_return"][post]).sum()),
        "post_cost_diff_sum": float(
            (daily_f["cost"][post] - daily_c["cost"][post]).sum()),
        "nav_end_diff": float(daily_f["net_nav"].iloc[-1]
                              - daily_c["net_nav"].iloc[-1]),
    }


# ============================================================
# CLI 阶段
# ============================================================


def _mh1_dates(window: str) -> pd.DatetimeIndex:
    sig = pd.read_parquet(SOURCE_DATA / f"signals_{window}.parquet")
    return pd.DatetimeIndex(sorted(pd.to_datetime(sig["date"].unique())))


def cmd_preflight() -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    from kronos_qlib import QlibProvider

    inputs = {str(G1_MEAN_PARQUET[w]): sha256_file(G1_MEAN_PARQUET[w])
              for w in WINDOW_BOUNDS}
    inputs[str(SOURCE_DATA / "signals_W3.parquet")] = sha256_file(
        SOURCE_DATA / "signals_W3.parquet")
    inputs[str(SOURCE_DATA / "signals_W4.parquet")] = sha256_file(
        SOURCE_DATA / "signals_W4.parquet")
    manifest: dict = {
        "batch": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "window_bounds": WINDOW_BOUNDS,
        "forward_cutoff": FORWARD_CUTOFF,
        "g1_file_sha256": {w: sha256_file(G1_MEAN_PARQUET[w])
                           for w in WINDOW_BOUNDS},
        "v2_config": asdict(
            __import__("paper_replication.engine_v2", fromlist=["EngineConfigV2"])
            .EngineConfigV2()),
        "arms": list(ARMS),
        "inputs": inputs,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    for wname in WINDOW_BOUNDS:
        provider = QlibProvider("csi300", *WINDOW_BOUNDS[wname])
        cal = provider.trading_days(*WINDOW_BOUNDS[wname])
        mh1 = _mh1_dates(wname)
        raw = pd.read_parquet(G1_MEAN_PARQUET[wname])
        raw_idx = pd.DatetimeIndex(raw.index)
        manifest[wname] = {
            "n_cal_days": int(len(cal)),
            "n_g1_signal_days_in_window": int(
                len(raw_idx[(raw_idx >= cal[0]) & (raw_idx <= cal[-1])])),
            "n_mh1_dates": int(len(mh1)),
            "mh1_dates": [str(d.date()) for d in mh1],
            "n_universe": len(frozen_universe(wname)),
        }
        assert cal[-1] <= pd.Timestamp(FORWARD_CUTOFF)
    # 预检即做等价性验证（同一构造在 replay 再验一次）
    for wname in WINDOW_BOUNDS:
        raw = pd.read_parquet(G1_MEAN_PARQUET[wname])
        raw.index = pd.DatetimeIndex(raw.index)
        provider = None
        mh1 = _mh1_dates(wname)
        full, cropped = build_arms(raw, pd.DatetimeIndex(raw.index),
                                   list(raw.columns), mh1)
        keep = full.index.isin(mh1)
        pd.testing.assert_frame_equal(full[keep], cropped[keep])
    (GRID_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info(f"[preflight] manifest 冻结：{len(inputs)} 输入哈希；"
                + "；".join(f"{w} {manifest[w]['n_cal_days']} 日历/"
                           f"{manifest[w]['n_g1_signal_days_in_window']} 信号/"
                           f"{manifest[w]['n_mh1_dates']} MH1/"
                           f"{manifest[w]['n_universe']} 列"
                           for w in WINDOW_BOUNDS))


def cmd_replay() -> None:
    mpath = GRID_DIR / "manifest.json"
    if not mpath.is_file():
        raise RuntimeError("先运行 --stage preflight")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    verify_manifest_hashes(manifest)

    from kronos_qlib import QlibProvider
    from paper_replication.benchmark import probe_index_benchmark
    from paper_replication.engine_v2 import build_pool_equal_weight_benchmark_v2

    (GRID_DIR / "daily").mkdir(parents=True, exist_ok=True)
    (GRID_DIR / "transactions").mkdir(parents=True, exist_ok=True)
    summary: dict = {"batch": manifest["batch"], "windows": {}}

    for wname, (start, end) in WINDOW_BOUNDS.items():
        universe = frozen_universe(wname)
        provider = QlibProvider("csi300", start, end)
        px, trd, uls = fetch_market(provider, start, end, universe)
        bench_idx = probe_index_benchmark(provider, start, end)
        raw = pd.read_parquet(G1_MEAN_PARQUET[wname])
        raw.index = pd.DatetimeIndex(raw.index)
        mh1_dates = _mh1_dates(wname)
        full, cropped = build_arms(raw, pd.DatetimeIndex(px.index),
                                   universe, mh1_dates)
        # 运行时等价性再验证（含 NaN 位置）
        keep = full.index.isin(mh1_dates)
        pd.testing.assert_frame_equal(full[keep], cropped[keep])

        sig_dates = pd.DatetimeIndex(
            [d for d in full.index if full.loc[d].notna().any()])
        extra = report_extra_dates(px.index, sig_dates, mh1_dates)

        results: dict[str, dict] = {}
        dailies: dict[str, pd.DataFrame] = {}
        for arm_name, grid in (("FULL", full), ("CROPPED", cropped)):
            res = replay_one(grid, px, trd, uls)
            daily = decompose(res["net"], res["trades"])
            save_transactions(res["trades"],
                              GRID_DIR / "transactions" / f"{wname}_{arm_name}.parquet")
            bench_ew_arm = build_pool_equal_weight_benchmark_v2(
                px, trd, grid, fix_mask=True)
            bench_ew_common = common_mask_ew(px, trd, full, cropped)
            row = evaluate_arm(arm_name, daily, res["trades"], bench_idx,
                               bench_ew_arm, bench_ew_common)
            daily["bench_idx_ret"] = bench_idx.reindex(daily.index)
            daily["bench_ew_own_ret"] = bench_ew_arm.reindex(daily.index)
            daily["bench_ew_common_ret"] = bench_ew_common.reindex(daily.index)
            daily.to_parquet(GRID_DIR / "daily" / f"{wname}_{arm_name}.parquet")
            dailies[arm_name] = daily
            results[arm_name] = row
            logger.info(
                f"[{wname}] {arm_name}: 自身毛 {_cum(daily['gross_return']):+.2%}"
                f"/净 {_cum(daily['net_return']):+.2%} | idx 超额净累计 "
                f"{row['idx_excess_net_cum']:+.2%} | 成本Σ "
                f"{row['cost_sum']:.4f} | 交易日 {row['n_trade_days']}")

        # CROPPED 复现锚：与 22c6677 correction 的 G1_mean 逐位一致
        anchor = pd.read_parquet(
            Path(__file__).resolve().parent / "data" / "correction_20260908" /
            "daily" / f"{wname}_G1_mean.parquet")
        anchor_err = float((dailies["CROPPED"]["net_return"]
                            - anchor["net_return"].reindex(
                                dailies["CROPPED"].index)).abs().max())
        assert anchor_err <= 1e-12, (
            f"{wname} CROPPED 复现锚失败：max|Δ|={anchor_err:.3e}")
        logger.info(f"[{wname}] CROPPED 复现锚 max|Δ|={anchor_err:.1e} OK")

        split = split_analysis(full, cropped, dailies["FULL"], dailies["CROPPED"])
        # 尾部/缺口交易取证：以实际 trades 证明（不能从"无信号"推断）
        last_mh1 = max(mh1_dates)
        tail_trades = {}
        for arm_name in ARMS:
            tf = pd.read_parquet(
                GRID_DIR / "transactions" / f"{wname}_{arm_name}.parquet")
            tail = tf[tf["date"] > last_mh1]
            tail_trades[arm_name] = {
                "n_rows": int(len(tail)),
                "n_fill_days": int((tail["cost"] > 0).sum()),
                "n_bought_legs": int(sum(len(x) for x in tail["bought"])),
                "n_sold_legs": int(sum(len(x) for x in tail["sold"])),
                "cost_sum": float(tail["cost"].sum()),
            }
        summary["windows"][wname] = {
            "n_cal_days": int(px.shape[0]),
            "n_universe": len(universe),
            "extra_dates": extra,
            "split": split,
            "tail_trades_after_last_mh1": tail_trades,
            "results": results,
        }
        logger.info(
            f"[{wname}] 分歧日 {split['diverge_date']}：此前 {split['pre_days']} "
            f"日逐日净收益最大差 {split['pre_max_abs_net_err']:.1e}；此后净差和 "
            f"{split['post_net_diff_sum']:+.4%}（毛 "
            f"{split['post_gross_diff_sum']:+.4%}/成本 "
            f"{split['post_cost_diff_sum']:+.4%}）；尾部（>{last_mh1.date()}）"
            f"CROPPED 成交日 {tail_trades['CROPPED']['n_fill_days']}")

    (GRID_DIR / "grid_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    verify_manifest_hashes(manifest)
    logger.info("[replay] 完成：两臂两窗对照落盘，输入哈希前后一致")


def main() -> int:
    parser = argparse.ArgumentParser(description="G1 信号日期对照 CLI")
    parser.add_argument("--stage", required=True, choices=["preflight", "replay"])
    args = parser.parse_args()
    log_dir = Path(__file__).resolve().parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dir / f"grid_audit_{args.stage}.log"), enqueue=False)
    if args.stage == "preflight":
        cmd_preflight()
    else:
        cmd_replay()
    return 0


if __name__ == "__main__":
    sys.exit(main())
