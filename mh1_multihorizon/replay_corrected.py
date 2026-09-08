"""MH1 纠偏：真正的 engine_v2 重放 + 同路径毛/净成本分解（20260908 纠偏计划）。

背景（计划 §1）：原 ``engine_attachment``/``nav.py`` 经
``baseline_suite.pipeline.run_group`` 走的是**旧引擎**（``paper_replication.engine``），
JSON 里的 ``delay=1`` 等元数据与实际调用无关。本模块直接调用
``paper_replication.engine_v2`` 公共接口重放既有 MH1 信号（零训练、零新推理），
从**同一次收费运行**的交易日志分解毛收益/成本/净收益。

纪律：窗口/行情/信号/标签读取上界 ≤ ``FORWARD_CUTOFF``（2026-07-24）；原始
证据（信号/checkpoint/判读 JSON/净值/日志）只读并前后哈希对拍；纠正产物只写
``data/correction_20260908/``；不改共享引擎、不改原始结果数值。

用法（仓库根目录）::

    python -m mh1_multihorizon.replay_corrected --stage preflight
    python -m mh1_multihorizon.replay_corrected --stage replay
    python -m mh1_multihorizon.replay_corrected --stage pool-audit
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

from mh1_multihorizon.config import HORIZONS, MAIN_HORIZON_IDX, sha256_file

# 原始数据只读根（主工作树，绝对路径——纠偏 worktree 内未跟踪数据不在此处）
SOURCE_ROOT = Path("/home/user/workspace/Kronos")
SOURCE_DATA = SOURCE_ROOT / "mh1_multihorizon" / "data"

FORWARD_CUTOFF = "2026-07-24"
REJECTED_FORWARD_MSG = "越过 forward 封存线"
RUNS = ("S42", "M42", "S43", "M43", "S44", "M44")
H10 = f"h{HORIZONS[MAIN_HORIZON_IDX]}"
EXPECT_PROTOCOL_SHA = ("2516998a2106fa8b2630b27bb34cf6742b70c"
                       "3c6e8e33b564b46b4e28297f9ff")
WINDOW_BOUNDS = {"W3": ("2025-07-01", "2025-12-31"),
                 "W4": ("2026-01-01", "2026-07-24")}
CORR_DIR = Path(__file__).resolve().parent / "data" / "correction_20260908"

G1_MEAN_PARQUET = {
    "W3": SOURCE_ROOT / "g5_head" / "data" / "daily_signals_2025h2_G1_mean.parquet",
    "W4": SOURCE_ROOT / "finetune_suite" / "data" / "g1" /
          "daily_signals_backtest_G1_mean.parquet",
}
# 旧 MH1 冻结宇宙清单（G1/F1/F0/M 信号族列并集，W4↔backtest、W3↔2025h2 文件族）
def _frozen_universe_files() -> dict[str, list[Path]]:
    r4 = SOURCE_ROOT / "finetune_suite" / "data"
    variants = ("min", "max", "last", "mean")
    return {
        "W4": [r4 / "g1" / f"daily_signals_backtest_G1_{v}.parquet" for v in variants]
        + [r4 / f"daily_signals_backtest_F1_{v}.parquet" for v in variants]
        + [r4 / f"daily_signals_backtest_F0_{v}.parquet" for v in variants]
        + [r4 / "daily_signals_backtest_M.parquet"],
        "W3": [r4 / "g0" / f"daily_signals_2025h2_G0_{v}.parquet" for v in variants]
        + [r4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet" for v in variants]
        + [r4 / "g0" / "daily_signals_2025h2_M.parquet"],
    }


def frozen_universe(window: str) -> list[str]:
    files = _frozen_universe_files()[window]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"冻结宇宙清单缺文件：{missing}")
    return sorted(set().union(*[set(pd.read_parquet(f).columns) for f in files]))


# ============================================================
# 证据哈希（任务 A：前后不变对拍）
# ============================================================


def evidence_paths() -> list[Path]:
    paths = [
        SOURCE_DATA / "signals_W3.parquet", SOURCE_DATA / "signals_W4.parquet",
        SOURCE_DATA / "protocol.json", SOURCE_DATA / "protocol.json.sha256",
        SOURCE_DATA / "judge_summary.json", SOURCE_DATA / "engine_v2_attachment.json",
        SOURCE_DATA / "timing.json", SOURCE_DATA / "manifest.json",
        SOURCE_DATA / "nav_W3.parquet", SOURCE_DATA / "nav_W4.parquet",
        SOURCE_DATA / "stage_state.json", SOURCE_DATA / "gpu_budget.json",
    ]
    paths += [SOURCE_DATA / f"run_{r}_best.pt" for r in RUNS]
    paths += [SOURCE_DATA / f"run_{r}_history.json" for r in RUNS]
    paths += [SOURCE_DATA / "logs" / f for f in
              ("preflight.log", "smoke.log", "train.log", "evaluate.log", "nav.log")]
    paths += [G1_MEAN_PARQUET["W3"], G1_MEAN_PARQUET["W4"]]
    paths += [f for w in ("W3", "W4") for f in _frozen_universe_files()[w]]
    return paths


def hash_evidence(paths) -> dict[str, str]:
    return {str(p): sha256_file(p) for p in paths if Path(p).is_file()}


def verify_evidence(manifest: dict[str, str]) -> None:
    for path, expect in manifest.items():
        got = sha256_file(Path(path))
        if got != expect:
            raise RuntimeError(f"原始证据被改动：{path}")


# ============================================================
# 输入校验（CLI 拒绝条件，计划 §8）
# ============================================================


def validate_inputs(
    *,
    window_bounds: dict[str, tuple[str, str]] | None = None,
    runs_present: list[str] | None = None,
    source_sha256: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    """拒绝越封存线窗口 / 六臂不全 / 来源哈希不匹配。"""
    if window_bounds:
        for wname, (_s, e) in window_bounds.items():
            if str(e) > FORWARD_CUTOFF or any(
                    str(x) > FORWARD_CUTOFF for x in (_s, e)):
                raise ValueError(f"{wname} {REJECTED_FORWARD_MSG}：{(_s, e)}")
    if runs_present is not None and sorted(runs_present) != sorted(RUNS):
        raise ValueError(f"六臂不全：缺 {sorted(set(RUNS) - set(runs_present))}")
    if (source_sha256 is not None and expected_sha256 is not None
            and source_sha256 != expected_sha256):
        raise ValueError(
            f"来源哈希不匹配：{source_sha256[:16]}… vs {expected_sha256[:16]}…")


# ============================================================
# 核心重放（直连 engine_v2 真实接口，任务 B）
# ============================================================


def prepare_signal_grid(wide: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """信号宽表重索引到完整逐日交易日历；非信号日 NaN，不 ffill（计划 §4）。"""
    grid = wide.reindex(calendar)
    return grid


def replay_one(
    signal_wide: pd.DataFrame,
    px_wide: pd.DataFrame,
    tradeable: pd.DataFrame,
    uls_wide: pd.DataFrame,
    *,
    top_k: int = 50, drop_n: int = 5, min_hold: int = 5, cost_bps: float = 15.0,
) -> dict:
    """单臂单窗一次收费 v2 运行（配置与调用模块随产物落盘，禁止手写元数据）。"""
    from paper_replication.engine_v2 import (
        EngineConfigV2, build_limit_masks, run_portfolio_v2,
    )

    cfg = EngineConfigV2(top_k=top_k, drop_n=drop_n, min_hold=min_hold,
                         cost_bps=cost_bps)
    buy_blocked, sell_blocked = build_limit_masks(uls_wide=uls_wide)
    net, trades = run_portfolio_v2(
        signal_wide, px_wide, tradeable, cfg=cfg,
        buy_blocked=buy_blocked, sell_blocked=sell_blocked)
    return {
        "net": net, "trades": trades, "cfg": asdict(cfg),
        "modules": {
            "run": "paper_replication.engine_v2.run_portfolio_v2",
            "limits": "paper_replication.engine_v2.build_limit_masks",
        },
    }


def decompose(net: pd.Series, trades) -> pd.DataFrame:
    """同路径毛/净分解（任务 C）：gross = net + 同一成交路径的逐日费用。"""
    tf = trades.to_frame()
    daily_cost = (tf.groupby("date")["cost"].sum()
                  .reindex(net.index, fill_value=0.0))
    gross = net + daily_cost
    out = pd.DataFrame({
        "gross_return": gross, "net_return": net, "cost": daily_cost})
    out["gross_nav"] = (1.0 + out["gross_return"]).cumprod()
    out["net_nav"] = (1.0 + out["net_return"]).cumprod()
    out["nav_drag"] = out["gross_nav"] - out["net_nav"]
    for col in ("turnover_one_side", "turnover_double"):
        out[col] = (tf.groupby("date")[col].sum()
                    .reindex(net.index, fill_value=0.0))
    # 恒等式自检（≤1e-12，防手误）
    assert (out["net_return"] - (out["gross_return"] - out["cost"])).abs().max() <= 1e-12
    assert np.allclose(
        tf["cost"], (tf["freed"] + tf["bought_amt"]) * 0.0015)
    return out


def fetch_market(provider, start: str, end: str, cols: list[str]):
    """一次拉窗口行情：close / tradestatuscode / up_down_limit_status。

    语义复刻 ``paper_replication.replay_v2.fetch_window``（含 uls 全零拒绝）。
    """
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = start
        provider._end_date = end
        provider.instruments_ = list(cols)
        df = provider.fetch(
            ["$close", "$tradestatuscode", "$up_down_limit_status"], freq="day")
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig
    px = df["close"].unstack("instrument").sort_index()
    tsc = (df["tradestatuscode"].unstack("instrument").sort_index()
           .reindex_like(px))
    uls = (df["up_down_limit_status"].unstack("instrument").sort_index()
           .reindex_like(px))
    if (uls != 0).sum().sum() == 0:
        raise RuntimeError("up_down_limit_status 全零——涨跌停状态缺失，停止真实重放")
    trd = (tsc == -1).fillna(False) & px.notna()
    return px, trd, uls


# ============================================================
# 评价汇总（任务 C：毛/净 × 双基准，同有效日期）
# ============================================================


def _perf(series: pd.Series, trades, name: str) -> dict:
    from paper_replication.engine_v2 import compute_perf_v2

    p = compute_perf_v2(series.dropna(), trades, name=name)
    return p.to_dict()


def _rule_desc(gross_aer: float, net_aer: float) -> str:
    if gross_aer <= 0:
        return "扣费前已无正超额"
    if net_aer <= 0:
        return "费用使该项由正转非正"
    return "该窗口扣费后仍正"


def evaluate_arm(
    name: str, daily: pd.DataFrame, trades,
    bench_idx: pd.Series, bench_ew: pd.Series,
) -> dict:
    """单臂单窗：自身毛/净绩效 + 双基准毛/净超额（各自共同有效日期）。"""
    from paper_replication.engine_v2 import attach_benchmark_v2

    gross_ex_idx = attach_benchmark_v2(daily["gross_return"], bench_idx)
    net_ex_idx = attach_benchmark_v2(daily["net_return"], bench_idx)
    gross_ex_ew = attach_benchmark_v2(daily["gross_return"], bench_ew)
    net_ex_ew = attach_benchmark_v2(daily["net_return"], bench_ew)
    return {
        "arm": name,
        "self": {"gross": _perf(daily["gross_return"], trades, f"{name}|gross"),
                 "net": _perf(daily["net_return"], trades, f"{name}|net")},
        "excess_idx": {
            "n_common": int(len(net_ex_idx.dropna())),
            "gross": _perf(gross_ex_idx, trades, f"{name}|gi"),
            "net": _perf(net_ex_idx, trades, f"{name}|ni"),
            "rule": _rule_desc(_perf(gross_ex_idx, trades, "")["aer"],
                               _perf(net_ex_idx, trades, "")["aer"])},
        "excess_ew": {
            "n_common": int(len(net_ex_ew.dropna())),
            "gross": _perf(gross_ex_ew, trades, f"{name}|ge"),
            "net": _perf(net_ex_ew, trades, f"{name}|ne"),
            "rule": _rule_desc(_perf(gross_ex_ew, trades, "")["aer"],
                               _perf(net_ex_ew, trades, "")["aer"])},
        "sum_cost": float(daily["cost"].sum()),
        "nav_drag_end": float(daily["nav_drag"].iloc[-1]),
        "n_trade_days": int(len(daily.dropna(subset=["net_return"]))),
    }


# ============================================================
# CLI 阶段
# ============================================================


def _read_protocol_sha() -> str:
    return (SOURCE_DATA / "protocol.json.sha256").read_text(encoding="utf-8").strip()


def replay_window(
    wname: str,
    frames_wide: dict[str, pd.DataFrame],
    *,
    fetcher=None,
    runs: tuple[str, ...] = RUNS,
    with_g1_mean: bool = True,
) -> tuple[dict, dict[str, dict]]:
    """单窗六臂（+G1_mean）真实 v2 重放。

    :param frames_wide: ``{window: 长格式信号}``。
    :param fetcher: 注入式行情源 ``f(window, universe_cols) -> (px, trd, uls,
        bench_idx)``；None 时走 qlib（冻结宇宙清单 + 指数基准）。
    :returns: ``(meta_w, {run: {daily, trades, eval, engine}})``。
    """
    from paper_replication.benchmark import probe_index_benchmark
    from paper_replication.engine_v2 import build_pool_equal_weight_benchmark_v2

    start, end = WINDOW_BOUNDS[wname] if wname in WINDOW_BOUNDS else (None, None)
    if fetcher is None:
        from kronos_qlib import QlibProvider

        provider = QlibProvider("csi300", start, end)
        universe = frozen_universe(wname)
        px, trd, uls = fetch_market(provider, start, end, universe)
        bench_idx = probe_index_benchmark(provider, start, end)
    else:
        universe = sorted(frames_wide[wname]["instrument"].unique())
        px, trd, uls, bench_idx = fetcher(wname, universe)

    runs_map: dict[str, pd.DataFrame] = {}
    for run in runs:
        runs_map[run] = _load_signal_wide(frames_wide, wname, run)
    if with_g1_mean:
        g1 = pd.read_parquet(G1_MEAN_PARQUET[wname])
        mh1_dates = pd.DatetimeIndex(sorted(runs_map[runs[0]].index))
        runs_map["G1_mean"] = g1.reindex(mh1_dates)

    out: dict[str, dict] = {}
    for name, wide in runs_map.items():
        grid = prepare_signal_grid(wide, pd.DatetimeIndex(px.index))
        res = replay_one(grid, px, trd, uls)
        daily = decompose(res["net"], res["trades"])
        bench_ew = build_pool_equal_weight_benchmark_v2(
            px, trd, grid, fix_mask=True)
        daily["bench_idx_ret"] = bench_idx.reindex(daily.index)
        daily["bench_ew_ret"] = bench_ew.reindex(daily.index)
        out[name] = {"daily": daily, "trades": res["trades"],
                     "eval": evaluate_arm(name, daily, res["trades"],
                                          bench_idx, bench_ew),
                     "engine": {"config": res["cfg"], "modules": res["modules"]}}
    meta = {
        "n_universe": len(universe), "n_rebalances": int(px.shape[0]),
        "n_cal_days": int(px.shape[0]),
        "engine": {"modules": out[runs[0]]["engine"]["modules"],
                   "config": out[runs[0]]["engine"]["config"],
                   "benchmarks": ["000300.SH 指数", "同池等权(v2 掩码)"]},
    }
    return meta, out


def _load_signal_wide(frames_wide: dict[str, pd.DataFrame], window: str,
                      run: str) -> pd.DataFrame:
    df = frames_wide[window]
    arm, seed = run[0], int(run[1:])
    sub = df[(df["arm"] == arm) & (df["seed"] == seed)]
    wide = sub.pivot(index="date", columns="instrument", values=H10)
    wide.index = pd.DatetimeIndex(wide.index)   # parquet date 列为字符串
    return wide.sort_index()


def cmd_preflight() -> None:
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    # 运行批次：保留上一轮记录，不覆盖
    batch = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = hash_evidence(evidence_paths())
    missing = [str(p) for p in evidence_paths() if not Path(p).is_file()]
    if missing:
        logger.warning(f"缺失证据文件（不阻塞代码/测试交付）：\n" + "\n".join(missing))

    # 协议哈希 + 信号覆盖核验
    proto_sha = _read_protocol_sha()
    assert proto_sha == EXPECT_PROTOCOL_SHA, f"主协议哈希漂移：{proto_sha}"
    runs_present: list[str] = []
    decision_dates: dict[str, set] = {}
    for wname in WINDOW_BOUNDS:
        sig = pd.read_parquet(SOURCE_DATA / f"signals_{wname}.parquet")
        assert not sig.duplicated(["date", "instrument", "arm", "seed"]).any()
        assert (sig["protocol_sha256"] == proto_sha).all(), "信号协议哈希列不一致"
        assert np.isfinite(sig[H10]).all(), "h10 含非有限值"
        runs_present += [f"{a}{s}" for a, s in zip(sig["arm"], sig["seed"].astype(str))]
        decision_dates[wname] = set(sig["date"].unique())
    validate_inputs(window_bounds=WINDOW_BOUNDS,
                    runs_present=sorted(set(runs_present)))
    # W3/W4 日期与旧 manifest 对拍
    old_manifest = json.loads((SOURCE_DATA / "manifest.json").read_text(encoding="utf-8"))
    for wname in WINDOW_BOUNDS:
        old_days = set(old_manifest[wname]["per_day"].keys())
        assert decision_dates[wname] == old_days, (
            f"{wname} 信号日期与旧 manifest 不一致")

    config = {
        "batch": batch,
        "window_bounds": WINDOW_BOUNDS,
        "forward_cutoff": FORWARD_CUTOFF,
        "runs": list(RUNS),
        "g1_mean_parquet": {w: str(p) for w, p in G1_MEAN_PARQUET.items()},
        "protocol_sha256": proto_sha,
        "evidence_manifest": manifest,
        "n_evidence_files": len(manifest),
        "missing_evidence": missing,
        "engine": "paper_replication.engine_v2（EngineConfigV2 默认六修正全开）",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (CORR_DIR / "manifest_before.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info(f"[preflight] 证据 {len(manifest)} 件哈希入档；协议 {proto_sha[:16]}…；"
                f"W3 {len(decision_dates['W3'])} 日 / W4 {len(decision_dates['W4'])} 日"
                "与旧 manifest 一致；六臂齐全")


def _require_preflight() -> dict:
    p = CORR_DIR / "manifest_before.json"
    if not p.is_file():
        raise RuntimeError("先运行 --stage preflight（证据未锁定）")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    verify_evidence(cfg["evidence_manifest"])
    validate_inputs(source_sha256=_read_protocol_sha(),
                    expected_sha256=cfg["protocol_sha256"])
    return cfg


def cmd_replay() -> None:
    cfg = _require_preflight()
    frames = {w: pd.read_parquet(SOURCE_DATA / f"signals_{w}.parquet")
              for w in WINDOW_BOUNDS}
    (CORR_DIR / "daily").mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    identity_check = {"max_abs_identity_err": 0.0, "max_abs_cost_formula_err": 0.0}

    for wname in WINDOW_BOUNDS:
        meta_w, per_run = replay_window(wname, frames)
        logger.info(f"[{wname}] 宇宙 {meta_w['n_universe']} 列 × "
                    f"{meta_w['n_cal_days']} 个交易日")
        for name, payload in per_run.items():
            daily = payload["daily"]
            daily.to_parquet(CORR_DIR / "daily" / f"{wname}_{name}.parquet")
            row = dict(payload["eval"])
            row["window"] = wname
            row["engine_config"] = payload["engine"]["config"]
            row["engine_modules"] = payload["engine"]["modules"]
            summary_rows.append(row)
            tf = payload["trades"].to_frame()
            identity_check["max_abs_identity_err"] = max(
                identity_check["max_abs_identity_err"],
                float((daily["net_return"] - (daily["gross_return"]
                                              - daily["cost"])).abs().max()))
            identity_check["max_abs_cost_formula_err"] = max(
                identity_check["max_abs_cost_formula_err"],
                float((tf["cost"] - (tf["freed"] + tf["bought_amt"])
                       * 0.0015).abs().max()))
            logger.info(
                f"[{wname}] {name}: 自身毛 {daily['gross_nav'].iloc[-1] - 1:+.2%} / "
                f"净 {daily['net_nav'].iloc[-1] - 1:+.2%} | 成本合计 "
                f"{daily['cost'].sum():.4f} | 净超额 idx "
                f"{row['excess_idx']['net']['aer']:+.2%} ew "
                f"{row['excess_ew']['net']['aer']:+.2%} | "
                f"{row['excess_idx']['rule']}")

    out = {"batch": cfg["batch"],
           "created_at": datetime.now().isoformat(timespec="seconds"),
           "protocol_sha256": cfg["protocol_sha256"],
           "window_bounds": WINDOW_BOUNDS,
           "identity_check": identity_check,
           "rows": summary_rows}
    (CORR_DIR / "replay_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    # 结束后证据对拍（前后不变）
    verify_evidence(cfg["evidence_manifest"])
    logger.info(f"[replay] 完成：{len(summary_rows)} 臂窗行；恒等式最大误差 "
                f"{identity_check['max_abs_identity_err']:.2e}；证据哈希前后一致")


def save_transactions(trades, path: Path) -> None:
    """真实交易日志落盘（计划 §3 任务 A：全字段可 round-trip，非字符串化）。

    ``sold``/``bought`` 以 pyarrow list 列原生存储，读回即 Python list——
    禁止把列表转成不可解析字符串。
    """
    df = trades.to_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


GRID_AUDIT_DIR = Path(__file__).resolve().parent / "data" / "grid_audit_20260908"


def cmd_transactions() -> None:
    """补证：14 固定臂窗只重放一次，落盘真实交易日志并与 correction 对拍。

    逐日净收益与 ``correction_20260908/daily/`` 逐值对拍（绝对误差 ≤1e-12）；
    差异非零先查数据版本/接线（断言失败即停），不覆盖历史结果。
    """
    cfg = _require_preflight()
    frames = {w: pd.read_parquet(SOURCE_DATA / f"signals_{w}.parquet")
              for w in WINDOW_BOUNDS}
    tx_dir = GRID_AUDIT_DIR / "transactions"
    tx_dir.mkdir(parents=True, exist_ok=True)
    max_err = 0.0
    n_files = 0
    for wname in WINDOW_BOUNDS:
        _meta, per_run = replay_window(wname, frames)
        for name, payload in per_run.items():
            save_transactions(payload["trades"],
                              tx_dir / f"{wname}_{name}.parquet")
            n_files += 1
            daily = payload["daily"]["net_return"]
            old = pd.read_parquet(
                Path(__file__).resolve().parent / "data" /
                "correction_20260908" / "daily" / f"{wname}_{name}.parquet")
            err = float((daily - old["net_return"].reindex(daily.index))
                        .abs().max())
            max_err = max(max_err, err)
            assert err <= 1e-12, (
                f"{wname} {name} 逐日净收益与 correction 不一致：max|Δ|={err:.3e}"
                "（先查数据版本与接线，不覆盖历史结果）")
            logger.info(f"[tx] {wname} {name}: 交易日志落盘，逐日净收益对拍"
                        f"max|Δ|={err:.1e} OK")
    (GRID_AUDIT_DIR / "transactions_summary.json").write_text(
        json.dumps({"batch": cfg["batch"], "n_files": n_files,
                    "max_abs_daily_net_err": max_err,
                    "created_at": datetime.now().isoformat(timespec="seconds")},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[tx] 补证完成：{n_files} 份交易日志；逐日净收益最大误差 {max_err:.1e}")


def cmd_pool_audit() -> None:
    """任务 D：训练池 PIT 偏差取证（只读重建样本键 + 逐日交集，零训练）。"""
    from kronos_qlib import QlibProvider
    from mh1_multihorizon.data import SymbolTable, scan_train_days, union_calendar
    from h1_readout.corpus_loader import CorpusUnpickler

    def _load_source_tables() -> dict[str, SymbolTable]:
        feats = ["open", "high", "low", "close", "vol", "amt"]
        tables: dict[str, SymbolTable] = {}
        for split in ("train", "val"):
            path = (SOURCE_ROOT / "finetune_suite" / "data" /
                    ("train_data.pkl" if split == "train" else "val_data.pkl"))
            with open(path, "rb") as f:
                data = CorpusUnpickler(f).load()
            for code, df in data.items():
                tables.setdefault(code, []).append(df)
        out: dict[str, SymbolTable] = {}
        for code, parts in tables.items():
            df = pd.concat(parts)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            out[code] = SymbolTable(
                vals=df[feats].values.astype(np.float32),
                dates=pd.DatetimeIndex(df.index))
        return out

    tables = _load_source_tables()
    calendar = union_calendar(tables)
    days, stats = scan_train_days(
        tables, calendar, start="2014-01-02", label_end="2024-12-31")
    logger.info(f"重建训练样本键：{stats.n_days} 日 / {stats.n_samples:,} 样本"
                f"（决策 {stats.decision_min}~{stats.decision_max}）")
    provider = QlibProvider("csi300", "2014-01-02", "2024-12-31")
    rows = []
    for i, day in enumerate(days):
        ds = str(day.date.date())
        members = set(provider.list_pool_at("csi300", ds))
        codes = set(day.codes)
        inter = codes & members
        rows.append({"date": ds, "n_samples": len(codes),
                     "n_members": len(members),
                     "n_member_samples": len(inter),
                     "n_nonmember_samples": len(codes) - len(inter),
                     "nonmember_ratio": 1.0 - len(inter) / max(len(codes), 1)})
        if (i + 1) % 500 == 0:
            logger.info(f"  pool-audit [{i + 1}/{len(days)}]")
    df = pd.DataFrame(rows)
    df.to_parquet(CORR_DIR / "pool_audit.parquet", index=False)
    audit = {
        "n_days": int(len(df)),
        "samples_min_max": [int(df.n_samples.min()), int(df.n_samples.max())],
        "samples_mean": float(df.n_samples.mean()),
        "nonmember_ratio_mean": float(df.nonmember_ratio.mean()),
        "nonmember_ratio_max": float(df.nonmember_ratio.max()),
        "decision_range": [df.date.iloc[0], df.date.iloc[-1]],
        "note": "历史并集训练口径 vs t 日 PIT 成分交集，逐日统计（只读取证）",
    }
    (CORR_DIR / "pool_audit_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[pool-audit] {audit['n_days']} 日：样本 "
                f"{audit['samples_min_max']}（均值 {audit['samples_mean']:.1f}），"
                f"非当日成员占比均值 {audit['nonmember_ratio_mean']:.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MH1 纠偏 v2 重放 CLI")
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "replay", "pool-audit", "transactions"])
    args = parser.parse_args()
    log_dir = Path(__file__).resolve().parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dir / f"correction_{args.stage}.log"), enqueue=False)
    if args.stage == "preflight":
        cmd_preflight()
    elif args.stage == "replay":
        cmd_replay()
    elif args.stage == "transactions":
        cmd_transactions()
    else:
        cmd_pool_audit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
