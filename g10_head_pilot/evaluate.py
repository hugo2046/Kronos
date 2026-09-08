"""G10H-pilot 评价：FULL 逐日生成式推理 → mean 信号 → IC + 真实 v2（计划 §2/§4/§5）。

纪律：推理到两窗**末日**（交易信号不得被标签裁剪）；IC 标签 t+10 终点留段
内（尾部 10 个决策日无标签，属标签可用性而非信号缺失）；v2 接线先以既有
FULL G1 信号复现 ``grid_audit_20260908`` 的 FULL 逐日净收益（≤1e-12 门禁）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from g10_head_pilot.config import (
    FORWARD_CUTOFF, G1_FULL_SIGNALS, GRID_AUDIT_FULL_DAILY, IC_HORIZON,
    IC_MIN_STOCKS, INFERENCE, NW_LAG, RUN_DIR, WINDOW_BOUNDS,
)

STATES = ("bestCE", "e15")


# ============================================================
# FULL 逐日生成式推理（canonical：build_inference_windows + 分块 predict）
# ============================================================


def score_full_window(predictor, provider, wname: str) -> pd.DataFrame:
    """单窗逐决策日（全部交易日）PIT csi300 → mean 信号宽表。

    未来时间戳只用交易日历日期（y_ts 为未来 H 个交易日的时间特征），
    不访问任何未来观测值。
    """
    from kronos_qlib import build_inference_windows
    from paper_replication.signal import (
        compute_signal_from_preds, predict_batch_chunked,
    )

    start, end = WINDOW_BOUNDS[wname]
    rebalances = provider.trading_days(start, end)
    rows: dict[str, float] = {}
    wide = {}
    for i, d in enumerate(rebalances):
        ds = str(d.date())
        df_list, x_ts, y_ts, codes, stats = build_inference_windows(
            provider, ds, lookback=INFERENCE["lookback"],
            predict_len=INFERENCE["pred_len"], pool="csi300")
        if not df_list:
            logger.warning(f"[{wname}] {ds} 无可用窗口")
            continue
        preds = predict_batch_chunked(
            predictor, df_list, x_ts, y_ts,
            pred_len=INFERENCE["pred_len"], T=INFERENCE["T"],
            top_k=INFERENCE["top_k"], top_p=INFERENCE["top_p"],
            sample_count=INFERENCE["sample_count"], chunk_size=32)
        day_sig = {}
        for code, pred in zip(codes, preds):
            last_close = float(df_list[codes.index(code)]["close"].iloc[-1])
            day_sig[code] = compute_signal_from_preds(
                pred["close"], last_close)
        wide[ds] = day_sig
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"[{wname}] 推理 [{i + 1}/{len(rebalances)}] {ds}："
                        f"{len(day_sig)} 只")
    return pd.DataFrame(wide)


# ============================================================
# IC（k=10，标签终点留段内，code 对齐）
# ============================================================


def rank_ic(scores: dict, labels: dict,
            min_stocks: int = IC_MIN_STOCKS):
    common = [c for c in scores if c in labels and np.isfinite(labels[c])
              and np.isfinite(scores[c])]
    if len(common) < min_stocks:
        return None, f"common {len(common)} < {min_stocks}"
    x = pd.Series([scores[c] for c in common]).rank().to_numpy()
    y = pd.Series([labels[c] for c in common]).rank().to_numpy()
    x, y = x - x.mean(), y - y.mean()
    denom = np.sqrt((x @ x) * (y @ y))
    if denom <= 0:
        return None, "constant"
    return float((x @ y) / denom), None


def window_labels(provider, wname: str) -> dict[str, dict[str, float]]:
    """逐决策日 k=10 标签（终点=日历 t+10 且 ≤ 窗末；缺行→该股无标签）。"""
    start, end = WINDOW_BOUNDS[wname]
    calendar = provider.trading_days(start, FORWARD_CUTOFF)
    cal_pos = {d: i for i, d in enumerate(calendar)}
    days = provider.trading_days(start, end)
    out: dict[str, dict] = {}
    for d in days:
        c = cal_pos[d]
        if c + IC_HORIZON >= len(calendar) or calendar[c + IC_HORIZON] > pd.Timestamp(end):
            continue           # 标签终点越窗末 → 尾部 10 日无标签（披露）
        members = provider.list_pool_at("csi300", str(d.date()))
        win = provider.trading_days(
            str(calendar[max(0, c - 89)].date()), str(calendar[c].date()))
        fetch_start = str(win[0].date())
        fetch_end = str(calendar[c + IC_HORIZON].date())
        orig = (provider._start_date, provider._end_date, provider.instruments_)
        try:
            provider._start_date = fetch_start
            provider._end_date = fetch_end
            provider.instruments_ = list(members)
            raw = provider.fetch(["$close"], freq="day")
        finally:
            provider._start_date, provider._end_date, provider.instruments_ = orig
        lab: dict[str, float] = {}
        for code in members:
            try:
                sub = raw.xs(code, level="instrument").sort_index()
            except KeyError:
                continue
            if d not in sub.index:
                continue
            base = float(sub.loc[d, "close"])
            ep = calendar[c + IC_HORIZON]
            if ep not in sub.index or not (np.isfinite(base) and base > 0):
                continue
            nxt = float(sub.loc[ep, "close"])
            if np.isfinite(nxt) and nxt > 0:
                lab[code] = nxt / base - 1.0
        out[str(d.date())] = lab
    return out


def daily_ic_series(wide: pd.DataFrame, labels: dict) -> list[float]:
    vals = []
    cols = {str(pd.Timestamp(c).date()) if not isinstance(c, str) else c
            for c in wide.columns}
    for ds in sorted(cols):
        if ds not in labels:
            continue
        col = [c for c in wide.columns
               if (str(pd.Timestamp(c).date()) if not isinstance(c, str)
                   else c) == ds][0]
        ic, _ = rank_ic(dict(wide[col].dropna().items()), labels[ds])
        if ic is not None:
            vals.append(ic)
    return vals


def nw_t(values: list[float], lag: int = NW_LAG) -> float:
    """单窗 Bartlett NW t（空序列返回 NaN，不除零）。"""
    if not len(values):
        return float("nan")
    from paper_replication.ic_horizon_profile import _nw_tvalue

    return float(_nw_tvalue(np.asarray(values, dtype=float), lag))


# ============================================================
# v2 接线复现门禁（对 grid_audit FULL 逐日净收益 ≤1e-12）
# ============================================================


def verify_full_gate() -> dict:
    """用既有 FULL G1 信号走本包引擎接线，对拍 grid_audit FULL 日净收益。"""
    from mh1_multihorizon.replay_corrected import (
        SOURCE_DATA, decompose, fetch_market, frozen_universe, replay_one,
    )

    errs = {}
    for wname, (start, end) in WINDOW_BOUNDS.items():
        raw = pd.read_parquet(G1_FULL_SIGNALS[wname])
        raw.index = pd.DatetimeIndex(raw.index)
        from kronos_qlib import QlibProvider

        provider = QlibProvider("csi300", start, end)
        universe = frozen_universe(wname)
        px, trd, uls = fetch_market(provider, start, end, universe)
        grid = raw.reindex(index=px.index, columns=universe)
        res = replay_one(grid, px, trd, uls)
        daily = decompose(res["net"], res["trades"])
        ref = pd.read_parquet(GRID_AUDIT_FULL_DAILY / f"{wname}_FULL.parquet")
        err = float((daily["net_return"]
                     - ref["net_return"].reindex(daily.index)).abs().max())
        assert err <= 1e-12, (
            f"{wname} FULL G1 复现门禁失败：{err:.3e}（先查接线，不进收益解释）")
        errs[wname] = err
        logger.info(f"[gate] {wname} FULL G1 引擎接线复现 max|Δ|={err:.1e} OK")
    return errs


# ============================================================
# v2 交易评价（复用 replay_corrected 真实接口）
# ============================================================


def trade_arm_window(wide: pd.DataFrame, wname: str, tag: str = "") -> dict:
    """单臂窗 v2 交易：FULL 网格信号 → 毛/净/成本 + 指数累计超额。"""
    from kronos_qlib import QlibProvider
    from mh1_multihorizon.replay_corrected import (
        decompose, fetch_market, frozen_universe, replay_one,
        save_transactions,
    )
    from paper_replication.benchmark import probe_index_benchmark
    from paper_replication.engine_v2 import build_pool_equal_weight_benchmark_v2

    start, end = WINDOW_BOUNDS[wname]
    provider = QlibProvider("csi300", start, end)
    universe = frozen_universe(wname)
    px, trd, uls = fetch_market(provider, start, end, universe)
    bench_idx = probe_index_benchmark(provider, start, end)
    # 入参约定：日期×股票宽表（date index, code columns；索引可为字符串）
    idx = pd.DatetimeIndex(wide.index)
    grid = pd.DataFrame(wide.values, index=idx,
                        columns=list(wide.columns)).reindex(
        index=px.index, columns=universe)
    res = replay_one(grid, px, trd, uls)
    daily = decompose(res["net"], res["trades"])
    bench_ew = build_pool_equal_weight_benchmark_v2(
        px, trd, grid, fix_mask=True)
    daily["bench_idx_ret"] = bench_idx.reindex(daily.index)
    daily["bench_ew_ret"] = bench_ew.reindex(daily.index)
    ex_idx_net = (daily["net_return"] - daily["bench_idx_ret"]).dropna()
    ex_idx_gross = (daily["gross_return"] - daily["bench_idx_ret"]).dropna()
    ex_ew_net = (daily["net_return"] - daily["bench_ew_ret"]).dropna()

    def _cum(s):
        v = s.dropna()
        return float((1 + v).prod() - 1) if len(v) else float("nan")

    def _mdd(s):
        v = s.dropna()
        nav = (1 + v).cumprod()
        return float((nav / nav.cummax() - 1).min()) if len(v) else float("nan")

    out_dir = RUN_DIR / "trade"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{tag}_{wname}" if tag else wname
    daily.to_parquet(out_dir / f"{name}.parquet")
    return {
        "self_gross_cum": _cum(daily["gross_return"]),
        "self_net_cum": _cum(daily["net_return"]),
        "idx_excess_gross_cum": _cum(ex_idx_gross),
        "idx_excess_net_cum": _cum(ex_idx_net),
        "idx_excess_net_aer": (float((1 + ex_idx_net).prod()
                                     ** (252 / len(ex_idx_net)) - 1)
                               if len(ex_idx_net) else float("nan")),
        "ew_own_excess_net_cum": _cum(ex_ew_net),
        "ew_own_n_days": int(len(ex_ew_net)),
        "strategy_mdd_net": _mdd(daily["net_return"]),
        "cost_sum": float(daily["cost"].sum()),
        "turnover_double_sum": float(daily["turnover_double"].sum()),
        "n_cal_days": int(px.shape[0]),
    }


# ============================================================
# 判据 V/I/P/S/E（计划 §5，一次冻结代入）
# ============================================================


def judge(stats: dict) -> dict:
    ic = stats["ic"]          # {(arm,state,w): mean}
    tr = stats["trade"]       # {(arm,state,w): idx_excess_net_cum}
    crit: dict = {}
    i_ok = all(ic[("A1", s, w)] > 0 for s in STATES for w in WINDOW_BOUNDS)
    drop_a1 = {w: ic[("A1", "bestCE", w)] - ic[("A1", "e15", w)]
               for w in WINDOW_BOUNDS}
    i_ok = i_ok and all(drop_a1[w] <= 0.01 for w in WINDOW_BOUNDS)
    crit["I_info_retained"] = bool(i_ok)
    crit["I_drops_A1"] = drop_a1
    p_ok = all((stats["ic_diff_e15"][w] > 0) for w in WINDOW_BOUNDS) and (
        stats["ic_diff_e15"]["combined_t"] > 2)
    crit["P_vs_A0_e15"] = bool(p_ok)
    drop_a0 = {w: ic[("A0", "bestCE", w)] - ic[("A0", "e15", w)]
               for w in WINDOW_BOUNDS}
    s_ok = all(drop_a1[w] <= drop_a0[w] for w in WINDOW_BOUNDS) and any(
        drop_a1[w] < drop_a0[w] for w in WINDOW_BOUNDS)
    crit["S_stability"] = bool(s_ok)
    crit["S_drops_A0"] = drop_a0
    e_ok = all(tr[("A1", s, w)]["idx_excess_net_cum"] > 0
               for s in STATES for w in WINDOW_BOUNDS)
    crit["E_econ_filter"] = bool(e_ok)
    crit["E_values_A1_idx_excess_net_cum"] = {
        f"{s}|{w}": tr[("A1", s, w)]["idx_excess_net_cum"]
        for s in STATES for w in WINDOW_BOUNDS}
    return crit
