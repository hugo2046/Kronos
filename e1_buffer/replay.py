"""E1 阶段 3.1b：G1 三种子 × 两窗重放——canonical vs 缓冲带全表落盘（不判读）。

输入 = 既有 G1 种子族（s100=G1 / s101=G2S101 / s102=G2S102）mean 信号 parquet，
两窗全用（2025H2 + backtest；E1 不涉训练，无准样本内问题）。零 GPU 纯重放。

口径对齐（与 run_g1_backtest / run_g2_judge / run_g5_eval 逐字一致，保证
canonical 复跑可与冻结数字对拍）：

    - 窗口：backtest 2026-01-01~2026-07-24 / 2025h2 2025-07-01~2025-12-31；
    - px/可交易掩码/双基准：``build_px_tradeable`` + ``build_dual_benchmarks``，
      列宇宙 = 三种子 + F0 + M 信号列并集（= M 的 338 列，与各轮原始全表一致）；
    - rebalances = 当窗 M 信号索引（与原始轮次一致）；
    - canonical 引擎 = ``paper_replication.engine.run_portfolio``（只读复用）。

三段分解（计划 §2 呈现条款）：成本节约（换手降幅 × 30bp，确定性）/ 毛 AER
变化（噪声域）/ 净 AER（两者合成）。**本脚本只落盘，不判读**——判据
E1-1~E1-4 统一在 3.4 一次开封。

落盘（e1_buffer/data/，不入库）：
    e1_replay_results.json（全表 + 对拍 + 中位数汇总，无判据字段）
    e1_replay_table.csv（人读全表）

用法::

    /home/user/miniconda3/envs/quant/bin/python -m e1_buffer.replay
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
from e1_buffer.engine import BufferEngineConfig, run_buffer_portfolio
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from paper_replication.engine import (
    TRADING_DAYS_PER_YEAR,
    EngineConfig,
    attach_benchmark,
    compute_perf,
    run_portfolio,
)

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DATA_DIR = PKG_DIR / "data"
FS = REPO_ROOT / "finetune_suite" / "data"

WINDOWS: dict[str, tuple[str, str]] = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": ("2025-07-01", "2025-12-31"),
}

SEEDS = ("s100", "s101", "s102")

# G1 种子族 mean 信号在盘位置（只读；s100 的 2025H2 = g5 轮补生成的 G1 权重推理）
SIGNAL_PARQUET: dict[tuple[str, str], Path] = {
    ("s100", "backtest"): FS / "g1" / "daily_signals_backtest_G1_mean.parquet",
    ("s100", "2025h2"): REPO_ROOT / "g5_head" / "data" / "daily_signals_2025h2_G1_mean.parquet",
    ("s101", "backtest"): FS / "g2" / "s101" / "daily_signals_backtest_G2S101_mean.parquet",
    ("s101", "2025h2"): FS / "g2" / "s101" / "daily_signals_2025h2_G2S101_mean.parquet",
    ("s102", "backtest"): FS / "g2" / "s102" / "daily_signals_backtest_G2S102_mean.parquet",
    ("s102", "2025h2"): FS / "g2" / "s102" / "daily_signals_2025h2_G2S102_mean.parquet",
}

# 列宇宙组成（对齐原始轮次：臂信号 ∪ F0 ∪ M ⇒ 338 列）
def _universe_files(window: str) -> list[Path]:
    f0 = FS / "g0" / f"daily_signals_{window}_F0_mean.parquet" \
        if window == "2025h2" else FS / f"daily_signals_backtest_F0_mean.parquet"
    m = FS / "g0" / f"daily_signals_{window}_M.parquet" \
        if window == "2025h2" else FS / "daily_signals_backtest_M.parquet"
    return list(SIGNAL_PARQUET[(s, window)] for s in SEEDS) + [f0, m]


# canonical 冻结参照（对拍门禁：重放链路正确性检查，非判读）
def load_frozen_refs() -> dict[tuple[str, str], float]:
    g2j = json.loads((FS / "g2" / "g2_judge_results.json").read_text(encoding="utf-8"))
    g5 = json.loads(
        (REPO_ROOT / "g5_head" / "data" / "g5_stage2_results.json").read_text(encoding="utf-8")
    )
    return {
        ("s100", "backtest"): g2j["perf"]["backtest"]["s100"]["aer_ew"],
        ("s101", "backtest"): g2j["perf"]["backtest"]["s101"]["aer_ew"],
        ("s102", "backtest"): g2j["perf"]["backtest"]["s102"]["aer_ew"],
        ("s100", "2025h2"): g5["results"]["G1_mean"]["2025h2"]["perf_ew"]["aer"],
        ("s101", "2025h2"): g2j["perf"]["h2_new_seeds"]["s101"]["aer_ew"],
        ("s102", "2025h2"): g2j["perf"]["h2_new_seeds"]["s102"]["aer_ew"],
    }


def _perf_block(daily_ret, trades, bench_idx, bench_ew, *, name) -> dict:
    perf_idx = compute_perf(attach_benchmark(daily_ret, bench_idx), trades, name=name)
    perf_ew = compute_perf(attach_benchmark(daily_ret, bench_ew), trades, name=name)
    tdf = trades.to_frame()
    return {
        "aer_ew": perf_ew.aer,
        "aer_idx": perf_idx.aer,
        "ir_ew": perf_ew.ir,
        "ir_idx": perf_idx.ir,
        "n_days": perf_ew.n_days,
        "daily_turnover": float(tdf["turnover_ratio"].mean()),
    }


def replay_window(window: str, start: str, end: str) -> tuple[dict, dict]:
    """单窗：三种子 canonical(net/gross) + E1(net/gross) + 对拍。返回 (rows, meta)。"""
    from kronos_qlib import QlibProvider

    cfg = replace(
        BaselineConfig.load(window="oos"),
        window=f"e1_{window}", backtest_start=start, backtest_end=end,
    )
    signals = {s: pd.read_parquet(SIGNAL_PARQUET[(s, window)]) for s in SEEDS}
    uni_files = _universe_files(window)
    all_cols = sorted(set().union(*[set(pd.read_parquet(p).columns) for p in uni_files]))
    m_idx = pd.DatetimeIndex(
        pd.read_parquet(uni_files[-1]).index
    )  # rebalances = 当窗 M 索引（原始轮次口径）

    provider = QlibProvider(cfg.pool, start, end)
    px, trd = build_px_tradeable(provider, cfg, m_idx, all_cols)
    bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

    ec_net = EngineConfig(top_k=cfg.top_k, drop_n=cfg.drop_n, min_hold=cfg.min_hold, cost_bps=cfg.cost_bps)
    ec_gross = replace(ec_net, cost_bps=0.0)
    bc_net = BufferEngineConfig()
    bc_gross = replace(bc_net, cost_bps=0.0)

    rows: dict[str, dict] = {}
    for s in SEEDS:
        sig = signals[s].reindex(index=px.index, columns=px.columns)

        # —— canonical：net 走 run_group（与原始轮次同一入口），gross 同引擎零成本 ——
        pi, pe, dr_c, _, _ = run_group(sig, px, trd, bench_idx, bench_ew, cfg=cfg, name=f"{window}/{s}/canonical")
        dr_c0, _, tr_c0 = run_portfolio(sig, px, trd, cfg=ec_gross)
        can = {
            "net": {"aer_ew": pe.aer, "aer_idx": pi.aer, "ir_ew": pe.ir,
                    "daily_turnover": pe.daily_turnover, "n_days": pe.n_days},
            "gross": _perf_block(dr_c0, tr_c0, bench_idx, bench_ew, name=f"{window}/{s}/canonical_gross"),
        }

        # —— E1 缓冲带：net + gross ——
        dr_e, _, tr_e = run_buffer_portfolio(sig, px, trd, cfg=bc_net)
        dr_e0, _, tr_e0 = run_buffer_portfolio(sig, px, trd, cfg=bc_gross)
        e1 = {
            "net": _perf_block(dr_e, tr_e, bench_idx, bench_ew, name=f"{window}/{s}/e1"),
            "gross": _perf_block(dr_e0, tr_e0, bench_idx, bench_ew, name=f"{window}/{s}/e1_gross"),
        }
        tdf = tr_e.to_frame()
        e1["avg_holdings"] = float(tdf["n_holdings"].mean())
        e1["avg_holdings_after_d0"] = float(tdf["n_holdings"].iloc[1:].mean())

        # —— 三段分解（呈现条款；无判据字段）——
        to_drop = 1.0 - e1["net"]["daily_turnover"] / can["net"]["daily_turnover"]
        delta = {
            "turnover_drop_pct": to_drop,
            "frozen_formula_ann_saving": to_drop * can["net"]["daily_turnover"] * 30.0 * TRADING_DAYS_PER_YEAR / 1e4,
            "realized_ann_cost_saving": (can["net"]["daily_turnover"] - e1["net"]["daily_turnover"]) * 15.0 * TRADING_DAYS_PER_YEAR / 1e4,
            "gross_aer_ew_change": e1["gross"]["aer_ew"] - can["gross"]["aer_ew"],
            "net_aer_ew_change": e1["net"]["aer_ew"] - can["net"]["aer_ew"],
        }
        rows[s] = {"canonical": can, "e1": e1, "delta": delta}

    meta = {
        "period": [start, end],
        "universe_cols": len(all_cols),
        "n_days": int(len(m_idx)),
        "beta_gap": beta_gap,
    }
    return rows, meta


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    refs = load_frozen_refs()

    out_rows: dict[str, dict[str, dict]] = {}
    meta: dict[str, dict] = {}
    crosscheck: dict[str, dict] = {}
    for window, (start, end) in WINDOWS.items():
        rows, m = replay_window(window, start, end)
        out_rows[window] = rows
        meta[window] = m
        for s in SEEDS:
            rerun = rows[s]["canonical"]["net"]["aer_ew"]
            ref = refs[(s, window)]
            crosscheck[f"{s}@{window}"] = {
                "ref_frozen": ref, "rerun": rerun, "abs_diff": abs(rerun - ref),
                "ok": bool(abs(rerun - ref) < 5e-4),
            }
        logger.info(f"[{window}] 三种子重放完成（universe={m['universe_cols']} 列，{m['n_days']} 日）")

    bad = [k for k, v in crosscheck.items() if not v["ok"]]
    if bad:
        raise AssertionError(f"canonical 复跑与冻结数字不一致：{bad}（重放链路有 bug，停止）")

    # —— 中位数汇总（机械聚合，无判据字段；判读留 3.4）——
    medians = {}
    for window in WINDOWS:
        can_med = float(np.median([out_rows[window][s]["canonical"]["net"]["aer_ew"] for s in SEEDS]))
        e1_med = float(np.median([out_rows[window][s]["e1"]["net"]["aer_ew"] for s in SEEDS]))
        drops = [out_rows[window][s]["delta"]["turnover_drop_pct"] for s in SEEDS]
        medians[window] = {
            "canonical_net_aer_ew_median": can_med,
            "e1_net_aer_ew_median": e1_med,
            "net_aer_ew_median_delta": e1_med - can_med,
            "turnover_drop_pct_median": float(np.median(drops)),
            "turnover_drop_pct_min": float(min(drops)),
        }

    out = {
        "experiment": "e1_buffer_replay",
        "date": "2026-08-20",
        "protocol": {
            "canonical_engine": EngineConfig().__dict__,
            "buffer_engine": BufferEngineConfig().__dict__,
            "signal": "G1 种子族（s100=G1/s101=G2S101/s102=G2S102）mean，只读复用",
            "windows": {w: list(v) for w, v in WINDOWS.items()},
            "note": "三段分解呈现：成本节约(冻结公式=换手降幅×30bp)/毛AER变化/净AER；"
                    "realized 口径 = Δ换手×15bp（引擎实扣口径）",
        },
        "crosscheck_canonical_vs_frozen": crosscheck,
        "windows_meta": meta,
        "table": out_rows,
        "medians": medians,
    }
    json_path = DATA_DIR / "e1_replay_results.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # —— 人读 CSV 全表（先落盘后打印）——
    csv_rows = []
    for window in WINDOWS:
        for s in SEEDS:
            r = out_rows[window][s]
            csv_rows.append({
                "window": window, "seed": s,
                "can_net_aer_ew": r["canonical"]["net"]["aer_ew"],
                "can_gross_aer_ew": r["canonical"]["gross"]["aer_ew"],
                "can_turnover": r["canonical"]["net"]["daily_turnover"],
                "e1_net_aer_ew": r["e1"]["net"]["aer_ew"],
                "e1_gross_aer_ew": r["e1"]["gross"]["aer_ew"],
                "e1_turnover": r["e1"]["net"]["daily_turnover"],
                "e1_avg_holdings": r["e1"]["avg_holdings"],
                "turnover_drop_pct": r["delta"]["turnover_drop_pct"],
                "frozen_ann_saving": r["delta"]["frozen_formula_ann_saving"],
                "realized_ann_cost_saving": r["delta"]["realized_ann_cost_saving"],
                "gross_aer_change": r["delta"]["gross_aer_ew_change"],
                "net_aer_change": r["delta"]["net_aer_ew_change"],
            })
    csv_path = DATA_DIR / "e1_replay_table.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    # —— 全部落盘后才打印（阶段输出；不判读）——
    print("=== E1 重放对拍（canonical 复跑 vs 冻结，|diff|<5e-4）===")
    for k, v in crosscheck.items():
        print(f"  {k}: 复跑 {v['rerun']:+.4%} vs 冻结 {v['ref_frozen']:+.4%} "
              f"(diff={v['abs_diff']:.2e}) OK")
    print("=== 三段分解全表（已落盘，不判读）===")
    df = pd.DataFrame(csv_rows)
    show = df[["window", "seed", "can_net_aer_ew", "e1_net_aer_ew",
               "net_aer_change", "gross_aer_change", "can_turnover",
               "e1_turnover", "turnover_drop_pct"]].copy()
    for c in show.columns[2:]:
        show[c] = show[c].map(lambda x: f"{x:+.2%}")
    show["e1_avg_holdings"] = df["e1_avg_holdings"].map(lambda x: f"{x:.1f}")
    print(show.to_string(index=False))
    print(f"落盘：{json_path} / {csv_path}")


if __name__ == "__main__":
    main()
