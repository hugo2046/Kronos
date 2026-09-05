"""H1 IC 判读（修订计划 §1/§3.3-3.4，一次开封）。

依据 ``docs/H1计划修订_IC判据_20260905.md``（判据只落在 k=10）：

    - 统一累计 IC 表：信号日 t 对 close(t+k)/close(t)−1 的日截面 Spearman
      rank-IC，k=1..10，NW(lag=k−1) t 值——四 H1 臂（seed 42）+ R1 两头
      （seed 中位：R-lin_s44 / R-kda_s43，R1 冻结判读的中位种子）+ G1_s100
      + M，两窗（backtest 134 日 / 2025H2 126 日），csi300 池；
    - IC1：每 H1 臂 k=10 两窗均值同为正 ∧ 两窗合并（n=260）NW(lag=9) t > 2；
    - IC2：同头配对差 k=10（H1a−R1 / H1b−H1a），合并 NW t > 2 才许"显著"；
    - IC3：G1_s100 与 M 参照行只呈现；
    - 引擎 v2 附表（描述性，不进判据）：四 H1 臂 + G1_s100 + M，两窗双基准
      AER（``paper_replication/engine_v2.py`` 六修正全开；等权基准逐臂掩码）。

用法（四臂两窗信号落盘后）::

    /home/user/miniconda3/envs/quant/bin/python -m h1_readout.run_h1_ic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_replication.ic_horizon_profile import (
    MIN_CROSS_N,
    _fetch_px,
    _nw_tvalue,
)

PKG_DIR = Path(__file__).resolve().parent
REPO = PKG_DIR.parent
H1_DATA = PKG_DIR / "data"
R1_DATA = REPO / "r1_objective" / "data"
R4 = REPO / "finetune_suite" / "data"
G5_DATA = REPO / "g5_head" / "data"

SEED = 42
WINDOWS: dict[str, tuple[str, str, str]] = {
    "backtest": ("2026-01-01", "2026-07-24", "2026-08-07"),
    "2025h2": ("2025-07-01", "2025-12-31", "2026-01-31"),
}
K_RANGE = range(1, 11)
K_JUDGE = 10  # 判据只在 k=10（修订计划 §3 纪律）

H1_ARMS = ("H1a-lin", "H1a-kda", "H1b-lin", "H1b-kda")

# 统一表臂 → parquet 相对路径（窗口占位 {w}）
ARMS: dict[str, dict[str, str]] = {
    **{
        a: {
            w: f"h1_readout/data/{a}/daily_signals_{w}_{a}_s{SEED}.parquet"
            for w in WINDOWS
        }
        for a in H1_ARMS
    },
    "R-lin_s44": {w: f"r1_objective/data/daily_signals_{w}_R-lin_s44.parquet" for w in WINDOWS},
    "R-kda_s43": {w: f"r1_objective/data/daily_signals_{w}_R-kda_s43.parquet" for w in WINDOWS},
    "G1_s100": {
        "backtest": "finetune_suite/data/g1/daily_signals_backtest_G1_mean.parquet",
        "2025h2": "g5_head/data/daily_signals_2025h2_G1_mean.parquet",
    },
    "M": {
        "backtest": "finetune_suite/data/daily_signals_backtest_M.parquet",
        "2025h2": "finetune_suite/data/g0/daily_signals_2025h2_M.parquet",
    },
}
ARM_ORDER = (*H1_ARMS, "R-lin_s44", "R-kda_s43", "G1_s100", "M")

# IC2 配对（同头；k=10 合并配对差 NW t）
PAIRS = (
    ("H1a-lin", "R-lin_s44"),
    ("H1a-kda", "R-kda_s43"),
    ("H1b-lin", "H1a-lin"),
    ("H1b-kda", "H1a-kda"),
)


def _read_signal(rel: str) -> pd.DataFrame:
    p = (REPO / rel).resolve()
    if not p.is_relative_to(REPO.resolve()):
        raise ValueError(f"信号路径越界：{p}")
    return pd.read_parquet(p)


def cum_ic_series(sig: pd.DataFrame, px: pd.DataFrame, k: int) -> pd.Series:
    """累计收益 rank-IC 的逐日序列（行 t = close(t+k)/close(t)−1 的截面 IC）。"""
    target = px.shift(-k) / px - 1.0
    out = {}
    for t in sig.index:
        if t not in px.index:
            continue
        pos = px.index.get_loc(t)
        if pos + k >= len(px.index):
            continue
        r = target.loc[t]
        s = sig.loc[t]
        # 强制同序（scipy.spearmanr 位置配对不对齐索引；见 ic_horizon_profile 对齐修正）
        r = r.reindex(s.index)
        mask = s.notna() & r.notna()
        if int(mask.sum()) < MIN_CROSS_N:
            continue
        rho, _ = stats.spearmanr(s[mask].to_numpy(), r[mask].to_numpy())
        if pd.notna(rho):
            out[t] = float(rho)
    return pd.Series(out)


def main() -> None:
    from kronos_qlib import QlibProvider

    provider = QlibProvider("csi300", "2025-07-01", "2026-08-07")

    # —— 逐臂逐窗：累计 IC 逐日序列（缓存，供表格/合并/配对共用）——
    series: dict[str, dict[str, pd.Series]] = {a: {} for a in ARM_ORDER}
    for win, (start, end, fetch_end) in WINDOWS.items():
        cols = sorted(set().union(*[set(_read_signal(ARMS[a][win]).columns) for a in ARM_ORDER]))
        px = _fetch_px(provider, cols, start, fetch_end)
        logger.info(f"[{win}] px {px.shape[0]} 日 × {px.shape[1]} 列")
        for arm in ARM_ORDER:
            sig = _read_signal(ARMS[arm][win]).reindex(index=px.index, columns=px.columns)
            series[arm][win] = cum_ic_series(sig, px, K_JUDGE)
            logger.info(f"[{win}] {arm}: n={len(series[arm][win])} "
                        f"k=10 mean={series[arm][win].mean():+.4f}")

    # —— 统一表：每窗逐 k 的 mean / NW t ——
    table: dict[str, dict[str, list[dict]]] = {}
    for win, (start, end, fetch_end) in WINDOWS.items():
        cols = sorted(set().union(*[set(_read_signal(ARMS[a][win]).columns) for a in ARM_ORDER]))
        px = _fetch_px(provider, cols, start, fetch_end)
        table[win] = {}
        for arm in ARM_ORDER:
            sig = _read_signal(ARMS[arm][win]).reindex(index=px.index, columns=px.columns)
            rows = []
            for k in K_RANGE:
                s = cum_ic_series(sig, px, k)
                rows.append({
                    "k": k, "n_days": int(len(s)),
                    "mean": float(s.mean()) if len(s) else float("nan"),
                    "t": _nw_tvalue(s.to_numpy(), lag=k - 1) if len(s) > 2 else float("nan"),
                })
            table[win][arm] = rows
            logger.info(f"[{win}][表] {arm}: " + " ".join(
                f"k{r['k']}:{r['mean']:+.4f}({r['t']:+.1f})" for r in rows))

    # —— IC1：k=10 两窗同正 ∧ 合并 n=260 NW(lag=9) t>2 ——
    ic1: dict[str, dict] = {}
    for arm in H1_ARMS:
        s1, s2 = series[arm]["backtest"], series[arm]["2025h2"]
        pooled = pd.concat([s1, s2]).to_numpy()
        m1, m2 = float(s1.mean()), float(s2.mean())
        t_pool = _nw_tvalue(pooled, lag=K_JUDGE - 1)
        ic1[arm] = {
            "bt_mean": m1, "h2_mean": m2,
            "both_pos": bool(m1 > 0 and m2 > 0),
            "pooled_n": int(len(pooled)), "pooled_t": float(t_pool),
            "pass": bool(m1 > 0 and m2 > 0 and t_pool > 2.0),
        }
        logger.info(f"[IC1] {arm}: bt={m1:+.4f} h2={m2:+.4f} pooled_t={t_pool:+.2f} "
                    f"→ {'PASS' if ic1[arm]['pass'] else 'fail'}")

    # —— IC2：同头配对差 k=10 合并 NW t ——
    ic2: dict[str, dict] = {}
    for a, b in PAIRS:
        d1 = series[a]["backtest"] - series[b]["backtest"]
        d2 = series[a]["2025h2"] - series[b]["2025h2"]
        d1, d2 = d1.dropna(), d2.dropna()
        pooled = pd.concat([d1, d2]).to_numpy()
        ic2[f"{a}−{b}"] = {
            "diff_bt": float(d1.mean()) if len(d1) else float("nan"),
            "diff_h2": float(d2.mean()) if len(d2) else float("nan"),
            "pooled_n": int(len(pooled)),
            "pooled_diff": float(pooled.mean()) if len(pooled) else float("nan"),
            "pooled_t": _nw_tvalue(pooled, lag=K_JUDGE - 1) if len(pooled) > 2 else float("nan"),
            "significant": bool(len(pooled) > 2 and _nw_tvalue(pooled, lag=K_JUDGE - 1) > 2.0),
        }
        r = ic2[f"{a}−{b}"]
        logger.info(f"[IC2] {a}−{b}: diff bt={r['diff_bt']:+.4f} h2={r['diff_h2']:+.4f} "
                    f"pooled_t={r['pooled_t']:+.2f} → {'显著' if r['significant'] else '带内方向描述'}")

    # —— 引擎 v2 附表（描述性）——
    from paper_replication.benchmark import probe_index_benchmark
    from paper_replication.engine_v2 import (
        EngineConfigV2,
        build_limit_masks,
        build_pool_equal_weight_benchmark_v2,
        compute_perf_v2,
        run_portfolio_v2,
    )
    from paper_replication.replay_v2 import _FROZEN_FILES, fetch_window

    def _frozen_universe(win: str) -> list[str]:
        """冻结口径轮次并集宇宙（与 g5/r1/h1 引擎封存一致）。"""
        return sorted(set().union(*[
            set(pd.read_parquet((REPO / f).resolve()).columns) for f in _FROZEN_FILES[win]
        ]))

    engine_rows: dict[str, dict] = {}
    for win, (start, end, _fe) in WINDOWS.items():
        uni = _frozen_universe(win)
        px, trd, uls = fetch_window(provider, uni, start, end)
        masks = build_limit_masks(uls_wide=uls)
        bench_idx = probe_index_benchmark(provider, start, end)
        for arm in (*H1_ARMS, "G1_s100", "M"):
            sig = _read_signal(ARMS[arm][win]).reindex(index=px.index, columns=px.columns)
            ret, trades = run_portfolio_v2(
                sig, px, trd, cfg=EngineConfigV2(), buy_blocked=masks[0], sell_blocked=masks[1]
            )
            bench_ew = build_pool_equal_weight_benchmark_v2(px, trd, sig, fix_mask=True)
            pi = compute_perf_v2(_excess(ret, bench_idx), trades, name=arm)
            pe = compute_perf_v2(_excess(ret, bench_ew), trades, name=arm)
            engine_rows.setdefault(arm, {})[win] = {
                "aer_idx": pi.aer, "ir_idx": pi.ir, "aer_ew": pe.aer, "ir_ew": pe.ir,
            }
            logger.info(f"[v2附表][{win}] {arm}: ew={pe.aer:+.2%} idx={pi.aer:+.2%}")

    out = {
        "experiment": "h1_ic_judge", "seed": SEED, "date": "2026-09-05",
        "k_judge": K_JUDGE, "min_cross_n": MIN_CROSS_N,
        "arms": {a: ARMS[a] for a in ARM_ORDER},
        "windows": {w: WINDOWS[w][:2] for w in WINDOWS},
        "table": table, "IC1": ic1, "IC2": ic2,
        "engine_v2": engine_rows,
    }
    out_path = H1_DATA / "h1_ic_results.json"
    resolved = out_path.resolve()
    if not resolved.is_relative_to(REPO.resolve()):
        raise ValueError(f"落盘路径越界：{resolved}")
    resolved.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    logger.info(f"H1 IC 判读数据落盘 {resolved}")


def _excess(daily_ret: pd.Series, bench: pd.Series) -> pd.Series:
    common = daily_ret.index.intersection(bench.index)
    return daily_ret.loc[common] - bench.loc[common]


if __name__ == "__main__":
    main()
