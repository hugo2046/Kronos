"""多窗 IC 剖面（docs/多窗IC剖面计划_20260906.md §4.2，纯描述性、零判据）。

四窗（半年切、互不重叠，各 +10 交易日缓冲）：

    W1 2024H2（2024-07-01~2024-12-31）、W2 2025H1、W3 2025H2、
    W4 2026H1（=backtest，至 07-24）。

臂 × 可用窗（计划 §1 冻结）：

    - F0_mean（官方权重 zero-shot，canonical）：W1~W4——W1/W2 由 paper 窗
      baseline mean 逐日宽表按窗切分（4.1 已确认在盘，不触发 §1 重算例外）；
    - M 动量：W1~W4（同上切分）；
    - G1 三种子 mean：W3~W4（W1/W2 与其 train/val 重叠，禁用）；
    - F1_mean：W3~W4（W3 用 g0 G0_mean parquet——按扩语料实验定义即
      F1 权重 @2025H2 重打分）；
    - H1a-kda（seed 42）：W3~W4（仅并列呈现）。

量（与 ic_horizon_profile 同口径 + 对齐修正 49e6875）：csi300 池、
k∈{1,5,10} 累计收益 rank-IC、NW(lag=k−1) t；每臂另给**四窗合并**（F0/M
n≈500、其余 n=260）的 IC 与 NW t。落盘
``paper_replication/data/ic_multiwindow.json``。

用法：``/home/user/miniconda3/envs/quant/bin/python -m paper_replication.ic_multiwindow``
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

from paper_replication.common import DATA_DIR, REPO_ROOT
from paper_replication.ic_horizon_profile import MIN_CROSS_N, _fetch_px, _nw_tvalue

WIN_ORDER = ("W1", "W2", "W3", "W4")
WINDOWS: dict[str, tuple[str, str, str]] = {
    # 窗 → (start, end, fetch_end)；fetch_end ≥ end + 10 个交易日，无尾部截断
    "W1": ("2024-07-01", "2024-12-31", "2025-01-31"),
    "W2": ("2025-01-01", "2025-06-30", "2025-07-20"),
    "W3": ("2025-07-01", "2025-12-31", "2026-01-31"),
    "W4": ("2026-01-01", "2026-07-24", "2026-08-07"),
}

B = "baseline_suite/data"
FS = "finetune_suite/data"
G5 = "g5_head/data"
H1 = "h1_readout/data"

# 臂 → {窗: (parquet 相对路径, 切分标志)}；切分标志=True 时按窗口起止切行
ARMS: dict[str, dict[str, tuple[str, bool]]] = {
    "F0_mean": {
        "W1": (f"{B}/daily_signals_paper_mean.parquet", True),
        "W2": (f"{B}/daily_signals_paper_mean.parquet", True),
        "W3": (f"{FS}/g0/daily_signals_2025h2_F0_mean.parquet", False),
        "W4": (f"{FS}/daily_signals_backtest_F0_mean.parquet", False),
    },
    "M": {
        "W1": (f"{B}/daily_signals_paper_M.parquet", True),
        "W2": (f"{B}/daily_signals_paper_M.parquet", True),
        "W3": (f"{FS}/g0/daily_signals_2025h2_M.parquet", False),
        "W4": (f"{FS}/daily_signals_backtest_M.parquet", False),
    },
    "G1_s100": {
        "W3": (f"{G5}/daily_signals_2025h2_G1_mean.parquet", False),
        "W4": (f"{FS}/g1/daily_signals_backtest_G1_mean.parquet", False),
    },
    "G1_s101": {
        "W3": (f"{FS}/g2/s101/daily_signals_2025h2_G2S101_mean.parquet", False),
        "W4": (f"{FS}/g2/s101/daily_signals_backtest_G2S101_mean.parquet", False),
    },
    "G1_s102": {
        "W3": (f"{FS}/g2/s102/daily_signals_2025h2_G2S102_mean.parquet", False),
        "W4": (f"{FS}/g2/s102/daily_signals_backtest_G2S102_mean.parquet", False),
    },
    "F1_mean": {
        "W3": (f"{FS}/g0/daily_signals_2025h2_G0_mean.parquet", False),  # =F1 权重@2025H2
        "W4": (f"{FS}/daily_signals_backtest_F1_mean.parquet", False),
    },
    "H1a-kda_s42": {
        "W3": (f"{H1}/H1a-kda/daily_signals_2025h2_H1a-kda_s42.parquet", False),
        "W4": (f"{H1}/H1a-kda/daily_signals_backtest_H1a-kda_s42.parquet", False),
    },
}
ARM_ORDER = ("F0_mean", "M", "G1_s100", "G1_s101", "G1_s102", "F1_mean", "H1a-kda_s42")
K_SET = (1, 5, 10)

_cache: dict[str, pd.DataFrame] = {}


def _load(rel: str) -> pd.DataFrame:
    if rel not in _cache:
        p = (REPO_ROOT / rel).resolve()
        if not p.is_relative_to(REPO_ROOT.resolve()):
            raise ValueError(f"信号 parquet 路径越界：{p}")
        _cache[rel] = pd.read_parquet(p)
    return _cache[rel]


def cum_ic_series(sig: pd.DataFrame, px: pd.DataFrame, k: int) -> pd.Series:
    """累计收益 rank-IC 逐日序列（对齐修正版：r.reindex(s.index) 后按位配对）。"""
    target = px.shift(-k) / px - 1.0
    out: dict = {}
    for t in sig.index:
        if t not in px.index:
            continue
        if px.index.get_loc(t) + k >= len(px.index):
            continue
        r = target.loc[t]
        s = sig.loc[t]
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

    provider = QlibProvider("csi300", "2024-07-01", "2026-08-07")
    # series[arm][win][k] = 逐日 IC 序列（供窗内统计与跨窗合并共用）
    series: dict[str, dict[str, dict[int, pd.Series]]] = {a: {} for a in ARM_ORDER}
    n_days: dict[str, dict[str, int]] = {a: {} for a in ARM_ORDER}

    for win in WIN_ORDER:
        start, end, fetch_end = WINDOWS[win]
        loaded: dict[str, pd.DataFrame] = {}
        for arm in ARM_ORDER:
            if win not in ARMS[arm]:
                continue
            rel, sliced = ARMS[arm][win]
            wide = _load(rel)
            if sliced:
                wide = wide.loc[(wide.index >= pd.Timestamp(start))
                                & (wide.index <= pd.Timestamp(end))]
            loaded[arm] = wide
        cols = sorted(set().union(*[set(w.columns) for w in loaded.values()]))
        px = _fetch_px(provider, cols, start, fetch_end)
        logger.info(f"[{win}] {start}~{end} px {px.shape[0]} 日 × {px.shape[1]} 列")
        for arm, wide in loaded.items():
            sig = wide.reindex(index=px.index, columns=px.columns)
            series[arm][win] = {k: cum_ic_series(sig, px, k) for k in K_SET}
            n_days[arm][win] = int(len(series[arm][win][K_SET[-1]]))
            cells = " ".join(
                f"k{k}:{s.mean():+.4f}({_nw_tvalue(s.to_numpy(), lag=k - 1):+.1f})"
                for k, s in series[arm][win].items()
            )
            logger.info(f"[{win}] {arm}: n={n_days[arm][win]} {cells}")

    out = {
        "experiment": "ic_multiwindow", "date": "2026-09-06",
        "windows": {w: WINDOWS[w][:2] for w in WIN_ORDER},
        "k_set": list(K_SET), "min_cross_n": MIN_CROSS_N,
        "arms": {a: {w: ARMS[a][w][0] for w in ARMS[a]} for a in ARM_ORDER},
        "note_f1_w3": "F1_mean@W3 = finetune_suite g0 G0_mean parquet（扩语料实验定义：F1 权重 @2025H2 重打分）",
        "table": {}, "pooled": {},
    }
    for arm in ARM_ORDER:
        out["table"][arm] = {}
        for win in series[arm]:
            out["table"][arm][win] = {
                str(k): {
                    "n": int(len(s)), "mean": float(s.mean()),
                    "t": float(_nw_tvalue(s.to_numpy(), lag=k - 1)),
                }
                for k, s in series[arm][win].items()
            }
        out["pooled"][arm] = {
            str(k): {
                "n": int(sum(len(series[arm][w][k]) for w in series[arm])),
                "mean": float(pd.concat([series[arm][w][k] for w in series[arm]]).mean()),
                "t": float(_nw_tvalue(
                    pd.concat([series[arm][w][k] for w in series[arm]]).to_numpy(),
                    lag=k - 1)),
            }
            for k in K_SET
        }
        p10 = out["pooled"][arm]["10"]
        logger.info(f"[pooled] {arm}: k=10 n={p10['n']} mean={p10['mean']:+.4f} "
                    f"t={p10['t']:+.2f}")

    out_path = DATA_DIR / "ic_multiwindow.json"
    resolved = out_path.resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"落盘路径越界：{resolved}")
    resolved.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    logger.info(f"多窗 IC 剖面落盘 {resolved}")


if __name__ == "__main__":
    main()
