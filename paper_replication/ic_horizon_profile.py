"""信号持有日 IC 剖面（k=1..10）+ 推理取数窗口断言。

产出 ``docs/信号持有日IC剖面_20260905.md`` 的数据源（描述性分析，
不改任何预注册判据）：

    1. **单日收益 IC（默认模式）**：对 G1 三种子 mean 信号（backtest
       2026-01-01~2026-07-24 / 2025H2 2025-07-01~2025-12-31 两窗）与 M
       （10 日动量，同窗对照），计算信号日 t 对**第 k 个交易日单日收益**
       （k=1..10，ret(t+k) = close(t+k)/close(t+k−1) − 1）的横截面
       Spearman rank-IC，逐日取均值，t 值 = mean/std×√n。
    2. **累计收益 IC（``--cumulative``）**：信号日 t 对 t+1..t+k **累计收益**
       （close(t+k)/close(t) − 1）的 rank-IC，k=1..10；相邻信号日的 IC 因
       收益窗重叠而自相关，t 值用 Newey-West HAC（Bartlett 核，
       **lag = k−1**）。
    3. **取数窗口断言**：对 ``kronos_qlib.windows.build_inference_windows``
       （``paper_replication/signal.py`` K 组信号的唯一取数入口）在两窗
       抽样日实跑，断言每个推理输入窗口的最后一根 bar 日期 ≤ 信号日 t
       （防未来数据进入输入）。

种子→parquet 映射与 g5/g4/r1 的 G1_SIGNAL_PARQUETS 冻结口径一致
（s100=G1、s101/s102=G2S101/102）。

用法::

    python -m paper_replication.ic_horizon_profile               # 单日模式
    python -m paper_replication.ic_horizon_profile --cumulative  # 累计模式（合并入同一 JSON）
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from paper_replication.common import DATA_DIR, REPO_ROOT

WINDOWS: dict[str, tuple[str, str, str]] = {
    # 窗口 → (start, end, 取数缓冲末页)：end 之后需 ≥10 个交易日覆盖 k=10
    "backtest": ("2026-01-01", "2026-07-24", "2026-08-07"),
    "2025h2": ("2025-07-01", "2025-12-31", "2026-01-31"),
}

ARMS: dict[tuple[str, str], str] = {
    ("backtest", "G1_s100"): "finetune_suite/data/g1/daily_signals_backtest_G1_mean.parquet",
    ("backtest", "G1_s101"): "finetune_suite/data/g2/s101/daily_signals_backtest_G2S101_mean.parquet",
    ("backtest", "G1_s102"): "finetune_suite/data/g2/s102/daily_signals_backtest_G2S102_mean.parquet",
    ("backtest", "M"): "finetune_suite/data/daily_signals_backtest_M.parquet",
    ("2025h2", "G1_s100"): "g5_head/data/daily_signals_2025h2_G1_mean.parquet",
    ("2025h2", "G1_s101"): "finetune_suite/data/g2/s101/daily_signals_2025h2_G2S101_mean.parquet",
    ("2025h2", "G1_s102"): "finetune_suite/data/g2/s102/daily_signals_2025h2_G2S102_mean.parquet",
    ("2025h2", "M"): "finetune_suite/data/g0/daily_signals_2025h2_M.parquet",
}

ARM_ORDER = ("G1_s100", "G1_s101", "G1_s102", "M")
K_RANGE = range(1, 11)
MIN_CROSS_N = 30  # 单日截面有效样本下限（信号与收益均非缺失）


def _fetch_px(provider, cols, start, end) -> pd.DataFrame:
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = start
        provider._end_date = end
        provider.instruments_ = list(cols)
        df = provider.fetch(["$close"], freq="day")
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig
    px = df["close"].unstack("instrument").sort_index()
    return px


def _nw_tvalue(x: np.ndarray, lag: int) -> float:
    """均值的新协方差稳健（Newey-West HAC，Bartlett 核）t 值。

    相邻信号日的累计收益 IC 因收益窗重叠（重叠 k−1 日）自相关，普通
    t = mean/std×√n 会低估标准误；γ_l 用 1/n 归一（Bartlett 下保证 PSD），
    lag=k−1 时 k=1 退化为普通 t。
    """
    n = len(x)
    mean = float(x.mean())
    dev = x - mean
    var = float(dev @ dev) / n
    for l in range(1, min(lag, n - 1) + 1):
        g_l = float(dev[l:] @ dev[:-l]) / n
        var += 2.0 * (1.0 - l / (lag + 1)) * g_l
    if var <= 0:
        return float("nan")
    return mean / float(np.sqrt(var / n))


def ic_profile(sig: pd.DataFrame, px: pd.DataFrame, k: int, *, mode: str = "single") -> dict:
    """信号日 t 对第 k 个交易日后收益的横截面 rank-IC（逐日 → 均值/t 值）。

    :param mode: ``single`` = 第 k 日**单日**收益（t 值 = mean/std×√n，逐日
        IC 近似独立）；``cumulative`` = t+1..t+k **累计**收益（t 值用
        Newey-West(lag=k−1)，处理重叠窗自相关）。
    """
    if mode == "cumulative":
        target = px.shift(-k) / px - 1.0  # 行 t = close(t+k)/close(t) − 1
    else:
        rets = px.pct_change(fill_method=None)
        target = rets.shift(-k)  # 行 t = 第 pos+k 日的单日收益
    ics: list[float] = []
    n_skipped_tail = 0
    for t in sig.index:
        if t not in px.index:
            continue
        pos = px.index.get_loc(t)
        if pos + k >= len(px.index):
            n_skipped_tail += 1
            continue
        r = target.loc[t]
        s = sig.loc[t]
        # 对齐修正（2026-09-05 发现）：scipy.spearmanr 对 Series 按位置配对、
        # 不对齐索引——信号 parquet 列序普遍无序（G1/G2/R1/H1 实测），px 列
        # 有序，错位配对会产出 ~0 的伪 IC。先 r.reindex(s.index) 强制同序。
        r = r.reindex(s.index)
        mask = s.notna() & r.notna()
        if int(mask.sum()) < MIN_CROSS_N:
            continue
        rho, _ = stats.spearmanr(s[mask].to_numpy(), r[mask].to_numpy())
        if pd.notna(rho):
            ics.append(float(rho))
    arr = np.array(ics)
    if len(arr) < 2:
        return {"k": k, "mode": mode, "n_days": len(arr), "mean": float("nan"),
                "t": float("nan"), "skipped_tail": n_skipped_tail}
    mean = float(arr.mean())
    if mode == "cumulative":
        t_val = _nw_tvalue(arr, lag=k - 1)
    else:
        sd = float(arr.std(ddof=1))
        t_val = mean / sd * np.sqrt(len(arr)) if sd > 0 else float("nan")
    return {
        "k": k, "mode": mode, "n_days": len(arr), "mean": mean,
        "t": t_val, "skipped_tail": n_skipped_tail,
    }


def assert_inference_window(provider, sample_dates: list[str]) -> list[str]:
    """实跑 build_inference_windows，断言输入窗口最后一根 bar ≤ 信号日 t。"""
    from kronos_qlib import build_inference_windows

    lines = []
    for ds in sample_dates:
        df_list, _x, _y, codes, stats_d = build_inference_windows(
            provider, ds, lookback=90, predict_len=10, pool="csi300"
        )
        t = pd.Timestamp(ds)
        max_last = max(df.index.max() for df in df_list)
        ok = bool(max_last <= t)
        # 输入窗口行长也须 = lookback（窗口不足者已被跳过，不进 df_list）
        lens_ok = all(len(df) == 90 for df in df_list)
        lines.append(
            f"t={ds}: {len(df_list)} 窗（池 {stats_d['n_pool']}，跳停牌 "
            f"{stats_d['skipped_halt']} / 历史不足 {stats_d['skipped_short']}），"
            f"max(last_bar)={max_last.date()} ≤ t：{'PASS' if ok else 'FAIL'}；"
            f"行长=90 全体成立：{'PASS' if lens_ok else 'FAIL'}"
        )
        assert ok and lens_ok, f"取数窗口断言失败：t={ds} max_last={max_last}"
    return lines


def main() -> None:
    from kronos_qlib import QlibProvider

    results: dict[str, dict] = {}
    assert_lines: dict[str, list[str]] = {}

    for win, (start, end, fetch_end) in WINDOWS.items():
        sigs = {}
        for (w, a), rel in ARMS.items():
            if w != win:
                continue
            sig_path = (REPO_ROOT / rel).resolve()
            if not sig_path.is_relative_to(REPO_ROOT.resolve()):
                raise ValueError(f"信号 parquet 路径越界：{sig_path}")
            sigs[a] = pd.read_parquet(sig_path)
        cols = sorted(set().union(*[set(s.columns) for s in sigs.values()]))
        provider = QlibProvider("csi300", start, fetch_end)
        px = _fetch_px(provider, cols, start, fetch_end)
        logger.info(f"[{win}] px {px.shape[0]} 日 × {px.shape[1]} 列（{start}~{fetch_end}）")

        results[win] = {}
        for arm in ARM_ORDER:
            sig = sigs[arm]
            results[win][arm] = [ic_profile(sig, px, k) for k in K_RANGE]
            row = " ".join(f"k{r['k']}:{r['mean']:+.4f}({r['t']:+.1f})" for r in results[win][arm])
            logger.info(f"[{win}] {arm}: {row}")

        # —— 取数窗口断言：按信号日索引五分位抽样，每窗 5 日 ——
        idx = sigs["M"].index
        picks = sorted({idx[0], idx[len(idx) // 4], idx[len(idx) // 2],
                        idx[3 * len(idx) // 4], idx[-1]})
        assert_lines[win] = assert_inference_window(
            provider, [d.strftime("%Y-%m-%d") for d in picks]
        )
        for ln in assert_lines[win]:
            logger.info(f"[{win}][窗口断言] {ln}")

    out = {
        "windows": {w: WINDOWS[w][:2] for w in WINDOWS},
        "arms": {f"{w}|{a}": p for (w, a), p in ARMS.items()},
        "min_cross_n": MIN_CROSS_N,
        "ic": results,
        "window_assert": assert_lines,
    }
    out_path = DATA_DIR / "ic_horizon_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    logger.info(f"IC 剖面落盘 {out_path}")


if __name__ == "__main__":
    import sys

    if "--cumulative" in sys.argv:
        # 累计收益剖面（NW lag=k−1 t 值）——实现体见 ic_horizon_cum（门禁隔离）
        from paper_replication.ic_horizon_cum import main_cumulative

        main_cumulative()
    else:
        main()
