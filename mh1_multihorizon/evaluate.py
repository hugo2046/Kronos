"""MH1 评估与封盘统计（计划 §5 任务 C / §6 判据）。

纪律：全部六个模型完成、W3/W4 信号落盘并核对（无重复键、无 forward 日期、
模型齐全）之后，才允许统一一次计算研究指标；IC 一律按 code 对齐（历史
spearman 位置配对错位教训，见 ``docs/H1实验结果_20260905.md`` §0）。

统计口径（§6）：

- 每日每种子：M/S 共同股票上分别算 RankIC → 差值；
- ``delta_t`` = 三种子日均差；M 参照列 = 三种子日均 IC；
- 分窗 NW(lag=9)；合并统计 = 分段 HAC（跨窗不建滞后协方差、缺失日期按原
  交易日距离配对、V≤0 不可判）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from mh1_multihorizon.config import IC_MIN_STOCKS, NW_LAG


# ============================================================
# IC 原语（code 对齐，拒绝位置配对）
# ============================================================


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（numpy 实现，无 scipy 位置配对隐患）。"""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    if denom <= 0:
        return float("nan")
    return float((rx @ ry) / denom)


def rank_ic_by_code(
    scores: dict[str, float], rets: dict[str, float],
    *, min_stocks: int = IC_MIN_STOCKS,
) -> tuple[Optional[float], Optional[str]]:
    """按 code 对齐的截面 RankIC；缺失记 (None, 原因)，不填零。

    :param scores: ``{code: 分数}``。
    :param rets: ``{code: 原始收益}``（NaN 标签剔除后计数）。
    :param min_stocks: 共同有效股票下限。
    :returns: ``(ic, None)`` 或 ``(None, reason)``。
    """
    common = [c for c in scores if c in rets and np.isfinite(rets[c])
              and np.isfinite(scores[c])]
    if len(common) < min_stocks:
        return None, f"common stocks {len(common)} < {min_stocks}"
    x = np.array([scores[c] for c in common])
    y = np.array([rets[c] for c in common])
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, "constant scores/labels"
    return _spearman(x, y), None


def daily_ic_from_frame(
    df: pd.DataFrame, date: str, *, min_stocks: int = IC_MIN_STOCKS,
) -> tuple[Optional[float], Optional[str]]:
    """长格式信号帧的单日 IC；重复 (date, code) 必须拒绝。"""
    sub = df[df["date"] == date]
    if sub["instrument"].duplicated().any():
        dup = sub.loc[sub["instrument"].duplicated(), "instrument"].tolist()[:3]
        raise ValueError(f"{date} 重复 (date,code)：{dup}")
    scores = dict(zip(sub["instrument"], sub["score"].astype(float)))
    rets = dict(zip(sub["instrument"], sub["label"].astype(float)))
    return rank_ic_by_code(scores, rets, min_stocks=min_stocks)


def paired_daily_ic(
    scores_m: dict[str, float], scores_s: dict[str, float],
    labels: dict[str, float], *, min_stocks: int = IC_MIN_STOCKS,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """M/S 同日共同股票配对 IC（同一股票集合，逐臂 Spearman + 差值）。"""
    common = [c for c in scores_m if c in scores_s and c in labels
              and np.isfinite(labels[c])
              and np.isfinite(scores_m[c]) and np.isfinite(scores_s[c])]
    if len(common) < min_stocks:
        return None, None, None, f"common stocks {len(common)} < {min_stocks}"
    xm = np.array([scores_m[c] for c in common])
    xs = np.array([scores_s[c] for c in common])
    y = np.array([labels[c] for c in common])
    if np.all(xm == xm[0]) or np.all(xs == xs[0]) or np.all(y == y[0]):
        return None, None, None, "constant scores/labels"
    icm = _spearman(xm, y)
    ics = _spearman(xs, y)
    return icm, ics, icm - ics, None


def aggregate_seeds(
    daily: dict[str, dict[str, float]], seeds: Sequence[str],
) -> dict:
    """三种子日均序列：只用全部种子共同有效的日期（§6）。"""
    dates = sorted(daily)
    common_dates = [
        d for d in dates
        if all(s in daily[d] and np.isfinite(daily[d][s]) for s in seeds)
    ]
    values = [float(np.mean([daily[d][s] for s in seeds])) for d in common_dates]
    return {
        "n_days": len(common_dates),
        "dates": common_dates,
        "values": values,
        "mean": float(np.mean(values)) if values else float("nan"),
    }


# ============================================================
# 分段 HAC（Newey-West，Bartlett 核）
# ============================================================


@dataclass
class HacResult:
    mu: float
    V: float
    t: float
    N: int
    judgable: bool
    per_window: dict

    def to_dict(self) -> dict:
        return {
            "mu": self.mu, "V": self.V, "t": self.t, "N": self.N,
            "judgable": self.judgable, "per_window": self.per_window,
        }


def segmented_hac(
    windows: Sequence[tuple[str, Sequence[float], Sequence[int]]],
    lag: int = NW_LAG,
) -> dict:
    """分段 Newey-West(lag) 合并 t 值（计划 §6 公式）。

    ``V = [Σz² + 2·Σ_{l=1..lag}(1−l/(lag+1))·Σ_within z_t·z_{t−l}] / N²``，
    ``t = mu / √V``。要点：

    - 中心化用**合并**均值 mu（单窗时退化为窗均值 → 与
      ``paper_replication.ic_horizon_profile._nw_tvalue`` 数值恒等）；
    - 滞后积只在**窗内**配对——W3 尾日与 W4 首日不建协方差；
    - 每个观测携带交易日历位置，缺失日期不压缩：lag-l 配对要求日历位差
      恰为 l（不是序列位置差）；
    - ``V ≤ 0`` 或样本不足 → ``judgable=False``（不可判，不通过）。

    :param windows: ``[(名称, 值序列, 交易日历位置序列), ...]``。
    :returns: :class:`HacResult` 的 dict 形式。
    """
    all_vals = np.concatenate([np.asarray(w[1], dtype=float) for w in windows])
    N = len(all_vals)
    mu = float(all_vals.mean()) if N else float("nan")
    V = float("nan")
    judgable = False
    per_window: dict = {}
    if N == 0:
        return HacResult(mu, V, float("nan"), 0, False, per_window).to_dict()
    z_all = all_vals - mu
    sse = float(z_all @ z_all)
    cross = 0.0
    for name, vals, positions in windows:
        arr = np.asarray(vals, dtype=float)
        z = arr - mu
        per_window[name] = {
            "n": len(arr), "mean": float(arr.mean()) if len(arr) else float("nan")}
        pos = {int(p): float(zz) for p, zz in zip(positions, z)}
        # l 遍历到 lag（不按 len-1 截断）：有缺档时日历距离可超过 n-1；
        # 连续序列下 distance-l 配对恰在 l ≤ n-1 存在，与 _nw_tvalue 恒等
        for l in range(1, lag + 1):
            wgt = 2.0 * (1.0 - l / (lag + 1))
            for p, zv in pos.items():
                if p - l in pos:
                    cross += wgt * zv * pos[p - l]
    V = (sse + cross) / (N * N)
    if N < 2:
        t = float("nan")
    elif V > 0:
        t = mu / float(np.sqrt(V))
        judgable = True
    else:
        t = float("nan")
    return HacResult(mu, V, t, N, judgable, per_window).to_dict()


__all__ = [
    "rank_ic_by_code", "daily_ic_from_frame", "paired_daily_ic",
    "aggregate_seeds", "segmented_hac", "HacResult",
]

# ============================================================
# 以下为 evaluate 阶段执行段（计划 §5 任务 C：信号、主表、判据、参照、
# 引擎附表、计时）。判读纪律：六模型信号全部落盘并核对后统一一次计算。
# ============================================================

from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from mh1_multihorizon.config import (  # noqa: E402
    ARMS, DATA_DIR, HORIZONS, MAIN_HORIZON_IDX, POOL, SEEDS, W3_END, W3_START,
    W4_END, W4_START, FORWARD_CUTOFF, verify_protocol,
)

WINDOW_BOUNDS = {"W3": (W3_START, W3_END), "W4": (W4_START, W4_END)}
_RUNS = [f"{a}{s}" for s in SEEDS for a in ARMS]  # S42..M44 顺序无关此处


def load_models(protocol: dict) -> dict[str, dict]:
    """载入六个新头 checkpoint（协议/完成态/底座哈希三重门禁）。"""
    from mh1_multihorizon.train import load_checkpoint

    out: dict[str, dict] = {}
    for run in _RUNS:
        path = DATA_DIR / f"run_{run}_best.pt"
        if not path.is_file():
            raise RuntimeError(f"checkpoint 缺失：{path}（先完成 --stage train）")
        ck = load_checkpoint(path)
        if not ck.get("done"):
            raise RuntimeError(f"{run} 训练未完成（done=False），拒绝评估")
        if ck["protocol_sha256"] != protocol.get("_sha256"):
            raise RuntimeError(f"{run} 协议哈希与当前协议不一致，拒绝评估")
        for k, v in ck["backbone_refs"].items():
            if protocol["g1_weights"].get(k) != v:
                raise RuntimeError(f"{run} 底座哈希 {k} 与协议不一致")
        out[run] = ck
    return out


def build_windows(provider) -> dict[str, list]:
    """构造 W3/W4 PIT 评估日（标签 NaN 允许，IC 时按可用性匹配）。"""
    from mh1_multihorizon.data import build_pit_days

    windows: dict[str, list] = {}
    for wname, (start, end) in WINDOW_BOUNDS.items():
        days, stats = build_pit_days(
            provider, start=start, end=end, label_end=end,
            require_complete_labels=False)
        assert days, f"{wname} 无评估日"
        windows[wname] = days
        logger.info(f"[{wname}] {stats.n_days} 日（{stats.decision_min}~"
                    f"{stats.decision_max}），末日 {len(days[-1].codes)} 只")
    return windows


def score_and_write_signals(
    models: dict[str, dict], backbone, windows: dict[str, list],
    device: str, protocol_sha: str,
) -> dict[str, pd.DataFrame]:
    """六模型 × 两窗逐日打分 → 长格式信号 parquet（一次落盘，判读前置）。"""
    heads = {}
    from mh1_multihorizon.heads import MultiHorizonHead

    for run, ck in models.items():
        h = MultiHorizonHead(backbone.d_model).to(device)
        h.load_state_dict(ck["state_dict"])
        h.eval()
        heads[run] = h

    frames: dict[str, pd.DataFrame] = {}
    for wname, days in windows.items():
        rows: list[pd.DataFrame] = []
        for d in days:
            with torch.no_grad():
                hidden = backbone.extract(d.x_norm.to(device),
                                          d.stamp.to(device))
                for run, h in heads.items():
                    scores = h(hidden).cpu().numpy()   # [N,4]
                    df = pd.DataFrame({
                        "date": str(d.date.date()),
                        "instrument": d.codes,
                        "arm": run[0], "seed": int(run[1:]),
                        "protocol_sha256": protocol_sha})
                    for hi, k in enumerate(HORIZONS):
                        df[f"h{k}"] = scores[:, hi]
                    rows.append(df)
        long_df = pd.concat(rows, ignore_index=True)
        out = DATA_DIR / f"signals_{wname}.parquet"
        long_df.to_parquet(out, index=False)
        # 落盘门禁：无重复键、无 forward 日期、六模型齐全
        assert not long_df.duplicated(
            ["date", "instrument", "arm", "seed"]).any(), f"{wname} 重复键"
        assert long_df["date"].max() <= FORWARD_CUTOFF, f"{wname} 越过封存线"
        got = sorted(set(long_df["arm"] + long_df["seed"].astype(str)))
        assert got == sorted(_RUNS), f"{wname} 模型不齐：{set(got) ^ set(_RUNS)}"
        logger.info(f"[{wname}] 信号落盘 {out.name}：{len(long_df):,} 行 "
                    f"（{long_df['date'].nunique()} 日 × "
                    f"{long_df['instrument'].nunique()} 股上限）")
        frames[wname] = long_df
    return frames


def _day_tables(d) -> dict[int, tuple[dict, dict]]:
    """单日 {期限下标: (labels dict)}。"""
    tabs = {}
    for hi in range(len(HORIZONS)):
        tabs[hi] = ({c: float(l) for c, l in zip(d.codes, d.labels[:, hi])})
    return tabs


def compute_daily_paired_ic(
    frames: dict[str, pd.DataFrame], windows: dict[str, list],
) -> dict:
    """逐日配对 IC：M/S 共同股票、三种子；输出 delta_t 与 M 日均序列。

    主统计（§6）：``delta_t = mean_seed(IC_M − IC_S)``、M 三种子日均 IC——
    均只取三种子共同有效日期；S 未受训辅助输出不评价（只记主期限）。
    """
    out: dict = {"per_window": {}, "seeds": {str(s): {} for s in SEEDS}}
    for wname, days in windows.items():
        df = frames[wname]
        delta_daily: dict[str, dict[str, float]] = {}
        m_daily: dict[str, dict[str, float]] = {}
        s_daily: dict[str, dict[str, float]] = {}
        m_aux: dict[int, dict[str, dict[str, float]]] = {
            hi: {} for hi in range(len(HORIZONS))}
        for d in days:
            ds = str(d.date.date())
            sub = df[df["date"] == ds]
            tabs = _day_tables(d)
            lab_main = tabs[MAIN_HORIZON_IDX]
            for seed in SEEDS:
                sm = sub[(sub["arm"] == "M") & (sub["seed"] == seed)]
                ss = sub[(sub["arm"] == "S") & (sub["seed"] == seed)]
                m_scores = dict(zip(sm["instrument"], sm[f"h{HORIZONS[MAIN_HORIZON_IDX]}"]))
                s_scores = dict(zip(ss["instrument"], ss[f"h{HORIZONS[MAIN_HORIZON_IDX]}"]))
                icm, ics, delta, reason = paired_daily_ic(
                    m_scores, s_scores, lab_main)
                key = str(seed)
                if delta is not None:
                    delta_daily.setdefault(ds, {})[key] = delta
                    m_daily.setdefault(ds, {})[key] = icm
                    s_daily.setdefault(ds, {})[key] = ics
                else:
                    logger.debug(f"{wname} {ds} s{seed} 配对缺失：{reason}")
                # M 各期限描述性 IC（只记 M 受训输出）
                for hi in range(len(HORIZONS)):
                    ic_h, r_h = rank_ic_by_code(
                        dict(zip(sm["instrument"], sm[f"h{HORIZONS[hi]}"])),
                        tabs[hi])
                    if ic_h is not None:
                        m_aux[hi].setdefault(ds, {})[key] = ic_h
        out["per_window"][wname] = {
            "delta": delta_daily, "m_main": m_daily, "s_main": s_daily,
            "m_aux": {f"h{HORIZONS[hi]}": v for hi, v in m_aux.items()},
        }
    return out


def judge(paired: dict, calendar: pd.DatetimeIndex) -> dict:
    """§6 主表与判据 1~4（一次开封）。"""
    cal_pos = {d: i for i, d in enumerate(calendar)}
    seeds_str = [str(s) for s in SEEDS]
    result: dict = {"windows": {}, "criteria": {}, "per_seed": {}}

    def _series(daily: dict) -> tuple[list[str], list[float], list[int]]:
        agg = aggregate_seeds(daily, seeds_str)
        dates = agg["dates"]
        return (dates, agg["values"],
                [cal_pos[pd.Timestamp(d)] for d in dates])

    series = {"delta": {}, "m_main": {}}
    for wname in WINDOW_BOUNDS:
        pw = paired["per_window"][wname]
        for kind in ("delta", "m_main"):
            dates, vals, pos = _series(pw[kind])
            series[kind][wname] = (dates, vals, pos)
            result["windows"].setdefault(kind, {})[wname] = {
                "n_days": len(vals),
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "nw_t": segmented_hac([(wname, vals, pos)])["t"],
                "dates_min": dates[0] if dates else None,
                "dates_max": dates[-1] if dates else None,
            }
    # 合并分段 HAC（跨窗不建滞后协方差）
    for kind in ("delta", "m_main"):
        combined = segmented_hac(
            [(w, series[kind][w][1], series[kind][w][2]) for w in WINDOW_BOUNDS])
        result["windows"][kind]["combined"] = combined

    # 每种子每窗 M−S 均值（判据 3）
    per_seed_pos: dict[str, dict[str, float]] = {}
    for wname in WINDOW_BOUNDS:
        pw = paired["per_window"][wname]
        dates = sorted(set(pw["delta"]))
        for seed in seeds_str:
            vals = [pw["delta"][d][seed] for d in dates
                    if seed in pw["delta"][d]]
            per_seed_pos.setdefault(seed, {})[wname] = (
                float(np.mean(vals)) if vals else float("nan"))
    result["per_seed"] = per_seed_pos

    w_cfg = result["windows"]
    c1 = all(w_cfg["m_main"][w]["mean"] > 0 for w in WINDOW_BOUNDS) and (
        w_cfg["m_main"]["combined"]["judgable"]
        and w_cfg["m_main"]["combined"]["t"] > 2)
    c2 = all(w_cfg["delta"][w]["mean"] > 0 for w in WINDOW_BOUNDS) and (
        w_cfg["delta"]["combined"]["judgable"]
        and w_cfg["delta"]["combined"]["t"] > 2)
    c3 = sum(
        all(v[w] > 0 for w in WINDOW_STATES)
        for v in per_seed_pos.values()) >= 2
    result["criteria"] = {
        "1_m_main_pos_both_windows_and_combined_t2": bool(c1),
        "2_delta_pos_both_windows_and_combined_t2": bool(c2),
        "3_two_of_three_seeds_pos_both_windows": bool(c3),
    }
    return result


WINDOW_STATES = ("W3", "W4")


def reference_ic(windows: dict[str, list]) -> dict:
    """参照行（G1_mean / H1a-lin / H1a-kda s42）：本地信号按本轮同日期/
    同股票/同标签重算；缺失标记，不重训、不复制旧表。

    旧文件窗口命名映射：W3↔``2025h2``、W4↔``backtest``（H1 打分产物键）。
    """
    _old_key = {"W3": "2025h2", "W4": "backtest"}
    refs = {
        "G1_mean": {
            "W3": REPO_ROOT_REF / "g5_head" / "data" /
                  "daily_signals_2025h2_G1_mean.parquet",
            "W4": REPO_ROOT_REF / "finetune_suite" / "data" / "g1" /
                  "daily_signals_backtest_G1_mean.parquet",
        },
        "H1a-lin_s42": {
            w: REPO_ROOT_REF / "h1_readout" / "data" / "H1a-lin" /
               f"daily_signals_{_old_key[w]}_H1a-lin_s42.parquet"
            for w in WINDOW_BOUNDS},
        "H1a-kda_s42": {
            w: REPO_ROOT_REF / "h1_readout" / "data" / "H1a-kda" /
               f"daily_signals_{_old_key[w]}_H1a-kda_s42.parquet"
            for w in WINDOW_BOUNDS},
    }
    out: dict = {}
    for name, per_w in refs.items():
        out[name] = {}
        for wname, path in per_w.items():
            if not path.is_file():
                out[name][wname] = {"status": "missing", "path": str(path)}
                continue
            sig = pd.read_parquet(path)
            vals, dates_used = [], []
            for d in windows[wname]:
                ds = str(d.date.date())
                if ds not in sig.index:
                    continue
                row = sig.loc[ds].dropna()
                lab = {c: l for c, l in zip(
                    d.codes, d.labels[:, MAIN_HORIZON_IDX]) if np.isfinite(l)}
                common = [c for c in row.index if c in lab]
                if len(common) < 30:
                    continue
                ic, _ = rank_ic_by_code(
                    {c: float(row[c]) for c in common},
                    {c: lab[c] for c in common})
                if ic is not None:
                    vals.append(ic)
                    dates_used.append(ds)
            if vals:
                cal = pd.DatetimeIndex(dates_used)
                # 参照 NW 用位置序列（窗内连续决策日，缺档按位距离近似披露）
                t = segmented_hac([(wname, vals, list(range(len(vals))))])["t"]
                out[name][wname] = {
                    "status": "ok", "n_days": len(vals), "mean": float(np.mean(vals)),
                    "nw_t": t, "note": "NW 位置距离（参照行披露口径）",
                }
            else:
                out[name][wname] = {"status": "no-overlap"}
    return out


REPO_ROOT_REF = Path(__file__).resolve().parent.parent


def timing_benchmark(
    models: dict[str, dict], backbone, provider, device: str, day,
) -> dict:
    """耗时与显存（计划 §5：10 warmup + 30 计时，GPU 前后 synchronize）。

    两条读出路径：完整历史→分数（backbone.extract + 头）与已有 hidden→分数；
    另测 G1 生成式参照（pred_len=10、sample_count=20、T=1.0、top_p=0.9，
    端到端含 predictor 内部预处理；次数因成本缩减为 1 warmup + 3 计时，
    与读出路径的 10+30 协议不同，披露为不可直接对比的参照量级）。

    :param day: 计时用 PIT 决策日（取 W4 中间日，避免假日空窗）。
    """
    import time as _time

    d = day
    n = min(128, len(d.codes))
    x = d.x_norm[:n].to(device)
    st = d.stamp[:n].to(device)
    from mh1_multihorizon.heads import MultiHorizonHead

    def _bench(fn, warmup=10, reps=30) -> dict:
        for _ in range(warmup):
            fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(reps):
            t0 = _time.perf_counter()
            fn()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append((_time.perf_counter() - t0) * 1000)
        ts_np = np.array(ts)
        peak = (torch.cuda.max_memory_allocated() / 2**30
                if device.startswith("cuda") else 0.0)
        return {"p50_ms": float(np.percentile(ts_np, 50)),
                "p95_ms": float(np.percentile(ts_np, 95)),
                "peak_gb": peak, "n_samples": int(n), "reps": reps}

    out: dict = {"n_samples": int(n)}
    with torch.no_grad():
        hidden = backbone.extract(x, st)
        for run in ("S42", "M42"):
            h = MultiHorizonHead(backbone.d_model).to(device)
            h.load_state_dict(models[run]["state_dict"])
            h.eval()
            out[f"{run}_full_history_to_score"] = _bench(
                lambda: h(backbone.extract(x, st)))
            out[f"{run}_hidden_to_score"] = _bench(lambda: h(hidden))
        # G1 生成式参照（canonical：L=90/H=10/N=20/T=1.0/top_p=0.9）
        from model import KronosPredictor

        pred = KronosPredictor(
            model=backbone.kronos, tokenizer=backbone.tokenizer, device=device)
        # 原始（未归一化）窗口：从 qlib 重取该日 90 日窗（上界封存线内）
        x_cal_end = pd.Timestamp(d.date)
        full_cal = provider.trading_days(end=FORWARD_CUTOFF)
        c = int(full_cal.searchsorted(x_cal_end))
        fetch_start = str(full_cal[c - 89].date())
        fetch_end = str(full_cal[c].date())
        codes = d.codes[:n]
        orig = (provider._start_date, provider._end_date, provider.instruments_)
        try:
            provider._start_date = fetch_start
            provider._end_date = fetch_end
            provider.instruments_ = codes
            raw = provider.fetch(["$open", "$high", "$low", "$close",
                                  "$volume", "$amount"], freq="day")
        finally:
            provider._start_date, provider._end_date, provider.instruments_ = orig
        df_list = []
        for code in codes:
            sub = raw.xs(code, level="instrument").sort_index()
            df_list.append(sub)
        # predict_batch 要求价格/量列无 NaN：剔除缺数股票（计时样本量相应披露）
        keep_idx = [i for i, df in enumerate(df_list)
                    if not df[["open", "high", "low", "close",
                               "volume", "amount"]].isnull().values.any()]
        df_list = [df_list[i] for i in keep_idx]
        # 时间戳须为 datetime 序列（calc_time_stamps 用 .dt 访问器）
        x_dates = pd.Series(pd.DatetimeIndex(full_cal[c - 89: c + 1]))
        y_dates = pd.Series(pd.DatetimeIndex(full_cal[c + 1: c + 11]))
        x_stamp = [x_dates] * len(df_list)
        y_stamp = [y_dates] * len(df_list)
        out["g1_generative_ref"] = _bench(
            lambda: pred.predict_batch(
                df_list, x_stamp, y_stamp, pred_len=10, T=1.0, top_p=0.9,
                sample_count=20, verbose=False),
            warmup=1, reps=3)
        out["g1_generative_ref"]["n_samples_kept"] = len(df_list)
        out["g1_generative_ref"]["params"] = {
            "pred_len": 10, "sample_count": 20, "T": 1.0, "top_p": 0.9,
            "protocol": "1 warmup + 3 reps（成本缩减，非 10+30）",
            "includes": "predictor 端到端（含内部预处理/归一化/解码）",
        }
    return out


def engine_attachment(frames: dict[str, pd.DataFrame],
                      windows: dict[str, list] | None = None, *,
                      fetcher=None,
                      window_bounds: dict[str, tuple[str, str]] | None = None,
                      with_g1_mean: bool = True) -> dict:
    """引擎附表——**纠偏版（20260908）**：直连 ``paper_replication.engine_v2``
    真实接口（经 :func:`mh1_multihorizon.replay_corrected.replay_window` 委托），
    六新头 + G1_mean 同网格参照 × 两窗，主期限 10 日分数进引擎。

    历史勘误：2026-09-07 版本经 ``baseline_suite.pipeline.run_group`` 实际
    走**旧引擎**（``paper_replication.engine``），其 JSON 中的 ``delay=1``
    等配置元数据与实际调用无关；该产物已标注为旧引擎口径（见
    ``docs/MH1纠偏与收益成本诊断结果_20260908.md``）。本函数不再引用
    ``baseline_suite.pipeline``，配置元数据由 ``dataclasses.asdict`` 于实际
    调用点生成。不据此改 IC 判据。

    :param frames: ``{window: 长格式信号}``。
    :param windows: 兼容旧签名的评估日容器（引擎不再消费，仅保留占位）。
    :param fetcher: 测试注入行情源（生产 None → qlib 冻结宇宙 + 指数基准）。
    :param window_bounds: 旧签名兼容（重放窗口由 replay_corrected 冻结值决定）。
    """
    from mh1_multihorizon.replay_corrected import replay_window as _replay

    frames_wide = {w: frames[w] for w in frames}
    results: dict = {run: {} for run in _RUNS}
    engine_meta: dict = {}
    for wname in frames_wide:
        meta_w, per_run = _replay(wname, frames_wide, fetcher=fetcher,
                                  with_g1_mean=with_g1_mean)
        engine_meta[wname] = meta_w
        for name, payload in per_run.items():
            if name == "G1_mean":
                engine_meta[wname]["g1_mean_same_grid"] = True
                continue
            ev = payload["eval"]
            results[name][wname] = {
                "perf_idx": ev["excess_idx"]["net"],
                "perf_ew": ev["excess_ew"]["net"],
                "perf_idx_gross": ev["excess_idx"]["gross"],
                "perf_ew_gross": ev["excess_ew"]["gross"],
                "self_gross": ev["self"]["gross"], "self_net": ev["self"]["net"],
                "rule_idx": ev["excess_idx"]["rule"],
                "rule_ew": ev["excess_ew"]["rule"],
                "sum_cost": ev["sum_cost"],
                "nav_drag_end": ev["nav_drag_end"],
            }
        # G1_mean 参照行并入 results（键不与六臂冲突）
        if "G1_mean" in per_run:
            ev = per_run["G1_mean"]["eval"]
            results.setdefault("G1_mean", {})[wname] = {
                "perf_idx": ev["excess_idx"]["net"],
                "perf_ew": ev["excess_ew"]["net"],
                "perf_idx_gross": ev["excess_idx"]["gross"],
                "perf_ew_gross": ev["excess_ew"]["gross"],
                "rule_idx": ev["excess_idx"]["rule"],
                "rule_ew": ev["excess_ew"]["rule"],
                "note": "G1_mean 限定同调仓网格（同节奏交易）",
            }
    return {"meta": engine_meta, "results": results}


__all__ += [
    "WINDOW_BOUNDS", "load_models", "build_windows",
    "score_and_write_signals", "compute_daily_paired_ic", "judge",
    "reference_ic", "timing_benchmark", "engine_attachment",
]
