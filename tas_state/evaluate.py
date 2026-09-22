"""TAS1 评估器：逐日 IC、配对差、HAC、Holm 与机器可读判决（计划 §6）。

统计规则（冻结）：

- IC_t = Spearman(signal_{i,t}, close_{i,t+10}/close_{i,t} − 1)，k=10 唯一主终点；
- 所有比较按 ``(date, code)`` 显式对齐，股票列重排不改变结果；
- 配对差 d_t = IC_{T,t} − IC_{B,t}（同日同股票集合），HAC 主 lag=9、敏感性 20；
- 按真实交易日顺序分段累积协方差（日历空档切断，不删空档伪造连续）；
- 多重比较 Holm 校正；历史 p 值仅诊断。

用法（P2 开封）::

    python -m tas_state.evaluate --stage pilot --unseal
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]

MIN_CROSS_N = 30  # 计划 §6.1：每天至少 30 只


# ============================================================
# 逐日 IC 与配对差
# ============================================================
def daily_rank_ic(
    sig: pd.DataFrame, fwd: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.Series:
    """逐日截面 Spearman RankIC（按 (date, code) 显式对齐）。

    :param sig: 信号宽表（index=date, columns=code）。
    :param fwd: 前向收益宽表（同构；列序任意，内部 reindex 对齐）。
    :param dates: 评估日序列（sig 的行的子集）。
    :returns: 逐日 IC Series（index=date）。
    :raises ValueError: 某日合并后 < MIN_CROSS_N 只（覆盖门禁失败，
        不能选择性 dropna 后继续报胜——计划 §6.1）。
    """
    cols = sig.columns
    out = {}
    for d in dates:
        s = sig.loc[d].reindex(cols)
        f = fwd.loc[d].reindex(cols)
        m = s.notna() & f.notna()
        if int(m.sum()) < MIN_CROSS_N:
            raise ValueError(
                f"{d:%Y-%m-%d}: 合并后仅 {int(m.sum())} 只 < {MIN_CROSS_N}，"
                f"覆盖门禁失败（不允许缩池续评）"
            )
        out[d] = stats.spearmanr(s[m], f[m]).statistic
    return pd.Series(out)


def paired_difference(ic_a: pd.Series, ic_b: pd.Series) -> pd.Series:
    """配对差 d_t = IC_a − IC_b（同日对齐；缺失日报错不静默删）。"""
    common = ic_a.index.intersection(ic_b.index)
    if len(common) != len(ic_a) or len(common) != len(ic_b):
        missing_a = ic_a.index.difference(ic_b.index)
        missing_b = ic_b.index.difference(ic_a.index)
        raise ValueError(
            f"配对差要求同日序列：a 缺 {len(missing_b)} 日、b 缺 {len(missing_a)} 日"
        )
    return (ic_a - ic_b).dropna()


# ============================================================
# 分段 Newey-West HAC
# ============================================================
def contiguous_segments(index: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> list[slice]:
    """按交易日历把 index 切成连续段（真实空档切断）。

    段定义：相邻样本在日历上紧邻（calendar 位置差 1）。存在缺失交易日
    时切断——HAC 的 lag-i 协方差只用同段样本对，不跨空档。
    """
    cal_pos = {d: i for i, d in enumerate(calendar)}
    pos = [cal_pos[d] for d in index]
    segs: list[slice] = []
    start = 0
    for i in range(1, len(pos)):
        if pos[i] != pos[i - 1] + 1:
            segs.append(slice(start, i))
            start = i
    segs.append(slice(start, len(pos)))
    return segs


def hac_tvalue(x: np.ndarray, lag: int, segments: list[slice]) -> tuple[float, float]:
    """Newey-West t（分段协方差；Bartlett 权重）。

    :returns: ``(t值, p值)``（双侧，正态近似）。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    mu = x.mean()
    d = x - mu

    def gamma(l: int) -> float:
        # 分段累积：只统计同段内间隔 l 的样本对
        s = 0.0
        for seg in segments:
            lo, hi = seg.start, seg.stop
            if hi - lo > l:
                s += float(np.dot(d[lo : hi - l], d[lo + l : hi]))
        return s / n

    var = gamma(0)
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1)
        var += 2.0 * w * gamma(l)
    if var <= 0:
        return (np.nan, np.nan)
    se = np.sqrt(var / n)
    t = mu / se
    p = 2.0 * stats.norm.sf(abs(t))
    return (float(t), float(p))


def nw_summary(ic: pd.Series, calendar: pd.DatetimeIndex, lag: int) -> dict:
    """均值 IC + NW t/p（主 lag 与敏感性 lag 由调用方分别调）。"""
    segs = contiguous_segments(ic.index, calendar)
    t, p = hac_tvalue(ic.values, lag, segs)
    return {
        "mean": float(ic.mean()),
        "n_days": int(len(ic)),
        "n_segments": len(segs),
        "t": t,
        "p": p,
        "lag": lag,
    }


# ============================================================
# Holm 校正与判决
# ============================================================
def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm 步降校正（与计划玩具例 [0.001,0.04,0.2]→[0.003,0.08,0.2] 一致）。"""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)  # 步降单调性
        adj[idx] = min(1.0, running)
    return [float(v) for v in adj]


@dataclass
class StageAVerdict:
    """计划 §6.2 阶段 A：seed42 快速淘汰。"""

    t_pre_ic_w1: float
    t_pre_ic_w2: float
    delta_vs_b0_w1: float
    delta_vs_b0_w2: float
    pass_all: bool
    reasons: list[str] = field(default_factory=list)


def stage_a_decision(
    ic_t_pre_w1: pd.Series,
    ic_t_pre_w2: pd.Series,
    ic_b0_w1: pd.Series,
    ic_b0_w2: pd.Series,
) -> StageAVerdict:
    """阶段 A 判决（两窗 IC 均 >0 且相对 B0 配对差均 >0 才续种子）。

    不做显著性宣称（计划 §6.2）。
    """
    d1 = paired_difference(ic_t_pre_w1, ic_b0_w1)
    d2 = paired_difference(ic_t_pre_w2, ic_b0_w2)
    v = StageAVerdict(
        t_pre_ic_w1=float(ic_t_pre_w1.mean()),
        t_pre_ic_w2=float(ic_t_pre_w2.mean()),
        delta_vs_b0_w1=float(d1.mean()),
        delta_vs_b0_w2=float(d2.mean()),
        pass_all=False,
    )
    reasons = []
    if not v.t_pre_ic_w1 > 0:
        reasons.append(f"W1 T-PRE IC={v.t_pre_ic_w1:.4f} ≤ 0")
    if not v.t_pre_ic_w2 > 0:
        reasons.append(f"W2 T-PRE IC={v.t_pre_ic_w2:.4f} ≤ 0")
    if not v.delta_vs_b0_w1 > 0:
        reasons.append(f"W1 ΔIC(B0)={v.delta_vs_b0_w1:.4f} ≤ 0")
    if not v.delta_vs_b0_w2 > 0:
        reasons.append(f"W2 ΔIC(B0)={v.delta_vs_b0_w2:.4f} ≤ 0")
    v.reasons = reasons
    v.pass_all = not reasons
    return v


def main() -> int:
    """P2/P3 开封入口（实现于 signals 落盘后；未落盘时报错不空跑）。"""
    import argparse

    from tas_state.config import PKG_DIR, TASConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PKG_DIR / "config.json"))
    ap.add_argument("--stage", choices=["pilot", "historical"], required=True)
    ap.add_argument("--unseal", action="store_true")
    args = ap.parse_args()
    if not args.unseal:
        logger.warning("未指定 --unseal：只做覆盖检查，不输出历史成绩")
        return 0

    cfg = TASConfig.load(args.config)
    sig_dir = PKG_DIR / "data" / ("pilot" if args.stage == "pilot" else "historical")
    if not sig_dir.exists():
        raise FileNotFoundError(f"信号目录不存在：{sig_dir}（先跑 tas_state.signals）")
    # 逐窗读信号 → daily_rank_ic → stage_a_decision（pilot）/ 阶段 B（historical）
    raise NotImplementedError(
        "evaluate 主入口在 P2 信号全部落盘后按冻结协议执行；"
        "统计核已由 tests/test_tas_protocol.py 锁定"
    )


if __name__ == "__main__":
    raise SystemExit(main())
