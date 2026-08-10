"""阶段 2→3 编排：信号生成（K/M/R/P）+ 四组同引擎回测 + 预注册判读。

入口：

    - :func:`run_n_sensitivity` —— N 敏感性试点（计划 §3，先做）：
      1 个月 × N∈{5, 20}，对比日均 RankIC 与截面相关。
    - :func:`run_full_replication` —— 全区间四组信号 + 同引擎回测。

断点续跑：K 组（Kronos 推理，N=20，约 2.2 小时）逐日落盘到
``data/daily_signals_K.parquet``，重跑自动跳过已存在的日期。

判读（§4，预注册，跑 K 组前冻结）：

    1. 引擎门禁：P 组不得有显著正 AER（AER < +3%）——否则引擎有 bug，停止；
       （负成本拖是预期内的，见 tests/test_paper_replication.py 门禁口径注记）
    2. 锚点比较：K 组 AER/IR 与论文 base 规格（Table 10 CSI300：AER 0.1911 /
       IR 1.3782）同量级（AER 差 <10pp 且同号）→ 复现成功；
    3. 定性判定：K 组 AER>0 且 IR>0.5 且 K 组优于 M、R → 论文方向可复现；
    4. 与既有 RankIC（论文窗口 B0 +0.0389）互为印证。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from paper_replication.common import DATA_DIR, ReplicationConfig, ensure_data_dir
from paper_replication.engine import (
    EngineConfig,
    PerfStats,
    attach_benchmark,
    compute_perf,
    run_portfolio,
)

# 论文 Table 10 CSI300 base 规格锚点（阶段 0 提取，2026-08-10）
PAPER_BASE_AER = 0.1911
PAPER_BASE_IR = 1.3782


def _engine_cfg(cfg: ReplicationConfig) -> EngineConfig:
    """从复现配置构造引擎参数。"""
    return EngineConfig(
        top_k=cfg.top_k, drop_n=cfg.drop_n, min_hold=cfg.min_hold, cost_bps=cfg.cost_bps
    )


def _align_columns(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    """对齐多张宽表的列（取并集，按列名排序），保证四组同池可比。"""
    cols = sorted(set().union(*[set(f.columns) for f in frames]))
    return [f.reindex(columns=cols) for f in frames]


def run_group(
    signal_wide: pd.DataFrame,
    px_wide: pd.DataFrame,
    tradeable: pd.DataFrame,
    bench_idx: pd.Series,
    bench_ew: pd.Series,
    *,
    cfg: ReplicationConfig,
    name: str,
) -> tuple[PerfStats, PerfStats, pd.Series, pd.Series]:
    """对单组信号跑引擎 + 算双基准绩效。

    双基准（见 benchmark.py）：
        - ``bench_idx`` = csi300 指数（论文口径，AER 锚点比较用）；
        - ``bench_ew`` = 同池等权（引擎门禁用，剥离等权-beta 溢价）。

    :returns: ``(perf_idx, perf_ew, daily_ret, excess_idx)``：
        ``perf_idx`` = 相对指数基准的 AER/IR/MDD/换手；
        ``perf_ew`` = 相对同池等权基准的 AER/IR/MDD/换手（门禁判据）；
        ``daily_ret`` = 组合逐日净收益；``excess_idx`` = 相对指数的超额日收益。
    """
    from paper_replication.benchmark import build_pool_equal_weight_benchmark

    ec = _engine_cfg(cfg)
    sig = signal_wide.reindex(index=px_wide.index, columns=px_wide.columns)
    daily_ret, _, trades = run_portfolio(sig, px_wide, tradeable, cfg=ec)

    excess_idx = attach_benchmark(daily_ret, bench_idx)
    excess_ew = attach_benchmark(daily_ret, bench_ew)
    perf_idx = compute_perf(excess_idx, trades, name=name)
    perf_ew = compute_perf(excess_ew, trades, name=name)
    logger.info(
        f"[{name}] AER(指数)={perf_idx.aer:+.2%} IR={perf_idx.ir:+.3f} | "
        f"AER(等权)={perf_ew.aer:+.2%} IR={perf_ew.ir:+.3f} | "
        f"MDD={perf_idx.max_drawdown:.2%} 日均换手={perf_idx.daily_turnover:.2%} (n={perf_idx.n_days})"
    )
    return perf_idx, perf_ew, daily_ret, excess_idx


def compute_signal_rankic(
    signal_wide: pd.DataFrame, fwd_ret_wide: pd.DataFrame
) -> tuple[float, float]:
    """逐日截面 RankIC（信号 vs 次日收益），返回 (mean, daily_corr_mean)。

    用于 N 敏感性试点（§3）：对比 N=5 vs N=20 的截面区分力。
    """
    common = signal_wide.index.intersection(fwd_ret_wide.index)
    rhos = []
    for d in common:
        s = signal_wide.loc[d]
        r = fwd_ret_wide.loc[d]
        mask = s.notna() & r.notna()
        if mask.sum() < 5:
            continue
        rho, _ = stats.spearmanr(s[mask], r[mask])
        if pd.notna(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), float("nan")
    arr = np.array(rhos)
    return float(arr.mean()), float(arr.std(ddof=1))


def judge(
    perf_k_idx: PerfStats,
    perf_k_ew: PerfStats,
    perf_m_idx: PerfStats,
    perf_r_idx: PerfStats,
    perf_p_ew: PerfStats,
    *,
    beta_gap: float,
) -> dict:
    """预注册判读（§4，按顺序）。

    双基准门禁（见 benchmark.py / pipeline.run_group）：
        - 规则 1（引擎门禁）用**同池等权**基准：P 组 AER(等权) < +3% → 引擎无 bug。
          指数基准的 P 组 AER 含 +4% 等权-beta 溢价，不能直接判门禁。
        - 规则 2/3（锚点 / 定性）用**指数**基准：与论文口径一致。

    :param beta_gap: 等权-beta 溢价（指数基准 AER − 等权基准 AER 的结构性差），并列报告。
    :returns: 判读结论字典。
    """
    out: dict = {"beta_gap": beta_gap}

    # 规则 1：引擎门禁（P 组相对**同池等权**基准不得有显著正 AER）
    gate_ok = perf_p_ew.aer < 0.03
    out["engine_gate"] = {
        "p_aer_ew": perf_p_ew.aer,
        "p_aer_ew_threshold": 0.03,
        "passed": gate_ok,
        "note": (
            f"P 组 AER(等权)={perf_p_ew.aer:+.2%} < +3% → 引擎无 bug（等权基准剥离 beta），继续"
            if gate_ok
            else f"P 组 AER(等权)={perf_p_ew.aer:+.2%} ≥ +3% → 引擎制造选股 alpha，停止"
        ),
    }

    # 规则 2：锚点比较（指数基准，base 规格锚点 = Table 10 CSI300）
    # 注意：指数基准 AER 含 +beta_gap 的结构性等权溢价，判"同量级"时把它作为基线偏移
    k_effective = perf_k_idx.aer  # 指数基准口径，与论文锚点同口径
    same_sign = np.sign(k_effective) == np.sign(PAPER_BASE_AER)
    within_band = abs(k_effective - PAPER_BASE_AER) < 0.10
    out["anchor_compare"] = {
        "paper_base_aer": PAPER_BASE_AER,
        "paper_base_ir": PAPER_BASE_IR,
        "k_aer_index": perf_k_idx.aer,
        "k_aer_eqw": perf_k_ew.aer,
        "k_ir_index": perf_k_idx.ir,
        "beta_gap": beta_gap,
        "replicated": bool(same_sign and within_band),
        "note": (
            f"K 组 AER(指数)={perf_k_idx.aer:+.2%}（其中 +{beta_gap:.2%} 为等权 beta 溢价，"
            f"选股 alpha = AER(等权)={perf_k_ew.aer:+.2%}）与论文 base {PAPER_BASE_AER:+.2%} "
            f"同号且差<10pp → 复现成功"
            if (same_sign and within_band)
            else f"K 组 AER(指数)={perf_k_idx.aer:+.2%}（选股 alpha={perf_k_ew.aer:+.2%}）"
            f"未达锚点同量级（差 {perf_k_idx.aer - PAPER_BASE_AER:+.2%}）"
        ),
    }

    # 规则 3：定性判定（指数基准 K>0 & IR>0.5 & K 优于 M、R）
    beats_baselines = (perf_k_idx.aer > perf_m_idx.aer) and (perf_k_idx.aer > perf_r_idx.aer)
    qualitative_pass = (perf_k_idx.aer > 0) and (perf_k_idx.ir > 0.5) and beats_baselines
    out["qualitative"] = {
        "k_positive": perf_k_idx.aer > 0,
        "k_ir_gt_05": perf_k_idx.ir > 0.5,
        "k_beats_M_R": beats_baselines,
        "passed": qualitative_pass,
        "note": (
            "K 组 AER(指数)>0 且 IR>0.5 且优于 M、R → 论文方向可复现（base 规格）"
            if qualitative_pass
            else "K 组不敌免费基线或 IR/方向不达标 → 论文结果在本数据上不可复现"
        ),
    }

    # 总判定
    if not gate_ok:
        out["verdict"] = "引擎门禁未通过，停止"
    elif out["anchor_compare"]["replicated"]:
        out["verdict"] = "复现成功（达论文 base 锚点同量级）"
    elif qualitative_pass:
        out["verdict"] = "论文方向可复现（base 规格），但未达锚点同量级"
    else:
        out["verdict"] = "论文结果在本数据/口径上不可复现"
    return out
