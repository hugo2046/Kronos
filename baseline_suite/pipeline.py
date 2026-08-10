"""四变体 + KDA 三臂 + 对照的回测编排与判读（计划 §2-4）。

复用 ``paper_replication.engine``（top-k/drop-n long-only）与
``paper_replication.benchmark``（双基准）。本模块只做：

- :func:`run_variant_group` —— 单变体信号过引擎，返回双基准绩效；
- :func:`run_kda_arms` —— 载入 KDA 三臂 checkpoint 出 long-only 信号宽表；
- :func:`judge_oos` —— 样本外预注册判定（计划 §4，跑前冻结）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import VARIANTS, BaselineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
from paper_replication.benchmark import (
    build_pool_equal_weight_benchmark,
    probe_index_benchmark,
)
from paper_replication.engine import (
    EngineConfig,
    PerfStats,
    attach_benchmark,
    compute_perf,
    run_portfolio,
)


def _engine_cfg(cfg: BaselineConfig) -> EngineConfig:
    """从配置构造引擎参数。"""
    return EngineConfig(
        top_k=cfg.top_k, drop_n=cfg.drop_n, min_hold=cfg.min_hold, cost_bps=cfg.cost_bps
    )


def build_dual_benchmarks(
    provider, cfg: BaselineConfig, px_wide: pd.DataFrame, tradeable: pd.DataFrame
) -> tuple[pd.Series, pd.Series, float]:
    """构造双基准 + beta_gap（指数 / 同池等权 / 二者累计差）。

    :returns: ``(bench_idx, bench_ew, beta_gap)``。
    """
    bench_idx = probe_index_benchmark(provider, cfg.backtest_start, cfg.backtest_end)
    bench_ew = build_pool_equal_weight_benchmark(px_wide, tradeable)
    common = bench_idx.index.intersection(bench_ew.index)
    beta_gap = float(
        (1 + bench_idx.loc[common]).prod() - (1 + bench_ew.loc[common]).prod()
    )
    logger.info(
        f"双基准 [{cfg.window}]：指数累计 {(1+bench_idx.loc[common]).prod()-1:+.2%} / "
        f"等权累计 {(1+bench_ew.loc[common]).prod()-1:+.2%} / beta_gap={beta_gap:+.2%}"
    )
    return bench_idx, bench_ew, beta_gap


def run_group(
    signal_wide: pd.DataFrame,
    px_wide: pd.DataFrame,
    tradeable: pd.DataFrame,
    bench_idx: pd.Series,
    bench_ew: pd.Series,
    *,
    cfg: BaselineConfig,
    name: str,
) -> tuple[PerfStats, PerfStats, pd.Series, pd.Series, pd.Series]:
    """单组信号跑引擎 + 双基准绩效（与 paper_replication.pipeline.run_group 同构）。

    :returns: ``(perf_idx, perf_ew, daily_ret, excess_idx, excess_ew)``。
        新增返回 ``excess_ew`` 供门禁 / 样本外判读（§4 主判据用等权基准）。
    """
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
        f"MDD={perf_idx.max_drawdown:.2%} 日均换手={perf_idx.daily_turnover:.2%} "
        f"(n={perf_idx.n_days})"
    )
    return perf_idx, perf_ew, daily_ret, excess_idx, excess_ew


def run_kda_arms(
    provider, cfg: BaselineConfig, rebalances: pd.DatetimeIndex, *, device: str
) -> dict[str, pd.DataFrame]:
    """载入 KDA 三臂冻结 checkpoint，逐日出 long-only 截面得分宽表（计划 §3）。

    输入口径与训练一致（窗口 z-score + clip5；B2/B3 经冻结 Kronos 主干取隐状态）。
    三臂 forward 返回 score[B]——本函数把 score 作为截面排序信号（越大越买），
    直接送进 long-only 引擎（不改引擎）。

    :returns: ``{"B1": wide, "B2": wide, "B3": wide}``，与四变体同形状。
    """
    import torch

    from cross_section_kda import (
        B1SupervisedHead,
        B2LinearProbe,
        B3KronosKdaHead,
        KronosFrozenBackbone,
    )
    from cross_section_kda.data import build_daily_samples

    tokenizer = None
    kronos = None
    backbone = None
    models: dict[str, torch.nn.Module] = {}

    # B1 不需要主干；B2/B3 共用一个冻结主干
    b23_needed = True  # 两个都要，加载一次

    # 惰性加载主干（仅 B2/B3 需要）
    from model import Kronos, KronosTokenizer

    kda_data_dir = REPO_ROOT / "cross_section_kda" / "data"
    arms = ("B1", "B2", "B3")
    for arm in arms:
        ckpt_path = kda_data_dir / f"{arm}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"KDA checkpoint 缺失：{ckpt_path}")
        if arm in ("B2", "B3") and backbone is None and b23_needed:
            tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name).to(device)
            kronos = Kronos.from_pretrained(cfg.model_name).to(device)
            backbone = KronosFrozenBackbone(tokenizer=tokenizer, kronos=kronos, device=device)
        if arm == "B1":
            m = B1SupervisedHead().to(device)
        elif arm == "B2":
            m = B2LinearProbe(backbone).to(device)
        else:
            m = B3KronosKdaHead(backbone).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models[arm] = m
        logger.info(f"载入 {arm} checkpoint：{ckpt_path.name}")

    rows: dict[str, list[dict]] = {arm: [] for arm in arms}
    n_done = 0
    for i, d in enumerate(rebalances):
        ds = d.strftime("%Y-%m-%d")
        b = build_daily_samples(provider, date=ds, pool=cfg.pool)
        if b is None:
            logger.warning(f"{ds}: KDA 无可用样本")
            for arm in arms:
                rows[arm].append({})
            continue
        with torch.no_grad():
            for arm, m in models.items():
                score = m(b.x_norm.to(device), b.stamp.to(device)).cpu().numpy()
                rows[arm].append({c: float(s) for c, s in zip(b.codes, score)})
        n_done += 1
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"KDA 三臂 [{i + 1}/{len(rebalances)}] {ds}: {len(b.codes)} 只")

    wide = {arm: pd.DataFrame(rows[arm], index=rebalances) for arm in arms}
    for arm in arms:
        logger.info(
            f"KDA {arm} 信号 [{cfg.window}]：{wide[arm].shape[0]} 日 × "
            f"平均 {wide[arm].notna().sum(axis=1).mean():.0f} 只/日"
        )
    return wide


def judge_oos(
    mean_perf_ew: PerfStats,
    mean_perf_idx: PerfStats,
    min_perf_ew: PerfStats,
    placeholder_perf_ew: PerfStats,
    *,
    paper_min_perf_ew: PerfStats | None = None,
    paper_mean_perf_ew: PerfStats | None = None,
) -> dict:
    """样本外预注册判定（计划 §4，跑前冻结）。

    四条判据（顺序固定）：
        1. 主判据：mean 样本外 AER(等权) > 0 **且** AER(指数) > 0 → "正超额在样本外成立"；
        2. IR(指数) ≥ 0.3 → "强度未塌方"；< 0.3 但 AER>0 → "方向存活、强度衰减"；
        3. mean 样本外 AER(等权) ≤ 0 → "论文窗口正结果不能外推，疑似窗口运气"；
        4. min 变体：若论文窗 + 样本外两段都显著强于 mean（AER(等权) 差 > 3pp）→
           "min 含风险信息"候选发现，本轮不改 canonical。

    :param paper_min_perf_ew / paper_mean_perf_ew: 论文窗口的 min/mean 等权绩效，
        用于判据 4 的"两段都强"比较（None 时跳过该判据）。
    """
    out: dict = {}

    # 引擎门禁：P 组 AER(等权) < +3%（新窗口顺带复验）
    gate_ok = placeholder_perf_ew.aer < 0.03
    out["engine_gate"] = {
        "p_aer_ew": placeholder_perf_ew.aer,
        "threshold": 0.03,
        "passed": bool(gate_ok),
        "note": (
            f"P 组 AER(等权)={placeholder_perf_ew.aer:+.2%} < +3% → 引擎无 bug"
            if gate_ok
            else f"P 组 AER(等权)={placeholder_perf_ew.aer:+.2%} ≥ +3% → 引擎制造 alpha，停止"
        ),
    }

    # 判据 1 + 3：mean 正超额是否外推
    mean_aer_ew_pos = mean_perf_ew.aer > 0
    mean_aer_idx_pos = mean_perf_idx.aer > 0
    main_holds = mean_aer_ew_pos and mean_aer_idx_pos
    out["criterion_main"] = {
        "mean_aer_ew": mean_perf_ew.aer,
        "mean_aer_idx": mean_perf_idx.aer,
        "both_positive": bool(main_holds),
        "note": (
            f"mean AER(等权)={mean_perf_ew.aer:+.2%} > 0 且 AER(指数)={mean_perf_idx.aer:+.2%} > 0 → "
            f"正超额在样本外成立"
            if main_holds
            else f"mean AER(等权)={mean_perf_ew.aer:+.2%} ≤ 0 或 AER(指数)={mean_perf_idx.aer:+.2%} ≤ 0 → "
            f"论文窗口正结果不能外推，疑似窗口运气（不做参数搜索）"
        ),
    }

    # 判据 2：IR(指数) 强度
    ir_ok = mean_perf_idx.ir >= 0.3
    strength = "强度未塌方" if ir_ok else "方向存活、强度衰减" if mean_aer_idx_pos else "强度塌方"
    out["criterion_strength"] = {
        "mean_ir_idx": mean_perf_idx.ir,
        "threshold": 0.3,
        "note": f"IR(指数)={mean_perf_idx.ir:+.3f} → {strength}",
    }

    # 判据 4：min 两段都显著强于 mean
    out["criterion_min"] = {"note": "min vs mean 两段对比未做（缺论文窗口绩效）"}
    if paper_min_perf_ew is not None and paper_mean_perf_ew is not None:
        paper_min_minus_mean = paper_min_perf_ew.aer - paper_mean_perf_ew.aer
        oos_min_minus_mean = min_perf_ew.aer - mean_perf_ew.aer
        both_strong = (
            paper_min_minus_mean > 0.03 and oos_min_minus_mean > 0.03
        )
        out["criterion_min"] = {
            "paper_min_minus_mean": paper_min_minus_mean,
            "oos_min_minus_mean": oos_min_minus_mean,
            "both_strong": bool(both_strong),
            "note": (
                f"min 两段都强于 mean（论文窗 {paper_min_minus_mean:+.2%} / 样本外 "
                f"{oos_min_minus_mean:+.2%} 均 > 3pp）→ 候选发现，本轮不改 canonical"
                if both_strong
                else f"min 未两段都强于 mean（论文窗 {paper_min_minus_mean:+.2%} / "
                f"样本外 {oos_min_minus_mean:+.2%}）→ 不构成候选发现"
            ),
        }

    # 总判定
    if not gate_ok:
        verdict = "引擎门禁未通过，停止"
    elif main_holds:
        verdict = "正超额在样本外成立" + ("，强度未塌方" if ir_ok else "，方向存活强度衰减")
    else:
        verdict = "论文窗口正结果不能外推，疑似窗口运气（封盘，不做参数搜索）"
    out["verdict"] = verdict
    return out
