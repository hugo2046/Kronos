"""五臂信号装配 + 评估（计划 §3）。

两段执行，**纪律**：
    - ``run_train_eval``：训练/早停段评估，可反复跑（调参只在此段内）。
    - ``run_final_validation``：最终验证段（2024-07-01 之后 50 期），**只在 B1/B2/B3
      全部定型后跑一次，封盘**。无论结果好坏如实报告。

五臂共用同一份 :mod:`cross_section.evaluate`（import，不复制粘贴）；信号长表统一为
``[date, code, <factor_col>, fwd_ret_10d]``，直接喂 ``evaluate_factor``。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy import stats

from cross_section.common import ExperimentConfig
from cross_section.evaluate import evaluate_factor
from cross_section_kda import KronosFrozenBackbone
from cross_section_kda.data import build_daily_samples, build_final_grid
from cross_section_kda.train import (
    B0_SIGNALS_PATH,
    DATA_DIR,
    TrainConfig,
    build_split_cache,
    save_checkpoint,
    train_arm,
)


# ============================================================
# 信号长表装配
# ============================================================


def _model_signals_long(
    model, provider, dates: list[pd.Timestamp], pool: str, device: str,
    factor_col: str,
) -> pd.DataFrame:
    """逐日构样本 → 模型打分 → 长表 [date, code, <factor_col>, fwd_ret_10d]。

    fwd_ret_10d 取未截面化的原始前向收益（与 B0 signals parquet 同口径）。
    """
    model.eval()
    rows: list[dict] = []
    for d in dates:
        b = build_daily_samples(provider, date=d.strftime("%Y-%m-%d"), pool=pool)
        if b is None:
            continue
        with torch.no_grad():
            score = model(b.x_norm.to(device), b.stamp.to(device)).cpu().numpy()
        for code, s, fr in zip(b.codes, score, b.fwd_ret_raw):
            rows.append({
                "date": d, "code": code,
                factor_col: float(s), "fwd_ret_10d": float(fr),
            })
    return pd.DataFrame(rows)


def _b0_signals_long(factor_col: str = "signal_B0") -> pd.DataFrame:
    """从 B0 signals parquet 读取（直接复用，不重算）。"""
    df = pd.read_parquet(B0_SIGNALS_PATH)
    out = df[["date", "code", "signal", "fwd_ret_10d"]].rename(columns={"signal": factor_col})
    return out


# ============================================================
# 训练/早停段评估（可反复跑）
# ============================================================


def run_train_eval(
    cfg: ExperimentConfig,
    *,
    device: str,
    arms: tuple[str, ...] = ("B1", "B2", "B3"),
) -> dict:
    """训练 B1/B2/B3 并在早停段评估 RankIC/ICIR/分组（调参用）。

    **不触碰最终验证段。**
    """
    from kronos_qlib import QlibProvider
    from model import Kronos, KronosTokenizer

    p = QlibProvider(cfg.pool, cfg.backtest_start, cfg.data_end)

    # 加载 Kronos 主干（B2/B3 用；B1 不需要但加载成本低）
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name).to(device)
    kronos = Kronos.from_pretrained(cfg.model_name).to(device)
    backbone = KronosFrozenBackbone(tokenizer=tokenizer, kronos=kronos, device=device)

    # 训练 / 早停样本（逐交易日）
    train_batches = build_split_cache(
        p, start=cfg.backtest_start, end="2023-12-15", pool=cfg.pool, rebalance_only=False,
    )
    es_batches = build_split_cache(
        p, start="2024-01-02", end="2024-06-14", pool=cfg.pool, rebalance_only=False,
    )

    results: dict[str, dict] = {}
    for arm in arms:
        logger.info("=" * 70)
        logger.info(f"训练/早停段评估：{arm}")
        logger.info("=" * 70)
        bb = backbone if arm in ("B2", "B3") else None
        model, info = train_arm(arm, bb, train_batches, es_batches,
                                device=device, cfg=TrainConfig(arm=arm))
        save_checkpoint(model, arm, info)

        # 早停段长表评估（同 evaluate.py 口径）
        es_dates = sorted({b.date for b in es_batches})
        sig = _model_signals_long(model, p, es_dates, cfg.pool, device, factor_col=f"signal_{arm}")
        if len(sig) == 0:
            logger.warning(f"[{arm}] 早停段无信号，跳过评估")
            continue
        ic, grp = evaluate_factor(sig, f"signal_{arm}", cfg)
        results[arm] = {
            "info": info,
            "early_stop": {
                "n_periods": ic.n_periods,
                "rankic_mean": ic.rankic_mean,
                "icir": ic.icir,
                "t_stat": ic.t_stat,
                "rankic_positive_ratio": ic.rankic_positive_ratio,
                "long_short_annualized_net": grp.annualized,
                "long_short_max_drawdown_net": grp.max_drawdown,
            },
        }
    out_path = DATA_DIR / "train_eval_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    logger.info(f"训练/早停段评估汇总 → {out_path}")
    return results


# ============================================================
# 最终验证段（只跑一次，封盘）
# ============================================================


def run_final_validation(
    cfg: ExperimentConfig,
    *,
    device: str,
    arms: tuple[str, ...] = ("B1", "B2", "B3"),
) -> dict:
    """**最终验证段（2024-07-01 之后 50 期）：五臂总表 + 归因 + 判定。**

    纪律：只在 B1/B2/B3 全部定型后跑一次。跑完即封盘。
    """
    from kronos_qlib import QlibProvider
    from model import Kronos, KronosTokenizer

    p = QlibProvider(cfg.pool, cfg.backtest_start, cfg.data_end)
    final_dates = build_final_grid(B0_SIGNALS_PATH)
    logger.info(f"最终验证段：{len(final_dates)} 期（{final_dates[0].date()}~"
                f"{final_dates[-1].date()}），封盘运行")

    # 加载主干 + 各臂定型 checkpoint
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name).to(device)
    kronos = Kronos.from_pretrained(cfg.model_name).to(device)
    backbone = KronosFrozenBackbone(tokenizer=tokenizer, kronos=kronos, device=device)

    models: dict[str, torch.nn.Module] = {}
    for arm in arms:
        from cross_section_kda import B1SupervisedHead, B2LinearProbe, B3KronosKdaHead
        if arm == "B1":
            m = B1SupervisedHead().to(device)
        elif arm == "B2":
            m = B2LinearProbe(backbone).to(device)
        else:
            m = B3KronosKdaHead(backbone).to(device)
        ckpt = torch.load(DATA_DIR / f"{arm}_best.pt", map_location=device, weights_only=True)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models[arm] = m

    # 五臂信号长表（最终验证段）
    frames = []
    # B0
    b0 = _b0_signals_long("signal_B0")
    b0 = b0[b0["date"] >= pd.Timestamp("2024-07-01")].copy()
    frames.append(("B0", b0, "signal_B0"))
    # B1/B2/B3
    for arm, m in models.items():
        col = f"signal_{arm}"
        sig = _model_signals_long(m, p, final_dates, cfg.pool, device, factor_col=col)
        frames.append((arm, sig, col))

    # 内连接到共同行（保证五臂同样本）
    merged = None
    for arm, df, col in frames:
        sub = df[["date", "code", col, "fwd_ret_10d"]]
        merged = sub if merged is None else merged.merge(
            sub[["date", "code", col]], on=["date", "code"], how="inner"
        )
    logger.info(f"五臂内连接：{len(merged)} 行 × {merged['date'].nunique()} 期")

    # 逐臂评估
    metrics: dict[str, dict] = {}
    for arm, _, col in frames:
        ic, grp = evaluate_factor(merged, col, cfg)
        metrics[arm] = {
            "factor_col": col, "n_periods": ic.n_periods,
            "rankic_mean": ic.rankic_mean, "rankic_std": ic.rankic_std,
            "icir": ic.icir, "t_stat": ic.t_stat, "p_value": ic.p_value,
            "rankic_positive_ratio": ic.rankic_positive_ratio,
            "group_mean_returns": grp.group_mean_returns.round(5).to_dict(),
            "long_short_annualized_net": grp.annualized,
            "long_short_max_drawdown_net": grp.max_drawdown,
        }

    # 论文窗口子段（2024-07~2025-06，次要报告）
    paper_mask = merged["date"] <= pd.Timestamp("2025-06-30")
    paper_metrics: dict[str, dict] = {}
    if paper_mask.any():
        for arm, _, col in frames:
            sub = merged[paper_mask]
            if len(sub) < 20:
                continue
            ic, _ = evaluate_factor(sub, col, cfg)
            paper_metrics[arm] = {
                "n_periods": ic.n_periods, "rankic_mean": ic.rankic_mean,
                "icir": ic.icir, "t_stat": ic.t_stat,
            }

    # 归因判定（计划 §0）
    b0_rankic = metrics["B0"]["rankic_mean"]
    b3_rankic = metrics["B3"]["rankic_mean"]
    b1_rankic = metrics["B1"]["rankic_mean"]
    b2_rankic = metrics["B2"]["rankic_mean"]
    verdict = {
        "改造有效(B3>B0 且 B3>B1)": bool(b3_rankic > b0_rankic and b3_rankic > b1_rankic),
        "KDA头有独立贡献(B3>B2)": bool(b3_rankic > b2_rankic),
        "B0_RankIC": b0_rankic, "B1_RankIC": b1_rankic,
        "B2_RankIC": b2_rankic, "B3_RankIC": b3_rankic,
        "预注册判据_B0_final_RankIC": 0.0185,  # 计划 §3 预注册值
    }

    out = {
        "segment": {"start": str(final_dates[0].date()), "end": str(final_dates[-1].date()),
                     "n_periods": len(final_dates)},
        "metrics": metrics,
        "paper_window_2024-07_2025-06": paper_metrics,
        "verdict": verdict,
    }
    out_path = DATA_DIR / "final_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    logger.info(f"最终验证段结果 → {out_path}")
    return out


__all__ = ["run_train_eval", "run_final_validation"]
