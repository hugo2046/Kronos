"""阶段 2 入口：四组信号生成（K/M/R/P）+ N 敏感性试点。

用法（解释器 /home/user/miniconda3/envs/quant/bin/python）::

    # N 敏感性试点（先做，约 10 分钟）
    python -m paper_replication.run_signals n-sensitivity

    # 全量 K 组信号（断点续跑，约 2.2 小时）
    python -m paper_replication.run_signals kronos

    # M / R / P 组（快，纯取数 + 随机）
    python -m paper_replication.run_signals baselines

K 组落盘 ``data/daily_signals_K.parquet``（date × code signal 宽表，不入库）；
M/R/P 组落 ``data/daily_signals_M.parquet`` 等。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from paper_replication.common import DATA_DIR, ReplicationConfig, ensure_data_dir
from paper_replication.signal import (
    build_px_tradeable,
    run_kronos_signals,
    run_momentum_reversal,
    run_placeholder,
)


def build_provider(cfg: ReplicationConfig):
    """构造 QlibProvider（覆盖窗口 + 回看缓冲）。"""
    from kronos_qlib import QlibProvider

    fetch_start = (
        pd.Timestamp(cfg.backtest_start) - pd.Timedelta(days=cfg.lookback * 2)
    ).strftime("%Y-%m-%d")
    return QlibProvider(cfg.pool, fetch_start, cfg.backtest_end)


def load_predictor(cfg: ReplicationConfig):
    """加载 KronosPredictor（与 cross_section/stage2_pilot.py 同口径）。"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def build_rebalances(cfg: ReplicationConfig):
    """每日调仓日序列（论文窗口内的每个交易日）。"""
    from kronos_qlib import QlibProvider

    p = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
    cal = p.trading_days(cfg.backtest_start, cfg.backtest_end)
    return cal


def cmd_baselines(cfg: ReplicationConfig) -> None:
    """生成 M / R / P 组信号。"""
    provider = build_provider(cfg)
    rebalances = build_rebalances(cfg)
    logger.info(f"调仓日（每日）：{len(rebalances)} 个，{rebalances[0].date()}~{rebalances[-1].date()}")

    mom_wide, rev_wide = run_momentum_reversal(provider, cfg, rebalances)
    mom_wide.to_parquet(DATA_DIR / "daily_signals_M.parquet")
    rev_wide.to_parquet(DATA_DIR / "daily_signals_R.parquet")
    logger.info(f"M/R 信号落盘 {DATA_DIR / 'daily_signals_M.parquet'}")

    # P 组：列与 M 对齐
    p_wide = run_placeholder(cfg, rebalances, list(mom_wide.columns), seed=cfg.seed)
    p_wide.to_parquet(DATA_DIR / "daily_signals_P.parquet")
    logger.info(f"P 信号落盘 {DATA_DIR / 'daily_signals_P.parquet'}")


def cmd_kronos(cfg: ReplicationConfig, *, sample_count: int | None = None) -> None:
    """生成 K 组信号（断点续跑）。

    :param sample_count: 覆盖配置的 sample_count（N 敏感性试点用）。
    """
    # 断点续跑：若已有部分日期的信号，跳过
    ckpt_path = DATA_DIR / "daily_signals_K.parquet"
    if ckpt_path.exists() and sample_count is None:
        existing = pd.read_parquet(ckpt_path)
        done = set(pd.to_datetime(existing.index))
        logger.info(f"断点续跑：已有 {len(done)} 日信号，跳过")
    else:
        existing = pd.DataFrame()
        done = set()

    rebalances = build_rebalances(cfg)
    pending = [d for d in rebalances if d not in done]
    logger.info(
        f"K 组信号：{len(rebalances)} 日总计，{len(pending)} 日待跑"
        + (f"（N={sample_count}）" if sample_count else f"（N={cfg.sample_count}）")
    )
    if not pending:
        logger.info("K 组信号已全部完成")
        return

    provider = build_provider(cfg)
    predictor = load_predictor(cfg)
    # 临时覆盖 sample_count
    if sample_count is not None:
        from dataclasses import replace

        cfg = replace(cfg, sample_count=sample_count)

    new_wide = run_kronos_signals(
        predictor,
        provider,
        cfg,
        pd.DatetimeIndex(pending),
        checkpoint_path=ckpt_path,
    )
    combined = pd.concat([existing, new_wide]).sort_index()
    combined.to_parquet(ckpt_path)
    logger.info(f"K 组信号落盘 {ckpt_path}（累计 {len(combined)} 日）")


def cmd_n_sensitivity(cfg: ReplicationConfig) -> None:
    """N 敏感性试点（计划 §3）：1 个月 × N∈{5, 20}。

    对比信号截面相关与日均 RankIC——论文声称随 N 单调改善。
    若 N=20 反而更差，先停下排查。
    """
    # 取窗口前 1 个月（约 21 交易日）
    provider = build_provider(cfg)
    rebalances = build_rebalances(cfg)
    pilot = rebalances[:21]
    logger.info(f"N 敏感性试点：{len(pilot)} 日，{pilot[0].date()}~{pilot[-1].date()}")

    # H 日前向收益做 RankIC 标签（与信号公式 mean(pred[t+1..t+H]) 同口径，
    # 也与 cross_section 的 fwd_ret_10d 一致，可与论文窗口 +0.0389 直接对照）。
    # **勘误（2026-08-10）**：首轮误用 1 日次日收益 shift(-1)，与 H=10 信号口径
    # 不匹配，IC 失真。改为 close[t+H]/close[t]-1，挂在 t 行。
    from kronos_qlib import QlibProvider

    fetch_end = (
        pd.Timestamp(cfg.backtest_start) + pd.Timedelta(days=cfg.predict_len * 3)
    ).strftime("%Y-%m-%d")
    p2 = QlibProvider(cfg.pool, cfg.backtest_start, fetch_end)
    px_raw = p2.fetch(["$close"], freq="day")
    if "instrument" in px_raw.index.names:
        px = px_raw["close"].unstack("instrument").sort_index()
    else:
        px = px_raw["close"].sort_index()
    # H 日前向收益：fwd[t] = close[t+H]/close[t]-1，挂在决策日 t
    fwd = px.shift(-cfg.predict_len) / px - 1.0

    results = {}
    for n in (5, 20):
        logger.info(f"=== N={n} ===")
        provider = build_provider(cfg)
        predictor = load_predictor(cfg)
        from dataclasses import replace

        cfg_n = replace(cfg, sample_count=n)
        sig_wide = run_kronos_signals(predictor, provider, cfg_n, pilot, progress_every=5)

        common_days = sig_wide.index.intersection(fwd.index)
        from paper_replication.pipeline import compute_signal_rankic

        rankic_mean, rankic_std = compute_signal_rankic(
            sig_wide.loc[common_days], fwd.loc[common_days]
        )
        results[n] = {
            "rankic_mean": rankic_mean,
            "rankic_std": rankic_std,
            "icir": rankic_mean / rankic_std if rankic_std > 0 else float("nan"),
            "signal_std_mean": float(
                pd.Series(
                    [sig_wide.loc[d].std() for d in common_days]
                ).mean()
            ),
            "n_days_used": len(common_days),
            "label": f"fwd_ret_{cfg.predict_len}d (close[t+H]/close[t]-1)",
        }
        logger.info(
            f"N={n}: RankIC 均值={rankic_mean:+.4f} ICIR={results[n]['icir']:+.3f} "
            f"信号截面 std={results[n]['signal_std_mean']:.4f} (n={len(common_days)})"
        )

    # 落盘
    import json

    out_path = DATA_DIR / "n_sensitivity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"N 敏感性结果落盘 {out_path}")

    # 判读：N=20 应 ≥ N=5（论文单调改善）。但 n≈21 试点样本极小，
    # IC 序列 std≈0.06 → 均值标准误 ≈ 0.013，量级显著大于 N 间差值。
    # 故试点只作"明显反常才停"的粗筛，不作定量结论。
    delta = results[20]["rankic_mean"] - results[5]["rankic_mean"]
    se = results[5]["rankic_std"] / np.sqrt(max(results[5]["n_days_used"], 1))
    logger.info(f"N 间差 ΔRankIC={delta:+.4f}，N=5 均值标准误≈{se:.4f}（试点样本小，仅粗筛）")
    if delta < -2 * se:  # 明显反常（差值 < -2SE）才停
        logger.warning(
            f"⚠️ N=20 RankIC ({results[20]['rankic_mean']:+.4f}) 显著低于 N=5 "
            f"({results[5]['rankic_mean']:+.4f})，超 2 倍标准误，建议先停下排查"
        )
    else:
        logger.info("✓ N=20 vs N=5 差异在试点样本噪声内，与论文单调改善无显著矛盾，继续全量")


def main() -> None:
    parser = argparse.ArgumentParser(description="论文口径复现 · 信号生成")
    parser.add_argument(
        "cmd",
        choices=["kronos", "baselines", "n-sensitivity"],
        help="kronos=K组 | baselines=M/R/P | n-sensitivity=N试点",
    )
    args = parser.parse_args()
    cfg = ReplicationConfig.load()
    ensure_data_dir()
    if args.cmd == "kronos":
        cmd_kronos(cfg)
    elif args.cmd == "baselines":
        cmd_baselines(cfg)
    elif args.cmd == "n-sensitivity":
        cmd_n_sensitivity(cfg)


if __name__ == "__main__":
    main()
