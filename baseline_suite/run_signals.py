"""阶段 1 入口：四变体 baseline 信号生成 + mean 对拍门禁（计划 §2）。

用法（解释器 /home/user/miniconda3/envs/quant/bin/python）::

    # 论文窗口四变体（断点续跑，约 2 小时）
    python -m baseline_suite.run_signals variants --window paper

    # M/R/P 对照（快，纯取数 + 随机）
    python -m baseline_suite.run_signals baselines --window paper

落盘（data/ 不入库）：
    - ``daily_signals_paper_{last,mean,max,min}.parquet``
    - ``daily_signals_{M,R,P}.parquet``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import DATA_DIR, VARIANTS, BaselineConfig, ensure_dirs
from baseline_suite.signal import (
    run_momentum_reversal,
    run_placeholder,
    run_variant_signals,
)


def build_provider(cfg: BaselineConfig):
    """构造 QlibProvider（覆盖窗口 + 回看缓冲）。"""
    from kronos_qlib import QlibProvider

    fetch_start = (
        pd.Timestamp(cfg.backtest_start) - pd.Timedelta(days=cfg.lookback * 2)
    ).strftime("%Y-%m-%d")
    return QlibProvider(cfg.pool, fetch_start, cfg.backtest_end)


def load_predictor(cfg: BaselineConfig):
    """加载 KronosPredictor（与 paper_replication 同口径）。"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def build_rebalances(cfg: BaselineConfig) -> pd.DatetimeIndex:
    """每日调仓日序列（窗口内每个交易日）。"""
    from kronos_qlib import QlibProvider

    p = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
    return p.trading_days(cfg.backtest_start, cfg.backtest_end)


def cmd_variants(cfg: BaselineConfig) -> None:
    """生成四变体信号（last/mean/max/min）。"""
    rebalances = build_rebalances(cfg)
    logger.info(
        f"调仓日（每日）[{cfg.window}]：{len(rebalances)} 个，"
        f"{rebalances[0].date()}~{rebalances[-1].date()}"
    )

    provider = build_provider(cfg)
    predictor = load_predictor(cfg)
    wide = run_variant_signals(
        predictor, provider, cfg, rebalances, checkpoint_dir=DATA_DIR
    )
    for v in VARIANTS:
        out = DATA_DIR / f"daily_signals_{cfg.window}_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"{v} 信号落盘 {out}（{wide[v].shape[0]} 日）")


def cmd_baselines(cfg: BaselineConfig) -> None:
    """生成 M / R / P 组信号（与 paper_replication 同口径，仅窗口不同）。"""
    provider = build_provider(cfg)
    rebalances = build_rebalances(cfg)
    logger.info(
        f"调仓日（每日）[{cfg.window}]：{len(rebalances)} 个，"
        f"{rebalances[0].date()}~{rebalances[-1].date()}"
    )

    mom_wide, rev_wide = run_momentum_reversal(provider, cfg, rebalances)
    out_m = DATA_DIR / f"daily_signals_{cfg.window}_M.parquet"
    out_r = DATA_DIR / f"daily_signals_{cfg.window}_R.parquet"
    mom_wide.to_parquet(out_m)
    rev_wide.to_parquet(out_r)
    logger.info(f"M/R 信号落盘 {out_m}")

    p_wide = run_placeholder(cfg, rebalances, list(mom_wide.columns), seed=cfg.seed)
    out_p = DATA_DIR / f"daily_signals_{cfg.window}_P.parquet"
    p_wide.to_parquet(out_p)
    logger.info(f"P 信号落盘 {out_p}")


def cmd_gate(cfg: BaselineConfig) -> None:
    """mean 对拍门禁（§2.2）：新跑的 mean 与既有 daily_signals_K.parquet 逐位对拍。

    同 seed 同 N 同推理链路 → 应逐位一致。不一致先查随机性来源，不许带差异继续。
    """
    new_path = DATA_DIR / f"daily_signals_{cfg.window}_mean.parquet"
    ref_path = REPO_ROOT / "paper_replication" / "data" / "daily_signals_K.parquet"
    if not new_path.exists():
        raise FileNotFoundError(f"新 mean 信号缺失：{new_path}（先跑 variants）")
    if not ref_path.exists():
        raise FileNotFoundError(f"对拍基准缺失：{ref_path}（先在 paper_replication 跑 K 组）")

    new = pd.read_parquet(new_path)
    ref = pd.read_parquet(ref_path)
    logger.info(f"新 mean：{new.shape} | 既有 K：{ref.shape}")

    # 索引对齐（日期）
    common_dates = new.index.intersection(ref.index)
    if len(common_dates) == 0:
        raise AssertionError(f"无公共日期：new {new.index[0]}~{new.index[-1]} vs ref {ref.index[0]}~{ref.index[-1]}")
    logger.info(f"公共日期：{len(common_dates)} / 新 {len(new.index)} / 既有 {len(ref.index)}")

    # 列对齐
    common_cols = new.columns.intersection(ref.columns)
    logger.info(f"公共列：{len(common_cols)} / 新 {new.shape[1]} / 既有 {ref.shape[1]}")

    a = new.loc[common_dates, common_cols]
    b = ref.loc[common_dates, common_cols]
    # 只在两边都有值的位置比（NaN 应一致）
    both_valid = a.notna() & b.notna()
    diff = (a - b).where(both_valid)
    max_abs_diff = float(np.nanmax(np.abs(diff.values)))
    n_diff_gt_1e8 = int((np.abs(diff) > 1e-8).sum().sum())
    n_cells = int(both_valid.sum().sum())

    # NaN 一致性
    new_na = a.isna()
    ref_na = b.isna()
    na_mismatch = int((new_na != ref_na).sum().sum())

    logger.info(f"对拍结果：有效单元 {n_cells}，max|Δ|={max_abs_diff:.3e}，"
                f"|Δ|>1e-8 单元 {n_diff_gt_1e8}，NaN 不一致 {na_mismatch}")

    gate_pass = (max_abs_diff < 1e-8) and (n_diff_gt_1e8 == 0) and (na_mismatch == 0)
    if gate_pass:
        logger.info(f"✓ 对拍门禁通过：mean 与既有 K 组逐位一致（max|Δ|={max_abs_diff:.3e}）")
    else:
        raise AssertionError(
            f"✗ 对拍门禁未通过：max|Δ|={max_abs_diff:.3e}（{n_diff_gt_1e8} 单元 >1e-8），"
            f"NaN 不一致 {na_mismatch}——先查随机性来源，不许带差异继续"
        )


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="baseline 四变体信号生成 + 对拍门禁")
    parser.add_argument(
        "cmd",
        choices=["variants", "baselines", "gate"],
        help="variants=四变体 | baselines=M/R/P | gate=mean对拍门禁",
    )
    parser.add_argument(
        "--window", choices=["paper", "oos"], default="paper",
        help="窗口：paper=2024-07~2025-06 | oos=2025-07~2026-07",
    )
    args = parser.parse_args()
    cfg = BaselineConfig.load(window=args.window)
    ensure_dirs()
    if args.cmd == "variants":
        cmd_variants(cfg)
    elif args.cmd == "baselines":
        cmd_baselines(cfg)
    elif args.cmd == "gate":
        cmd_gate(cfg)


if __name__ == "__main__":
    main()
