"""阶段 1.1：G0 补测信号生成（计划 §3，20260815）。

G0 = 第 4 轮 F1 权重（**不动一字，只读**）推理 2025-07-01~2025-12-31：
oos1 内、F1 从未在此段出过数、训练/val 窗（≤2025-06-30）之外——纯跨时段
稳定性补测。

- G0 四变体 = canonical 推理（L=90/H=10/N=20/T=1.0/top_p=0.9/seed=42，
  参数逐字来自 paper_replication/config.yaml）逐日 csi300，断点续跑；
- F0 四变体 / M = 从既有 baseline_suite oos parquet 切 2025H2 子集（不重推理）；
- 9 张宽表日期索引对齐断言。

落盘（finetune_suite/data/g0/，不入库）：
    daily_signals_2025h2_G0_{last,mean,max,min}.parquet
    daily_signals_2025h2_F0_{last,mean,max,min}.parquet
    daily_signals_2025h2_M.parquet
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import DATA_DIR as BL_DATA_DIR, VARIANTS, BaselineConfig
from baseline_suite.run_signals import build_provider, build_rebalances
from baseline_suite.signal import run_variant_signals
from finetune_suite.run_f1_signals import load_f1_predictor

PKG_DIR = Path(__file__).resolve().parent
G0_DIR = PKG_DIR / "data" / "g0"

# 计划 §1 冻结窗口：G0 评估 2025H2（oos1 前半段）
G0_START = "2025-07-01"
G0_END = "2025-12-31"


def build_g0_config() -> BaselineConfig:
    """oos 口径 + G0 窗 + **唯一变量=评估时段**（权重 = 第 4 轮 F1，只读）。"""
    from finetune_suite.config import Config as SuiteConfig

    suite = SuiteConfig()
    return replace(
        BaselineConfig.load(window="oos"),
        window="2025h2_G0",  # checkpoint/日志标签（断点续跑文件名即最终名）
        backtest_start=G0_START,
        backtest_end=G0_END,
        model_name=suite.finetuned_predictor_path,
        tokenizer_name=suite.finetuned_tokenizer_path,
    )


def cut_subset(tag: str) -> pd.DataFrame:
    """从既有 baseline_suite oos parquet 切 2025H2 子集（F0 四变体 / M）。"""
    src = BL_DATA_DIR / f"daily_signals_oos_{tag}.parquet"
    df = pd.read_parquet(src)
    sub = df.loc[G0_START:G0_END]
    out = G0_DIR / (
        f"daily_signals_2025h2_F0_{tag}.parquet" if tag in VARIANTS
        else "daily_signals_2025h2_M.parquet"
    )
    sub.to_parquet(out)
    logger.info(f"{tag} 子集切取：{sub.shape[0]} 日 × {sub.shape[1]} 列 → {out.name}")
    return sub


def main() -> None:
    G0_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_g0_config()
    logger.info(
        f"G0 臂配置：window={cfg.window} [{cfg.backtest_start}~{cfg.backtest_end}] "
        f"pool={cfg.pool} N={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} seed={cfg.seed}"
    )
    logger.info(f"G0 权重（第 4 轮 F1，只读）：model={cfg.model_name}")
    logger.info(f"                      tokenizer={cfg.tokenizer_name}")

    # —— G0 四变体推理（断点续跑，checkpoint 名 = 最终落盘名）——
    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = load_f1_predictor(cfg)
    wide = run_variant_signals(
        predictor, provider, cfg, rebalances, checkpoint_dir=G0_DIR
    )
    for v in VARIANTS:
        out = G0_DIR / f"daily_signals_2025h2_G0_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"G0 {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    # —— F0 四变体 + M 子集 ——
    arms: dict[str, pd.DataFrame] = {f"G0_{v}": wide[v] for v in VARIANTS}
    for v in VARIANTS:
        arms[f"F0_{v}"] = cut_subset(v)
    arms["M"] = cut_subset("M")

    # —— 对齐断言：9 表日期索引完全一致 ——
    ref_idx = arms["G0_mean"].index
    assert len(ref_idx) > 0, "G0 信号为空"
    for name, df in arms.items():
        assert df.index.equals(ref_idx), (
            f"{name} 日期索引与 G0_mean 不一致："
            f"{df.index.min()}~{df.index.max()} vs {ref_idx.min()}~{ref_idx.max()}"
        )
    logger.info(
        f"对齐断言通过：9 表日期索引一致（{len(ref_idx)} 日，"
        f"{ref_idx.min().date()}~{ref_idx.max().date()}）"
    )
    logger.info(
        f"列覆盖：G0 平均 {arms['G0_mean'].notna().sum(axis=1).mean():.0f} 只/日 | "
        f"F0 {arms['F0_mean'].shape[1]} 列 | M {arms['M'].shape[1]} 列"
    )


if __name__ == "__main__":
    main()
