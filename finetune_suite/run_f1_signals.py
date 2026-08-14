"""阶段 3.1：F1 四变体信号生成 + F0（四变体）/M 子集切取（计划 §5，修订2）。

- F1 = 微调 tokenizer+predictor（本地 best_model 目录），canonical 推理
  （L=90/H=10/N=20/T=1.0/top_p=0.9/seed=42，参数逐字来自
  paper_replication/config.yaml），backtest 窗 2026-01-01~2026-07-24 逐日 csi300；
  一次推理四聚合一次算全（mean/last/max/min，同第 1 轮 run_variant_signals）；
- F0/M = 从既有 baseline_suite oos 四变体 parquet 切 2026-01-01~2026-07-24 子集
  （不重推理）；
- 三臂（9 张宽表）日期索引对齐断言。

落盘（finetune_suite/data/，不入库）：
    daily_signals_backtest_F1_{last,mean,max,min}.parquet
    daily_signals_backtest_F0_{last,mean,max,min}.parquet
    daily_signals_backtest_M.parquet
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
from finetune_suite.config import Config as SuiteConfig

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"

# 计划 §1 冻结窗口：backtest = oos1 后半段（半污染）
BACKTEST_START = "2026-01-01"
BACKTEST_END = "2026-07-24"


def build_f1_config() -> BaselineConfig:
    """oos 口径 + backtest 窗 + **唯一变量=权重**（微调本地 checkpoint）。"""
    suite = SuiteConfig()
    return replace(
        BaselineConfig.load(window="oos"),
        window="backtest",
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        model_name=suite.finetuned_predictor_path,
        tokenizer_name=suite.finetuned_tokenizer_path,
    )


def cut_subset(tag: str) -> pd.DataFrame:
    """从既有 baseline_suite oos parquet 切 backtest 子集（F0 四变体 / M）。"""
    src = BL_DATA_DIR / f"daily_signals_oos_{tag}.parquet"
    df = pd.read_parquet(src)
    sub = df.loc[BACKTEST_START:BACKTEST_END]
    out = DATA_DIR / (
        f"daily_signals_backtest_F0_{tag}.parquet" if tag in VARIANTS
        else "daily_signals_backtest_M.parquet"
    )
    sub.to_parquet(out)
    logger.info(f"{tag} 子集切取：{sub.shape[0]} 日 × {sub.shape[1]} 列 → {out.name}")
    return sub


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_f1_config()
    logger.info(
        f"F1 臂配置：window={cfg.window} [{cfg.backtest_start}~{cfg.backtest_end}] "
        f"pool={cfg.pool} N={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} seed={cfg.seed}"
    )
    logger.info(f"F1 权重（唯一变量）：model={cfg.model_name}")
    logger.info(f"                      tokenizer={cfg.tokenizer_name}")

    # —— F1 四变体推理（断点续跑）——
    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = load_f1_predictor(cfg)
    wide = run_variant_signals(
        predictor, provider, cfg, rebalances, checkpoint_dir=DATA_DIR
    )
    for v in VARIANTS:
        out = DATA_DIR / f"daily_signals_backtest_F1_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"F1 {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    # —— F0 四变体 + M 子集 ——
    arms: dict[str, pd.DataFrame] = {f"F1_{v}": wide[v] for v in VARIANTS}
    for v in VARIANTS:
        arms[f"F0_{v}"] = cut_subset(v)
    arms["M"] = cut_subset("M")

    # —— 对齐断言：9 张表日期索引完全一致 ——
    ref_idx = arms["F1_mean"].index
    assert len(ref_idx) > 0, "F1 信号为空"
    for name, df in arms.items():
        assert df.index.equals(ref_idx), (
            f"{name} 日期索引与 F1_mean 不一致："
            f"{df.index.min()}~{df.index.max()} vs {ref_idx.min()}~{ref_idx.max()}"
        )
    logger.info(
        f"对齐断言通过：9 表日期索引一致（{len(ref_idx)} 日，"
        f"{ref_idx.min().date()}~{ref_idx.max().date()}）"
    )
    logger.info(
        f"列覆盖：F1 平均 {arms['F1_mean'].notna().sum(axis=1).mean():.0f} 只/日 | "
        f"F0 {arms['F0_mean'].shape[1]} 列 | M {arms['M'].shape[1]} 列"
    )


def load_f1_predictor(cfg: BaselineConfig):
    """加载微调权重（build_predictor 的路径与字段同构，仅权重不同）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载微调 tokenizer：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载微调 predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


if __name__ == "__main__":
    main()
