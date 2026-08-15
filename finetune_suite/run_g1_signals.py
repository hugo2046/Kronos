"""阶段 2.3：G1 四变体信号生成（计划 §4，20260815）。

- G1 = 全 A 微调权重（``finetune_{tokenizer,predictor}_g1``，阶段 2.1/2.2 产物），
  canonical 推理（L=90/H=10/N=20/T=1.0/top_p=0.9/seed=42，参数逐字来自
  paper_replication/config.yaml），backtest 窗 2026-01-01~2026-07-24 逐日 csi300
  ——**与第 4 轮同窗可比；回测池仍为 csi300（训练语料变宽 ≠ 评估池变化）**；
- F0 四变体 / M **只读复用**第 4 轮 ``finetune_suite/data/`` 的 backtest 子集
  （同窗同文件，不重推理不重切取）；
- 日期索引对齐断言（G1 四表 + 第 4 轮 F0/M）。

落盘（finetune_suite/data/g1/，不入库）：
    daily_signals_backtest_G1_{last,mean,max,min}.parquet
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.run_signals import build_provider, build_rebalances
from baseline_suite.signal import run_variant_signals
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START

PKG_DIR = Path(__file__).resolve().parent
G1_DIR = PKG_DIR / "data" / "g1"
ROUND4_DATA = PKG_DIR / "data"


def build_g1_config() -> BaselineConfig:
    """oos 口径 + backtest 窗 + **唯一变量=权重**（G1 全 A 微调 checkpoint）。"""
    from finetune_suite.train_g1 import G1Config

    g1 = G1Config()
    return replace(
        BaselineConfig.load(window="oos"),
        window="backtest_G1",  # checkpoint/日志标签（断点续跑文件名即最终名）
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        model_name=g1.finetuned_predictor_path,
        tokenizer_name=g1.finetuned_tokenizer_path,
    )


def main() -> None:
    G1_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_g1_config()
    logger.info(
        f"G1 臂配置：window={cfg.window} [{cfg.backtest_start}~{cfg.backtest_end}] "
        f"pool={cfg.pool} N={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} seed={cfg.seed}"
    )
    logger.info(f"G1 权重（唯一变量）：model={cfg.model_name}")
    logger.info(f"                      tokenizer={cfg.tokenizer_name}")

    # —— G1 四变体推理（断点续跑，checkpoint 名 = 最终落盘名）——
    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_g1_predictor(cfg)
    wide = run_variant_signals(
        predictor, provider, cfg, rebalances, checkpoint_dir=G1_DIR
    )
    for v in VARIANTS:
        out = G1_DIR / f"daily_signals_backtest_G1_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"G1 {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    # —— 对齐断言：G1 四表 + 第 4 轮 F0/M（只读复用）索引逐日一致 ——
    arms: dict[str, pd.DataFrame] = {f"G1_{v}": wide[v] for v in VARIANTS}
    for v in VARIANTS:
        arms[f"F0_{v}"] = pd.read_parquet(ROUND4_DATA / f"daily_signals_backtest_F0_{v}.parquet")
    arms["M"] = pd.read_parquet(ROUND4_DATA / "daily_signals_backtest_M.parquet")

    ref_idx = arms["G1_mean"].index
    assert len(ref_idx) > 0, "G1 信号为空"
    for name, df in arms.items():
        assert df.index.equals(ref_idx), (
            f"{name} 日期索引与 G1_mean 不一致："
            f"{df.index.min()}~{df.index.max()} vs {ref_idx.min()}~{ref_idx.max()}"
        )
    logger.info(
        f"对齐断言通过：{len(arms)} 表日期索引一致（{len(ref_idx)} 日，"
        f"{ref_idx.min().date()}~{ref_idx.max().date()}）"
    )
    logger.info(
        f"列覆盖：G1 平均 {arms['G1_mean'].notna().sum(axis=1).mean():.0f} 只/日 | "
        f"F0 {arms['F0_mean'].shape[1]} 列 | M {arms['M'].shape[1]} 列"
    )


def _load_g1_predictor(cfg: BaselineConfig):
    """加载 G1 微调权重（与 run_f1_signals.load_f1_predictor 同构，仅权重不同）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G1 tokenizer：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 G1 predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


if __name__ == "__main__":
    main()
