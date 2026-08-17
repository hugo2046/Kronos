"""G5 阶段 0 前置：补生成 G1@2025H2 四变体信号（20260817）。

§1 的 0.1/0.2 要求 backtest + 2025H2 双窗归因，但仓库中 **G1 的 2025H2 信号
不存在**（data/g0/ 是 F1 权重、data/g2/ 只有 s101~104）。本脚本以冻结的
canonical 协议补齐：与 run_g1_signals.py 唯一差别是窗口换成 2025h2（窗口常量
**逐字 import** 自 run_g2_signals.WINDOW_DEFS，不自定义）。

披露（G5 计划 §1 写"零新推理"，此为计划撰写时的事实性误设——G1@2025H2
parquet 缺失，双窗归因无法只用既有文件）：本生成发生在预承诺提交之后、
任何归因数字计算之前，推理协议零自由度（同 L/H/N/T/top_p/seed=42）。

落盘（g5_head/data/，不入库）：
    daily_signals_2025h2_G1_{last,mean,max,min}.parquet

用法::

    /home/user/miniconda3/envs/quant/bin/python -m g5_head.gen_g1_2025h2
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
from finetune_suite.run_g2_signals import WINDOW_DEFS

PKG_DIR = Path(__file__).resolve().parent
OUT_DIR = PKG_DIR / "data"


def build_g1_2025h2_config() -> BaselineConfig:
    """oos 口径 + 2025H2 窗 + G1 微调权重（与 run_g1_signals 同构，仅窗口不同）。"""
    from finetune_suite.train_g1 import G1Config

    g1 = G1Config()
    start, end = WINDOW_DEFS["2025h2"]
    return replace(
        BaselineConfig.load(window="oos"),
        window="2025h2_G1",  # checkpoint/日志标签（断点续跑文件名即最终名）
        backtest_start=start,
        backtest_end=end,
        model_name=g1.finetuned_predictor_path,
        tokenizer_name=g1.finetuned_tokenizer_path,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_g1_2025h2_config()
    logger.info(
        f"G1@2025H2 配置：window={cfg.window} [{cfg.backtest_start}~{cfg.backtest_end}] "
        f"pool={cfg.pool} N={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} seed={cfg.seed}"
    )
    logger.info(f"G1 权重：model={cfg.model_name}")
    logger.info(f"           tokenizer={cfg.tokenizer_name}")

    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_g1_predictor(cfg)
    wide = run_variant_signals(predictor, provider, cfg, rebalances, checkpoint_dir=OUT_DIR)
    for v in VARIANTS:
        out = OUT_DIR / f"daily_signals_2025h2_G1_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"G1@2025H2 {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    # —— 对齐断言：与 2025H2 既有 M/F0 索引逐日一致（同窗可比）——
    g0_dir = PKG_DIR.parent / "finetune_suite" / "data" / "g0"
    m_h2 = pd.read_parquet(g0_dir / "daily_signals_2025h2_M.parquet")
    f0_h2 = pd.read_parquet(g0_dir / "daily_signals_2025h2_F0_mean.parquet")
    ref = wide["mean"].index
    assert len(ref) > 0, "G1@2025H2 信号为空"
    for name, df in (("M_2025h2", m_h2), ("F0_mean_2025h2", f0_h2)):
        assert df.index.equals(ref), (
            f"{name} 日期索引与 G1@2025H2 不一致："
            f"{df.index.min()}~{df.index.max()} vs {ref.min()}~{ref.max()}"
        )
    logger.info(f"对齐断言通过：3 表日期索引一致（{len(ref)} 日，{ref.min().date()}~{ref.max().date()}）")


def _load_g1_predictor(cfg: BaselineConfig):
    """加载 G1 微调权重（与 run_g1_signals._load_g1_predictor 同构）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G1 tokenizer：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 G1 predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


if __name__ == "__main__":
    main()
