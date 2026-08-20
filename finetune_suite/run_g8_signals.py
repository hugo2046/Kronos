"""G8 信号生成（计划 §1/§3.3，20260820 G8+E1 计划）。

- 臂 = G8 predictor（语料终点 +6 个月，seed=100/101/102；tokenizer 共享
  ``finetune_tokenizer_g8``），canonical 推理 L=90/H=10/N=20/T=1.0/top_p=0.9/
  **推理 seed 恒 42**（与训练种子无关，G2 同款）；
- **仅 backtest 窗**（2026-01-01~2026-07-24）：2025H2 已被 G8 用作早停窗，
  对 G8 是准样本内，禁止用作其跨窗检验（计划 §1 冻结）；
- 回测池恒 csi300（训练语料变宽 ≠ 评估池变化，G1 同款）；
- 日期索引对齐断言（与第 4 轮 F0/M backtest 子集只读对齐）。

落盘（finetune_suite/data/g8/s{seed}/，不入库）：
    daily_signals_backtest_G8S{seed}_{last,mean,max,min}.parquet

用法::

    /home/user/miniconda3/envs/quant/bin/python -m finetune_suite.run_g8_signals --seed 100
"""
from __future__ import annotations

import argparse
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
G8_DIR = PKG_DIR / "data" / "g8"
ROUND4_DATA = PKG_DIR / "data"


def arm_tag(seed: int) -> str:
    """种子臂标签：100 → G8S100。"""
    return f"G8S{seed}"


def build_g8_config(seed: int) -> BaselineConfig:
    """oos 口径 + backtest 窗 + **唯一变量=语料终点产出的权重**（推理 seed 恒 42）。"""
    from finetune_suite.train_g8 import G8Config

    g8 = G8Config(seed=seed)
    return replace(
        BaselineConfig.load(window="oos"),
        window=f"backtest_{arm_tag(seed)}",  # checkpoint/日志标签（断点续跑文件名即最终名）
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        model_name=g8.finetuned_predictor_path,
        tokenizer_name=g8.finetuned_tokenizer_path,  # 三种子共享 G8 tokenizer
    )


def _load_g8_predictor(cfg: BaselineConfig):
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G8 tokenizer（共享）：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 G8 predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def run_one(seed: int) -> None:
    out_dir = G8_DIR / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_g8_config(seed)
    logger.info(
        f"G8 臂配置：seed={seed} window={cfg.window} "
        f"[{cfg.backtest_start}~{cfg.backtest_end}] pool={cfg.pool} "
        f"N={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} 推理seed={cfg.seed}"
    )
    logger.info(f"G8 权重（唯一变量=语料终点）：model={cfg.model_name}")
    logger.info(f"                              tokenizer={cfg.tokenizer_name}（共享）")

    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_g8_predictor(cfg)
    wide = run_variant_signals(predictor, provider, cfg, rebalances, checkpoint_dir=out_dir)
    for v in VARIANTS:
        out = out_dir / f"daily_signals_backtest_{arm_tag(seed)}_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"{arm_tag(seed)} {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    # —— 对齐断言：G8 四表 + 第 4 轮 F0/M（只读）索引逐日一致 ——
    arms: dict[str, pd.DataFrame] = {f"{arm_tag(seed)}_{v}": wide[v] for v in VARIANTS}
    for v in VARIANTS:
        arms[f"F0_{v}"] = pd.read_parquet(ROUND4_DATA / f"daily_signals_backtest_F0_{v}.parquet")
    arms["M"] = pd.read_parquet(ROUND4_DATA / "daily_signals_backtest_M.parquet")
    ref_idx = arms[f"{arm_tag(seed)}_mean"].index
    assert len(ref_idx) > 0, "G8 信号为空"
    for name, df in arms.items():
        assert df.index.equals(ref_idx), (
            f"{name} 日期索引与 {arm_tag(seed)}_mean 不一致："
            f"{df.index.min()}~{df.index.max()} vs {ref_idx.min()}~{ref_idx.max()}"
        )
    logger.info(
        f"对齐断言通过：9 表日期索引一致（{len(ref_idx)} 日，"
        f"{ref_idx.min().date()}~{ref_idx.max().date()}）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="G8 语料新鲜度臂信号生成（仅 backtest 窗）")
    parser.add_argument("--seed", type=int, choices=[100, 101, 102], required=True)
    args = parser.parse_args()
    run_one(args.seed)


if __name__ == "__main__":
    main()
