"""G2.2：种子臂两窗四变体信号生成（计划 §1，20260816 计划）。

- 臂 = G1 predictor 以 seed=101/102 重训的权重（``finetune_predictor_g2_s{seed}``，
  唯一变量 = 训练种子；tokenizer 复用 G1 共享工件）；
- 每种子两窗四变体：backtest（2026-01-01~2026-07-24）+ 2025H2
  （2025-07-01~2025-12-31），canonical 推理（L=90/H=10/N=20/T=1.0/top_p=0.9/
  seed=42——**推理种子恒 42，与训练种子无关**），回测池恒 csi300；
- F0/M 对照只读复用既有子集（backtest=第 4 轮 data/，2025H2=data/g0/）；
- 日期索引对齐断言。

落盘（finetune_suite/data/g2/s{101,102}/，不入库）：
    daily_signals_backtest_G2S101_{last,mean,max,min}.parquet
    daily_signals_2025h2_G2S101_{last,mean,max,min}.parquet（s102 同构）

用法::

    python finetune_suite/run_g2_signals.py --seed 101 --window backtest
    python finetune_suite/run_g2_signals.py --seed 101 --window 2025h2
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
G2_DIR = PKG_DIR / "data" / "g2"

WINDOW_DEFS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": ("2025-07-01", "2025-12-31"),
}


def arm_tag(seed) -> str:
    """种子臂标签：101 → G2S101；'dtok' → DTOK（增补 37aba7d）。"""
    return f"G2S{seed}" if isinstance(seed, int) else "DTOK"


def build_g2_config(seed, window: str) -> BaselineConfig:
    """oos 口径 + 指定窗 + **唯一变量=训练种子产出的权重**（推理 seed 恒 42）。

    seed ∈ {101,102}（核心，两窗）| {103,104}（增补 D-seed+，仅 backtest）
    | 'dtok'（增补 D-tok 全管线 seed=101，仅 backtest）。
    """
    if isinstance(seed, int):
        from finetune_suite.train_g2 import G2Config

        g2 = G2Config(seed=seed)
        tokenizer_path = g2.finetuned_tokenizer_path  # G1 tokenizer 共享复用
    else:
        from finetune_suite.train_dtok import DtokConfig

        g2 = DtokConfig()
        tokenizer_path = g2.finetuned_tokenizer_path  # D-tok 自训 tokenizer

    start, end = WINDOW_DEFS[window]
    return replace(
        BaselineConfig.load(window="oos"),
        window=f"{window}_{arm_tag(seed)}",  # checkpoint/日志标签（断点续跑文件名即最终名）
        backtest_start=start,
        backtest_end=end,
        model_name=g2.finetuned_predictor_path,
        tokenizer_name=tokenizer_path,
    )


def run_one(seed, window: str) -> None:
    sub = "dtok" if seed == "dtok" else f"s{seed}"
    out_dir = G2_DIR / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_g2_config(seed, window)
    logger.info(
        f"G2 臂配置：seed={seed} window={cfg.window} "
        f"[{cfg.backtest_start}~{cfg.backtest_end}] pool={cfg.pool} "
        f"N={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} 推理seed={cfg.seed}"
    )
    logger.info(f"G2 权重（唯一变量=训练种子）：model={cfg.model_name}")
    logger.info(f"                             tokenizer={cfg.tokenizer_name}（共享复用）")

    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_g2_predictor(cfg)
    wide = run_variant_signals(predictor, provider, cfg, rebalances, checkpoint_dir=out_dir)
    for v in VARIANTS:
        out = out_dir / f"daily_signals_{window}_{arm_tag(seed)}_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"{arm_tag(seed)} {window} {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")


def _load_g2_predictor(cfg: BaselineConfig):
    """加载 G2 种子权重（与 run_g1_signals._load_g1_predictor 同构）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G2 tokenizer（G1 共享）：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 G2 predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 种子臂两窗四变体推理")
    parser.add_argument("--seed", required=True,
                        help="101/102（核心两窗）| 103/104（增补 D-seed+）| dtok（增补 D-tok）")
    parser.add_argument("--window", choices=list(WINDOW_DEFS), required=True)
    args = parser.parse_args()
    seed = args.seed if args.seed == "dtok" else int(args.seed)
    # 增补条款（37aba7d）：增补臂只跑 backtest 窗（省预算），2025H2 不做
    if (seed in (103, 104) or seed == "dtok") and args.window != "backtest":
        parser.error("增补臂（103/104/dtok）仅跑 backtest 窗（增补条款）")
    run_one(seed, args.window)


if __name__ == "__main__":
    main()
