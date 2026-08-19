"""N50 阶段 3.2：采样放大推理信号生成（N50 计划 §3，20260819）。

- 唯一变量 ``sample_count``：20 → 50（论文敏感性曲线延长线，测到 N=20 为止，
  N=20→50 估计噪声只再缩 ~37%，预期效应小）；其余推理口径逐字 canonical
  （L=90/H=10/T=1.0/top_p=0.9/推理 seed=42）——L/H 议题已由 G7 终审关闭，
  本轮**不碰**窗口参数；
- G1 族三种子权重**只读**复用（s100=G1、s101=G2S101、s102=G2S102，
  tokenizer 共享 G1）——N 是纯推理期参数，换配置零训练；
- 推理链路**逐字复用** ``baseline_suite.signal.run_variant_signals``（第 4 轮/
  G1/G2/G4/G5/G7 同款）：唯一差别 = ``cfg.sample_count``；
- 窗口常量逐字 import（backtest=第 4 轮 BACKTEST_START/END）——与在位者同界
  可比；**2025h2 不跑**（预算裁定：N=50 成本 ≈2.5×，跑前声明，runner 不提供）；
- 断点续跑（checkpoint 名 = 最终落盘名）；落盘后与在位者 G1 族同窗信号
  日期索引逐日对齐断言。

落盘（n50_amplify/data/s{seed}/，不入库）：
    daily_signals_backtest_G1N50S{seed}_{last,mean,max,min}.parquet

用法::

    /home/user/miniconda3/envs/quant/bin/python -m n50_amplify.run_n50_signals \
        --seed 100
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
DATA_DIR = PKG_DIR / "data"

SEEDS = (100, 101, 102)

# 窗口常量逐字 import（不自定义）：与在位者 G1 族同界可比；仅 backtest（预算裁定）
WINDOW_DEFS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
}

# G1 族三种子权重映射（g4_features.run_g4_signals._assert_aligned 同款口径）
G1_FAMILY_ARMS = {100: "G1", 101: "G2S101", 102: "G2S102"}

# N=50 显存分块（等序列数预算）：predict_batch_chunked 按股票数分块（默认 32，
# N=20 实测安全 = 32×20=640 序列/块）；N=50 沿用 32 → 1600 序列/块，首跑第 3 日
# CUDA OOM（碎片累积）。缩到 12×50=600 ≤ 640。仅显存管理，非引擎参数。
N50_CHUNK_SIZE = 12


def _patch_chunk_size_for_n50() -> None:
    """运行时把 baseline_suite.signal 中的分块引用绑定为 chunk_size=N50_CHUNK_SIZE。

    只替换本包进程内的命名空间引用（既有文件零改动）；分块对逐股预测
    数学透明，固定 chunk_size 下全程确定性可复现。2026-08-19 首跑以
    默认 32 分块存下的 1 日 checkpoint 已删除，134 日全部以统一 12 分块重算。
    """
    from functools import partial

    import baseline_suite.signal as bs
    from paper_replication.signal import predict_batch_chunked

    bs.predict_batch_chunked = partial(
        predict_batch_chunked, chunk_size=N50_CHUNK_SIZE
    )


def _g1_family_paths(seed: int) -> tuple[str, str]:
    """G1 族权重路径：s100=G1；s101/102=G2S{seed}（tokenizer 共享 G1）。"""
    if seed == 100:
        from finetune_suite.train_g1 import G1Config

        g = G1Config()
    else:
        from finetune_suite.train_g2 import G2Config

        g = G2Config(seed=seed)
    return g.finetuned_predictor_path, g.finetuned_tokenizer_path


def arm_tag(seed: int) -> str:
    return f"G1N50S{seed}"


def build_n50_config(seed: int) -> BaselineConfig:
    """canonical(oos) + G1 族权重 + backtest 窗 + **唯一自由度 N=50**。

    除 sample_count（及必然随臂变的权重路径与窗口字段）外，
    其余字段与 ``BaselineConfig.load(window="oos")`` 逐字相等——
    ``tests/test_n50_config.py`` 门禁。
    """
    model, tokenizer = _g1_family_paths(seed)
    return replace(
        BaselineConfig.load(window="oos"),
        window=f"backtest_{arm_tag(seed)}",  # checkpoint/日志标签（断点续跑文件名即最终名）
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        model_name=model,
        tokenizer_name=tokenizer,
        sample_count=50,
    )


def run_one(seed: int) -> None:
    out_dir = DATA_DIR / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_n50_config(seed)
    logger.info(
        f"N50 臂配置：seed={seed} window={cfg.window} "
        f"[{cfg.backtest_start}~{cfg.backtest_end}] pool={cfg.pool} "
        f"L={cfg.lookback} H={cfg.predict_len} N={cfg.sample_count} T={cfg.T} "
        f"top_p={cfg.top_p} 推理seed={cfg.seed}"
    )
    logger.info(f"G1 族权重（只读复用）：model={cfg.model_name}")
    logger.info(f"                        tokenizer={cfg.tokenizer_name}")

    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_predictor(cfg)
    _patch_chunk_size_for_n50()
    wide = run_variant_signals(predictor, provider, cfg, rebalances, checkpoint_dir=out_dir)
    for v in VARIANTS:
        out = out_dir / f"daily_signals_backtest_{arm_tag(seed)}_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"{arm_tag(seed)} backtest {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    _assert_aligned(rebalances)


def _assert_aligned(rebalances: pd.DatetimeIndex) -> None:
    """与在位者 G1 族同窗信号日期索引逐日一致（同窗可比门禁）。"""
    r4 = PKG_DIR.parent / "finetune_suite" / "data"
    refs = {
        100: r4 / "g1" / "daily_signals_backtest_G1_mean.parquet",
        101: r4 / "g2" / "s101" / "daily_signals_backtest_G2S101_mean.parquet",
        102: r4 / "g2" / "s102" / "daily_signals_backtest_G2S102_mean.parquet",
    }
    for s, p in refs.items():
        if not p.exists():
            logger.warning(f"对齐参照缺失（跳过）：{p}")
            continue
        ref_idx = pd.read_parquet(p).index
        assert ref_idx.equals(rebalances), (
            f"在位者 s{s} backtest 日期索引与 N50 不一致："
            f"{ref_idx.min()}~{ref_idx.max()} vs {rebalances.min()}~{rebalances.max()}"
        )
    logger.info(f"对齐断言通过：{len(refs)} 个在位者参照日期索引一致（{len(rebalances)} 日）")


def _load_predictor(cfg: BaselineConfig):
    """加载 G1 族权重（与 run_g1_signals._load_g1_predictor 同构，只读）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G1 族 tokenizer：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 G1 族 predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N50 采样放大推理（N=50，G1 族三种子，仅 backtest 窗）"
    )
    parser.add_argument("--seed", type=int, choices=list(SEEDS), required=True)
    args = parser.parse_args()
    run_one(args.seed)


if __name__ == "__main__":
    main()
