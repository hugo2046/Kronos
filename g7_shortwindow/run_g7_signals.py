"""G7 阶段 3.2：W85 短窗推理信号生成（G7 计划 §3，20260818）。

- W85 = L=8/H=5（用户经验先验"8 天看 5 天"，2026-08-18 跑前确认；W83 已撤除），
  G1 族三种子权重**只读**复用（s100=G1、s101=G2S101、s102=G2S102，
  tokenizer 共享 G1）——L/H 是纯推理期参数，换配置零训练；
- 推理链路**逐字复用** ``baseline_suite.signal.run_variant_signals``（第 4 轮/
  G1/G2/G5 同款）：唯一差别 = ``cfg.lookback``/``cfg.predict_len``；
  N=20/T=1.0/top_p=0.9/推理 seed=42（恒 42，与训练种子无关）、池 csi300、
  引擎口径（k=50/n=5/min_hold=5/15bp）逐字 canonical（test_g7_config 门禁）；
- 两窗窗口常量逐字 import（backtest=第 4 轮 BACKTEST_START/END、
  2025h2=run_g2_signals.WINDOW_DEFS）——与在位者同界可比；H 短导致的
  多余可结算日不使用；
- 断点续跑（checkpoint 名 = 最终落盘名）；落盘后与在位者 G1 族同窗信号
  日期索引逐日对齐断言。

落盘（g7_shortwindow/data/s{seed}/，不入库）：
    daily_signals_backtest_W85S{seed}_{last,mean,max,min}.parquet
    daily_signals_2025h2_W85S{seed}_{last,mean,max,min}.parquet

用法::

    /home/user/miniconda3/envs/quant/bin/python -m g7_shortwindow.run_g7_signals \
        --seed 100 --window backtest
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
from finetune_suite.run_g2_signals import WINDOW_DEFS as G2_WINDOW_DEFS

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"

SEEDS = (100, 101, 102)

# 窗口常量逐字 import（不自定义）：与在位者 G1 族同界可比
WINDOW_DEFS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": G2_WINDOW_DEFS["2025h2"],
}

# G1 族三种子权重映射（g4_features.run_g4_signals._assert_aligned 同款口径）
G1_FAMILY_ARMS = {100: "G1", 101: "G2S101", 102: "G2S102"}


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
    return f"W85S{seed}"


def build_w85_config(seed: int, window: str) -> BaselineConfig:
    """canonical(oos) + G1 族权重 + 指定窗 + **唯一自由度 L=8/H=5**。

    除 lookback/predict_len（及必然随臂变的权重路径与窗口字段）外，
    其余字段与 ``BaselineConfig.load(window="oos")`` 逐字相等——
    ``tests/test_g7_config.py`` 门禁。
    """
    model, tokenizer = _g1_family_paths(seed)
    start, end = WINDOW_DEFS[window]
    return replace(
        BaselineConfig.load(window="oos"),
        window=f"{window}_{arm_tag(seed)}",  # checkpoint/日志标签（断点续跑文件名即最终名）
        backtest_start=start,
        backtest_end=end,
        model_name=model,
        tokenizer_name=tokenizer,
        lookback=8,
        predict_len=5,
    )


def run_one(seed: int, window: str) -> None:
    out_dir = DATA_DIR / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_w85_config(seed, window)
    logger.info(
        f"W85 臂配置：seed={seed} window={cfg.window} "
        f"[{cfg.backtest_start}~{cfg.backtest_end}] pool={cfg.pool} "
        f"L={cfg.lookback} H={cfg.predict_len} N={cfg.sample_count} T={cfg.T} "
        f"top_p={cfg.top_p} 推理seed={cfg.seed}"
    )
    logger.info(f"G1 族权重（只读复用）：model={cfg.model_name}")
    logger.info(f"                        tokenizer={cfg.tokenizer_name}")

    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_w85_predictor(cfg)
    wide = run_variant_signals(predictor, provider, cfg, rebalances, checkpoint_dir=out_dir)
    for v in VARIANTS:
        out = out_dir / f"daily_signals_{window}_{arm_tag(seed)}_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"{arm_tag(seed)} {window} {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    _assert_aligned(window, rebalances)


def _assert_aligned(window: str, rebalances: pd.DatetimeIndex) -> None:
    """与在位者 G1 族同窗信号日期索引逐日一致（同窗可比门禁）。"""
    r4 = PKG_DIR.parent / "finetune_suite" / "data"
    g5 = PKG_DIR.parent / "g5_head" / "data"
    incumbent = {
        "backtest": {
            100: r4 / "g1" / "daily_signals_backtest_G1_mean.parquet",
            101: r4 / "g2" / "s101" / "daily_signals_backtest_G2S101_mean.parquet",
            102: r4 / "g2" / "s102" / "daily_signals_backtest_G2S102_mean.parquet",
        },
        "2025h2": {
            100: g5 / "daily_signals_2025h2_G1_mean.parquet",
            101: r4 / "g2" / "s101" / "daily_signals_2025h2_G2S101_mean.parquet",
            102: r4 / "g2" / "s102" / "daily_signals_2025h2_G2S102_mean.parquet",
        },
    }
    refs = incumbent[window]
    for s, p in refs.items():
        if not p.exists():
            logger.warning(f"对齐参照缺失（跳过）：{p}")
            continue
        ref_idx = pd.read_parquet(p).index
        assert ref_idx.equals(rebalances), (
            f"在位者 s{s} {window} 日期索引与 W85 不一致："
            f"{ref_idx.min()}~{ref_idx.max()} vs {rebalances.min()}~{rebalances.max()}"
        )
    logger.info(f"对齐断言通过：{len(refs)} 个在位者参照日期索引一致（{len(rebalances)} 日）")


def _load_w85_predictor(cfg: BaselineConfig):
    """加载 G1 族权重（与 run_g1_signals._load_g1_predictor 同构，只读）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G1 族 tokenizer：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 G1 族 predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def main() -> None:
    parser = argparse.ArgumentParser(description="G7 W85 短窗推理（L=8/H=5，G1 族三种子）")
    parser.add_argument("--seed", type=int, choices=list(SEEDS), required=True)
    parser.add_argument("--window", choices=list(WINDOW_DEFS), required=True)
    args = parser.parse_args()
    run_one(args.seed, args.window)


if __name__ == "__main__":
    main()
