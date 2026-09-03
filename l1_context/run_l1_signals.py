"""L1 各臂信号生成（计划 §1/§4.3）。

- 臂 = ARMS 表（冻结）：L250-zs×三种子 / L500-zs(s100) / L250-ft(s100 重训)；
  tokenizer 恒 G1 s100 冻结共享；**唯一变量 = 推理 lookback（+ ft 臂的 predictor）**；
- canonical 推理逐字：H=10/N=20/T=1.0/top_p=0.9/推理 seed=42；csi300；
- 窗口（§1 冻结）：全臂 backtest；L250-zs 三种子加跑 2025H2（可比性）；
- 断点续跑（checkpoint 文件名即最终名）；F0/M 对照窗口子集只读对齐断言；
- **纪律 §5**：评估数字在各臂信号全部落盘前不看——本脚本只产信号，零绩效。

落盘（l1_context/data/{tag}/，不入库）：
    daily_signals_{window}_{arm_tag}_{last,mean,max,min}.parquet

用法::

    /home/user/miniconda3/envs/quant/bin/python -m l1_context.run_l1_signals --arm L250ZS100 --window backtest
    /home/user/miniconda3/envs/quant/bin/python -m l1_context.run_l1_signals --arm L500ZS100 --window backtest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import DATA_DIR as BL_DATA_DIR, VARIANTS
from baseline_suite.run_signals import build_provider, build_rebalances
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START

from l1_context.config import ARMS, WINDOW_DEFS, arm_tag, build_arm_config
from l1_context.signal_lb import lb_chunk_size, run_lb_variant_signals

PKG_DIR = Path(__file__).resolve().parent
L1_DATA_DIR = PKG_DIR / "data"
ROUND4_DATA = PKG_DIR.parent / "finetune_suite" / "data"


def _load_predictor(cfg):
    """加载 tokenizer + predictor（权重只读）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 tokenizer（G1 s100 冻结共享，全臂）：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 predictor（只读）：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def _cut_f0_m(window: str) -> dict[str, pd.DataFrame]:
    """F0 四变体 + M 的窗口子集（只读，g9 同款切取源）。"""
    start, end = WINDOW_DEFS[window]
    arms: dict[str, pd.DataFrame] = {}
    if window == "backtest":
        src_dir, prefix = ROUND4_DATA, "daily_signals_backtest_F0"
        m_name = "daily_signals_backtest_M.parquet"
    else:
        src_dir, prefix = BL_DATA_DIR, "daily_signals_oos"
        m_name = "daily_signals_oos_M.parquet"
    for v in VARIANTS:
        arms[f"F0_{v}"] = pd.read_parquet(src_dir / f"{prefix}_{v}.parquet").loc[start:end]
    arms["M"] = pd.read_parquet(src_dir / m_name).loc[start:end]
    return arms


def run_one(tag: str, window: str) -> None:
    spec = ARMS[tag]
    out_dir = L1_DATA_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_arm_config(tag, window)
    logger.info(
        f"L1 臂配置：arm={tag} window={window} [{cfg.backtest_start}~{cfg.backtest_end}] "
        f"pool={cfg.pool} L={cfg.lookback} N={cfg.sample_count} T={cfg.T} "
        f"top_p={cfg.top_p} 推理seed={cfg.seed}"
    )
    logger.info(f"权重（只读）：model={cfg.model_name}")
    logger.info(f"               tokenizer={cfg.tokenizer_name}")

    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_predictor(cfg)
    wide = run_lb_variant_signals(
        predictor, provider, cfg, rebalances,
        chunk_size=lb_chunk_size(cfg.lookback), checkpoint_dir=out_dir,
    )
    atag = arm_tag(tag)
    for v in VARIANTS:
        out = out_dir / f"daily_signals_{window}_{atag}_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"{atag} {window} {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    # —— 对齐断言：本臂四表 + F0 四变体 + M 窗口子集（只读）索引逐日一致 ——
    arms: dict[str, pd.DataFrame] = {f"{atag}_{v}": wide[v] for v in VARIANTS}
    arms.update(_cut_f0_m(window))
    ref_idx = arms[f"{atag}_mean"].index
    assert len(ref_idx) > 0, f"L1 {tag} {window} 信号为空"
    for name, df in arms.items():
        assert df.index.equals(ref_idx), (
            f"{name} 日期索引与 {atag}_mean 不一致："
            f"{df.index.min()}~{df.index.max()} vs {ref_idx.min()}~{ref_idx.max()}"
        )
    logger.info(
        f"对齐断言通过：9 表日期索引一致（{len(ref_idx)} 日，"
        f"{ref_idx.min().date()}~{ref_idx.max().date()}）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="L1 臂信号生成（唯一变量=推理 lookback）")
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--window", choices=list(WINDOW_DEFS), required=True)
    args = parser.parse_args()
    if args.window not in ARMS[args.arm]["windows"]:
        raise SystemExit(
            f"臂 {args.arm} 冻结窗口为 {ARMS[args.arm]['windows']}，不含 {args.window}")
    run_one(args.arm, args.window)


if __name__ == "__main__":
    main()
