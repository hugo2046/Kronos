"""G9 信号生成（计划 §1/§4.3，20260821 G9 计划）。

- 臂 = 全 epoch 重训（seed=100）predictor 的第 N 个 epoch checkpoint
  （E1/E5/E10/E15）+ E0（官方 Kronos-base predictor + G1 tokenizer，零训练
  免费诊断臂）；tokenizer 恒为 **G1 s100（冻结只读共享）**；
- 窗口（计划 §1 臂表冻结）：E1/E15 双窗（backtest 2026-01-01~2026-07-24 +
  2025H2 2025-07-01~2025-12-31）；E5/E10/E0 仅 backtest（描述性曲线臂）；
  ——计划文字"六次推理"实为臂×窗枚举 7 次（E1×2、E15×2、E5/E10/E0×backtest），
  按枚举如数执行（结果文档如实记录）；
- canonical 推理逐字：L=90/H=10/N=20/T=1.0/top_p=0.9/**推理 seed 恒 42**
  （与训练种子无关，G2/G8 同款）；回测池恒 csi300；
- 断点续跑（checkpoint 文件名即最终名）；F0/M 对照子集只读对齐断言；
- **纪律 §5：评估数字在六组信号全部落盘前不看**——本脚本只产信号，零绩效。

落盘（g9_ckpt/data/{arm}/，不入库）：
    daily_signals_{window}_G9{arm}_{last,mean,max,min}.parquet

用法::

    /home/user/miniconda3/envs/quant/bin/python -m g9_ckpt.run_g9_signals --arm E1 --window backtest
    /home/user/miniconda3/envs/quant/bin/python -m g9_ckpt.run_g9_signals --arm E15 --window 2025h2
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import DATA_DIR as BL_DATA_DIR, VARIANTS, BaselineConfig
from baseline_suite.run_signals import build_provider, build_rebalances
from baseline_suite.signal import run_variant_signals
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START

PKG_DIR = Path(__file__).resolve().parent
G9_DATA_DIR = PKG_DIR / "data"
G9_CKPT_ROOT = PKG_DIR / "outputs" / "models" / "finetune_predictor_g9" / "checkpoints"

WINDOW_DEFS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": ("2025-07-01", "2025-12-31"),
}

# 臂表（计划 §1 冻结）：epoch 臂 + E0 官方底座；双窗臂仅 E1/E15
EPOCH_ARMS: dict[str, int] = {"E1": 1, "E5": 5, "E10": 10, "E15": 15}
TWO_WINDOW_ARMS = ("E1", "E15")
OFFICIAL_PREDICTOR = "NeoQuasar/Kronos-base"

# 第 4 轮 F0/M backtest 宽表（只读对齐基准）
ROUND4_DATA = PKG_DIR.parent / "finetune_suite" / "data"


def arm_tag(arm: str) -> str:
    """臂标签：E1 → G9E1（DuckDB 归档同款）。"""
    return f"G9{arm}"


def arm_model_path(arm: str) -> str:
    """臂的 predictor 权重路径：E0=官方底座，其余=epoch_N checkpoint。"""
    if arm == "E0":
        return OFFICIAL_PREDICTOR
    return str(G9_CKPT_ROOT / f"epoch_{EPOCH_ARMS[arm]}")


def build_g9_config(arm: str, window: str) -> BaselineConfig:
    """oos 口径 + 指定窗 + **唯一变量=选用的 checkpoint**（tokenizer 恒 G1）。"""
    from finetune_suite.train_g1 import G1Config

    start, end = WINDOW_DEFS[window]
    return replace(
        BaselineConfig.load(window="oos"),
        window=f"{window}_{arm_tag(arm)}",  # checkpoint/日志标签（断点续跑文件名即最终名）
        backtest_start=start,
        backtest_end=end,
        model_name=arm_model_path(arm),
        tokenizer_name=G1Config().finetuned_tokenizer_path,  # G1 tokenizer 冻结共享
    )


def _load_g9_predictor(cfg: BaselineConfig, arm: str):
    """加载 tokenizer + predictor（E0=官方底座 + G1 tokenizer 组合）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G1 tokenizer（冻结只读，全臂共享）：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 G9-{arm} predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def _cut_f0_m(window: str) -> dict[str, pd.DataFrame]:
    """F0 四变体 + M 的窗口子集（只读；G0 cut_subset 同款切取源）。

    - backtest：第 4 轮 F0/M backtest 宽表（索引即该窗，恒等切取）；
    - 2025h2：baseline_suite oos 宽表切取（四变体文件名无 F0 前缀——即
      zero-shot K 组，G0 cut_subset 同款命名）。
    """
    start, end = WINDOW_DEFS[window]
    arms: dict[str, pd.DataFrame] = {}
    if window == "backtest":
        src_dir, prefix = ROUND4_DATA, "daily_signals_backtest_F0"
    else:
        src_dir, prefix = BL_DATA_DIR, "daily_signals_oos"
    for v in VARIANTS:
        arms[f"F0_{v}"] = pd.read_parquet(src_dir / f"{prefix}_{v}.parquet").loc[start:end]
    m_name = "daily_signals_backtest_M.parquet" if window == "backtest" else "daily_signals_oos_M.parquet"
    arms["M"] = pd.read_parquet(src_dir / m_name).loc[start:end]
    return arms


def run_one(arm: str, window: str) -> None:
    out_dir = G9_DATA_DIR / arm.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_g9_config(arm, window)
    logger.info(
        f"G9 臂配置：arm={arm} window={cfg.window} "
        f"[{cfg.backtest_start}~{cfg.backtest_end}] pool={cfg.pool} "
        f"N={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} 推理seed={cfg.seed}"
    )
    logger.info(f"G9 权重（唯一变量=选用 checkpoint）：model={cfg.model_name}")
    logger.info(f"                                 tokenizer={cfg.tokenizer_name}（G1 冻结）")

    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_g9_predictor(cfg, arm)
    wide = run_variant_signals(predictor, provider, cfg, rebalances, checkpoint_dir=out_dir)
    for v in VARIANTS:
        out = out_dir / f"daily_signals_{window}_{arm_tag(arm)}_{v}.parquet"
        wide[v].to_parquet(out)
        logger.info(f"{arm_tag(arm)} {window} {v} 落盘 {out.name}（{wide[v].shape[0]} 日）")

    # —— 对齐断言：本臂四表 + F0 四变体 + M 窗口子集（只读）索引逐日一致 ——
    arms: dict[str, pd.DataFrame] = {f"{arm_tag(arm)}_{v}": wide[v] for v in VARIANTS}
    arms.update(_cut_f0_m(window))
    ref_idx = arms[f"{arm_tag(arm)}_mean"].index
    assert len(ref_idx) > 0, f"G9 {arm} {window} 信号为空"
    for name, df in arms.items():
        assert df.index.equals(ref_idx), (
            f"{name} 日期索引与 {arm_tag(arm)}_mean 不一致："
            f"{df.index.min()}~{df.index.max()} vs {ref_idx.min()}~{ref_idx.max()}"
        )
    logger.info(
        f"对齐断言通过：9 表日期索引一致（{len(ref_idx)} 日，"
        f"{ref_idx.min().date()}~{ref_idx.max().date()}）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="G9 checkpoint 选择臂信号生成（零绩效，只产信号）")
    parser.add_argument("--arm", choices=["E0", "E1", "E5", "E10", "E15"], required=True)
    parser.add_argument("--window", choices=list(WINDOW_DEFS), required=True)
    args = parser.parse_args()
    # 计划 §1 臂表冻结：E5/E10/E0 仅 backtest（描述性），E1/E15 双窗
    if args.arm not in TWO_WINDOW_ARMS and args.window != "backtest":
        parser.error(f"臂 {args.arm} 仅跑 backtest 窗（计划 §1 臂表冻结）")
    run_one(args.arm, args.window)


if __name__ == "__main__":
    main()
