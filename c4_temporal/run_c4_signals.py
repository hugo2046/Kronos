"""C4 阶段 3.2：三种子 C4 信号生成落盘（C4 计划 §3，20260820）。

零训练零推理——对既有 G1 族三种子两窗 mean 信号 parquet（**只读**）施加
冻结变换（三值化 ±2% → 半衰期 λ=0.5^(1/10)/窗 30 加权），合并 260 日
连续宽表逐种子落盘 ``c4_temporal/data/s{s}/daily_signals_merged_C4S{s}_c4.parquet``。

落盘内容含预热期前 30 行（完整变换工件；评估层负责剔除——落盘前置纪律
只约束"评估数字在全部落盘前不看"，信号本身整段如实保存）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m c4_temporal.run_c4_signals``
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c4_temporal.transform import (
    C4_LAMBDA,
    C4_WINDOW,
    HALF_LIFE_DAYS,
    TRINARIZE_THRESHOLD,
    WARMUP_DAYS,
    build_c4,
    load_g1_mean_merged,
    trinarize,
)

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
SEEDS = (100, 101, 102)


def main() -> None:
    logger.info(
        f"C4 冻结超参：阈值 ±{TRINARIZE_THRESHOLD:.0%} / 半衰期 {HALF_LIFE_DAYS} 日"
        f"（λ={C4_LAMBDA:.10f}）/ 窗口 {C4_WINDOW} 日 / 预热烧 {WARMUP_DAYS} 日"
    )
    for s in SEEDS:
        src = load_g1_mean_merged(s)  # G1 信号 parquet 只读
        c4 = build_c4(src)
        assert c4.shape == src.shape, f"s{s} C4 形状漂移 {c4.shape} != {src.shape}"
        out_dir = DATA_DIR / f"s{s}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"daily_signals_merged_C4S{s}_c4.parquet"
        c4.to_parquet(out)
        # 只做形态统计（非评估数字）：三值票分布 + 评估段值域
        tri = trinarize(src)
        n_pos = int((tri > 0).sum().sum())
        n_neg = int((tri < 0).sum().sum())
        n_zero = int((tri == 0).sum().sum())
        n_nan = int(src.isna().sum().sum())
        ev = c4.iloc[WARMUP_DAYS:]
        logger.info(
            f"s{s} 落盘 {out.name}：{c4.shape[0]} 日 × {c4.shape[1]} 列 | "
            f"三值票 +1:{n_pos} / 0:{n_zero} / −1:{n_neg} / NaN:{n_nan} | "
            f"评估段（{WARMUP_DAYS}:）值域 [{ev.min().min():+.3f}, {ev.max().max():+.3f}]"
            f"（理论 ±13.066）"
        )
    logger.info("三种子 C4 信号全部落盘（评估数字解锁条件达成）")


if __name__ == "__main__":
    main()
