"""R1 打分落盘（计划 §4.2 后半，**不判读**）。

两窗（backtest 2026-01-01~2026-07-24 / 2025H2 2025-07-01~2025-12-31）逐日构样本
（L=90，与 G5 完全同口径——R1 唯一变量是头的损失函数）→ G1 底座单次前向取隐状态
→ 6 头（R-lin/R-kda × s42/43/44）decode 打分。信号落盘 ``r1_objective/data/``。

**纪律 §5**：6 checkpoint 全部定型后才可运行（入口断言）；本脚本零绩效、零判读。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m r1_objective.run_r1_signals
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from finetune_suite.run_g2_signals import WINDOW_DEFS as G2_WINDOWS

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"

from r1_objective.run_r1_train import R1_ARMS, R1_SEEDS

MODEL_NAMES = [f"{arm}_s{seed}" for arm in R1_ARMS for seed in R1_SEEDS]

WINDOW_BOUNDS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": G2_WINDOWS["2025h2"],
}


def score_window(models: dict, backbone, window: str, device: str) -> dict[str, pd.DataFrame]:
    """单窗逐日打分：底座隐状态一次前向，6 头 decode（镜像 g5 score_window）。"""
    from cross_section_kda.data import build_daily_samples
    from g5_head.heads import decode_score
    from kronos_qlib import QlibProvider

    start, end = WINDOW_BOUNDS[window]
    provider = QlibProvider("csi300", start, end)
    rebalances = provider.trading_days(start, end)

    rows: dict[str, list[dict]] = {n: [] for n in models}
    for i, d in enumerate(rebalances):
        ds = d.strftime("%Y-%m-%d")
        b = build_daily_samples(provider, date=ds, pool="csi300")
        if b is None:
            logger.warning(f"{ds}: 无可用样本")
            for n in models:
                rows[n].append({})
            continue
        with torch.no_grad():
            hidden = backbone.extract(b.x_norm.to(device), b.stamp.to(device))
            for n, m in models.items():
                score = decode_score(m, hidden).cpu().numpy()
                rows[n].append({c: float(s) for c, s in zip(b.codes, score)})
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  score [{window}] [{i + 1}/{len(rebalances)}] {ds}: {len(b.codes)} 只")

    wide = {n: pd.DataFrame(rows[n], index=rebalances) for n in models}
    for n, df in wide.items():
        df.to_parquet(DATA_DIR / f"daily_signals_{window}_{n}.parquet")
        logger.info(f"[{window}] {n} 信号落盘（{df.shape[0]} 日，"
                    f"平均 {df.notna().sum(axis=1).mean():.0f} 只/日）")
    return wide


def main() -> None:
    device = "cuda:0"
    from g5_head.backbone_g1 import load_g1_backbone
    from r1_objective.cache_loader import load_hidden_cache  # noqa: F401  (装载能力声明)
    from r1_objective.run_r1_train import _make_head

    # —— 产物门禁：6 checkpoint 全部定型才许打分（§5 判读前置纪律）——
    missing = [n for n in MODEL_NAMES if not (DATA_DIR / f"{n}_best.pt").exists()]
    assert not missing, f"6 checkpoint 未全部定型，禁止打分：缺 {missing}"

    backbone = load_g1_backbone(device)
    models = {}
    for name in MODEL_NAMES:
        arm, seed = name.rsplit("_s", 1)
        m = _make_head(arm, backbone).to(device)
        ckpt = torch.load(DATA_DIR / f"{name}_best.pt", map_location="cpu", weights_only=True)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models[name] = m
        logger.info(f"载入 {name} ← {name}_best.pt（es_RankIC={ckpt['info']['best_es_rankic']:+.4f}）")

    for window in WINDOW_BOUNDS:
        logger.info("=" * 70)
        logger.info(f"窗口 {window}：{WINDOW_BOUNDS[window]}")
        score_window(models, backbone, window, device)
    logger.info("==== R1 打分落盘完成（判读统一 run_r1_judge 一次开封） ====")


if __name__ == "__main__":
    main()
