"""最终验证段运行入口（计划 §5 第 5 步，**只跑一次，封盘**）。

纪律：在 B1/B2/B3 全部定型（run_train_eval 完成、checkpoint 落盘）后运行。
跑完即封盘，无论结果好坏如实报告。

用法：
    /home/user/miniconda3/envs/quant/bin/python -m cross_section_kda.run_final_validation
"""
from __future__ import annotations

import sys

from loguru import logger

from cross_section.common import ExperimentConfig
from cross_section_kda.evaluate_arms import run_final_validation


def main() -> int:
    cfg = ExperimentConfig.load()
    device = cfg.device
    logger.info(f"最终验证段（封盘运行）：device={device} pool={cfg.pool}")
    out = run_final_validation(cfg, device=device)
    v = out["verdict"]
    logger.info("=" * 70)
    logger.info(f"改造有效(B3>B0 且 B3>B1)：{v['改造有效(B3>B0 且 B3>B1)']}")
    logger.info(f"KDA头独立贡献(B3>B2)：{v['KDA头有独立贡献(B3>B2)']}")
    logger.info(f"B0={v['B0_RankIC']:+.4f} B1={v['B1_RankIC']:+.4f} "
                f"B2={v['B2_RankIC']:+.4f} B3={v['B3_RankIC']:+.4f}")
    logger.info("✅ 最终验证段完成（封盘）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
