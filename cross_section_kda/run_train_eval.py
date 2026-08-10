"""训练/早停段运行入口（计划 §5 第 4 步）。

调参**只在此段内**——最终验证段（2024-07-01 之后）本入口不触碰。

用法：
    /home/user/miniconda3/envs/quant/bin/python -m cross_section_kda.run_train_eval
"""
from __future__ import annotations

import sys

from loguru import logger

from cross_section.common import ExperimentConfig
from cross_section_kda.evaluate_arms import run_train_eval


def main() -> int:
    cfg = ExperimentConfig.load()
    device = cfg.device
    logger.info(f"训练/早停段：device={device} pool={cfg.pool}")
    run_train_eval(cfg, device=device)
    logger.info("✅ 训练/早停段完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
