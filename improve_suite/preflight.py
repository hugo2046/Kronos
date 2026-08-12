"""环境断言（计划 §2.0.5 阶段 0 预检）。

三项硬断言，全过才许进后续阶段：

    1. csi500 在 2024-07-01 / 2025-07-01 两个时点 ``list_pool_at`` 非空且 ≥400 只
       （阶段 2 B3 跨池验证、阶段 5 候选用 csi500，跑前确认数据可取）；
    2. ``assert_oos_within_data()`` 通过（样本外末日 + H 结算 ≤ 数据末日）；
    3. GPU 可用（推理 / 训练在 cuda:0）。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.preflight
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from improve_suite.common import ImproveConfig, ensure_dirs


def check_gpu() -> bool:
    """GPU 可用性断言。"""
    import torch

    ok = torch.cuda.is_available()
    if ok:
        logger.info(f"✓ GPU 可用：{torch.cuda.get_device_name(0)}")
    else:
        logger.error("✗ GPU 不可用——推理 / 训练需 cuda:0")
    return ok


def check_pool_size(pool: str, t: str, min_size: int = 400) -> int:
    """某池在某时点的成分股数量断言（≥ min_size）。

    :returns: 实际成分股数。
    """
    from kronos_qlib import QlibProvider

    p = QlibProvider(pool, t, t)
    members = p.list_pool_at(pool, t)
    n = len(members)
    if n >= min_size:
        logger.info(f"✓ {pool} @ {t}：{n} 只（≥ {min_size}）")
    else:
        logger.error(f"✗ {pool} @ {t}：仅 {n} 只（< {min_size}）")
    return n


def check_oos_window() -> bool:
    """样本外窗口结算不越界断言（复用 BaselineConfig.assert_oos_within_data）。"""
    from baseline_suite.common import BaselineConfig

    cfg = BaselineConfig.load(window="oos")
    try:
        cfg.assert_oos_within_data()
        return True
    except AssertionError as e:
        logger.error(f"✗ 样本外结算越界：{e}")
        return False


def main() -> None:
    ensure_dirs()
    logger.info("==== improve_suite 预检（计划 §2.0.5）====")

    results = {}
    # 1. csi500 两时点成分股
    for t in ("2024-07-01", "2025-07-01"):
        results[f"csi500@{t}"] = check_pool_size("csi500", t, min_size=400)
    # 2. 样本外窗口
    results["oos_within_data"] = check_oos_window()
    # 3. GPU
    results["gpu"] = check_gpu()

    # 汇总
    pool_ok = all(v >= 400 for k, v in results.items() if k.startswith("csi500@"))
    all_ok = pool_ok and results["oos_within_data"] and results["gpu"]
    logger.info(f"==== 预检结果：{'全过' if all_ok else '存在失败项'} ====")
    logger.info(f"csi500 成分股：{ {k: v for k, v in results.items() if k.startswith('csi500@')} }")
    if not all_ok:
        raise SystemExit("预检未全过，检查上方 ✗ 项")
    logger.info("✓ 预检全过，可进阶段 1+")


if __name__ == "__main__":
    main()
