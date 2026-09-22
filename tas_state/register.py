"""TAS1 独立新前向登记（计划 §6.3 / P4）。

- 不读写既有 forward 登记文件（finetune_suite/registry 等）；
- 原协议封存边界：``2026-07-25`` 起的旧登记不读不写；历史评估只允许
  读取既有协议已公开的白名单结算缓冲；
- 新登记首个决策日必须晚于 manifest 冻结时刻；不追溯回填。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from tas_state.config import PKG_DIR, TASConfig

FORWARD_SEAL_START = "2026-07-25"  # 计划 §5.1：原 forward 封存起点
REGISTRY_DIR = PKG_DIR / "data" / "forward_registry"


def assert_forward_seal(
    date: pd.Timestamp,
    *,
    what: str,
    early_eval: bool = False,
    settlement_buffer: bool = False,
) -> None:
    """访问门禁：旧 forward 封存 + 新 forward 不可提前评价。

    :param date: 被访问的信号/结算日期。
    :param what: 用途描述（进错误信息）。
    :param early_eval: True 表示"新 forward 提前评价"检查——任何不早于
        登记冻结日的未来信号都不得读取成绩。
    :param settlement_buffer: True 表示访问的是**价格结算缓冲**
        （计划 §5.1 白名单：既有协议已公开用于旧决策日结算的价格数据，
        允许晚于封存边界；但不是旧 forward 信号/成绩）。
    :raises PermissionError: 越界访问。
    """
    seal = pd.Timestamp(FORWARD_SEAL_START)
    today = pd.Timestamp(pd.Timestamp.now().date())
    if early_eval and date > today:
        raise PermissionError(
            f"{what} {date:%Y-%m-%d} 未成熟：新 forward 只记录到时可获得的"
            f"输入和信号，标签成熟前不得提前评价"
        )
    if settlement_buffer:
        if date > today:
            raise PermissionError(
                f"{what} {date:%Y-%m-%d} 超出真实数据末日：价格缓冲同样"
                f"不得访问未来"
            )
        return  # 白名单：旧决策日的结算价格缓冲（非 forward 信号）
    if date >= seal:
        raise PermissionError(
            f"{what} {date:%Y-%m-%d} ≥ 封存边界 {FORWARD_SEAL_START}："
            f"原 forward 按协议封存，新实验不得读取其信号或成绩"
        )


def create_forward_manifest(
    cfg: TASConfig,
    *,
    frozen_at: pd.Timestamp,
    first_decision_day: pd.Timestamp,
    n_days: int = 126,
    arms: list[str],
) -> Path:
    """P4 阶段：生成全新前向 manifest（冻结代码/权重/集合/统计规则哈希）。

    首个决策日必须晚于冻结时刻；登记 126 个交易日 + 最后结算日
    （+10 交易日标签成熟）。只允许在 P3 阶段 B 全部通过后调用。
    """
    if first_decision_day <= frozen_at:
        raise ValueError(
            f"首个决策日 {first_decision_day:%Y-%m-%d} 必须晚于冻结时刻 "
            f"{frozen_at:%Y-%m-%d %H:%M}"
        )
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_sha256": cfg.sha256(),
        "frozen_at": f"{frozen_at:%Y-%m-%dT%H:%M:%S}",
        "first_decision_day": f"{first_decision_day:%Y-%m-%d}",
        "n_decision_days": n_days,
        "label_horizon": cfg.predict_len,
        "arms": arms,
        "statistics": {
            "primary": "k=10 RankIC，配对差 HAC(lag=9)，Holm 7 项族",
            "min_delta_ic": cfg.paired_ic_threshold,
            "hac_lag_sensitivity": cfg.hac_lag_sensitivity,
        },
        "rules": [
            "不追溯回填过去日期", "不加入旧 forward 冒充原生登记臂",
            "两个 63 日半窗只在最终一起开封", "日常只检查完整性不展示成绩",
        ],
    }
    out = REGISTRY_DIR / f"forward_manifest_{first_decision_day:%Y%m%d}.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    logger.info(f"新前向 manifest 已冻结：{out}")
    return out
