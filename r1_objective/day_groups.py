"""冻结 g5 缓存的日边界重建（R1 IC 损失的数据完整性前提）。

问题：``g5_head/data/hidden_cache_train_es.pt`` 的 train 段是**扁平**的
``{"hidden": [Ntr,90,832], "y": [Ntr]}``（跨日随机批 MSE 用），而 IC 损失需要
**按日截面**分组。缓存文件只读（§5），不能改结构。

方案：日边界可从数据层确定性重建——``build_split_cache(TRAIN_START, TRAIN_END)``
逐日构造样本（与缓存构建同一调用、同一剔除规则），各日长度做累积切片。构建时与
冻结缓存**逐位对拍**（每日 y_z 切片 vs 缓存 y 逐元素一致），对拍记录落盘
``r1_objective/data/train_day_groups.json``；此后训练只读 JSON，不再触 DDB。

g5 缓存（只读，2026-08-17 构建）在 2022~2023-12 历史窗上确定性成立：DDB 历史数据
与 PIT 池成分不变 → 日长度序列与缓存构建时一致；y 逐位对拍失败即硬断言（拒绝训练）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
HIDDEN_CACHE_PATH = PKG_DIR.parent / "g5_head" / "data" / "hidden_cache_train_es.pt"
DAY_GROUPS_JSON = DATA_DIR / "train_day_groups.json"

# IC 损失退化保护：截面 < MIN_CROSS_SECTION 只的日（构建时断言不存在）
MIN_CROSS_SECTION = 5


def build_and_verify_day_groups() -> dict:
    """重建 train 日边界 + 与冻结缓存逐位对拍 + 落盘 JSON。

    返回文档结构::

        {"n_train_samples": int, "groups": [{"date", "start", "end", "length"}, ...],
         "verify": {"y_allclose": bool, "max_abs_diff": float,
                    "hidden_rows": int, "n_days": int}}
    """
    from cross_section_kda.data import TRAIN_END, TRAIN_START
    from cross_section_kda.train import build_split_cache
    from kronos_qlib import QlibProvider
    from r1_objective.cache_loader import load_hidden_cache

    logger.info(f"日边界重建：build_split_cache({TRAIN_START}~{TRAIN_END})（数据层，无 GPU）")
    provider = QlibProvider("csi300", TRAIN_START, TRAIN_END)
    batches = build_split_cache(provider, start=TRAIN_START, end=TRAIN_END,
                                pool="csi300", rebalance_only=False)
    logger.info(f"train 日数 {len(batches)}（与 g5 缓存构建同调用路径）")

    cache = load_hidden_cache(HIDDEN_CACHE_PATH)
    hidden_rows = int(cache["train"]["hidden"].shape[0])
    y_cache = cache["train"]["y"].numpy()
    assert hidden_rows == y_cache.shape[0], "缓存自身 hidden/y 行数不一致"

    groups, start, y_days = [], 0, []
    for b in batches:
        length = len(b.codes)
        assert length >= MIN_CROSS_SECTION, (
            f"{b.date.date()} 截面仅 {length} 只 < {MIN_CROSS_SECTION}，IC 损失退化"
        )
        y_days.append(b.y_z)
        groups.append({"date": str(b.date.date()), "start": start,
                       "end": start + length, "length": length})
        start += length
    assert start == hidden_rows, (
        f"日边界重建总样本 {start} ≠ 缓存 {hidden_rows}——数据层与缓存构建时漂移，拒绝训练"
    )

    max_diff = 0.0
    for g, y_day in zip(groups, y_days):
        seg = y_cache[g["start"]:g["end"]]
        max_diff = max(max_diff, float(np.max(np.abs(seg - y_day))))
    assert max_diff < 1e-6, (
        f"日边界与冻结缓存 y 对拍失败 max|Δ|={max_diff:.3e}——切片错位，拒绝训练"
    )
    logger.info(f"日边界对拍：{len(groups)} 日 max|Δy|={max_diff:.3e} → 逐位一致")

    doc = {
        "n_train_samples": start,
        "groups": groups,
        "verify": {"y_allclose": True, "max_abs_diff": max_diff,
                   "hidden_rows": hidden_rows, "n_days": len(groups)},
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DAY_GROUPS_JSON.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    logger.info(f"日边界落盘 → {DAY_GROUPS_JSON}")
    return doc


def load_day_groups() -> dict:
    """读日边界 JSON（训练入口）；缺失则先构建（含深度对拍）。"""
    if not DAY_GROUPS_JSON.exists():
        return build_and_verify_day_groups()
    return json.loads(DAY_GROUPS_JSON.read_text(encoding="utf-8"))


__all__ = ["build_and_verify_day_groups", "load_day_groups", "DAY_GROUPS_JSON",
           "HIDDEN_CACHE_PATH", "MIN_CROSS_SECTION"]
