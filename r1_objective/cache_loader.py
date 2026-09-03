"""g5 隐状态缓存的安全装载器（Mimosa 门禁合规）。

``g5_head/data/hidden_cache_train_es.pt`` 是本仓库 2026-08-17 自产的可信缓存
（构建于 ``g5_head.run_g5_head.build_and_save_cache``，含 pd.Timestamp / numpy
数组等非张量载荷）。历史代码用 ``weights_only=False`` 装载（当时无门禁）；本包
新代码按 Mimosa 门禁建议改用 ``weights_only=True`` + **最小显式 allowlist**——
只放行缓存实际包含的纯数据重建函数（时间戳反序列化 / numpy 数组重建 / dtype），
不放行任意全局。allowlist 之外任何类型出现即装载失败（fail-closed）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import numpy.dtypes as npdt
from numpy._core.multiarray import _reconstruct
from pandas._libs.tslibs.timestamps import _unpickle_timestamp

CACHE_PATH = Path(__file__).resolve().parent.parent / "g5_head" / "data" / "hidden_cache_train_es.pt"

# 缓存实际包含的非张量类型（构建端 ``build_and_save_cache`` 的载荷清单）：
# train{Tensor,Tensor} / es[{Timestamp, Tensor, ndarray, list[str]}] / refs[{Tensor×3}]
_ALLOWLIST = [
    pd.Timestamp,
    _unpickle_timestamp,
    _reconstruct,
    np.ndarray,
    np.dtype,
    *[getattr(npdt, n) for n in dir(npdt) if n.endswith("DType")],
]


def load_hidden_cache(path: Path | str | None = None, *, map_location: str = "cpu") -> dict:
    """安全装载 g5 隐状态缓存（只读；weights_only=True + 最小 allowlist）。"""
    p = Path(path) if path is not None else CACHE_PATH
    torch.serialization.add_safe_globals(_ALLOWLIST)
    return torch.load(p, map_location=map_location, weights_only=True)


__all__ = ["load_hidden_cache", "CACHE_PATH"]
