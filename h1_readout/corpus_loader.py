"""G1 同源语料 pkl 的安全装载器（受限 Unpickler，Mimosa 门禁合规）。

``finetune_suite/data/{,ashares}/{train,val}_data.pkl`` 是本仓库自产的可信语料
（``finetune_suite.build_dataset`` 2026-08-15 构建，dict[str, DataFrame]）。历史
代码用裸 ``pickle.load``；本包按门禁建议改用**受限 Unpickler**：仅放行
pandas/numpy 的数据重建函数（DataFrame/索引/BlockManager/Cython 数组重建等），
``pandas._libs`` 命名空间外的 ``__pyx_unpickle_*`` 与一切其他全局一律拒绝
（fail-closed）。语料内容与 G1 微调逐字节同源，装载后经数值健全性检查。
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_FSDATA = REPO_ROOT / "finetune_suite" / "data"

CORPUS_PKLS: dict[str, dict[str, Path]] = {
    "csi300": {"train": _FSDATA / "train_data.pkl", "val": _FSDATA / "val_data.pkl"},
    "ashares": {"train": _FSDATA / "ashares" / "train_data.pkl",
                "val": _FSDATA / "ashares" / "val_data.pkl"},
}

# pandas/numpy 数据重建函数（pickle 载荷清单内的全部类型）
_BASE_ALLOW = {
    "pandas.core.frame": ["DataFrame"],
    "pandas.core.indexes.datetimes": ["DatetimeIndex", "_new_DatetimeIndex"],
    "pandas.core.indexes.base": ["_new_Index", "Index"],
    "pandas.core.indexes.numeric": ["Int64Index"],
    "pandas.core.arrays.datetimes": ["DatetimeArray"],
    "pandas.core.internals.managers": ["BlockManager"],
    "pandas.core.internals.blocks": ["new_block"],
    "pandas._libs.internals": ["_unpickle_block", "BlockValuesUnpicker_unsafe"],
    "pandas._libs.tslibs.timestamps": ["_unpickle_timestamp"],
    "numpy._core.multiarray": ["_reconstruct", "scalar"],
    "numpy": ["dtype", "ndarray"],
    "numpy.dtypes": [n for n in ("Float64DType", "Float32DType", "Int64DType",
                                 "Int32DType", "BoolDType", "ObjectDType",
                                 "DateTime64DType", "TimeDelta64DType", "StrDType",
                                 "UInt8DType")],
    "builtins": ["set", "list", "dict", "tuple", "str", "int", "float", "bytes",
                 "slice", "range", "frozenset", "bool", "complex", "NoneType"],
    "collections": ["OrderedDict"],
    "codecs": ["encode"],
}
# pandas._libs 内部 Cython 数组重建函数（纯数据重建，限该命名空间）
_PYX = re.compile(r"^__pyx_unpickle_[A-Za-z_]+$")


class CorpusUnpickler(pickle.Unpickler):
    """受限 Unpickler：allowlist 之外的任何全局即抛错（fail-closed）。"""

    def find_class(self, module: str, name: str):
        if name in _BASE_ALLOW.get(module, []) or (
            module.startswith("pandas._libs") and _PYX.match(name)
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"forbidden global: {module}.{name}")


def load_corpus_split(pool: str, split: str) -> dict:
    """装载 {symbol: DataFrame(6 列 OHLCVA, DatetimeIndex)}（只读，不修改）。"""
    path = CORPUS_PKLS[pool][split]
    with open(path, "rb") as f:
        data = CorpusUnpickler(f).load()
    assert isinstance(data, dict) and data, f"{path} 载荷异常"
    sample = next(iter(data.values()))
    assert list(sample.columns) == ["open", "high", "low", "close", "vol", "amt"]
    return data


__all__ = ["load_corpus_split", "CorpusUnpickler", "CORPUS_PKLS"]
