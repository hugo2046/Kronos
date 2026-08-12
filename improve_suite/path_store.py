"""逐路径落盘 / 读回（计划 §2 阶段 0）。

长表 schema：``date, code, path_id, step, pred_close``（约 260日×300只×20路×10步
≈ 1.6e7 行，parquet 压缩后 <100MB，不入库）。

落盘路径约定 ``improve_suite/data/paths_<window>_<config>.parquet``。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"

# 长表列（固定顺序，便于下游消费）
COLUMNS = ["date", "code", "path_id", "step", "pred_close"]
_DTYPES = {
    "code": "str",
    "path_id": "int16",
    "step": "int16",
    "pred_close": "float32",
}


def write_paths(df: pd.DataFrame, path: str | Path) -> Path:
    """逐路径长表落盘为 parquet。

    :param df: 长表，至少含 :data:`COLUMNS` 列。
    :param path: 输出 parquet 路径（父目录自动创建）。
    :returns: 落盘的 :class:`Path`。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col, dt in _DTYPES.items():
        out[col] = out[col].astype(dt)
    out.to_parquet(path, index=False)
    return path


def read_paths(path: str | Path) -> pd.DataFrame:
    """读回逐路径长表（date 列还原为 datetime）。"""
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def paths_to_wide_pred_close(
    df: pd.DataFrame, date, code: str
) -> pd.DataFrame:
    """取某 (date, code) 的逐路径 close 宽表 ``(path_id × step)``。

    便于分布信号消费：行=路径，列=预测步。
    """
    sub = df[(df["date"] == pd.Timestamp(date)) & (df["code"] == code)]
    return sub.pivot(index="path_id", columns="step", values="pred_close").sort_index()
