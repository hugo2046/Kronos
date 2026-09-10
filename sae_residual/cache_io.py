"""缓存 I/O 原语：原子写、身份校验、批量加载（计划 §4/§7）。

chunk 为 ``npz``，内嵌 ``identity_json``；协议版本 / G1 权重 SHA / 数据段
任一不匹配即拒绝复用（``CacheMismatchError``），防止旧缓存混入。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sae_residual import config as C


class CacheMismatchError(RuntimeError):
    """缓存身份不匹配（拒绝复用）。"""


def sha256_file(path: Path) -> str:
    """文件 SHA256（1MiB 分块）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def identity_of(path: Path) -> dict:
    """读取 chunk 内嵌身份（缺身份 → ``CacheMismatchError``）。"""
    with np.load(path, allow_pickle=False) as z:
        if "identity_json" not in z:
            raise CacheMismatchError(f"{path.name} 缺身份字段")
        return json.loads(str(z["identity_json"]))


def write_chunk(path: Path, payload: dict, identity: dict) -> None:
    """原子写 chunk：先 ``.tmp`` 再 rename；身份内嵌为 JSON 字符串。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, identity_json=np.array(json.dumps(identity, ensure_ascii=False)),
             **payload)
    tmp.replace(path)


def load_chunk(path: Path, expect: dict) -> dict:
    """加载并校验身份；``expect`` 的每个字段必须与内嵌身份逐值相等。

    缺字段/不匹配 → ``CacheMismatchError``（调用方重建或停止）。
    """
    got = identity_of(path)
    for k, v in expect.items():
        if got.get(k) != v:
            raise CacheMismatchError(
                f"{path.name} 身份字段 {k} 不匹配：{got.get(k)!r} != {v!r}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "identity_json"}


def chunk_manifest_path(art_dir: Path) -> Path:
    return art_dir / "cache_manifest.json"


def samples_path(art_dir: Path, split: str) -> Path:
    return art_dir / f"sample_keys_{split}.parquet"


def load_split_arrays(split: str, expect: dict | None = None) -> dict:
    """整段加载全部 chunk，拼接为数组并二次校验身份。

    :returns: train/val → ``{dates, instruments, h, s_g1, y_mean, close_t}``；
        W3/W4 → ``{dates, instruments, h, s_g1}``（s_g1 为基线原信号值）。
    """
    d = C.CACHE_DIR / split
    chunks = sorted(p for p in d.glob(f"{split}_*.npz")
                    if not p.name.endswith(".tmp.npz"))
    if not chunks:
        raise FileNotFoundError(f"{split} 缓存不存在：{d}")
    expect = expect or {}
    out: dict[str, list] = {}
    for p in chunks:
        arrs = load_chunk(p, expect)
        for k, v in arrs.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def wide_from_arrays(dates: np.ndarray, instruments: np.ndarray,
                     values: np.ndarray) -> pd.DataFrame:
    """(date, instrument, value) → date×instrument 宽表。"""
    df = pd.DataFrame({"datetime": pd.to_datetime(dates),
                       "instrument": instruments, "value": values})
    return df.pivot(index="datetime", columns="instrument",
                    values="value").sort_index()


__all__ = ["CacheMismatchError", "sha256_file", "identity_of", "write_chunk",
           "load_chunk", "load_split_arrays", "wide_from_arrays",
           "samples_path", "chunk_manifest_path"]
