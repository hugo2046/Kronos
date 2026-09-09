"""G10H 运行时状态：RNG 快照/隔离、版本化 checkpoint 与信号 cache 身份。

修复三缺陷（20260909 收尾计划 §5/§6）：①RNG 隔离移到"训练 epoch 完成后、
验证前"且覆盖 Python/numpy/Torch-CPU/已初始化 CUDA；②epoch checkpoint 带
完整随机状态与版本化元数据（``g10h-runtime-v2``），原子写；③信号 cache
只在全部身份匹配时复用，窗口恢复遵循 RNG 链。仅 G10 复用，不建设全仓框架。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

RUNTIME_SCHEMA = "g10h-runtime-v2"
VAL_SEED = 100


# ============================================================
# RNG 快照（全部已使用随机源；CUDA 未初始化则跳过）
# ============================================================


def _numpy_state_to_primitives(state) -> tuple:
    """numpy MT19937 状态 → weights_only 安全原语（keys 转 int 列表）。"""
    name, keys, pos, has_gauss, cached = state
    return (str(name), [int(x) for x in keys.tolist()], int(pos),
            int(has_gauss), float(cached))


def _numpy_state_from_primitives(prim: tuple):
    return (prim[0], np.array(prim[1], dtype=np.uint32), prim[2], prim[3],
            prim[4])


def capture_rng() -> dict:
    """捕获当前全部随机状态（Python/numpy/Torch-CPU/已初始化 CUDA 设备）。"""
    snap: dict = {
        "python": repr(random.getstate()),
        "numpy": _numpy_state_to_primitives(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_initialized():
        snap["cuda"] = [
            torch.cuda.get_rng_state(i)
            for i in range(torch.cuda.device_count())
        ]
    return snap


def restore_rng(snap: dict) -> None:
    """恢复 :func:`capture_rng` 捕获的状态（缺键拒绝，不静默半恢复）。"""
    random.setstate(eval(snap["python"]))  # noqa: S307 — 自身捕获的元组
    np.random.set_state(_numpy_state_from_primitives(snap["numpy"]))
    torch.set_rng_state(snap["torch_cpu"])
    if "cuda" in snap:
        for i, st in enumerate(snap["cuda"]):
            torch.cuda.set_rng_state(st, i)


class validation_rng:
    """验证边界 RNG 隔离：捕获训练态 → seed100 → ``finally`` 恢复。"""

    def __init__(self) -> None:
        self._snap: dict | None = None

    def __enter__(self) -> "validation_rng":
        self._snap = capture_rng()
        random.seed(VAL_SEED)
        np.random.seed(VAL_SEED)
        torch.manual_seed(VAL_SEED)
        return self

    def __exit__(self, *exc) -> None:
        restore_rng(self._snap)


# ============================================================
# 选点与身份哈希
# ============================================================


def select_best(history: list[dict]) -> tuple[int, float]:
    """bestCE = 验证 CE 最低 epoch（严格小于→并列最早）；非有限不可选。

    :returns: ``(epoch, val_ce)``；无有限值抛 ``ValueError``。
    """
    best_epoch, best_ce = None, float("inf")
    for row in history:
        v = row.get("val_ce")
        if v is None or not np.isfinite(v):
            continue
        if v < best_ce:
            best_ce, best_epoch = float(v), int(row["epoch"])
    if best_epoch is None:
        raise ValueError("无有限验证 CE")
    return best_epoch, best_ce


def normalized_state_hash(state_dict: dict) -> str:
    """数值身份哈希：参数名+dtype+shape+固定顺序 CPU 连续字节。"""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ============================================================
# 版本化 checkpoint（原子写）与信号 cache 身份
# ============================================================


def save_versioned_checkpoint(path: Path | str, payload: dict) -> str:
    """先写临时文件再原子替换；返回本文件 SHA256。

    payload 需含模型/优化器状态与 ``meta``；载入侧按 :data:`RUNTIME_SCHEMA`
    校验后恢复。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def load_versioned_checkpoint(path: Path | str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("meta", {}).get("runtime_schema") != RUNTIME_SCHEMA:
        raise RuntimeError(
            f"checkpoint 非 {RUNTIME_SCHEMA}（旧 schema 只读审计，不续跑）")
    return ck


def write_signal_meta(path: Path | str, meta: dict) -> None:
    Path(str(path) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


def cache_reusable(signal_path: Path | str, expected_meta: dict) -> bool:
    """cache 复用门禁：元数据全匹配 + 输出文件 SHA 一致，否则拒绝。

    expected_meta 由实际加载对象与调用参数构造（checkpoint 内容哈希/epoch、
    tokenizer 哈希、协议/运行时、推理参数、窗口与形状摘要），不手写声明。
    """
    signal_path = Path(signal_path)
    meta_path = Path(str(signal_path) + ".meta.json")
    if not signal_path.is_file() or not meta_path.is_file():
        return False
    on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, val in expected_meta.items():
        if on_disk.get(key) != val:
            return False
    if on_disk.get("signal_sha256") != sha256_file(signal_path):
        return False
    return True


def signal_meta_for(checkpoint_sha: str, epoch: int, tokenizer_sha: str,
                    protocol_sha: str, inference: dict, window: str,
                    signal_path: Path | str) -> dict:
    """构造信号 cache 元数据（含输出文件 SHA，供下次复用核验）。"""
    return {
        "runtime_schema": RUNTIME_SCHEMA,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(epoch),
        "tokenizer_sha256": tokenizer_sha,
        "protocol_sha256": protocol_sha,
        "inference": dict(inference),
        "window": window,
        "signal_sha256": sha256_file(signal_path),
    }


__all__ = [
    "RUNTIME_SCHEMA", "VAL_SEED", "capture_rng", "restore_rng",
    "validation_rng", "select_best", "normalized_state_hash", "sha256_file",
    "save_versioned_checkpoint", "load_versioned_checkpoint",
    "write_signal_meta", "cache_reusable", "signal_meta_for",
]
