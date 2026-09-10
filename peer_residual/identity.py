"""PEER checkpoint / 缓存身份门禁（计划 20260910 §6.1）。

新 checkpoint schema：保存并加载校验 run_id、协议内容 SHA、base
tokenizer/predictor SHA、peer/SAE 清单 SHA、norm_stats SHA、arm/seed/epoch；
缺字段拒绝、内容逐项比较、加载前校验权重文件 SHA。旧权重不自动升级；
历史诊断走只读 legacy 路径（仅接收交接清单精确 SHA，记录
``legacy_identity_incomplete``，禁止用于新训练续跑或新实验验收）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from peer_residual import config as C
from peer_residual.model import build_head
from sae_residual.cache_io import sha256_file

# 新 schema 必备身份字段（缺一拒绝；不自动补写/升级旧权重）
HEAD_IDENTITY_FIELDS = ("run_id", "protocol_digest", "tokenizer_sha",
                        "predictor_sha", "sae_manifest_sha",
                        "peer_manifest_sha", "norm_stats_sha",
                        "arm", "seed", "epoch")


def protocol_digest() -> str:
    """协议内容 SHA（冻结实验自由度的规范化 JSON）。"""
    payload = {"protocol": C.PROTOCOL_VERSION, "run_id": C.RUN_ID,
               "head": C.HEAD, "train": {**C.TRAIN},
               "test_windows": {k: list(v) for k, v in C.TEST_WINDOWS.items()},
               "forward_cutoff": C.FORWARD_CUTOFF, "pool": C.POOL,
               "lookback": C.LOOKBACK, "predict_len": C.PREDICT_LEN}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def current_identity() -> dict:
    """从当前协议和真实文件取得生产身份。

    :returns: 不含 arm/seed/epoch 的公共身份；仅计算 SHA，不加载底座。
    """
    from peer_residual import data as PD

    return {"run_id": C.RUN_ID, "protocol_digest": protocol_digest(),
            **PD.g1_weight_shas(),
            "sae_manifest_sha": sha256_file(C.SAE_CACHE_MANIFEST),
            "peer_manifest_sha": sha256_file(C.ART_DIR / "peer_cache_manifest.json"),
            "norm_stats_sha": sha256_file(C.SAE_NORM_STATS)}


def save_head_checkpoint(model, path: Path, identity: dict) -> Path:
    """按新 schema 保存头（身份字段齐全才允许写）。"""
    missing = [k for k in HEAD_IDENTITY_FIELDS if k not in identity]
    if missing:
        raise RuntimeError(f"新 checkpoint 身份字段缺失：{missing}——拒绝保存")
    torch.save({"state_dict": model.state_dict(),
                "identity": {k: identity[k] for k in HEAD_IDENTITY_FIELDS}},
               path)
    return path


def load_head_checkpoint(path: Path, expect: dict, d_in: int,
                         expected_file_sha: str | None = None):
    """加载并逐项校验身份；文件 SHA 与冻结信号来源 manifest 对应。

    :param expect: 期望身份（每个字段与 ckpt 内 identity 逐项相等）。
    :param expected_file_sha: 权重文件 SHA（字节级门禁，可选但生产必传）。
    :raises RuntimeError: 缺字段 / 字段不匹配 / 文件字节被改。
    """
    missing_expect = set(HEAD_IDENTITY_FIELDS) - set(expect)
    if missing_expect:
        raise RuntimeError(f"期望身份缺字段：{sorted(missing_expect)}")
    if expected_file_sha is not None:
        got_sha = sha256_file(path)
        if got_sha != expected_file_sha:
            raise RuntimeError(
                f"头文件字节与冻结清单不一致：{path.name} {got_sha[:16]}… != "
                f"{expected_file_sha[:16]}…")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "identity" not in ckpt:
        raise RuntimeError(f"{path.name} 缺 identity 字段（旧 schema）——拒绝，"
                           f"不自动升级；历史文件走 legacy 只读路径")
    ident = ckpt["identity"]
    missing = [k for k in HEAD_IDENTITY_FIELDS if k not in ident]
    if missing:
        raise RuntimeError(f"{path.name} identity 缺字段 {missing}——拒绝")
    for k, v in expect.items():
        if ident.get(k) != v:
            raise RuntimeError(
                f"{path.name} 身份字段 {k} 不匹配：{ident.get(k)!r} != {v!r}")
    model = build_head(int(ident["seed"]), d_in)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_legacy_head(path: Path, expected_sha256: str, arm: str, seed: int,
                     epoch: int, d_in: int):
    """只读 legacy 路径：仅接收交接清单精确 SHA 的旧 best 文件。

    旧 meta 只有 arm/seed/epoch/protocol/d_in（无 run_id/base/manifest SHA）
    → 记录 ``legacy_identity_incomplete``；禁止用于新训练续跑或验收
    （:func:`assert_training_eligible` 会拒绝）。
    """
    got = sha256_file(path)
    if got != expected_sha256:
        raise RuntimeError(f"legacy 头 SHA 不匹配：{got[:16]}… != "
                           f"{expected_sha256[:16]}…")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt.get("meta", {})
    for k, v in (("arm", arm), ("seed", seed), ("epoch", epoch)):
        if meta.get(k) != v:
            raise RuntimeError(f"legacy 头 meta {k}={meta.get(k)!r} != {v!r}")
    model = build_head(seed, d_in)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    note = {"legacy_identity_incomplete": True,
            "legacy_fields": sorted(meta.keys()),
            "missing_fields": [k for k in HEAD_IDENTITY_FIELDS
                               if k not in meta],
            "file_sha256": got}
    return model, note


def assert_training_eligible(note: dict) -> None:
    """门禁：legacy/身份不完整的头不得用于新训练续跑或新实验验收。"""
    if note.get("legacy_identity_incomplete"):
        raise RuntimeError("legacy_identity_incomplete：旧头只读诊断可用，"
                           "禁止用于新训练续跑或新实验验收")


def verify_cache_dir(cache_dir: Path, manifest: dict, split: str) -> dict:
    """按冻结清单校验目录：完整键集合 + 逐 chunk SHA（元数据对但字节变也拒）。

    :param manifest: 冻结 manifest（``splits.<split>.chunks[].file/sha256``）。
    :raises RuntimeError: 缺 chunk / 多余 chunk / SHA 不匹配。
    """
    spec = manifest["splits"][split]
    expected = {c["file"]: c["sha256"] for c in spec["chunks"]}
    actual_files = sorted(p.name for p in cache_dir.iterdir()
                          if p.suffix == ".npz"
                          and not p.name.endswith(".tmp.npz"))
    missing = sorted(set(expected) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected))
    if missing:
        raise RuntimeError(f"{split} 缓存缺 chunk（清单权威，不以 glob 子集"
                           f"代替）：{missing[:5]}")
    if extra:
        raise RuntimeError(f"{split} 缓存出现清单外 chunk：{extra[:5]}")
    for name, sha in expected.items():
        got = sha256_file(cache_dir / name)
        if got != sha:
            raise RuntimeError(f"{split} chunk 字节与清单不一致：{name} "
                               f"{got[:16]}… != {sha[:16]}…")
    return {"n_chunks": len(expected), "split": split}


def verify_reusable_head(heads_dir: Path, arm: str, seed: int, epoch: int,
                         frozen_head_shas: dict[str, str]) -> str:
    """复用门禁：旧"best.json 存在就跳过"不能绕过身份。

    :param frozen_head_shas: 冻结清单 {文件名: SHA256}；清单缺该文件或
        头文件字节被改 → 拒绝复用（重训或不跳过由调用方决定）。
    :returns: 校验通过的文件 SHA。
    """
    name = f"head_{arm}_s{seed}_e{epoch:03d}.pt"
    p = heads_dir / name
    if not p.is_file():
        raise RuntimeError(f"复用头缺失：{p}")
    if name not in frozen_head_shas:
        raise RuntimeError(f"复用头不在冻结清单：{name}")
    got = sha256_file(p)
    if got != frozen_head_shas[name]:
        raise RuntimeError(f"复用头字节与冻结清单不一致：{name}")
    return got


__all__ = ["HEAD_IDENTITY_FIELDS", "protocol_digest", "current_identity",
           "save_head_checkpoint", "load_head_checkpoint",
           "load_legacy_head", "assert_training_eligible",
           "verify_cache_dir", "verify_reusable_head"]
