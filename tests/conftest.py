"""PEER 生产入口测试的合成文件身份，无真实权重或行情依赖。"""
from __future__ import annotations

import json
import pandas as pd
import pytest
import torch
from peer_residual import config as C
from peer_residual import data as PD
from peer_residual import identity as I
from peer_residual import train as T
from peer_residual.model import build_head
from sae_residual.cache_io import sha256_file

@pytest.fixture
def peer_runtime(tmp_path, monkeypatch):
    """构造实际文件身份，让生产路径自行计算哈希而非 mock 校验器。"""
    monkeypatch.setattr(C, "ART_DIR", tmp_path)
    monkeypatch.setattr(C, "HEADS_DIR", tmp_path)
    for key in ("G1_TOKENIZER", "G1_PREDICTOR"):
        d = tmp_path / key
        d.mkdir()
        (d / "model.safetensors").write_bytes(key.encode())
        monkeypatch.setattr(C, key, d)
    for key in ("SAE_CACHE_MANIFEST", "SAE_NORM_STATS"):
        p = tmp_path / f"{key}.json"
        p.write_text("{}")
        monkeypatch.setattr(C, key, p)
    (tmp_path / "peer_cache_manifest.json").write_text("{}")
    wide = pd.DataFrame({"a": [0.1, 0.2], "b": [0.2, 0.1]},
                        index=pd.bdate_range("2025-01-02", periods=2))
    baseline = {}
    for w in C.TEST_WINDOWS:
        p = tmp_path / f"baseline_{w}.parquet"
        wide.to_parquet(p)
        baseline[w] = p
    monkeypatch.setattr(C, "BASELINE_SIGNALS", baseline)
    monkeypatch.setattr(C, "BASELINE_SHA256",
                        {w: sha256_file(p) for w, p in baseline.items()})
    ident = {"run_id": C.RUN_ID, "protocol_digest": I.protocol_digest(),
             **PD.g1_weight_shas(),
             "sae_manifest_sha": sha256_file(C.SAE_CACHE_MANIFEST),
             "peer_manifest_sha": sha256_file(tmp_path / "peer_cache_manifest.json"),
             "norm_stats_sha": sha256_file(C.SAE_NORM_STATS)}
    for arm in C.ARMS:
        p = T.head_file(tmp_path, arm, 100, 1)
        I.save_head_checkpoint(build_head(100, 16), p,
                               {**ident, "arm": arm, "seed": 100, "epoch": 1})
        (tmp_path / f"best_{arm}_s100.json").write_text(json.dumps({
            "best_epoch": 1, "identity": ident,
            "epoch_sha256": {"1": sha256_file(p)}}))
    return {"identity": ident, "wide": wide, "path": tmp_path}
