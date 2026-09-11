"""PATH1 收尾门禁：只使用合成文件，不读取实验标签或真实权重。"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from path_information import config as C
from path_information import evaluate as E
from path_information import model as M
from path_information import paths as P
from path_information import run as R
from path_information import train as T


@pytest.fixture
def path_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ART_DIR", tmp_path)
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(C, "HEADS_DIR", tmp_path / "heads")
    C.HEADS_DIR.mkdir()
    weights = {"tokenizer_sha": "tok", "predictor_sha": "pred"}
    monkeypatch.setattr(P, "g1_weight_shas", lambda: weights.copy())
    (tmp_path / "preflight_manifest.json").write_text(json.dumps({"g1_weights": weights}))
    stats = {"mu_g1": 0., "sigma_g1": 1., "sigma_e": 1.}
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    chunks = []
    for split, (date, _) in C.SEGMENTS.items():
        name = f"{split}_{date.replace('-', '')}.npz"
        p = P.write_path_chunk(C.CACHE_DIR / split, name,
            {"dates": np.array([date]), "instruments": np.array(["a"]),
             "s_g1": np.array([0.1]), "close_t": np.array([10.]),
             "close_paths": np.ones((1, 20, 10)) * 10.},
            {"protocol": C.PROTOCOL_VERSION, "split": split})
        chunks.append({"split": split, "file": f"{split}/{name}",
                       "sha256": P.sha256_file(p)})
    (tmp_path / "cache_manifest.json").write_text(json.dumps({"chunks": chunks}))
    return tmp_path, stats


def _scores(root):
    files = {}
    for arm in C.ARMS:
        for split, (date, _) in C.SEGMENTS.items():
            name = f"scores_{arm}_{split}.npz"
            p = P.write_path_chunk(root, name,
                {"dates": np.array([date]), "instruments": np.array(["a"]),
                 "s_final": np.array([0.1])},
                {"protocol": C.PROTOCOL_VERSION, "arm": arm, "split": split,
                 "kind": "scores", "seed": C.SEED})
            files[f"{arm}:{split}"] = {"file": name, "sha256": P.sha256_file(p)}
    return files


def ready_scores(runtime):
    """真实新上下文、小头与六份分数组成可通过生产校验的合成档案。"""
    root, _ = runtime
    context = T.current_context(root)
    heads = {}
    for arm in C.ARMS:
        p = T.head_file(C.HEADS_DIR, arm, C.SEED, 0)
        torch.save({"state_dict": M.PathHead().state_dict(), "context": context,
                    "meta": {"protocol": C.PROTOCOL_VERSION, "arm": arm,
                             "seed": C.SEED, "epoch": 0}}, p)
        sha = P.sha256_file(p)
        (C.HEADS_DIR / f"best_{arm}_s{C.SEED}.json").write_text(json.dumps({
            "best_epoch": 0, "best_head_file": p.name,
            "best_head_sha256": sha, "context": context}))
        heads[arm] = {"file": p.name, "sha256": sha, "epoch": 0}
    manifest = {"files": _scores(root), "context": context, "heads": heads}
    (root / "scores_manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("variant", ["empty", "missing", "extra", "alias"])
def test_incomplete_manifest_never_loads_labels(tmp_path, variant):
    files = _scores(tmp_path)
    if variant == "empty":
        files = {}
    elif variant == "missing":
        files.pop("PATH:dev_eval")
    elif variant == "extra":
        files["OTHER:dev_eval"] = files["PATH:dev_eval"]
    else:
        files["PATH:dev_eval"] = files["MEAN:dev_eval"]
    (tmp_path / "scores_manifest.json").write_text(json.dumps({"files": files}))
    calls = []
    with pytest.raises(RuntimeError):
        E.unseal(tmp_path, label_loader=lambda: calls.append(1))
    assert calls == []


def test_score_entry_must_pass_current_context(path_runtime, monkeypatch):
    root, stats = path_runtime
    def check_load(*args, **kwargs):
        assert kwargs.get("expect_context"), "实际打分入口没有传期望上下文"
        assert kwargs["expect_context"]["stats_sha256"] == P.sha256_file(root / "norm_stats.json")
        return M.PathHead(), {}
    monkeypatch.setattr(T, "load_best", check_load)
    R._score_segment("PATH", "dev_eval", stats)


def test_load_best_cannot_check_one_file_then_load_another(tmp_path):
    head = M.PathHead()
    actual = T.head_file(tmp_path, "PATH", 100, 4)
    torch.save({"state_dict": head.state_dict(), "meta": {
        "arm": "PATH", "seed": 100, "epoch": 4, "protocol": C.PROTOCOL_VERSION}}, actual)
    decoy = tmp_path / "decoy.pt"
    decoy.write_bytes(b"this is not the loaded checkpoint")
    (tmp_path / "best_PATH_s100.json").write_text(json.dumps({
        "best_epoch": 4, "best_head_file": decoy.name,
        "best_head_sha256": P.sha256_file(decoy), "context": {}}))
    with pytest.raises(RuntimeError):
        T.load_best(tmp_path, "PATH", 100, expect_context={})


def test_evaluate_refuses_existing_scores_before_any_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ART_DIR", tmp_path)
    (tmp_path / "norm_stats.json").write_text("{}")
    marker = tmp_path / "scores_manifest.json"
    marker.write_text('{"files": {}}')
    monkeypatch.setattr(R, "_score_segment", lambda *a: pytest.fail("不应重生分数"))
    with pytest.raises(RuntimeError):
        R.cmd_evaluate()
    assert marker.read_text() == '{"files": {}}'


@pytest.mark.parametrize("changed", ["stats", "cache", "head", "protocol"])
def test_full_manifest_passes_then_identity_change_blocks_labels(path_runtime, changed):
    root, _ = path_runtime
    ready_scores(path_runtime)
    calls = []
    E.unseal(root, label_loader=lambda: calls.append(1))
    if changed == "stats":
        (root / "norm_stats.json").write_text('{"different": true}')
    elif changed == "cache":
        p = next((root / "cache" / "train").glob("*.npz"))
        p.write_bytes(b"changed")
    elif changed == "head":
        T.head_file(C.HEADS_DIR, "PATH", C.SEED, 0).write_bytes(b"changed")
    else:
        p = root / "scores_manifest.json"
        m = json.loads(p.read_text())
        m["context"]["protocol"] = "wrong"
        p.write_text(json.dumps(m))
    with pytest.raises(RuntimeError):
        E.unseal(root, label_loader=lambda: calls.append(1))
    assert calls == [1]


def test_path_writer_refuses_to_replace_existing_artifact(tmp_path):
    p = P.write_path_chunk(tmp_path, "frozen.npz", {"value": np.array([1.])}, {})
    original = p.read_bytes()
    with pytest.raises(RuntimeError):
        P.write_path_chunk(tmp_path, "frozen.npz", {"value": np.array([2.])}, {})
    assert p.read_bytes() == original
