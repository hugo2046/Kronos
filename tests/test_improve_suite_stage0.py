"""improve_suite 阶段 0 单测（计划 §2）。

两条覆盖（均不触发 qlib / GPU，纯张量 / DataFrame）：

    1. ``test_paths_mean_bitmatch``：构造随机小张量 + 确定性 fake tokenizer/model，
       断言 ``auto_regressive_inference_paths`` 返回的均值与原
       ``auto_regressive_inference`` **逐位一致**（同 seed；numpy 逐位相等），
       且均值 == 路径张量沿 sample_count 维的 mean。
    2. ``test_path_store_roundtrip``：逐路径长表写入 → 读回，断言 shape 与数值一致。

这两个测试**先于实现**写成：模块未建时 import 失败 → FAIL；实现后 → PASS。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 确定性 fake tokenizer / model（仅供 bitmatch 测试，不依赖 GPU / HF 权重）
# ============================================================


class _FakeTokenizer:
    """最小确定性 tokenizer：encode→两路 token id，decode→float 张量。

    行为完全由输入决定（无内部 RNG），保证两条推理函数的唯一随机源是
    ``sample_from_logits``（torch.multinomial），从而同 seed → 同样本 → 均值逐位相等。
    """

    def __init__(self, vocab_size: int = 16, out_feat: int = 6):
        self.vocab_size = vocab_size
        self.out_feat = out_feat

    def to(self, device):
        return self

    def encode(self, x, half=True):
        # x: (B, seq, feat) float → 两路 token-id 张量（s1, s2）
        tok = (x.sum(-1).abs() * 1000).long() % self.vocab_size
        return [tok, tok.clone()]

    def decode(self, input_tokens, half=True):
        pre, post = input_tokens  # 各 (B, seq) long
        base = pre.float().unsqueeze(-1) + post.float().unsqueeze(-1) * 0.5
        return base.repeat(1, 1, self.out_feat)


class _FakeModel:
    """最小确定性 model：decode_s1/decode_s2 返回 logits（embed 内积），无内部采样。"""

    def __init__(self, vocab_size: int = 16, d_model: int = 8):
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        torch.nn.init.normal_(self.embed.weight, mean=0.0, std=0.3)

    def to(self, device):
        return self

    def decode_s1(self, pre, post, stamp):
        emb = self.embed(pre.long().clamp(0, self.vocab_size - 1))  # (B, seq, d)
        logits = emb @ self.embed.weight.T  # (B, seq, vocab)
        return logits, emb

    def decode_s2(self, context, sample_pre):
        s = self.embed(sample_pre.long().squeeze(-1).clamp(0, self.vocab_size - 1))  # (B, d)
        logits = (context + s.unsqueeze(1)) @ self.embed.weight.T  # (B, seq, vocab)
        return logits


# ============================================================
# §2.0.2 test_paths_mean_bitmatch
# ============================================================


class TestPathsMeanBitmatch:
    """路径版均值必须与原函数逐位一致（计划 §2.0.2）。

    ``auto_regressive_inference_paths`` 与原 ``auto_regressive_inference`` 的唯一差异
    在出口：路径版额外返回逐路径张量。RNG 调用序逐字不动 → 同 seed 下样本逐位相同。
    """

    def test_mean_bitmatches_original(self) -> None:
        from model.kronos import auto_regressive_inference
        from improve_suite.path_inference import auto_regressive_inference_paths

        torch.manual_seed(0)
        tok = _FakeTokenizer()
        model = _FakeModel()

        bs, seq_len, feat = 3, 12, 6
        pred_len = 5
        sample_count = 4
        max_context = 512
        rng = np.random.default_rng(7)
        x = torch.from_numpy(rng.standard_normal((bs, seq_len, feat)).astype(np.float32))
        x_stamp = torch.from_numpy(rng.standard_normal((bs, seq_len, 4)).astype(np.float32))
        y_stamp = torch.from_numpy(rng.standard_normal((bs, pred_len, 4)).astype(np.float32))

        # 原函数
        torch.manual_seed(42)
        mean_orig = auto_regressive_inference(
            tok, model, x, x_stamp, y_stamp, max_context, pred_len,
            clip=5, T=1.0, top_k=0, top_p=0.9, sample_count=sample_count, verbose=False,
        )

        # 路径版（同 seed 重置 → 同样本）
        torch.manual_seed(42)
        mean_paths, paths = auto_regressive_inference_paths(
            tok, model, x, x_stamp, y_stamp, max_context, pred_len,
            clip=5, T=1.0, top_k=0, top_p=0.9, sample_count=sample_count, verbose=False,
        )

        # 核心断言：均值逐位一致
        assert mean_orig.shape == mean_paths.shape, (
            f"shape 不一致：orig {mean_orig.shape} vs paths {mean_paths.shape}"
        )
        assert np.array_equal(mean_orig, mean_paths), (
            "均值未逐位一致：max|Δ|="
            f"{np.max(np.abs(mean_orig - mean_paths)):.3e}"
        )

    def test_paths_shape_and_mean_consistency(self) -> None:
        """路径张量形状 = (bs, sample_count, total_seq, d)；其 mean 等于返回的均值。"""
        from improve_suite.path_inference import auto_regressive_inference_paths

        tok = _FakeTokenizer()
        model = _FakeModel()

        bs, seq_len, feat = 2, 10, 6
        pred_len = 4
        sample_count = 3
        rng = np.random.default_rng(11)
        x = torch.from_numpy(rng.standard_normal((bs, seq_len, feat)).astype(np.float32))
        x_stamp = torch.from_numpy(rng.standard_normal((bs, seq_len, 4)).astype(np.float32))
        y_stamp = torch.from_numpy(rng.standard_normal((bs, pred_len, 4)).astype(np.float32))

        torch.manual_seed(42)
        mean_paths, paths = auto_regressive_inference_paths(
            tok, model, x, x_stamp, y_stamp, 512, pred_len,
            clip=5, T=1.0, top_k=0, top_p=0.9, sample_count=sample_count, verbose=False,
        )

        # 路径张量形状：(bs, sample_count, seq_len+pred_len, feat)
        assert paths.shape == (bs, sample_count, seq_len + pred_len, feat), (
            f"路径形状错：{paths.shape}，期望 {(bs, sample_count, seq_len + pred_len, feat)}"
        )
        # 均值 == 路径沿 sample_count 维 mean
        recomputed = np.mean(paths, axis=1)
        assert np.array_equal(recomputed, mean_paths), (
            "返回均值与路径张量的 mean 不一致："
            f"max|Δ|={np.max(np.abs(recomputed - mean_paths)):.3e}"
        )


# ============================================================
# §2.0.3 test_path_store_roundtrip
# ============================================================


class TestPathStoreRoundtrip:
    """逐路径长表写入 → 读回，shape 与数值一致（计划 §2.0.3）。"""

    def test_roundtrip_preserves_values(self, tmp_path: Path) -> None:
        from improve_suite.path_store import read_paths, write_paths

        dates = pd.date_range("2024-07-01", periods=3, freq="B")
        codes = ["000001.SZ", "000002.SZ", "600000.SH"]
        n_paths = 4
        steps = 5
        rng = np.random.default_rng(3)
        # 构造逐路径 pred_close：(date, code, path_id, step) → pred_close
        rows = []
        for d in dates:
            for c in codes:
                for p in range(n_paths):
                    for s in range(steps):
                        rows.append(
                            {
                                "date": d,
                                "code": c,
                                "path_id": p,
                                "step": s,
                                "pred_close": float(rng.uniform(5, 50)),
                            }
                        )
        df = pd.DataFrame(rows)

        out = tmp_path / "paths_test.parquet"
        write_paths(df, out)
        assert out.exists(), "落盘文件不存在"

        back = read_paths(out)
        # 列一致（数值列）
        assert set(["date", "code", "path_id", "step", "pred_close"]).issubset(back.columns)
        # 行数一致
        assert len(back) == len(df), f"行数不一致：{len(back)} vs {len(df)}"
        # 数值一致（按相同排序键比对；store 落盘 float32，故比对到 float32 精度）
        keys = ["date", "code", "path_id", "step"]
        a = df.sort_values(keys).reset_index(drop=True)
        b = back.sort_values(keys).reset_index(drop=True)
        assert np.allclose(
            a["pred_close"].values, b["pred_close"].values, rtol=0, atol=1e-5
        ), "pred_close 数值不一致（超出 float32 精度）"
