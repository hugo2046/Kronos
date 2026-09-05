"""任务2（方案 §7）：多期限头与冻结提取的离线模型测试（stub G1，无 HF/GPU）。

覆盖：
- 输入 [4,90,832]（stub 用小维度）→ 输出 [4,10]；
- 批顺序置换后输出按同序置换（批内独立）；
- future_stamp 合法范围校验（越界显式报错）；
- 修改单条样本不改变其他样本输出；
- 冻结提取：同输入两次输出逐位一致；参数全部 requires_grad=False。
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dhead_distill.head import MultiHorizonHead


def _make_head(d_model: int = 32, n_horizons: int = 10, head_dim: int = 16) -> MultiHorizonHead:
    return MultiHorizonHead(
        d_model=d_model, head_dim=head_dim, n_heads=4, n_horizons=n_horizons,
        calendar_cardinalities=(60, 24, 7, 32, 13),
    )


def _stamp(B: int, H: int = 10) -> torch.Tensor:
    """合法 future_stamp：[B,H,5]，各列在基数内（minute=0/hour=0 的日频口径）。"""
    torch.manual_seed(0)
    s = torch.zeros(B, H, 5, dtype=torch.long)
    s[..., 0] = 0                       # minute
    s[..., 1] = 0                       # hour
    s[..., 2] = torch.randint(0, 7, (B, H))
    s[..., 3] = torch.randint(1, 32, (B, H))
    s[..., 4] = torch.randint(1, 13, (B, H))
    return s


def test_head_output_shape() -> None:
    """[4,90,hidden] + [4,10,5] → [4,10]。"""
    head = _make_head(d_model=32)
    hidden = torch.randn(4, 90, 32)
    out = head(hidden, _stamp(4))
    assert out.shape == (4, 10)


def test_head_batch_permutation_equivariant() -> None:
    """批顺序置换后输出按同样方式置换（评估期批独立性）。"""
    head = _make_head(d_model=32)
    head.eval()
    hidden = torch.randn(8, 90, 32)
    stamp = _stamp(8)
    with torch.no_grad():
        out = head(hidden, stamp)
        perm = torch.randperm(8)
        out_p = head(hidden[perm], stamp[perm])
    torch.testing.assert_close(out_p, out[perm])


def test_head_sample_independence() -> None:
    """修改某条样本的输入不改变其他样本输出。"""
    head = _make_head(d_model=32)
    head.eval()
    hidden = torch.randn(4, 90, 32)
    stamp = _stamp(4)
    with torch.no_grad():
        out1 = head(hidden, stamp)
        hidden2 = hidden.clone()
        hidden2[2] += 10.0  # 只改第 2 条
        out2 = head(hidden2, stamp)
    torch.testing.assert_close(out2[[0, 1, 3]], out1[[0, 1, 3]])
    assert not torch.allclose(out2[2], out1[2])


def test_head_future_stamp_bounds() -> None:
    """future_stamp 越界（≥基数或负）显式报错。"""
    head = _make_head(d_model=32)
    hidden = torch.randn(2, 90, 32)
    bad = _stamp(2)
    bad[0, 0, 2] = 7      # weekday 基数 7，越界
    with pytest.raises(ValueError):
        head(hidden, bad)
    bad2 = _stamp(2)
    bad2[0, 0, 4] = -1    # 负值
    with pytest.raises(ValueError):
        head(hidden, bad2)


def test_head_trainable_params_recorded() -> None:
    """记录实测可训练参数数目（不依赖估计文字，§4.1）。"""
    head = _make_head(d_model=832, head_dim=128)
    n = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert n > 0
    info = head.parameter_summary()
    assert info["trainable_params"] == n
    assert info["d_model"] == 832 and info["n_horizons"] == 10


# ---------------------------------------------------------------- 冻结提取 ---

class _StubTokenizer(nn.Module):
    """离线 stub：encode(x, half=True) → (s1_ids, s2_ids)。"""

    def __init__(self, vocab_s1: int = 16, vocab_s2: int = 8):
        super().__init__()
        self.vocab_s1, self.vocab_s2 = vocab_s1, vocab_s2

    def encode(self, x, half=False):
        base = (x.sum(-1) * 1000).long()  # [B,T]
        s1 = (base * 3) % self.vocab_s1
        s2 = (base * 5) % self.vocab_s2
        return s1, s2


class _StubHierarchicalEmbedding(nn.Module):
    """接受 (s1_ids, s2_ids) 列表的 stub 嵌入（对齐 HierarchicalEmbedding 接口）。"""

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.emb_s1 = nn.Embedding(16, d_model)
        self.emb_s2 = nn.Embedding(8, d_model)

    def forward(self, token_ids):
        s1_ids, s2_ids = token_ids
        return self.emb_s1(s1_ids.long()) + self.emb_s2(s2_ids.long())


class _StubKronos(nn.Module):
    """复刻 Kronos 主干关键件（embedding/time_emb/transformer/norm）的小 stub。"""

    def __init__(self, d_model: int = 32, n_layers: int = 4):
        super().__init__()
        from model.module import RMSNorm, TransformerBlock

        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = _StubHierarchicalEmbedding(d_model)
        self.time_emb = nn.Linear(5, d_model)
        self.token_drop = nn.Dropout(0.0)
        self.transformer = nn.ModuleList(
            [TransformerBlock(d_model, 4, ff_dim=64) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)


def _make_stub_backbone():
    from dhead_distill.backbone import StudentBackbone

    tok, kro = _StubTokenizer(), _StubKronos(d_model=32, n_layers=4)
    bb = StudentBackbone(tokenizer=tok, kronos=kro, d_model_expected=32)
    return bb, tok, kro


def test_frozen_extract_deterministic_and_frozen() -> None:
    """冻结提取：两次输出逐位一致；全部参数 requires_grad=False。"""
    bb, tok, kro = _make_stub_backbone()
    x = torch.randn(3, 90, 6)
    stamp = torch.zeros(3, 90, 5)
    h1 = bb.extract(x, stamp)
    h2 = bb.extract(x, stamp)
    assert h1.shape == (3, 90, 32)
    torch.testing.assert_close(h1, h2)
    for p in bb.parameters():
        assert not p.requires_grad
    # eval 语义：train() 调用后主干仍 eval（token_drop 不生效）
    bb.train()
    assert not tok.training and not kro.training


def test_dmodel_mismatch_rejected() -> None:
    """主干 d_model 与配置不符时构造报错（§4.1 校验）。"""
    from dhead_distill.backbone import StudentBackbone

    with pytest.raises(ValueError):
        StudentBackbone(
            tokenizer=_StubTokenizer(), kronos=_StubKronos(d_model=32),
            d_model_expected=832,
        )
