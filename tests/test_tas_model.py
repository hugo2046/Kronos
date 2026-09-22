"""TAS1 模型行为测试（计划 §8 P1，全部用小型随机权重，不依赖 GPU/HF 下载）。

覆盖：状态无未来依赖、teacher-forcing 因果掩码、逐步↔批量对拍、PRE/POST
同状态、状态改变历史编码（PRE 变 / POST 不变）、冻结与梯度、S/q 不进
tokenizer.decode、baseline 旁路与原版逐位一致、显式因果 s2 与官方
training 路径逐位一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference
from tas_state.config import TASConfig
from tas_state.model import ConditionalKronos, StateEncoder

# 小型模型规格（结构同源官方，规模缩小；dropout=0 使 train/eval 无随机差异）
D_MODEL = 64
N_HEADS = 4
S1_BITS = 6
S2_BITS = 6


def _make_pair(seed: int = 0) -> tuple[Kronos, KronosTokenizer]:
    torch.manual_seed(seed)
    tok = KronosTokenizer(
        d_in=6, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=128,
        n_enc_layers=2, n_dec_layers=2, ffn_dropout_p=0.0,
        attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=S1_BITS, s2_bits=S2_BITS,
        beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
    )
    km = Kronos(
        s1_bits=S1_BITS, s2_bits=S2_BITS, n_layers=2, d_model=D_MODEL,
        n_heads=N_HEADS, ff_dim=128, ffn_dropout_p=0.0,
        attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=True,
    )
    return km, tok


@pytest.fixture()
def ck() -> ConditionalKronos:
    km, tok = _make_pair(0)
    torch.manual_seed(1)
    enc = StateEncoder(D_MODEL, hidden=64, n_states=4)
    return ConditionalKronos(km, tok, enc, TASConfig())


@pytest.fixture(scope="module")
def batch():
    """随机小 batch：x_norm [B,90,6]、stamp、未来标签 token。"""
    rng = np.random.default_rng(7)
    B = 4
    x = rng.normal(size=(B, 90, 6)).astype(np.float32)
    x = np.clip(x, -5, 5)
    idx = pd_range_stamp(90, B)
    y_stamp = pd_range_stamp(10, B)
    y_s1 = torch.randint(0, 2 ** S1_BITS, (B, 10))
    y_s2 = torch.randint(0, 2 ** S2_BITS, (B, 10))
    return (
        torch.tensor(x), idx, y_s1, y_s2, y_stamp,
    )


def pd_range_stamp(n: int, B: int) -> torch.Tensor:
    """合法范围时间特征（minute 0-59 / hour 0-23 / weekday 0-6 / day 1-31 / month 1-12）。"""
    rng = np.random.default_rng(3)
    minute = np.zeros((B, n), dtype=np.float32)
    hour = np.full((B, n), 15.0, dtype=np.float32)
    weekday = rng.integers(0, 7, (B, n)).astype(np.float32)
    day = rng.integers(1, 29, (B, n)).astype(np.float32)
    month = rng.integers(1, 13, (B, n)).astype(np.float32)
    return torch.tensor(np.stack([minute, hour, weekday, day, month], -1))


def _backbone(ck: ConditionalKronos, seq: torch.Tensor) -> torch.Tensor:
    h = seq
    for layer in ck.kronos.transformer:
        h = layer(h)
    return ck.kronos.norm(h)


def _hash_params(module: torch.nn.Module) -> str:
    import hashlib

    m = hashlib.sha256()
    for p in module.parameters():
        m.update(p.detach().cpu().numpy().tobytes())
    return m.hexdigest()


# ============================================================
# 1. 状态与第一步 logits 无未来依赖（计划 §8：test_future_mutation...）
# ============================================================
def test_future_mutation_does_not_change_state(ck, batch):
    x, stamp, y_s1, y_s2, y_stamp = batch
    S_a, _ = ck.encode_state(x, stamp)
    S_b, _ = ck.encode_state(x.clone(), stamp)
    assert torch.equal(S_a, S_b)

    def first_step_logits(y_st):
        # 第一步前缀为空：序列 [S][hist][q]，不含任何未来信息
        S, hist = ck.encode_state(x, stamp)
        q = ck.encoder.query.unsqueeze(0).expand(x.shape[0], -1)
        seq = ck.second_pass_input(
            hist, stamp, S=S, q=q, layout="pre"
        )
        return ck.kronos.head(_backbone(ck, seq))[:, -1, :]

    la = first_step_logits(y_stamp)
    lb = first_step_logits(torch.zeros_like(y_stamp))
    assert torch.equal(la, lb)


# ============================================================
# 2. teacher-forcing 因果掩码（计划 §8：test_teacher_forcing_future_mask）
# ============================================================
def test_teacher_forcing_future_mask(ck, batch):
    x, stamp, y_s1, y_s2, y_stamp = batch
    S, _ = ck.encode_state(x, stamp)
    kw = dict(S=S, layout="pre")
    out0 = ck.teacher_forced_forward(x, stamp, y_s1, y_s2, y_stamp, **kw)

    j = 5  # 第 j=5 日（0 基）及以后被替换
    y_s1_alt = y_s1.clone(); y_s1_alt[:, j:] = (y_s1[:, j:] + 7) % (2 ** S1_BITS)
    y_s2_alt = y_s2.clone(); y_s2_alt[:, j:] = (y_s2[:, j:] + 7) % (2 ** S2_BITS)
    out1 = ck.teacher_forced_forward(x, stamp, y_s1_alt, y_s2_alt, y_stamp, **kw)

    # 第 j 日粗 logits 不变（第 j 日读出位置的输入只到 y_{j-1}）
    assert torch.allclose(out0.s1_logits[:, j], out1.s1_logits[:, j], atol=1e-6)
    # 第 j 日之前的粗 logits 也不变；第 j 日之后被输入前缀改变
    assert torch.allclose(out0.s1_logits[:, :j], out1.s1_logits[:, :j], atol=1e-6)
    assert not torch.allclose(out0.s1_logits[:, j + 1], out1.s1_logits[:, j + 1], atol=1e-6)

    # 改第 j 日粗标签 → 该日细 logits 变（sibling 条件），更早位置不变
    y_s1_only = y_s1.clone(); y_s1_only[:, j] = (y_s1[:, j] + 3) % (2 ** S1_BITS)
    out2 = ck.teacher_forced_forward(x, stamp, y_s1_only, y_s2, y_stamp, **kw)
    assert not torch.allclose(out0.s2_logits[:, j], out2.s2_logits[:, j], atol=1e-6)
    assert torch.allclose(out0.s2_logits[:, :j], out2.s2_logits[:, :j], atol=1e-6)
    # y_j 同时是第 j+1 日的 teacher 输入：其后的粗 logits 允许变化（因果方向）
    assert torch.allclose(out0.s1_logits[:, : j + 1], out2.s1_logits[:, : j + 1], atol=1e-6)
    assert not torch.allclose(out0.s1_logits[:, j + 1], out2.s1_logits[:, j + 1], atol=1e-6)


# ============================================================
# 3. 逐步 teacher forcing ↔ 批量因果实现 logits 对拍（防粗细错位/泄漏）
# ============================================================
def test_stepwise_matches_batched_logits(ck, batch):
    x, stamp, y_s1, y_s2, y_stamp = batch
    S, hist_tokens = ck.encode_state(x, stamp)
    B = x.shape[0]
    q = ck.encoder.query.unsqueeze(0).expand(B, -1)

    s1_steps, s2_steps = [], []
    for k in range(10):
        seq = ck.second_pass_input(
            hist_tokens, stamp, S=S, q=q, layout="pre",
            prefix_s1=y_s1[:, :k] if k else None,
            prefix_s2=y_s2[:, :k] if k else None,
            prefix_stamp=y_stamp[:, :k] if k else None,
        )
        h = _backbone(ck, seq)
        s1_steps.append(ck.kronos.head(h)[:, -1, :])
        pos = torch.tensor([h.shape[1] - 1], device=h.device)
        s2_steps.append(
            ck.causal_s2_logits(h, pos, y_s1[:, k : k + 1]).squeeze(1)
        )
    s1_stepwise = torch.stack(s1_steps, dim=1)
    s2_stepwise = torch.stack(s2_steps, dim=1)

    out = ck.teacher_forced_forward(x, stamp, y_s1, y_s2, y_stamp, S=S, layout="pre")
    assert torch.allclose(s1_stepwise, out.s1_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(s2_stepwise, out.s2_logits, atol=1e-5, rtol=1e-5)


# ============================================================
# 4. PRE/POST 同状态、同 q、同采样流（计划 §8：test_pre_post_same_state）
# ============================================================
def test_pre_post_same_state(ck, batch):
    x, stamp, y_s1, y_s2, y_stamp = batch
    S, hist_tokens = ck.encode_state(x, stamp)
    B = x.shape[0]
    q = ck.encoder.query.unsqueeze(0).expand(B, -1)

    for layout in ("pre", "post"):
        seq = ck.second_pass_input(
            hist_tokens, stamp, S=S, q=q, layout=layout
        )
        assert seq.shape == (B, 95, D_MODEL)  # 4 + 90 + 1
    # q 读出位置两布局一致（94）
    pos = ck.query_pred_positions(10, x.device)
    assert pos[0].item() == 94 and pos[-1].item() == 103

    # 同 seed 下 PRE / POST 各自的生成确定性（同布局两次调用同结果）
    torch.manual_seed(11)
    g1s, g1f = ck.generate(x, stamp, y_stamp, S=S, layout="pre", sample_count=2)
    torch.manual_seed(11)
    g2s, g2f = ck.generate(x, stamp, y_stamp, S=S, layout="pre", sample_count=2)
    assert torch.equal(g1s, g2s) and torch.equal(g1f, g2f)


# ============================================================
# 5. 状态改变历史编码：PRE 变 / POST 不变（计划 §3.3 机制前提）
# ============================================================
def test_state_changes_historical_encoding(ck, batch):
    x, stamp, *_ = batch
    _, hist_tokens = ck.first_pass(x, stamp)
    B = x.shape[0]
    q = ck.encoder.query.unsqueeze(0).expand(B, -1)
    S_a = torch.randn(B, 4, D_MODEL)
    S_b = S_a + 1.0

    h_pre_a = _backbone(ck, ck.second_pass_input(hist_tokens, stamp, S=S_a, q=q, layout="pre"))
    h_pre_b = _backbone(ck, ck.second_pass_input(hist_tokens, stamp, S=S_b, q=q, layout="pre"))
    # PRE：历史位置 4..93 随 S 改变
    assert not torch.allclose(h_pre_a[:, 4:94], h_pre_b[:, 4:94], atol=1e-6)

    h_post_a = _backbone(ck, ck.second_pass_input(hist_tokens, stamp, S=S_a, q=q, layout="post"))
    h_post_b = _backbone(ck, ck.second_pass_input(hist_tokens, stamp, S=S_b, q=q, layout="post"))
    # POST：历史位置 0..89 因果地看不到后置 S，不变
    assert torch.allclose(h_post_a[:, :90], h_post_b[:, :90], atol=1e-6)


# ============================================================
# 6. 冻结与梯度（计划 §8：test_frozen_weights_and_adapter_gradient）
# ============================================================
def test_frozen_weights_and_adapter_gradient(ck, batch):
    x, stamp, y_s1, y_s2, y_stamp = batch
    ck.train()
    assert not ck.kronos.training  # 底座恒 eval（dropout=0.2 不得打开）
    base_hash = _hash_params(ck.kronos)

    out = ck.teacher_forced_forward(x, stamp, y_s1, y_s2, y_stamp, layout="pre")
    loss = F.cross_entropy(
        out.s1_logits.reshape(-1, 2 ** S1_BITS), y_s1.reshape(-1)
    ) + F.cross_entropy(
        out.s2_logits.reshape(-1, 2 ** S2_BITS), y_s2.reshape(-1)
    )
    loss.backward()

    for p in ck.kronos.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
    grads = [p.grad for p in ck.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.any(g != 0) for g in grads)

    opt = torch.optim.AdamW(ck.encoder.parameters(), lr=3e-4)
    opt.step()
    assert _hash_params(ck.kronos) == base_hash
    ck.eval()


# ============================================================
# 7. S/q 永不进入 tokenizer.decode（计划 §8：test_no_soft_tokens_in_decode）
# ============================================================
def test_no_soft_tokens_in_decode(ck, batch, monkeypatch):
    x, stamp, y_s1, y_s2, y_stamp = batch
    calls: list = []
    orig = ck.tokenizer.decode

    def spy(x_ids, half=False):
        calls.append(x_ids)
        return orig(x_ids, half=half)

    monkeypatch.setattr(ck.tokenizer, "decode", spy)
    torch.manual_seed(5)
    S, hist_tokens = ck.encode_state(x, stamp)
    gen_s1, gen_s2 = ck.generate(x, stamp, y_stamp, S=S, layout="pre", sample_count=2)
    means = torch.zeros(4, 6)
    stds = torch.ones(4, 6)
    ck.decode_tokens(hist_tokens, gen_s1, gen_s2, means, stds)

    assert len(calls) >= 1
    for arg in calls:
        s1_ids, s2_ids = arg
        # 只允许 90 历史 + 10 生成 = 100（S/q 不在序列里）
        assert s1_ids.shape[-1] == 100, f"decode 序列长度 {s1_ids.shape}"
        assert s1_ids.min() >= 0 and s1_ids.max() < 2 ** S1_BITS
        assert s2_ids.min() >= 0 and s2_ids.max() < 2 ** S2_BITS


# ============================================================
# 8. baseline 旁路与原版逐位一致（P0 对拍的玩具版）
# ============================================================
def test_baseline_bypass_matches_official_decode(ck, batch):
    x, stamp, *_ = batch
    s1_ids, s2_ids = ck.tokenizer.encode(x, half=True)

    seq = ck.second_pass_input((s1_ids, s2_ids), stamp, S=None, q=None, layout="baseline")
    logits_bypass = ck.kronos.head(_backbone(ck, seq))

    logits_official, _ = ck.kronos.decode_s1(s1_ids, s2_ids, stamp)
    assert torch.equal(logits_bypass, logits_official)


def test_baseline_generate_matches_auto_regressive(ck, batch):
    x, stamp, y_s1, y_s2, y_stamp = batch
    ck.eval()
    torch.manual_seed(21)
    g_s1, g_s2 = ck.generate(
        x, stamp, y_stamp, layout="baseline", sample_count=3,
        temperature=1.0, top_k=0, top_p=0.9,
    )
    # 原版 auto_regressive_inference（同 seed）
    torch.manual_seed(21)
    x_rep = x.unsqueeze(1).repeat(1, 3, 1, 1).reshape(-1, 90, 6)
    st_rep = stamp.unsqueeze(1).repeat(1, 3, 1, 1).reshape(-1, 90, 5)
    ys_rep = y_stamp.unsqueeze(1).repeat(1, 3, 1, 1).reshape(-1, 10, 5)
    # 捕获原版逐步采样的粗细 token：patch sample_from_logits 记录调用序列
    from model import kronos as kronos_mod

    recorded: list = []
    orig_sample = kronos_mod.sample_from_logits

    def spy_sample(logits, **kw):
        out = orig_sample(logits, **kw)
        recorded.append(out.detach().clone())
        return out

    kronos_mod.sample_from_logits = spy_sample
    try:
        auto_regressive_inference(
            ck.tokenizer, ck.kronos, x_rep, st_rep, ys_rep,
            max_context=512, pred_len=10, clip=5, T=1.0, top_k=0,
            top_p=0.9, sample_count=1, verbose=False,
        )
    finally:
        kronos_mod.sample_from_logits = orig_sample

    # 原版每步两次采样（s1、s2）→ 偶数次 s1、奇数次 s2
    ar_s1 = torch.cat([recorded[2 * i] for i in range(10)], dim=1)
    ar_s2 = torch.cat([recorded[2 * i + 1] for i in range(10)], dim=1)
    assert torch.equal(g_s1.reshape(-1, 10), ar_s1)
    assert torch.equal(g_s2.reshape(-1, 10), ar_s2)


# ============================================================
# 9. 显式因果 s2 与官方 training 路径逐位一致（wrapper 正确性锁）
# ============================================================
def test_causal_s2_matches_official_training(ck):
    torch.manual_seed(9)
    B, L = 3, 37
    context = torch.randn(B, L, D_MODEL)
    s1_ids = torch.randint(0, 2 ** S1_BITS, (B, L))

    ck.kronos.train()  # 官方 is_causal=self.training=True 路径
    try:
        official = ck.kronos.decode_s2(context, s1_ids)
    finally:
        ck.kronos.eval()
    wrapper = ck.causal_s2_logits(context, torch.arange(L), s1_ids)
    assert torch.equal(official, wrapper)


# ============================================================
# 10. 参数量公式（计划 §3.2：202d+64，最终以 numel 实测）
# ============================================================
def test_state_encoder_parameter_count():
    enc = StateEncoder(832, hidden=64, n_states=4)
    assert enc.trainable_parameter_count() == 202 * 832 + 64
    enc_small = StateEncoder(D_MODEL, hidden=64, n_states=4)
    assert enc_small.trainable_parameter_count() == 202 * D_MODEL + 64


# ============================================================
# 11. 底座 dropout 非 0 时 train() 仍强制底座 eval（官方 Kronos-base
#     ffn/resid dropout=0.2，训练模式下底座随机性必须关闭）
# ============================================================
def test_train_mode_keeps_base_eval():
    torch.manual_seed(4)
    tok = KronosTokenizer(
        d_in=6, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=128,
        n_enc_layers=2, n_dec_layers=2, ffn_dropout_p=0.0,
        attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=S1_BITS, s2_bits=S2_BITS,
        beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
    )
    km = Kronos(
        s1_bits=S1_BITS, s2_bits=S2_BITS, n_layers=2, d_model=D_MODEL,
        n_heads=N_HEADS, ff_dim=128, ffn_dropout_p=0.2,
        attn_dropout_p=0.0, resid_dropout_p=0.2, token_dropout_p=0.0,
        learn_te=True,
    )
    enc = StateEncoder(D_MODEL, hidden=64, n_states=4)
    c = ConditionalKronos(km, tok, enc, TASConfig())
    c.train()
    assert c.training
    assert not c.kronos.training
    assert not c.tokenizer.training
    assert c.encoder.training
    # 前向两次数值一致（底座无随机性）
    x = torch.randn(2, 90, 6)
    st = pd_range_stamp(90, 2)
    Sa, _ = c.encode_state(x, st)
    Sb, _ = c.encode_state(x, st)
    assert torch.equal(Sa, Sb)
