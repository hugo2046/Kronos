"""任务4（方案 §7）：末两层 q/v LoRA 注入与梯度隔离测试（stub 主干，无 GPU）。

覆盖方案 §5：LoRA 全零时学生隐藏输出与原 G1 对拍逐位一致；反传只更新
指定权重；教师权重不受训练影响。
"""
from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn as nn

from tests.test_dhead_model import _StubKronos, _StubTokenizer


def _hash_params(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _make_student():
    from dhead_distill.backbone import StudentBackbone

    torch.manual_seed(7)
    tok, kro = _StubTokenizer(), _StubKronos(d_model=32, n_layers=4)
    return StudentBackbone(tok, kro, d_model_expected=32, n_trainable_layers=2)


def test_lora_zero_init_forward_identical() -> None:
    """LoRA 全零（B=0）时：学生隐藏输出与冻结路径逐位一致。"""
    from dhead_distill.backbone import inject_lora

    bb = _make_student()
    x = torch.randn(2, 90, 6)
    stamp = torch.zeros(2, 90, 5)
    ref = bb.extract(x, stamp)

    lora_params = inject_lora(bb, rank=8, alpha=8, dropout=0.0,
                              targets=("q_proj", "v_proj"))
    assert len(lora_params) > 0
    out = bb.extract_trainable(x, stamp)
    torch.testing.assert_close(out, ref)

    # 冻结路径不受注入影响
    torch.testing.assert_close(bb.extract(x, stamp), ref)


def test_lora_backward_updates_only_targets() -> None:
    """反传+一步更新后：LoRA 参数有非零梯度；原始权重逐字节不变。"""
    from dhead_distill.backbone import inject_lora

    bb = _make_student()
    lora_named = inject_lora(bb, rank=8, alpha=8, dropout=0.0,
                             targets=("q_proj", "v_proj"))
    # 注入后快照：base.* 为原始权重，lora_* 为适配参数
    before = {k: v.clone() for k, v in bb.kronos.state_dict().items()}

    x = torch.randn(2, 90, 6)
    stamp = torch.zeros(2, 90, 5)
    h = bb.extract_trainable(x, stamp)
    loss = (h * torch.randn_like(h)).sum()
    loss.backward()

    grads = [p.grad for _, p in lora_named if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads), "LoRA 至少一个参数须有非零梯度"

    opt = torch.optim.AdamW([p for _, p in lora_named], lr=1e-3)
    opt.step()

    changed = [
        k for k in before
        if not torch.equal(before[k], bb.kronos.state_dict()[k])
    ]
    assert changed, "一步更新后 LoRA 参数必须变化"
    for k in changed:
        assert "lora_" in k, f"非 LoRA 权重被改动：{k}"
    for k, v in before.items():
        if "lora_" not in k:
            assert torch.equal(v, bb.kronos.state_dict()[k]), \
                f"原始权重 {k} 被改动（教师侧必须逐字节不变）"


def test_lora_wraps_only_last_two_layers_qv() -> None:
    """注入点仅末 2 层 self_attn 的 q_proj/v_proj（k/o 与其它层不注入）。"""
    from dhead_distill.backbone import LoRALinear, inject_lora

    bb = _make_student()
    inject_lora(bb, rank=8, alpha=8, dropout=0.0, targets=("q_proj", "v_proj"))
    n_layers = len(bb.kronos.transformer)
    for i, block in enumerate(bb.kronos.transformer):
        attn = block.self_attn
        is_last2 = i >= n_layers - 2
        for name in ("q_proj", "v_proj"):
            mod = getattr(attn, name)
            assert isinstance(mod, LoRALinear) == is_last2, \
                f"层 {i} {name} 注入状态错误"
        assert not isinstance(attn.k_proj, LoRALinear)
        assert not isinstance(attn.out_proj, LoRALinear)


def test_extract_trainable_builds_graph_for_lora_only() -> None:
    """可训路径：输出 requires_grad=True（LoRA 参与图）；无 LoRA 时冻结路径无图。"""
    from dhead_distill.backbone import inject_lora

    bb = _make_student()
    x = torch.randn(2, 90, 6)
    stamp = torch.zeros(2, 90, 5)
    assert not bb.extract(x, stamp).requires_grad

    inject_lora(bb, rank=8, alpha=8, dropout=0.0, targets=("q_proj", "v_proj"))
    h = bb.extract_trainable(x, stamp)
    assert h.requires_grad
    trainable = {n for n, p in bb.named_parameters() if p.requires_grad}
    assert trainable, "注入后必须存在可训参数"
    assert all("lora" in n for n in trainable), \
        f"可训参数必须全部是 LoRA：{trainable}"
