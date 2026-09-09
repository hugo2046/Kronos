"""LoRA-mean 合成契约测试（不依赖真实数据/GPU/DB）。"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from lora_mean_pilot import adapter as ad  # noqa: E402


class _Attn(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, n, nn.Linear(d, d))


class _Layer(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.self_attn = _Attn(d)


class _FakePredictor(nn.Module):
    """同属性布局的最小 predictor 替身（transformer[i].self_attn.*_proj）。"""

    def __init__(self, n_layers=2, d=16):
        super().__init__()
        self.transformer = nn.ModuleList([_Layer(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, 8)
        self.embedding = nn.Linear(d, d)

    def forward(self, x):
        for lay in self.transformer:
            a = lay.self_attn
            x = a.q_proj(x) + a.k_proj(x) + a.v_proj(x) + a.out_proj(x)
        return self.head(x)


def test_inject_zero_delta_and_freeze():
    torch.manual_seed(0)
    m = _FakePredictor()
    x = torch.randn(4, 8, 16)
    y0 = m(x).clone()
    trainable = ad.inject_lora(m, rank=8, alpha=8, seed=100)
    y1 = m(x)
    assert torch.equal(y0, y1), "零增量门禁失败：注入当刻应逐位一致"
    # 恰 2层×4投影×(A,B) 个张量；全部 lora_A/B
    assert len(trainable) == 2 * 4 * 2
    assert all(".lora_A." in n or ".lora_B." in n for n in trainable)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_params == 2 * 4 * 2 * (16 * 8)    # d=16, rank=8


def test_only_lora_changes_after_step(tmp_path):
    import copy

    torch.manual_seed(0)
    m = _FakePredictor()
    m_preinject = copy.deepcopy(m)     # 同基底对照（inject 会重播种全局流）
    base_before = ad.base_state_hash(m)
    w_before = {k: v.clone() for k, v in m.state_dict().items()
                if ".lora_" not in k}
    ad.inject_lora(m, seed=100)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], 0.1)
    x = torch.randn(4, 16)
    loss = m(x).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    assert ad.base_state_hash(m) == base_before, "基底哈希变化！"
    for k, v in w_before.items():
        wrapped = (k.replace(".weight", ".base.weight").replace(
            ".bias", ".base.bias")              # 仅被包装的投影移入 .base
            if any(f"{n}_proj." in k for n in ("q", "k", "v", "out")) else k)
        assert torch.equal(v, m.state_dict()[wrapped]), f"原权重被改：{k}"
    # adapter 保存→载入 round-trip（新注入实例）
    st = ad.adapter_state(m)
    sha = ad.save_adapter(tmp_path / "a.pt", {"adapter": st, "seed": 100})
    assert len(sha) == 64
    m2 = m_preinject                         # 同一基底权重
    ad.inject_lora(m2, seed=7)               # 不同 seed 注入（A 不同）
    ad.apply_adapter(m2, ad.load_adapter(tmp_path / "a.pt")["adapter"])
    x2 = torch.randn(3, 16)
    assert torch.equal(m(x2), m2(x2)), "adapter 载入后输出不一致"


def test_real_predictor_param_count():
    """12 层 × 4 投影 × (832×8 + 8×832) = 638,976（公式核对，不载真权重）。"""
    n = 12 * 4 * (832 * 8 + 8 * 832)
    assert n == 638976


def test_verdict_and_gates():
    from lora_mean_pilot import evaluate as ev

    r = {("G1_original_mean", w): {"curve_end": 0.10} for w in ev.WINDOW_BOUNDS}
    r.update({("G1_paired_100", w): {"curve_end": 0.11} for w in ev.WINDOW_BOUNDS})
    r.update({("LoRA_100", "W3"): {"curve_end": 0.12},
              ("LoRA_100", "W4"): {"curve_end": 0.09}})
    v = ev.paired_verdict(r, [100])
    assert v["100|W3"]["win_original"] and v["100|W3"]["win_paired"]
    assert not v["100|W4"]["win_original"]            # W4 输给原 G1
    assert not ev.pilot_gate(v, 100)
    # confirm：≥2 seed 两窗双胜 + 中位数
    r2 = dict(r)
    for s in (101, 102):
        r2.update({(f"G1_paired_{s}", w): {"curve_end": 0.11}
                   for w in ev.WINDOW_BOUNDS})
    r2.update({("LoRA_101", "W3"): {"curve_end": 0.13},
               ("LoRA_101", "W4"): {"curve_end": 0.12},
               ("LoRA_102", "W3"): {"curve_end": 0.125},
               ("LoRA_102", "W4"): {"curve_end": 0.115}})
    v2 = ev.paired_verdict(r2, [100, 101, 102])
    cg = ev.confirm_gate(v2, [100, 101, 102])
    assert cg["seed_wins"] == {100: False, 101: True, 102: True}
    assert cg["repeated_improvement"] is True


def test_run_source_no_forbidden():
    for f in ("run.py", "train.py", "evaluate.py", "adapter.py"):
        src = (REPO_ROOT / "lora_mean_pilot" / f).read_text(encoding="utf-8")
        assert "model.fit" not in src and "from_pretrained\n" not in src.replace(
            "Kronos.from_pretrained", "").replace(
            "KronosTokenizer.from_pretrained", ""), f"{f} 出现禁用调用"
