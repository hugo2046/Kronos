"""LoRA 注入与适配器管理（20260909 LoRA-mean 计划 §3）。

从 G1 s100 predictor 起步：原权重全部冻结，对
``transformer.<i>.self_attn.{q_proj,k_proj,v_proj,out_proj}`` 注入零初始增量
LoRA（rank8/alpha8/dropout0，``x + (alpha/r)·B(A(x))``）。零初始化 B 保证
注入当刻前向与原 G1 逐位一致（先于任何训练的门禁）。参考
``dhead_distill/backbone.py:LoRALinear`` 的纯模块语义，独立实现于本包。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    """nn.Linear 的低秩适配包装：``base(x) + (alpha/r)·B(A(x))``。

    A Kaiming 初始化（固定 seed 下确定）、B 零初始化；base 权重冻结逐字节
    不变。dropout=0（计划冻结值）。
    """

    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Linear(base.in_features, rank, bias=False,
                                device=dev, dtype=dt)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False,
                                device=dev, dtype=dt)
        nn.init.kaiming_normal_(self.lora_A.weight, nonlinearity="linear")
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(
            self.lora_dropout(self.lora_A(x)))


def inject_lora(predictor: nn.Module, *, rank: int = 8, alpha: int = 8,
                seed: int = 100) -> list[str]:
    """对全部注意力投影注入 LoRA；返回可训练参数名列表（A/B）。

    注入前以固定 seed 播种——同 seed 的 A 初始化确定（配对实验要求）。
    """
    torch.manual_seed(seed)
    n_layers = len(predictor.transformer)
    wrapped: list[str] = []
    for i in range(n_layers):
        attn = predictor.transformer[i].self_attn
        for name in TARGET_SUFFIXES:
            linear = getattr(attn, name)
            setattr(attn, name, LoRALinear(linear, rank=rank, alpha=alpha))
            wrapped.append(f"transformer.{i}.self_attn.{name}")
    # 除 LoRA A/B 外全部冻结（原基底、DualHead、embedding、norm、FFN 等）
    for pname, p in predictor.named_parameters():
        p.requires_grad_(".lora_" in pname)
    trainable = sorted(n for n, p in predictor.named_parameters()
                       if p.requires_grad)
    assert all((".lora_A" in n or ".lora_B" in n) for n in trainable)
    return trainable


def zero_delta_logits(model_a: nn.Module, model_b: nn.Module,
                      x: torch.Tensor, atol: float = 0.0) -> float:
    """两模型同输入 logits 最大绝对差（零增量门禁：应恰为 0）。"""
    with torch.no_grad():
        d = (model_a(x) - model_b(x)).abs().max().item()
    return d


def adapter_state(predictor: nn.Module) -> dict:
    """仅 LoRA A/B 张量（基底不入 adapter 文件，经基底 SHA 绑定）。"""
    return {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()
            if ".lora_" in k}


def base_state_hash(predictor: nn.Module) -> str:
    """基底（非 LoRA）state_dict 稳定哈希——训练前后必须不变。"""
    h = hashlib.sha256()
    import numpy as np

    for k in sorted(predictor.state_dict()):
        if ".lora_" in k:
            continue
        # 归一化包装名：q_proj.base.weight → q_proj.weight（包装前后同基底
        # 张量的哈希一致，防键名变化误报）
        k = k.replace(".base.", ".")
        raw_key = k
        t = predictor.state_dict()[
            raw_key if raw_key in predictor.state_dict()
            else k.replace(".weight", ".base.weight").replace(
                ".bias", ".base.bias")].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(np.ascontiguousarray(t.numpy()).tobytes())
    return h.hexdigest()


def save_adapter(path: Path | str, payload: dict) -> str:
    """原子写 adapter 文件（epoch 唯一文件；best 只是索引）。返回 SHA256。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def load_adapter(path: Path | str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def apply_adapter(predictor: nn.Module, state: dict) -> None:
    """载入 adapter 张量（名字须与注入后完全一致）。"""
    cur = adapter_state(predictor)
    assert set(state) == set(cur), "adapter 键与当前注入不匹配"
    predictor.load_state_dict(state, strict=False)
