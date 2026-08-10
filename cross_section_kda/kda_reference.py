# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# 【交接说明】用户提供的 Kimi Linear (KDA) 参考实现，原为 qlib contrib 模型。
# 本实验只取组件：RMSNorm / CausalDepthwiseConv1d / KimiDeltaAttention / SwiGLU /
# KimiLinearBlock。不要使用 KimiLinearModel 的 DatasetH 契约（我们的数据来自
# kronos_qlib 窗口）。原文件中的 qlib 相对导入已注释，组件本身仅依赖 torch。

"""A shallow Kimi Linear style model for Qlib time-series datasets.

This module adapts the recurrent core of Kimi Delta Attention (KDA) to short
financial sequences. It intentionally does not reproduce the 48B MoE model:
the reusable parts here are the fine-grained decay gate, delta-rule memory,
causal short convolution, and gated RMS-normalized output.

References
----------
Kimi Linear: An Expressive, Efficient Attention Architecture
https://arxiv.org/abs/2510.26692
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm implemented locally for compatibility with older PyTorch."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * self.weight.float()).to(input_dtype)


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal short convolution used before the KDA recurrence."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d returns T + kernel_size - 1 positions. Keeping the first T
        # makes output[t] depend only on input[:t+1].
        seq_len = x.size(1)
        x = self.conv(x.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        return F.silu(x)


class KimiDeltaAttention(nn.Module):
    """Pure-PyTorch recurrent Kimi Delta Attention.

    Optimized Kimi implementations evaluate the same recurrence in parallel
    chunks. Qlib's default window is short, so this reference version favors
    portability and numerical clarity over custom kernels.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 2,
        dropout: float = 0.0,
        conv_kernel: int = 4,
        gate_rank: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        if d_model < 1 or nhead < 1:
            raise ValueError("d_model and nhead must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if not 0 < dt_min <= dt_max:
            raise ValueError("expected 0 < dt_min <= dt_max")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel)

        gate_rank = self.head_dim if gate_rank is None else gate_rank
        if gate_rank < 1:
            raise ValueError("gate_rank must be positive")
        self.forget_down = nn.Linear(d_model, gate_rank, bias=False)
        self.forget_up = nn.Linear(gate_rank, d_model, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(nhead).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.empty(d_model))
        self._init_dt_bias(dt_min, dt_max)

        self.beta_proj = nn.Linear(d_model, nhead, bias=False)
        self.output_gate_down = nn.Linear(d_model, gate_rank, bias=False)
        self.output_gate_up = nn.Linear(gate_rank, d_model, bias=False)
        self.output_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _init_dt_bias(self, dt_min: float, dt_max: float) -> None:
        # Initialize log-uniform time constants through inverse softplus.
        with torch.no_grad():
            log_dt = torch.empty_like(self.dt_bias).uniform_(math.log(dt_min), math.log(dt_max))
            dt = log_dt.exp()
            inverse_softplus = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.copy_(inverse_softplus)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), x.size(1), self.nhead, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"expected [batch, time, {self.d_model}], got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        q = F.normalize(self._shape_heads(self.q_conv(self.q_proj(x))), p=2, dim=-1)
        k = F.normalize(self._shape_heads(self.k_conv(self.k_proj(x))), p=2, dim=-1)
        v = self._shape_heads(self.v_conv(self.v_proj(x)))

        raw_decay = self._shape_heads(self.forget_up(self.forget_down(x))).float()
        dt_bias = self.dt_bias.view(1, 1, self.nhead, self.head_dim).float()
        decay_rate = self.A_log.exp().view(1, 1, self.nhead, 1).float()
        log_alpha = -decay_rate * F.softplus(raw_decay + dt_bias)
        beta = torch.sigmoid(self.beta_proj(x).float())

        # KDA maintains a fixed-size [B, H, K, V] recurrent state.
        state = torch.zeros(
            batch_size,
            self.nhead,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
            device=x.device,
        )
        outputs = []
        for step in range(seq_len):
            q_t = q[:, step].float()
            k_t = k[:, step].float()
            v_t = v[:, step].float()
            beta_t = beta[:, step, :, None]

            state = state * log_alpha[:, step].exp().unsqueeze(-1)
            memory_value = torch.einsum("bhk,bhkv->bhv", k_t, state)
            delta = v_t - memory_value
            state = state + torch.einsum("bhk,bhv->bhkv", beta_t * k_t, delta)
            output_t = self.scale * torch.einsum("bhk,bhkv->bhv", q_t, state)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=1).to(x.dtype)
        output = self.output_norm(output)
        output_gate = torch.sigmoid(self._shape_heads(self.output_gate_up(self.output_gate_down(x))))
        output = (output * output_gate).reshape(batch_size, seq_len, self.d_model)
        return self.dropout(self.out_proj(output))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        if d_model < 1 or hidden_dim < 1:
            raise ValueError("d_model and hidden_dim must be positive")
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class KimiLinearBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float,
        conv_kernel: int,
        gate_rank: int | None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = KimiDeltaAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            conv_kernel=conv_kernel,
            gate_rank=gate_rank,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


__all__ = [
    "CausalDepthwiseConv1d",
    "KimiDeltaAttention",
    "KimiLinearBlock",
    "RMSNorm",
    "SwiGLU",
]
