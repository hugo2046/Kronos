"""小 Mamba（计划 §6 阶段 4）。

移植 mamba-minimal 的 ``MambaBlock / ResidualBlock / RMSNorm / selective_scan``
（数学不动：选择性 Δ/B/C + ZOH 离散化 + 顺序扫描），**纯 PyTorch 无 CUDA 核依赖**。

头部改造（计划 §6）：
    - 输入 = token 嵌入：``Embedding(2^s1_bits, d_model) + Embedding(2^s2_bits, d_model)``
      （两嵌入相加，vocab 从 tokenizer 配置读取）；
    - 出口 = ``Linear(d_model, 1)`` 取末步（回归，预测 5 日均收益）；
    - 删除 ``vocab_size padding / norm_attn / from_pretrained`` 死代码。

训练协议超参（跑前冻结，§6）：d_model=64、n_layer=2、d_state=16、expand=2、
lr=1e-4、weight_decay=1e-3、batch=2048、MSE 损失、seed=42。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    """跑前冻结的训练超参（计划 §6）。"""

    s1_vocab: int = 1024  # 2^s1_bits，运行时从 tokenizer 读
    s2_vocab: int = 262144  # 2^s2_bits，运行时从 tokenizer 读
    d_model: int = 64
    n_layer: int = 2
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    dt_rank: int = 16  # 取 ceil(d_model/16)，与 mamba-minimal 默认一致


class RMSNorm(nn.Module):
    """与 mamba-minimal 逐字一致（数学不动）。"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


def selective_scan(x, delta, A, B, C, D):
    """选择性 SSM 的顺序扫描（ZOH 离散化，数学等价 mamba-minimal，无 CUDA）。

    布局：``x, delta`` = ``(B, L, d_inner)``；``A`` = ``(d_inner, d_state)``；
    ``B, C`` = ``(B, L, d_state)``；``D`` = ``(d_inner,)``。

    递推：``h_t = ΔA_t · h_{t-1} + ΔB_t · x_t``；``y_t = C_t · h_t + D ⊙ x_t``，
    其中 ``ΔA = exp(Δ ⊗ A)``，``ΔB = Δ ⊗ B``（ZOH）。
    """
    (b, l, d_in) = x.shape
    n = A.shape[1]
    # ΔA: (B, L, d_in, d_state)
    deltaA = torch.exp(torch.einsum("b l d, d n -> b l d n", delta, A))
    # ΔB ⊗ x: (B, L, d_in, d_state)
    deltaB_x = torch.einsum("b l d, b l n, b l d -> b l d n", delta, B, x)
    # 顺序扫描
    h = A.new_zeros(b, d_in, n)
    ys = []
    for t in range(l):
        h = deltaA[:, t] * h + deltaB_x[:, t]
        ys.append(h)
    hs = torch.stack(ys, dim=1)  # (B, L, d_in, d_state)
    y = torch.einsum("b l n, b l d n -> b l d", C, hs)
    # skip connection
    y = y + x * D.view(1, 1, -1)
    return y


class MambaBlock(nn.Module):
    """与 mamba-minimal 逐字一致（数学不动）。"""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        d_model = cfg.d_model
        d_inner = d_model * cfg.expand
        self.cfg = cfg
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=cfg.d_conv, groups=d_inner, padding=cfg.d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, cfg.dt_rank + cfg.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(cfg.dt_rank, d_inner, bias=True)
        # A: (d_inner, d_state)，初始化为负数（与 mamba-minimal 一致）
        A = torch.arange(1, cfg.d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x_and_res = x_and_res.transpose(1, 2)  # (B, 2*d_inner, L)
        (x1, z) = x_and_res.split(self.cfg.d_model * self.cfg.expand, dim=1)
        # conv + silu
        x1 = self.conv1d(x1)[:, :, :l]  # 因果（截断右侧 padding）
        x1 = x1.transpose(1, 2)  # (B, L, d_inner)
        x1 = F.silu(x1)
        # 投影 Δ, B, C
        y = self.ssm(x1)
        # 门控
        y = y * F.silu(z.transpose(1, 2))
        out = self.out_proj(y)
        return out

    def ssm(self, x):
        (b, l, d) = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        (delta, B, C) = x_dbl.split(
            [self.cfg.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        return selective_scan(x, delta, A, B, C, D)


class ResidualBlock(nn.Module):
    """RMSNorm → MambaBlock → 残差（与 mamba-minimal 一致）。"""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MambaBlock(cfg)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaSeqRegressor(nn.Module):
    """双 token 嵌入 + Mamba 残差栈 + 末步回归头（计划 §6 头部改造）。"""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.emb_s1 = nn.Embedding(cfg.s1_vocab, cfg.d_model)
        self.emb_s2 = nn.Embedding(cfg.s2_vocab, cfg.d_model)
        self.layers = nn.ModuleList([ResidualBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, s1_tokens, s2_tokens):
        """s1/s2_tokens: (B, L) long → 末步标量 (B,)。"""
        x = self.emb_s1(s1_tokens) + self.emb_s2(s2_tokens)  # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        last = x[:, -1, :]  # (B, d_model)
        return self.head(last).squeeze(-1)  # (B,)
