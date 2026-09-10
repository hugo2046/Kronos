"""AE/SAE 残差头（计划 §5 冻结结构，维数 D 从真实 h 读取）。

结构（无 dropout、无额外特征、不解冻底座）::

    x[D] → Linear(D,64) → GELU → Linear(64,16) → Sigmoid → z[16]
    z → Linear(16,64) → GELU → Linear(64,D) → x_hat[D]
    z → Linear(16,1) → r_hat          # weight/bias 全 0 初始化

零初始化的残差出口保证训练前 ``r_hat ≡ 0`` → ``s_final`` 逐值等于 G1。
同一 seed 构造的 AE/SAE 初始 state_dict 完全一致（唯一差异 = 训练期 beta）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sae_residual import config as C


class ResidualHead(nn.Module):
    """重构 + 残差预测双出口头（AE 与 SAE 共用同一类，beta 由损失控制）。

    :param d_in: 输入维数 D（真实隐状态维数，预期 832）。
    """

    def __init__(self, d_in: int) -> None:
        super().__init__()
        enc = C.HEAD["enc_dim"]
        zdim = C.HEAD["z_dim"]
        self.encoder = nn.Sequential(
            nn.Linear(d_in, enc), nn.GELU(),
            nn.Linear(enc, zdim), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(zdim, enc), nn.GELU(),
            nn.Linear(enc, d_in))
        self.residual = nn.Linear(zdim, 1)
        # 残差出口零初始化：训练前输出恒 0（逐值等于 G1 的实现门禁）
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.d_in = d_in

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor]:
        """前向。

        :param x: 归一化隐状态 ``[N, D]``。
        :returns: ``(r_hat [N], x_hat [N,D], z [N,zdim])``。
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        r_hat = self.residual(z).squeeze(-1)
        return r_hat, x_hat, z


def build_head(seed: int, d_in: int) -> ResidualHead:
    """按 head seed 构造头；同 seed 两次构造 state_dict 逐位一致。"""
    torch.manual_seed(seed)
    return ResidualHead(d_in)


def sparse_kl(z: torch.Tensor, rho: float = C.LOSS["rho"]) -> torch.Tensor:
    """Bernoulli KL 稀疏约束 ``mean_j [rho·log(rho/p_j) + (1-rho)·log((1-rho)/(1-p_j))]``。

    ``p_j = clamp(mean_batch(z_j), 1e-6, 1-1e-6)``；z 经 Sigmoid 已在 (0,1)。
    当 ``p_j = rho`` 时取 0（最小），对任意维偏离 rho 均严格为正。
    """
    p = z.mean(dim=0).clamp(1e-6, 1 - 1e-6)
    rho_t = torch.as_tensor(rho, dtype=p.dtype, device=p.device)
    kl = (rho_t * torch.log(rho_t / p)
          + (1 - rho_t) * torch.log((1 - rho_t) / (1 - p)))
    return kl.mean()


def head_loss(r_hat: torch.Tensor, r: torch.Tensor,
              x_hat: torch.Tensor, x: torch.Tensor, z: torch.Tensor,
              beta: float) -> dict:
    """``L = L_pred + 0.1·L_recon + beta·L_sparse``（不搜索权重）。

    :returns: 各分量与总损失（标量 tensor）的字典；beta=0 时 KL 仍计算
        （诊断用），但总损失中系数为 0 → 精确退化为 AE。
    """
    l_pred = F.mse_loss(r_hat, r)
    l_recon = F.mse_loss(x_hat, x)
    l_sparse = sparse_kl(z)
    total = l_pred + C.LOSS["recon_weight"] * l_recon + beta * l_sparse
    return {"total": total, "l_pred": l_pred.detach(),
            "l_recon": l_recon.detach(), "l_sparse": l_sparse.detach()}


def activation_diagnostics(z: torch.Tensor) -> dict:
    """平均激活与低激活占比（仅诊断，不作选点依据）。"""
    with torch.no_grad():
        mean_act = z.mean(dim=0)
        return {
            "mean_activation": float(mean_act.mean()),
            "low_activation_frac": float(
                (mean_act < 0.05).float().mean()),
        }


def final_signal(s_g1: torch.Tensor, r_hat: torch.Tensor,
                 sigma_e: float) -> torch.Tensor:
    """``s_final = s_G1 + σe·r_hat``（还原收益率量纲后再排序）。"""
    return s_g1 + sigma_e * r_hat


__all__ = ["ResidualHead", "build_head", "sparse_kl", "head_loss",
           "activation_diagnostics", "final_signal"]
