"""同日集合 attention 残差头（计划 §5 冻结结构）。

结构（OFF/ON 同形状，唯一差异 = attention 掩码）::

    x[D] → Linear(D,64) → u
    v  = LayerNorm(u)
    a  = MultiheadAttention(64,4,dropout=0,batch_first=True)(v,v,v,mask)
    z  = u + a
    z  = z + Linear(128,64)(GELU(Linear(64,128)(LayerNorm(z))))
    r̂  = Linear(64,1)(LayerNorm(z))          # weight/bias 全 0 初始化

零初始化残差出口保证训练前 ``r_hat ≡ 0`` → ``s_final`` 逐值等于 G1。
日期维是 batch 维，attention 序列维 = 该日股票（严禁把多日拼进 N 轴）。
padding 股票从 key/value 屏蔽；padding query 统一只连该日首个有效 key
再把输出置零（防全行无合法 key 的 NaN），有效 query 的可见性不受影响。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from peer_residual import config as C


def build_blocked_mask(valid: torch.Tensor, mode: str) -> torch.Tensor:
    """构造布尔 attention 掩码（True = 禁止访问）。

    :param valid: ``[B, N]`` bool，该位置为真实股票。
    :param mode: ``"OFF"`` 只允许每个股票关注自己（对角）；
        ``"ON"`` 同日全部有效股票互相可见。
    :returns: ``[B, N, N]`` bool。padding query（两种模式）只连该日首个
        有效 key——其输出随后被置零，仅为避免全行被遮产生 NaN/梯度污染。
    """
    if mode not in C.ARMS:
        raise ValueError(f"未知交互模式：{mode}")
    b, n = valid.shape
    device = valid.device
    valid_key = valid[:, None, :].expand(b, n, n)      # key 有效性 [B,Q,K]
    if mode == "OFF":
        self_only = torch.eye(n, dtype=torch.bool, device=device)
        allowed = self_only[None].expand(b, n, n) & valid_key
    else:  # ON
        allowed = valid[:, :, None].expand(b, n, n) & valid_key
    # padding query：只允许该日首个有效 key（每行保证 ≥1 个合法 key）
    first = valid.to(torch.long).argmax(dim=1)          # [B] 首个有效下标
    pad_q = (~valid)[:, :, None].expand(b, n, n)        # query 为 padding
    first_key = torch.zeros(b, n, n, dtype=torch.bool, device=device)
    first_key.scatter_(2, first[:, None, None].expand(b, n, 1), True)
    allowed = torch.where(pad_q, first_key, allowed)
    return ~allowed


class PeerResidualHead(nn.Module):
    """同日跨股票交互残差头（不含任何 G1 参数；h 为预计算冻结输入）。

    :param d_in: 输入维数 D（真实隐状态维数，预期 832）。
    :param d_model: attention/隐层维数（默认 64）。
    :param n_heads: attention 头数（默认 4）。
    :param ffn_dim: 前馈隐层维数（默认 128）。
    """

    def __init__(self, d_in: int, d_model: int = C.HEAD["d_model"],
                 n_heads: int = C.HEAD["n_heads"],
                 ffn_dim: int = C.HEAD["ffn_dim"]) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)             # u
        self.ln1 = nn.LayerNorm(d_model)                 # v
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)
        # 残差出口零初始化：训练前输出恒 0（逐值等于 G1 的实现门禁）
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.n_heads = n_heads
        self.d_in = d_in

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                mode: str) -> torch.Tensor:
        """前向。

        :param x: 归一化隐状态 ``[B, N, D]``（B=日期，N=当日股票，含 padding）。
        :param valid: ``[B, N]`` bool 有效股票掩码（全 padding 日期非法）。
        :param mode: ``"OFF"`` / ``"ON"`` 交互掩码（唯一实验变量）。
        :returns: ``r_hat [B, N]``，padding 位置已置零。
        """
        blocked = build_blocked_mask(valid, mode)
        attn_mask = blocked.repeat_interleave(self.n_heads, dim=0)
        u = self.proj(x)
        v = self.ln1(u)
        a, _ = self.attn(v, v, v, attn_mask=attn_mask, need_weights=False)
        z = u + a
        z = z + self.ffn_out(F.gelu(self.ffn_in(self.ln2(z))))
        r_hat = self.out(self.ln3(z)).squeeze(-1)
        return r_hat.masked_fill(~valid, 0.0)


def build_head(seed: int, d_in: int) -> PeerResidualHead:
    """按 head seed 构造头；同 seed 两次构造 state_dict 逐位一致。"""
    torch.manual_seed(seed)
    return PeerResidualHead(d_in)


def param_table(d_in: int) -> dict[str, int]:
    """精确参数表（计划 §5：构造前列参数表与精确数目，预期约 8.7 万）。"""
    dm, nh, ff = C.HEAD["d_model"], C.HEAD["n_heads"], C.HEAD["ffn_dim"]
    rows = {
        f"proj:{d_in}x{dm}": d_in * dm + dm,
        "ln1": 2 * dm,
        f"attn.in_proj:{dm}x{dm}x3": 3 * dm * dm + 3 * dm,
        f"attn.out_proj:{dm}x{dm}": dm * dm + dm,
        "ln2": 2 * dm,
        f"ffn_in:{dm}x{ff}": dm * ff + ff,
        f"ffn_out:{ff}x{dm}": ff * dm + dm,
        "ln3": 2 * dm,
        f"out:{dm}x1": dm + 1,
    }
    rows["total"] = sum(rows.values())
    return rows


def final_signal(s_g1: torch.Tensor, r_hat: torch.Tensor,
                 sigma_e: float) -> torch.Tensor:
    """``s_final = s_G1 + σe·r_hat``（还原收益率量纲后再排序）。"""
    return s_g1 + sigma_e * r_hat


__all__ = ["PeerResidualHead", "build_head", "build_blocked_mask",
           "param_table", "final_signal"]
