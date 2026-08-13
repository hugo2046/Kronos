"""T1：Mamba 时序头（B3 延伸·候选主臂）。

计划 §2 冻结结构：

    冻结主干 → input_proj(832→256) → Mamba ResidualBlock×2
    (d_model=256, d_state=16, expand=2, dt_rank=16, d_conv=4)
    → RMSNorm → 末步 → linear(256→1)

- 块逐字复用 :mod:`improve_suite.mamba_min` 的 ``ResidualBlock / RMSNorm / MambaConfig``
  （阶段 4 已与 mamba-minimal 逐位对拍，**不改 mamba_min**）；
- ``forward(x_norm, stamp) -> score[B]`` 契约与 :class:`B3KronosKdaHead` 完全一致，
  保证 T1 vs B3/D1 唯一变量是「头内时序混合块」（Mamba 选择性 SSM vs KDA 线性注意力）；
- 主干冻结（``KronosFrozenBackbone`` 恒 eval + ``extract`` 在 ``no_grad`` 下），故同一输入
  确定可复现——允许把 ``extract`` 的产物缓存后喂给 :meth:`_decode`（计划 §2.1）。

**容量对齐（须测试断言）**：可训练参数 ≈ 1.09M ∈ [0.8M, 1.3M]
（input_proj 213K + 2×ResidualBlock≈438K×2 + norm 256 + head 257）。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from improve_suite.mamba_min import MambaConfig, RMSNorm, ResidualBlock

# T1 冻结超参（计划 §2，跑前冻结）
_T1_DMODEL = 256
_T1_NLAYERS = 2
_T1_DSTATE = 16
_T1_EXPAND = 2
_T1_DT_RANK = 16  # ceil(d_model/16) = ceil(256/16) = 16（与 mamba-minimal 默认一致）
_T1_DCONV = 4


class MambaTemporalHead(nn.Module):
    """T1：冻结 Kronos 主干 + Mamba 选择性 SSM 时序混合头。

    :param backbone: :class:`cross_section_kda.backbone.KronosFrozenBackbone`（冻结，
        ``extract`` 返回 ``[B,T,832]`` 逐步隐状态）。
    :param d_model: Mamba 块内部维度（默认 256，与计划 §2 一致）。
    :param n_layers: Mamba 残差块层数（默认 2）。
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        d_model: int = _T1_DMODEL,
        n_layers: int = _T1_NLAYERS,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        d_in = backbone.d_model  # 832（Kronos 隐状态维）
        self.input_proj = nn.Linear(d_in, d_model)

        cfg = MambaConfig(
            d_model=d_model,
            n_layer=n_layers,
            d_state=_T1_DSTATE,
            expand=_T1_EXPAND,
            dt_rank=_T1_DT_RANK,
            d_conv=_T1_DCONV,
        )
        self.layers = nn.ModuleList([ResidualBlock(cfg) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """input_proj / head 用 xavier_normal_（与 B3KronosKdaHead._init_weights 同口径）。"""
        for m in (self.input_proj, self.head):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """后骨干计算：[B,T,832] → input_proj → Mamba 残差栈 → RMSNorm → 末步 → score[B]。

        供缓存训练调用（计划 §2.1：隐状态落盘后，训练 loop 直接喂缓存 hidden，
        跳过冻结主干前向）。数学上与 :meth:`forward` 走在线 ``extract`` 等价
        （主干冻结 + no_grad → 同输入同输出）。
        """
        x = self.input_proj(hidden)        # [B, T, d_model]
        for blk in self.layers:
            x = blk(x)
        x = self.norm_f(x)
        last = x[:, -1]                    # [B, d_model]
        return self.head(last).squeeze(-1)  # [B]

    def forward(self, x_norm: torch.Tensor, stamp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向契约与 B3 一致。

        :param x_norm: 已窗口 z-score + clip5 的 ``[B,T,6]``（调用方负责归一化）。
        :param stamp: 时间特征 ``[B,T,5]``（同 ``calc_time_stamps``）；``None`` 不加时间嵌入。
        :returns: score ``[B]``。
        """
        hidden = self.backbone.extract(x_norm, stamp)   # [B, T, 832] no_grad
        return self._decode(hidden)


__all__ = ["MambaTemporalHead"]
