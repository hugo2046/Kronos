"""五臂对照实验的三臂网络（B1/B2/B3）。

计划 §0：

| # | 模型 | 输入 | 可训练部分 |
|---|---|---|---|
| B1 | KDA 纯监督 | [B,90,6]（窗口 z-score） | 全部 |
| B2 | 冻结 Kronos + linear probe | Kronos 末步隐状态 [832] | 仅 linear |
| B3 | 冻结 Kronos + KDA 浅层头 | Kronos 逐步隐状态 [90,832] | KDA×2 + linear |

统一 ``forward(x_norm, stamp) -> score[B]`` 契约，训练/评估代码统一调度。
B2/B3 内部对冻结主干用 ``no_grad``——梯度只流过头，保证「可训练部分」与计划一致。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from cross_section_kda.backbone import KronosFrozenBackbone
from cross_section_kda.kda_modules import KimiLinearBlock


# KDA 头默认尺寸（计划 §0：B3 KDA×2 + linear ~1M；B1 同结构但输入维=6）
# d_model=832（与 Kronos 隐状态同维），nhead=8，ffn=2*d_model
_KDA_DMODEL = 832
_KDA_NHEAD = 8
_KDA_FFN = 2 * _KDA_DMODEL
_KDA_CONV_KERNEL = 4
_KDA_NLAYERS = 2


class B1SupervisedHead(nn.Module):
    """B1：KDA 纯监督（无预训练）。

    输入 [B,90,6]（已窗口 z-score + clip5，**与 Kronos 同口径**，保证 B1 vs B2/B3
    唯一变量是「预训练表示 vs 原始特征」）→ linear(6→832) → KDA×2 → 末步 → linear(832→1)。
    全部可训练（~0.1M 量级）。
    """

    def __init__(
        self,
        *,
        d_in: int = 6,
        d_model: int = _KDA_DMODEL,
        n_layers: int = _KDA_NLAYERS,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([
            KimiLinearBlock(
                d_model=d_model, nhead=_KDA_NHEAD, ffn_dim=_KDA_FFN,
                dropout=0.0, conv_kernel=_KDA_CONV_KERNEL, gate_rank=d_model // _KDA_NHEAD,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_norm: torch.Tensor, stamp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向：原始特征 → KDA → 末步打分。

        :returns: score ``[B]``。
        """
        x = self.input_proj(x_norm)            # [B, T, d_model]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        last = x[:, -1]                         # [B, d_model]
        return self.head(last).squeeze(-1)      # [B]


class B2LinearProbe(nn.Module):
    """B2：冻结 Kronos + linear probe（归因用）。

    输入经冻结主干取**末步**隐状态 [832] → linear(832→1)（~0.8K 可训练）。
    """

    def __init__(self, backbone: KronosFrozenBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        d_model = backbone.d_model
        self.probe = nn.Linear(d_model, 1)
        nn.init.xavier_normal_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)

    def forward(self, x_norm: torch.Tensor, stamp: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.backbone.extract(x_norm, stamp)   # [B, T, d_model] no_grad
        last = hidden[:, -1]                             # [B, d_model]
        return self.probe(last).squeeze(-1)              # [B]


class B3KronosKdaHead(nn.Module):
    """B3：冻结 Kronos + KDA 浅层头（改造·主菜）。

    输入经冻结主干取**逐步**隐状态 [90,832] → input_proj(832→832) → KDA×2 →
    末步 [:,-1] → linear(832→1)（~1M 可训练）。

    KDA 头与 B1 同结构（2 层 KimiLinearBlock），唯一区别是输入来源
    （Kronos 隐状态 vs 原始特征），保证 B1 vs B3 的对照是「预训练表示」单一变量。
    """

    def __init__(
        self,
        backbone: KronosFrozenBackbone,
        *,
        n_layers: int = _KDA_NLAYERS,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        d_model = backbone.d_model
        self.input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            KimiLinearBlock(
                d_model=d_model, nhead=_KDA_NHEAD, ffn_dim=_KDA_FFN,
                dropout=0.0, conv_kernel=_KDA_CONV_KERNEL, gate_rank=d_model // _KDA_NHEAD,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.input_proj, self.head]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_norm: torch.Tensor, stamp: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.backbone.extract(x_norm, stamp)   # [B, T, d_model] no_grad
        x = self.input_proj(hidden)                     # [B, T, d_model]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        last = x[:, -1]                                 # [B, d_model]
        return self.head(last).squeeze(-1)              # [B]


def count_trainable(m: nn.Module) -> int:
    """可训练参数数（断言/日志用）。"""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


__all__ = [
    "B1SupervisedHead",
    "B2LinearProbe",
    "B3KronosKdaHead",
    "count_trainable",
]
