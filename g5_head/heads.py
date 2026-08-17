"""G5 三头（计划 §2，臂冻结）。

| 臂 | 结构（隐状态 [B,90,832] 之后） | 容量 | 实现 |
|---|---|---|---|
| H-lin | 末步 → Linear(832→1) | ~0.8K | 复用 ``cross_section_kda.models.B2LinearProbe``（只读 import） |
| H-kda | proj(832→256) + KimiLinearBlock×2 + 末步 Linear(256→1) | 数值核算断言 ∈[0.8M,1.3M] | 本模块 ``G5KdaHead`` |
| H-mamba | proj(832→256) + Mamba ResidualBlock×2 + RMSNorm + 末步 Linear(256→1) | 1.09M（既有断言） | 复用 ``improve_suite.mamba_head.MambaTemporalHead``（只读 import） |

**H-kda 容量分辨（数值核算为准，计划 §0 教训）**：计划 §2 表内规格
``ffn=512`` 经数值核算 = **1,603,153 参数（1.60M）∉ [0.8M, 1.3M]**——计划自身的
结构规格与容量断言不相容。按 §0"容量对齐以数值核算为准（docstring 不可信）"
的既定裁决规则，做**最小单超参修正**：ffn 512→256 → **1,209,937（1.21M）∈ 区间**
（与 H-mamba 1.09M 同量级，保住机制对照）。修正先于任何训练发生、不涉及任何
评估反馈，非头结构搜索（§6）。J5 触发条款不变。其余超参逐字冻结：
d_model=256 / nhead=8 / gate_rank=32 / conv_kernel=4 / dropout=0.0 / ×2 层。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from cross_section_kda.kda_modules import KimiLinearBlock

# H-kda 冻结超参（计划 §2 + 上述容量分辨）
_HKDA_DMODEL = 256
_HKDA_NHEAD = 8
_HKDA_FFN = 256  # 计划写 512；数值核算 1.60M 超区间，按"数值核算为准"修正（见模块 docstring）
_HKDA_GATE_RANK = 32
_HKDA_CONV_KERNEL = 4
_HKDA_NLAYERS = 2


class G5KdaHead(nn.Module):
    """H-kda：冻结 G1 底座 + KDA 线性注意力时序头（主臂 A）。

    ``forward(x_norm, stamp) -> score[B]`` 契约与 B3/T1 一致；
    ``_decode(hidden)`` 供缓存训练（数学等价：主干冻结 + no_grad）。
    """

    def __init__(self, backbone: nn.Module, *, n_layers: int = _HKDA_NLAYERS) -> None:
        super().__init__()
        self.backbone = backbone
        d_in = backbone.d_model  # 832
        self.input_proj = nn.Linear(d_in, _HKDA_DMODEL)
        self.blocks = nn.ModuleList([
            KimiLinearBlock(
                d_model=_HKDA_DMODEL, nhead=_HKDA_NHEAD, ffn_dim=_HKDA_FFN,
                dropout=0.0, conv_kernel=_HKDA_CONV_KERNEL, gate_rank=_HKDA_GATE_RANK,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(_HKDA_DMODEL, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.input_proj, self.head):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B,T,832] → proj → KDA×2 → 末步 → score[B]（缓存训练路径）。"""
        x = self.input_proj(hidden)
        for blk in self.blocks:
            x = blk(x)
        last = x[:, -1]
        return self.head(last).squeeze(-1)

    def forward(self, x_norm: torch.Tensor, stamp: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.backbone.extract(x_norm, stamp)  # [B, T, 832] no_grad
        return self._decode(hidden)


def decode_score(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """缓存隐状态 → score[B] 的统一入口（三头分发）。

    H-kda / H-mamba 有 ``_decode``；H-lin（B2LinearProbe）无——取末步过 probe。
    """
    if hasattr(model, "_decode"):
        return model._decode(hidden)
    return model.probe(hidden[:, -1]).squeeze(-1)


__all__ = ["G5KdaHead", "decode_score"]
