"""PATH1 集合小头（计划 §5）：753 参数、N 轴排列不变、末层零初始化。

结构（全部冻结，不搜索宽度）::

    每条 10 维路径 → Linear(10,16) → GELU → Linear(16,16) → GELU
    20 条路径表示等权平均 → 16 维
    拼接 1 维 b → Linear(17,16) → GELU → Linear(16,1) → r_hat
    最后 Linear 的 weight/bias 零初始化

两臂同构：MEAN 喂全零 shape、PATH 喂真实 shape，二者都看到相同原
G1 scalar ``b``；未训练时 ``r_hat ≡ 0``（三臂分数逐值等于原 G1）。
N 轴经等权平均天然排列不变；H 轴保留时间次序（Linear 逐元素于 H）。
"""
from __future__ import annotations

import torch
from torch import nn


class PathHead(nn.Module):
    """采样路径形状 → 残差修正 ``r_hat``（量纲：σe 单位）。

    :ivar per_path: 逐路径 MLP ``[H]→16→16``（GELU）。
    :ivar mix: 拼接 b 后的混合 MLP ``17→16→1``（末层零初始化）。
    """

    def __init__(self, path_dim: int = 10, hidden: int = 16):
        super().__init__()
        self.path_dim = path_dim
        self.per_path = nn.Sequential(
            nn.Linear(path_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.mix = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, shape: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """前向。

        :param shape: ``[B, N, H]`` 路径形状特征（MEAN 臂喂全零）。
        :param b: ``[B]`` 标准化后的原 G1 scalar。
        :returns: ``[B]`` ``r_hat``（未训练恒 0）。
        """
        z = self.per_path(shape)             # [B, N, hidden]
        z = z.mean(dim=1)                    # N 轴等权平均（排列不变）
        zb = torch.cat([z, b.reshape(-1, 1).to(z.dtype)], dim=1)
        return self.mix(zb).reshape(-1)


def arm_input(shape: torch.Tensor, arm: str) -> torch.Tensor:
    """臂语义：MEAN 喂全零、PATH 喂真实 shape（同一 ``b``）。"""
    if arm == "PATH":
        return shape
    if arm == "MEAN":
        return torch.zeros_like(shape)
    raise ValueError(f"未知臂 {arm!r}（合法：PATH/MEAN）")


__all__ = ["PathHead", "arm_input"]
