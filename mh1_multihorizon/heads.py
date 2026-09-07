"""MH1 共享四期限头与损失（计划 §3 核心实现定义，逐字对齐）。

S（单期限对照）与 M（多期限实验）使用**完全相同的模型**——仅 ``horizon_loss``
的臂参数决定哪些输出参与监督。共享非线性层（LayerNorm→Linear→GELU→Linear）
保证辅助期限梯度必须经过与主期限相同的可训练表征。
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from r1_objective.ic_loss import ic_loss


class MultiHorizonHead(nn.Module):
    """从最后历史隐状态读取四期限分数。

    :param d_model: 主干隐状态维数。
    """

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden),
            nn.GELU(), nn.Linear(hidden, 4),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """计算排序分数。

        :param hidden: 形状 [B,T,D] 的历史隐状态。
        :returns: 按 1/5/10/20 日顺序排列的 [B,4] 分数。
        """
        return self.net(hidden[:, -1, :])


def horizon_loss(
    scores: torch.Tensor, labels: torch.Tensor, arm: str,
) -> torch.Tensor:
    """计算单日截面监督损失。

    :param scores: [B,4] 预测分数。
    :param labels: [B,4] 有限的累计收益标签。
    :param arm: S 为单期限，M 为联合期限。
    :returns: 标量负 Pearson 损失。
    :raises ValueError: 臂名非法时抛出。
    """
    if arm == "S":
        return ic_loss(scores[:, 2], labels[:, 2])
    if arm == "M":
        # 四项损失对应四个固定任务；数据构造已过滤退化截面。
        return torch.stack([
            ic_loss(scores[:, 0], labels[:, 0]),
            ic_loss(scores[:, 1], labels[:, 1]),
            ic_loss(scores[:, 2], labels[:, 2]),
            ic_loss(scores[:, 3], labels[:, 3]),
        ]).mean()
    raise ValueError(f"未知实验臂：{arm}")


def horizon_names(horizons: Sequence[int]) -> list[str]:
    """期限元组 → 列名（h1/h5/h10/h20）。"""
    return [f"h{k}" for k in horizons]


__all__ = ["MultiHorizonHead", "horizon_loss", "horizon_names"]
