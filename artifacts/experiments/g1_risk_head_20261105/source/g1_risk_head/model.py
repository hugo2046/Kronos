"""冻结G1表示上的三分位数线性读出。"""
from __future__ import annotations
import torch
from torch import nn, Tensor


class QuantileHead(nn.Module):
    """预测已除以历史波动尺度的三个有序分位数。

    :param d_model: 隐状态维度。
    :param initial_quantiles: 训练样本Z的三个经验分位数。
    """
    def __init__(self,d_model: int,initial_quantiles: Tensor) -> None:
        super().__init__()
        if initial_quantiles.shape!=(3,) or not torch.isfinite(initial_quantiles).all():
            raise ValueError('需要三个有限经验分位数')
        self.linear=nn.Linear(d_model,3)
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad():self.linear.bias.copy_(initial_quantiles.sort().values)

    def forward(self,hidden: Tensor) -> Tensor:
        """输出从低到高的条件分位数。

        :param hidden: batch乘隐藏维度。
        :return: batch乘三个分位数。
        """
        return self.linear(hidden).sort(dim=-1).values


def pinball(pred: Tensor,target: Tensor) -> Tensor:
    """逐样本逐分位pinball损失。

    :param pred: batch乘3预测。
    :param target: batch真实Z收益。
    :return: batch乘3损失，不预先聚合日期。
    """
    if pred.ndim!=2 or pred.shape[1]!=3 or target.shape!=(len(pred),):
        raise ValueError('预测/标签形状不匹配')
    error=target[:,None]-pred
    quantiles=pred.new_tensor([.1,.5,.9])
    return torch.maximum(quantiles*error,(quantiles-1)*error)
