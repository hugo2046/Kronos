"""蒸馏/监督/排序损失（方案 §4.3，公式逐字）。

所有均方损失先除以期限尺度 ``scale[h]``（训练集各期限 std，冻结），
避免把极小量级的收益 MSE 与 IC 相加。IC 在同日截面内计算。
"""
from __future__ import annotations

import torch


def normalized_mse(
    pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """按期限尺度归一的均方误差。

    :param pred: ``[B,H]`` 预测（期限均值信号或整条路径）。
    :param target: ``[B,H]`` 目标（真实标签或教师 replica 目标）。
    :param scale: ``[H]`` 各期限尺度（训练集 std，下限 0.01，冻结）。
    :returns: 标量损失。
    """
    return ((pred - target) / scale).square().mean()


def daily_ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """同日截面 IC 损失（1 − cos 相似度，期限均值信号）。

    调用者保证一个 batch 是同一日、至少 32 只股票；target 是 10 日期限均值。
    预测方差为 0 时用 epsilon（clamp_min 1e-8）保持有限。

    :param pred: ``[B,H]`` 预测。
    :param target: ``[B,H]`` 目标。
    :returns: 标量损失，值域 [0,2]。
    """
    p = pred.mean(dim=-1)
    y = target.mean(dim=-1)
    p = p - p.mean()
    y = y - y.mean()
    denom = (p.square().sum() * y.square().sum()).sqrt().clamp_min(1e-8)
    return 1.0 - (p * y).sum() / denom


__all__ = ["normalized_mse", "daily_ic_loss"]
