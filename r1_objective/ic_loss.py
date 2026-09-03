"""IC 损失（计划 §2：每日截面批内 −Pearson(pred, label_z)，qlib 惯例）。

设计要点：

- 输入是**同一天截面**的 (pred, label_z)——R1 训练批 = 每日截面（见
  ``run_r1_train.R1_BATCHING``），本损失不做任何跨日混合；
- label_z 已按日截面 z-score（``cross_section_kda.data.build_daily_samples``），
  Pearson 对 label 的仿射变换天然免疫，与 z-score 口径自洽；
- 分母 eps 防 pred 方差塌缩（线性探测初始化期）导致除零；
- 对 pred 仿射不变（尺度/截距不敏感）——这是与 MSE 分支互斥的数学特征
  （``test_r1_loss_ic`` 以此对拍互斥性）。
"""
from __future__ import annotations

import torch

_EPS = 1e-8


def ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """−Pearson(pred, target)，标量。输入 [N] 同日截面。"""
    p = pred - pred.mean()
    t = target - target.mean()
    cov = (p * t).mean()
    denom = p.pow(2).mean().sqrt() * t.pow(2).mean().sqrt() + _EPS
    return -cov / denom


__all__ = ["ic_loss"]
