"""任务2（方案 §7）：损失函数数学断言（方案 §4.3 给定的两测试 + 独立手算）。

所有断言用数学独立手算验证，不只比较两个调用相同实现的函数。
"""
from __future__ import annotations

import math

import pytest
import torch

from dhead_distill.losses import daily_ic_loss, normalized_mse


def test_mean_signal_matches_price_path() -> None:
    """期限均值信号 = 平均价格路径收益（mean over N 等价 close_t 归一后均值）。"""
    close_t = torch.tensor(100.0)
    prices = torch.tensor([[101.0, 103.0]])
    returns = prices / close_t - 1.0
    torch.testing.assert_close(
        returns.mean(-1), prices.mean(-1) / close_t - 1.0
    )


def test_losses_exact_match() -> None:
    """pred == target：normalized_mse == 0、daily_ic_loss == 0（方案原文）。"""
    y = torch.arange(320, dtype=torch.float32).reshape(32, 10) / 1000
    scale = torch.ones(10)
    assert normalized_mse(y, y, scale).item() == 0.0
    torch.testing.assert_close(
        daily_ic_loss(y, y), torch.tensor(0.0), atol=1e-6, rtol=0
    )


def test_normalized_mse_hand_computed() -> None:
    """手算：pred=1, target=0, scale=0.5 → ((1-0)/0.5)^2 均值 = 4。"""
    pred = torch.ones(4, 10)
    target = torch.zeros(4, 10)
    scale = torch.full((10,), 0.5)
    val = normalized_mse(pred, target, scale).item()
    assert val == pytest.approx(4.0, rel=1e-6)


def test_normalized_mse_per_horizon_scale() -> None:
    """逐期限尺度：h=0 尺度 1、h=1 尺度 2 → 同误差下 h=1 贡献为 h=0 的 1/4。"""
    pred = torch.zeros(1, 2)
    target = torch.ones(1, 2)
    scale = torch.tensor([1.0, 2.0])
    # h=0: (1/1)^2=1；h=1: (1/2)^2=0.25 → mean=(1+0.25)/2=0.625
    assert normalized_mse(pred, target, scale).item() == pytest.approx(0.625, rel=1e-6)


def test_daily_ic_loss_opposite_ranking_is_two() -> None:
    """完全反序：cos 相似度 = -1 → loss = 1 - (-1) = 2（手算）。"""
    y = torch.arange(32, dtype=torch.float32).reshape(32, 1).repeat(1, 10)
    pred = -y.clone()
    val = daily_ic_loss(pred, y)
    assert val.item() == pytest.approx(2.0, abs=1e-5)


def test_daily_ic_loss_hand_computed_small_case() -> None:
    """独立手算 4 样本情形：p=[1,-1,2,-2], y=[3,-3,1,-1]。"""
    # 均值化：p̄=0, ȳ=0；Σpy = 1*3 + (-1)(-3) + 2*1 + (-2)(-1) = 3+3+2+2 = 10
    # Σp² = 1+1+4+4 = 10；Σy² = 9+9+1+1 = 20 → cos = 10/sqrt(200) = 0.7071…
    p = torch.tensor([1.0, -1.0, 2.0, -2.0]).unsqueeze(-1).repeat(1, 10)
    y = torch.tensor([3.0, -3.0, 1.0, -1.0]).unsqueeze(-1).repeat(1, 10)
    expected = 1.0 - 10.0 / math.sqrt(10.0 * 20.0)
    val = daily_ic_loss(p, y)
    assert val.item() == pytest.approx(expected, abs=1e-6)


def test_daily_ic_loss_constant_input_finite() -> None:
    """常量输入（预测方差 0）：loss 有限（epsilon 保护）。"""
    y = torch.randn(32, 10)
    pred = torch.full((32, 10), 0.37)
    val = daily_ic_loss(pred, y)
    assert torch.isfinite(val)
    # 常量 p：均值化后全 0 → 分子 0，分母被 clamp 到 1e-8 → loss = 1 - 0 = 1
    assert val.item() == pytest.approx(1.0, abs=1e-4)


def test_daily_ic_loss_uses_horizon_mean() -> None:
    """IC 用期限均值信号：两臂只在均值上不同时 loss 一致（§4.3）。"""
    base = torch.randn(32, 10)
    a = base + torch.randn(32, 1)      # 逐样本平移（改变均值信号）
    b = base + 100.0                    # 常量平移不改变均值化后的信号
    y = torch.randn(32, 10)
    la, lb = daily_ic_loss(a, y), daily_ic_loss(b, y)
    assert lb.item() == pytest.approx(daily_ic_loss(base + 0.0, y).item(), abs=1e-5)
    assert la.item() != pytest.approx(lb.item(), abs=1e-6)
