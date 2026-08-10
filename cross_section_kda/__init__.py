"""Kronos 表示 + KDA 头 · 五臂对照实验代码包（计划 §1）。

公共符号按子模块分批导出，避免循环 import。需要 DolphinDB / GPU 的重对象
（backbone、data 里的 provider）采用惰性 import，保证纯张量测试（因果性 /
形状）不触发 qlib 初始化。
"""
from __future__ import annotations

# 轻量组件：import 即用，无 qlib / 模型依赖
from cross_section_kda.kda_modules import (
    CausalDepthwiseConv1d,
    KimiDeltaAttention,
    KimiLinearBlock,
    RMSNorm,
    SwiGLU,
)

__all__ = [
    "CausalDepthwiseConv1d",
    "KimiDeltaAttention",
    "KimiLinearBlock",
    "RMSNorm",
    "SwiGLU",
]
