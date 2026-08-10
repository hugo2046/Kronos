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


def __getattr__(name: str):
    """惰性导出较重的网络类（导入 backbone/models 会拉 torch.nn，但不会触发 qlib）。

    避免 ``import cross_section_kda`` 时一次性把所有子模块拉起。
    """
    if name in ("B1SupervisedHead", "B2LinearProbe", "B3KronosKdaHead", "count_trainable"):
        from cross_section_kda import models as _models
        return getattr(_models, name)
    if name == "KronosFrozenBackbone":
        from cross_section_kda.backbone import KronosFrozenBackbone as _B
        return _B
    raise AttributeError(f"module 'cross_section_kda' has no attribute {name!r}")


__all__ = [
    "CausalDepthwiseConv1d",
    "KimiDeltaAttention",
    "KimiLinearBlock",
    "RMSNorm",
    "SwiGLU",
    "B1SupervisedHead",
    "B2LinearProbe",
    "B3KronosKdaHead",
    "KronosFrozenBackbone",
    "count_trainable",
]
