# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""KDA 组件的稳定入口。

组件实现留在 :mod:`cross_section_kda.kda_reference`（用户提供的 MIT 参考实现，
保持单一源头、不复制）。本模块只做按计划 §1 的「组件落入 kda_modules.py」的
稳定重导出，并集中放形状/构造期的不变式断言，供训练与测试统一引用。
"""
from __future__ import annotations

from cross_section_kda.kda_reference import (
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
