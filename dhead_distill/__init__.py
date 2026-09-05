"""DHead v1 包入口（无副作用：不 import 子模块 / qlib / HF / CUDA）。

模块职责见方案 §6 文件职责表；离线可 import 是最终交付检查项，
故入口不连带导入 torch 等重依赖。
"""
__all__: list[str] = []
