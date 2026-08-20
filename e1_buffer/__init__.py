"""E1 引擎缓冲带（计划 §2，20260820 G8+E1 计划）。

新包隔离条款（纪律 §4）：不改 ``paper_replication/engine.py`` 一行——
canonical 引擎与其助手函数一律 import 只读复用；本包只新增
缓冲带引擎（:mod:`e1_buffer.engine`）与三种子×两窗重放
（:mod:`e1_buffer.replay`）。
"""
