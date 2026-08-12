"""improve_suite —— 状态假说与改进路线预注册实验（计划 §2~§6）。

本包**只新增**文件，不改 ``model/``、``paper_replication/``、``baseline_suite/``、
``cross_section*/``、``kronos_qlib/`` 任何既有代码（计划 §10 纪律）。

子模块：
    - :mod:`improve_suite.common`：``ImproveConfig``（canonical 配置 + 网格覆盖字段）
    - :mod:`improve_suite.path_inference`：逐路径推理仪器（对拍门禁保证均值逐位一致）
    - :mod:`improve_suite.path_store`：逐路径长表落盘 / 读回
    - :mod:`improve_suite.preflight`：环境断言（csi500 池、窗口、GPU）
"""
from __future__ import annotations

__all__ = ["common", "path_inference", "path_store", "preflight"]
