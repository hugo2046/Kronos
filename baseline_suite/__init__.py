"""Canonical baseline 四变体 + KDA long-only 重评 + 样本外延展（计划 2026-08-10）。

本包**复用** ``paper_replication`` 的引擎与基准（import，不复制），只新增：

- 四变体信号聚合（``last/mean/max/min``，全部除以现价）；
- KDA 三臂 long-only 重评（同一引擎，仅记录）；
- 样本外窗口（2025-07-01~2026-07-31）的预注册判定。

**纪律**（计划 §6）：
    - canonical 主线永远是 mean；四变体是预注册族，禁止事后挑最好的当主线；
    - 不改 ``paper_replication/``、``cross_section*/``、``kronos_qlib/`` 既有文件；
    - 样本外窗口数字在四变体 + 三对照全部定型前不许查看；跑完一次封盘。
"""
