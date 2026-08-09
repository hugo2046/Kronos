"""cross_section —— Kronos 截面 zero-shot 因子实验脚本与配置。

对应 ``docs/截面路径zero-shot测试计划_20260809.md``（v2）。
按阶段组织：

    - ``config.yaml``：全口径锚定（采样 / 调仓 / 评估），阶段间不许漂移；
    - ``rebalance.py``：调仓日序列 + 可评估边界硬门禁（计划 §3.1）；
    - ``signal.py``：Kronos 预测 → H 日平均预期收益率信号；
    - ``baselines.py``：动量 / 反转基线因子（同口径同池同调仓日）；
    - ``evaluate.py``：标准单因子检验（RankIC / 分组 / 多空，含成本）；
    - ``pipeline.py``：全量信号生成（构窗 → predict_batch → signal → parquet）。

数据层依赖 ``kronos_qlib``（``feature/qlib-data-layer`` 落地，9/9 验收通过），
**不要改 ``kronos_qlib``**。
"""
