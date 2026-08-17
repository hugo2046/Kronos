"""G4：市场上下文特征微调（计划 docs/G4特征微调实验计划_20260817.md）。

改输入不改解码：输入 6 → 9 列（idx_ret / mkt_vol / ma200_gate），AR 解码链路
一字不动。本包为 G4 唯一新增目录；既有目录（含 finetune_suite/、g5_head/、
kronos_qlib/、model/）零改动，G1 权重只读。

模块：
    - market_context：市场三列的因果计算（口径与 G3 登记的 MA200 门控一致）；
    - build_dataset：9 列语料 = G1 ashares pkl（只读）右连接市场三列；
    - surgery：tokenizer 6→9 列零初始化 warm-start（等价门禁的手术面）；
    - infer：9 列推理窗封装 + G4Predictor（不改 kronos_qlib/ 与 model/）;
    - config：G4Config（feature_list 9 列，路径指向本包）；
    - train：两阶段训练入口（warm-start + 复用官方训练循环）。
"""
