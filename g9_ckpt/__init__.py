"""G9：过拟合墙的另一侧——predictor checkpoint 选择规则（预注册实验包）。

计划：docs/G9checkpoint选择实验计划_20260821.md（feature/g9-ckpt 分支首提交）。
纪律：只新增 ``g9_ckpt/``；G1 tokenizer/权重/信号只读；``finetune_suite/`` 与
引擎零改动；E15 为唯一预声明对照臂，禁止事后挑任何其它 epoch。
"""
