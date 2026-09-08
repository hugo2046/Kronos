"""G10H-pilot：原生输出头受限适配最小实验包（20260908 计划）。

A0（全参数）vs A1（仅 ``head.proj_s1/proj_s2`` 四张量）两臂 × seed=100，
官方 Kronos-base 起点 + G1 微调 tokenizer 冻结，G1 同源全 A 语料与训练配方
（DualHead CE / AdamW(0.9,0.95,wd0.1) / OneCycle 4e-5 / batch50 / 15×2000
更新）。评价 = FULL 逐日 W3/W4 生成式 mean 信号 × 真实 engine_v2。
"""
