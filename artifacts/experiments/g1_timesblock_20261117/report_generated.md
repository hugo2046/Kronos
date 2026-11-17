# G1周期残差分支研究报告

阶段：两臂及五对象诊断性回测完成；固定联合收益门槛通过=False，不自动采纳。

改动仅为冻结原G1上的90日历史残差编码；原粗细生成头、均值信号和交易规则保持原口径。TimesBlock与普通Temporal控制参数相同，但有效感受野和实测GPU成本不同。

协议SHA：`c8c0b36e08908ca57fe2cf6672cbd5436002420a8fe5e901bf8477f0e08fc6cf`。训练seed100；新臂推理seed42/100各20路径。分支codex/g1-timesblock-20261117。

## W3

![扣费累计收益（每日净收益累加）](../artifacts/experiments/g1_timesblock_20261117/figures/W3_main.png)

| 对象 | 年化 | 累计和 | 复利 | 夏普 | 最大回撤 | 成本 | 日均换手 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original G1 | 73.80% | 28.73% | 31.83% | 2.767 | -10.20% | 2.19% | 14.08% |
| G1 40 paths | 65.33% | 26.25% | 28.58% | 2.513 | -10.90% | 2.27% | 14.57% |
| State adapter MC40 | 83.20% | 31.50% | 35.35% | 2.867 | -10.17% | 2.15% | 13.79% |
| Temporal control | 81.90% | 30.75% | 34.87% | 3.411 | -6.77% | 2.17% | 13.96% |
| TimesBlock | 76.55% | 29.27% | 32.87% | 3.217 | -8.54% | 2.15% | 13.82% |

完整126日/37223股票日期格；Top50/drop5/hold5，次日开盘，9.5%简化成交阈值，原费用。

![五对象附图](../artifacts/experiments/g1_timesblock_20261117/figures/W3_all.png)
![差额与回撤](../artifacts/experiments/g1_timesblock_20261117/figures/W3_difference_drawdown.png)

## W4

![扣费累计收益（每日净收益累加）](../artifacts/experiments/g1_timesblock_20261117/figures/W4_main.png)

| 对象 | 年化 | 累计和 | 复利 | 夏普 | 最大回撤 | 成本 | 日均换手 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original G1 | 1.49% | 2.91% | 0.79% | 0.193 | -19.39% | 2.22% | 13.39% |
| G1 40 paths | 9.87% | 7.09% | 5.13% | 0.475 | -18.08% | 2.17% | 13.08% |
| State adapter MC40 | 5.22% | 4.08% | 2.74% | 0.336 | -14.32% | 2.38% | 14.36% |
| Temporal control | 4.15% | 3.80% | 2.18% | 0.287 | -14.84% | 2.15% | 13.00% |
| TimesBlock | -4.31% | -0.51% | -2.31% | -0.037 | -19.74% | 2.32% | 13.97% |

完整134日/39563股票日期格；Top50/drop5/hold5，次日开盘，9.5%简化成交阈值，原费用。

![五对象附图](../artifacts/experiments/g1_timesblock_20261117/figures/W4_all.png)
![差额与回撤](../artifacts/experiments/g1_timesblock_20261117/figures/W4_difference_drawdown.png)

## 逐指标差额与配对区间

| 窗口 | 参照 | 指标 | TimesBlock减参照 | 95%区间 |
|---|---|---|---:|---|
| W3 | Original G1 | annualized | 2.755百分点 | [-73.056, 36.408] |
| W3 | Original G1 | sharpe | 0.449 | [-0.573, 1.790] |
| W3 | Original G1 | mdd | 1.655百分点 | [-0.455, 3.653] |
| W3 | G1 40 paths | annualized | 11.221百分点 | [-8.368, 26.155] |
| W3 | G1 40 paths | sharpe | 0.704 | [0.086, 1.606] |
| W3 | G1 40 paths | mdd | 2.355百分点 | [0.356, 4.960] |
| W3 | State adapter MC40 | annualized | -6.643百分点 | [-46.586, 28.867] |
| W3 | State adapter MC40 | sharpe | 0.350 | [-0.669, 1.472] |
| W3 | State adapter MC40 | mdd | 1.631百分点 | [-1.372, 3.180] |
| W3 | Temporal control | annualized | -5.344百分点 | [-24.776, 14.236] |
| W3 | Temporal control | sharpe | -0.195 | [-0.821, 0.378] |
| W3 | Temporal control | mdd | -1.768百分点 | [-2.881, 0.752] |
| W4 | Original G1 | annualized | -5.796百分点 | [-40.120, 14.558] |
| W4 | Original G1 | sharpe | -0.230 | [-1.003, 0.589] |
| W4 | Original G1 | mdd | -0.355百分点 | [-4.435, 5.702] |
| W4 | G1 40 paths | annualized | -14.181百分点 | [-61.714, 12.788] |
| W4 | G1 40 paths | sharpe | -0.511 | [-1.501, 0.532] |
| W4 | G1 40 paths | mdd | -1.661百分点 | [-7.230, 5.842] |
| W4 | State adapter MC40 | annualized | -9.526百分点 | [-29.611, 18.997] |
| W4 | State adapter MC40 | sharpe | -0.373 | [-1.188, 0.505] |
| W4 | State adapter MC40 | mdd | -5.421百分点 | [-12.988, 0.891] |
| W4 | Temporal control | annualized | -8.456百分点 | [-21.980, 19.206] |
| W4 | Temporal control | sharpe | -0.324 | [-1.036, 0.538] |
| W4 | Temporal control | mdd | -4.907百分点 | [-11.885, 0.449] |

## 机制、区间与限制

2048验证窗口粗/细CE与残差幅度/频率/周期见diagnostics.json及diagnostics/*_validation.parquet；12日完整截面RankIC和Top50重合见diagnostic_daily.csv。20日循环块、5000次、seed100的逐指标配对区间见economic_result.json comparisons。

90日频谱可能主要反映趋势或噪声，谱峰存在不能证明交易机制；普通时序控制未胜时不得宣称周期机制优越。W3/W4被反复查看，仅为开发证据，非独立样本。current-final历史版本是假设，旧TTA越界实际返回范围未知，不声称历史全程未读。

执行修订：逐prefix90..100编码并核验全部token；非前缀一致时保持发行历史token不变，只追加成熟日prefix末位token。验证CE仅选择最早最低检查点，不筛除经济评价。

复现：`python -m g1_timesblock.pipeline all`。门禁由root独立签署；原始账户、信号、读取日志、预算及逐指标区间位于本轮OUT。实验分支不合并，发布索引由root执行。
