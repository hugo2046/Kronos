# Kronos 表示 + KDA 头 · 五臂对照改造实验计划

> 日期：2026-08-10
> 新分支：`feature/kda-repr-head`，**从 `feature/cross-section-zeroshot` 切出**
> （依赖其上的 `kronos_qlib` 数据层与 `cross_section/` 评估管线）
> 背景：[截面路径zero-shot测试结果_20260810.md](./截面路径zero-shot测试结果_20260810.md) §4.1 核对后确认
> 分歧主因是口径不可比；本计划落实东证报告的两条路径（表示→截面因子 + 轻量化微调），
> 下游头采用用户提供的 KDA（Kimi Delta Attention）浅层组件。
> **成功标准（用户定义）：同数据同区间下，改造版超过论文原版信号（B0）与纯监督框架（B1）即为改造有效。**

## 0. 实验矩阵（五臂，同数据/同区间/同评估代码）

| # | 模型 | 输入 | 可训练部分 | 性质 |
|---|---|---|---|---|
| B0 | Kronos zero-shot | 90 日 OHLCVA | 无 | 论文原版基线（**已测**：全区间 RankIC +0.0098，验证段见 §3） |
| B1 | KDA 纯监督 | 90 日 OHLCVA（窗口 z-score） | 全部（~0.1M） | "框架本身"基线，无预训练 |
| B2 | 冻结 Kronos + linear probe | Kronos 末步隐状态 [832] | 仅 linear（~0.8K） | 改造·最简版（归因用） |
| **B3** | **冻结 Kronos + KDA 浅层头** | Kronos 逐步隐状态 [90, 832] | KDA×2 + linear（~1M） | **改造·主菜** |
| B4（可选） | B3 + 生成路径统计特征 | 隐状态 + 路径特征拼接 | 同 B3 | 改造·全量版，B3 达标后才做 |

**归因逻辑**：B3>B0 且 B3>B1 → 改造有效；再看 B3 vs B2——若 B3≈B2 增益来自预训练表示，
若 B3>B2 则 KDA 头有独立贡献。B4 只在 B3 达标后作为增量探索。

## 1. 已有资产（直接复用，不重写）

- `kronos_qlib`：取数与窗口构造（9/9 验收通过，**不许改**）；
- `cross_section/evaluate.py`：RankIC / ICIR / 分组 / 多空评估（**五臂共用同一份**，保证可比）;
- `cross_section/data/signals_with_baselines.parquet`：B0 的 111 期信号已在此，直接读取，不重算；
- Kronos 隐状态提取：参照 `feature/direction-classifier` 分支 `model/kronos_classifier.py` 的
  `KronosProbeClassifier.forward`（tokenizer.encode → embedding → time_emb → transformer → norm），
  **重写这 ~15 行到新代码中，不要 merge 那个分支**；
- KDA 组件：用户提供的 `RMSNorm / CausalDepthwiseConv1d / KimiDeltaAttention / SwiGLU / KimiLinearBlock`
  单文件落入 `cross_section_kda/kda_modules.py`（保留 MIT 头注释）。**只取组件**，
  不要套用其 `KimiLinearModel` 的 qlib DatasetH 契约——我们的数据来自 `kronos_qlib` 窗口。

## 2. 数据与切分（预注册，不许事后挪动）

- 池：csi300，point-in-time（同前）；样本：**每个交易日**逐股票一条（不限于调仓日，扩大训练集）；
- 特征窗：≤t 的 90 个交易日；标签：`close[t+10]/close[t] - 1`，**按日截面 z-score** 后做回归目标（MSE）；
- 停牌/不足 90 日的剔除规则与 `build_inference_windows` 完全一致；
- **切分（含 10 交易日 purge，防标签窗口重叠泄露）**：
  - 训练：2022-01-04 ~ 2023-12-15；
  - 早停验证：2024-01-02 ~ 2024-06-14（与训练段间隔 ≥10 交易日）；
  - **最终验证：2024-07-01 ~ 2026-07-22（50 个调仓期，只跑一次，出结果前不许看）**；
  - 最终验证段同时是论文的时间隔离窗口起点之后，与论文口径对齐。
- 训练超参预注册：AdamW lr=1e-3（B2）/3e-4（B1/B3），weight_decay=0.01，batch=1024，
  epochs≤50，早停 patience=5（按早停验证段的日均 RankIC），seed=42。
  **超参只许在训练/早停段内调，最终验证段一次定型。**

## 3. 评估（与 B0 严格同口径）

- 信号：模型输出的截面得分；调仓日取每 10 个交易日（与 B0 的 50 个验证期完全相同的日期网格）；
- 指标：RankIC 均值 / ICIR / t 值 / 5 分组单调性 / 多空净年化（同 `evaluate.py`，成本口径不变）；
- **主判定（预注册）**：最终验证段 50 期上，
  `RankIC(B3) > RankIC(B0=+0.0185 该段实测)` **且** `RankIC(B3) > RankIC(B1)` → 改造有效；
- 次要报告：论文窗口 2024-07~2025-06 子段、B2 归因对照、各臂训练曲线；
- 全部五臂结果进同一张表，缺任何一臂不得下结论。

## 4. 工程要点

1. Kronos 隐状态**在线提取**（冻结主干 eval 模式 + no_grad，90-token 单次前向，无自回归），
   先不做磁盘缓存；若训练吞吐受限，可缓存 fp16 末段隐状态，但**特征内容不许变**；
2. B1 输入用与 Kronos 相同的窗口 z-score（clip 5），保证"预训练表示 vs 原始特征"是唯一变量；
3. KDA 的 90 步 Python 递归在日频短序列上可接受；若慢，先加大 batch，不许改架构；
4. 新代码入 `cross_section_kda/`；训练产物与信号 parquet 不入库；
5. 测试：KDA 组件形状/因果性单测（篡改 t 之后的输入，t 时刻输出必须不变——因果卷积
   padding 裁剪与递归都要覆盖）、隐状态提取与 `KronosProbeClassifier` 语义一致的对拍测试、
   切分 purge 间隔断言。**因果性测试必须做注入式验证**（去掉卷积裁剪应 FAIL）。

## 5. 纪律

- 最终验证段在五臂全部定型前**不许触碰**；跑完一次即封盘，无论结果好坏如实报告；
- 每阶段贴实际输出（`superpowers:verification-before-completion`）；
- 不改 `kronos_qlib` 与 `cross_section/` 既有文件（evaluate 若需复用以 import 方式，不复制粘贴）；
- 提交规范同 CLAUDE.md，`Co-Authored-By: Hugo <shen.lan123@gmail.com>`；
- 产出 `docs/KDA改造实验结果_<日期>.md`：五臂总表 + 归因 + 判定 + 局限。
