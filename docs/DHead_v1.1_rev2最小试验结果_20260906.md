# DHead v1.1 rev2 最小试验结果 · 2026-09-06

> 方案：`docs/多期限预测头蒸馏v1.1方案_20260905.md` rev2 节（判据跑前冻结）；
> 修订要求：`docs/DHead复核与最小试验修订要求_20260906.md`。
> 执行：zcode @ Debian（RTX 5090），分支 `codex/dhead-distill`。
> 命令与退出码见 §6；产物外置 `Kronos-dhead-artifacts/`。

## 1. 结论速览（按冻结措辞）

- **工程检查（末 epoch 训练 D ≤ R_train）：A 臂 NO（D=0.0534）、B 臂 NO
  （D=0.0309，=1.10×R_train）→ 两配置在本预算（200 epoch）下未通过最小
  拟合检查，原因待诊断**。不据此判容量不足、架构无效或扩大搜索。
- 诊断性观察（仅本批 8 日样本含义）：B（R2 仿射还原接口）显著优于 A
  （旧 raw-return 接口）——初始 D 5.60 vs 158.93（约 28×），末点 D 0.031
  vs 0.053，逐日 Spearman(vs 教师 r0) 0.827 vs 0.636；两臂均大幅优于可部署
  的全局逐期限均值基线（E=0.0884）。A/B 差异无泛化/收益/main 解锁含义。
- 教师模式敏感性（只记录，不加法分解）：eval 态教师在本 8 日的
  R_train=0.0280；v1 train 态（dropout 开启）在 val 窗的 R=0.0220——不同
  窗口不同模式，不可相减作 dropout 占比。

## 2. 数据与教师

| 项 | 值 |
|---|---|
| 子清单 | pilot train 前 8 决策日 × 512 样本（只读切片，hash `db662186f3e2`） |
| 教师 | eval 态断言后生成；N20/T1.0/top_p.9/top_k0、keyed RNG 原口径、3 replicas；namespace `v11-rev2`，512×3=1536 次 171.7s（8.94 次/s） |
| R_train | **0.0280**（同 8 日 replica1/2 按日等权 MSE_norm，冻结 scale） |
| 全局逐期限均值基线 E | **0.0884**（仅由 8 日训练 target 计算；可部署参照） |
| （事后 oracle 参照） | 当日教师截面均值未列为主要基线；如引用须按 rev2 标注"不可部署、事后参照" |
| replica 两两 Spearman | r0r1=0.783，r0r2=0.816，r1r2=0.800 |
| v1 产物隔离 | 旧 `teacher-pilot-*` 目录零改动（64+16 分片原样）；新目录 `teacher-v11-rev2-pilot-train-db662186f3e2`（8 分片，schema=2+内容hash） |

## 3. 两臂全表（seed=42，200 epoch 无早停，lr 3e-4/wd .01/clip 1，纯蒸馏 D；末 epoch 为冻结诊断点）

| 指标（按日等权） | A：raw_return | B：normalized_close_affine_return |
|---|---|---|
| epoch0（优化前）训练 D | 158.927 | 5.605 |
| 末点训练 D（固定权重回评） | **0.0534** | **0.0309** |
| 末点 E（vs replica1，独立） | 0.0590 | 0.0450 |
| D ≤ R_train(0.0280)？ | **NO**（1.91×） | **NO**（1.10×） |
| 逐日 Spearman(vs r0) 均值 | 0.636 | **0.827** |
| 有效 Spearman 日 | 8/8 | 8/8 |
| 学生信号截面方差（末点） | 5e-05 | 2.5e-04 |
| 末 epoch 梯度范数 | 54.4 | 4.19 |
| 非有限值累计 | 0 | 0 |
| 训练曲线（训练态 task loss） | 92.90 → 1.06(ep50) → 0.25(ep100) → 0.11(ep150) → 0.090(ep199) | 5.60 → 0.13 → 0.054 → 0.050 → 0.052 |
| 耗时 | 94.8s | 96.1s |

注：history 内 `train_loss` 是该 epoch 训练过程中（权重移动）的均值，
"末点训练 D"是末权重固定后的回评——两者口径不同，均已如实并列。
A 臂梯度范数（54）显著大于 B（4.2）且曲线在 ep199 仍未见平台——与
"接口尺度病态使 A 优化困难"的解读一致，但属诊断观察，非结论。

## 4. 允许结论边界（rev2 §3.3 冻结代入）

1. 两臂均未过工程检查（D≤R_train）→ **"本配置在本预算下未通过最小拟合
   检查，原因待诊断"**。B 臂距阈值 10%，且无逐股方差/排序达标的证据，
   不得声称逐股信息已学会；也不判容量不足/架构无效/扩大搜索。
2. sqrt(R_train)=0.167（scale 单位）是两 replica 差值 RMS，不是单 replica
   噪声；教师 replica 两两 Spearman≈0.8 表明其逐股信号中度可复现。
3. A/B 差异只有本批样本诊断含义。main、D1/D2、forward、经济评价保持锁定
   （main 入口已被 R4 门禁实际拦截：pilot 保真门禁未过）。
4. main 诊断窗保真、端到端速度、economic 真实 adapter：**未完成**（未运行），
   如实封锁；"全量实施已验收"未达成。

## 5. 交付物与核验

- 汇总：`Kronos-dhead-artifacts/v11rev2-minimal-fit-v11-rev2/summary.json`
  （含两臂逐 epoch 全 history、身份 hash：教师权重/代码/清单/scale/输出语义）；
  两臂训练目录 `v11rev2-{A,B}-s42/`（逐 epoch ckpt + result.json）。
- 离线回归：`pytest tests/ -k dhead` → **67 passed**（R1 多 epoch 反例、
  R2 +900 平移反例与教师公式逐位对拍、R3 分片篡改/namespace 隔离、
  R4 门禁实检各就位）。
- 真实资产预检（试验前）：G1 eval 断言 OK；R2 真实数据对拍（64 样本×
  z=0.7）最大偏差 1.7e-16。

## 6. 复现命令与退出码

```bash
export DHEAD_BASE_REPO=/home/user/workspace/Kronos
export DHEAD_ARTIFACT_ROOT=/home/user/workspace/Kronos-dhead-artifacts
cd /home/user/workspace/Kronos-dhead-distill
PY=/home/user/miniconda3/envs/quant/bin/python

$PY -m pytest tests/ -q -k dhead                    # → 67 passed, exit 0
$PY -m dhead_distill.cli minimal-fit --namespace v11-rev2
# 本轮实跑 exit 0；累计 GPU 195s（上限 3600s）：
#   教师重生成 171.7s（首次；缓存后即时装载）+ A 臂 94.8s + B 臂 96.1s
```

执行偏差披露：首跑因 `_day_metrics` 取键错误（E 应由 replica1 现算）与
`trainer.history` 笔误两次中断（exit 1），修复后干净重跑；两次中断未产出
任何计入本表的数据，A 臂残留 run 目录已按身份隔离原则清除重训。
