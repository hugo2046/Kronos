# G9：过拟合墙的另一侧（predictor checkpoint 选择规则）· 预注册实验计划

> 日期：2026-08-21
> 新分支：`feature/g9-ckpt`，从 `feature/g8-e1` 切出（首个提交 = 本计划）
> 立项依据：五次微调（F1/G1/G4/G8/D-tok）predictor 的 best checkpoint **全部 = epoch 1**
> （按 val token 交叉熵早停）。该选择规则继承自官方配置，从未被审视。交叉熵衡量
> "下一个 token 猜得准"，alpha 取决于"生成路径的截面排序质量"——两者可能脱钩
> （本项目已两次实证同类脱钩：早停段 RankIC ⊥ oos；val ⊥ oos）。**墙的另一侧从未
> 被评估过。** 附带命题：若 predictor 只训 2000 步，G1 的改善可能主要来自训满 15 epoch
> 的 tokenizer——本轮 E0 臂免费检验此命题。
> 执行解释器：`/home/user/miniconda3/envs/quant/bin/python`（DDB + RTX 5090）。
>
> **执行提示（agentic workers）**：按阶段顺序执行，checkbox 跟踪；每阶段贴实际输出。

## 0. 已确认事实（直接采信）

| 项 | 值 |
|---|---|
| 在位者 | G1 s100（epoch-1 checkpoint）：backtest AER 等权 **+14.33%** / 指数 +10.66%；2025H2 亦正 |
| G1 配方 | ashares 并集、train 2014-01-02~2024-12-31、val 2025-01-01~2025-06-30、两阶段、predictor epochs=15、lr 4e-5、batch 50、每 epoch 2000 步、seed 100；G1 s100 tokenizer 在盘（共享工件，本轮**不重训**） |
| 确定性 | G2 实证：同 seed 重训 epoch1-3 val 逐位一致——重训可复现在位者，配对对照成立 |
| 评估窗 | backtest 2026-01-01~2026-07-24 + 2025H2（配方与 G1 同、val 仍为 2025H1，2025H2 对本轮**合法**） |
| 噪声框架 | ±26pp（134 日窗）；判据符号型 + 配对；带内"不可判" |
| forward | 2026-07-25 起零判读不变 |

## 1. 设计（唯一变量 = 选用第几个 epoch 的 predictor checkpoint）

**训练**：加载 G1 s100 tokenizer（冻结），predictor 按 G1 配方逐字重训 seed 100，
**保存全部 15 个 epoch 的 checkpoint**（`finetune_suite/train_predictor.py` 只存 best——
复制为 `g9_ckpt/train_predictor_all_epochs.py`，唯一改动 = 每 epoch 落盘，diff 注释标明；
**不改** `finetune_suite/` 任何文件）。逐 epoch val CE 表必列。

**臂（冻结）**：

| 臂 | checkpoint | 角色 | 推理窗 |
|---|---|---|---|
| **E1** | epoch 1（重训） | 在位者复现锚（配对基线） | backtest + 2025H2 |
| **E15** | epoch 15（训满） | **唯一预声明对照臂** | backtest + 2025H2 |
| E5 / E10 | epoch 5 / 10 | 描述性曲线，**不进判据** | backtest |
| E0 | epoch 0 = 官方 predictor + G1 tokenizer（零训练） | 免费诊断："alpha 主要来自 tokenizer 吗" | backtest |

推理逐字 canonical（L=90/H=10/N=20/T=1.0/seed=42），mean 判据 + 四变体记录族；csi300。
单种子（s100）——已声明局限，K2 有利结果须经三种子确认后才允许改协议（§3 条件项）。

## 2. 判据（跑前冻结）

| # | 判据 | 冻结定义 |
|---|---|---|
| K0 复现锚 | E1 backtest AER(等权) 与冻结 +14.33% 差 ≤ 1pp → 重训复现在位者；否则如实披露，配对基线改用 E1 实测值（配对仍成立） |
| K1 存活 | E15 backtest AER(等权) > 0 且 AER(指数) > 0 |
| **K2 方向（核心决策规则）** | 配对差 (E15 − E1)：backtest 与 2025H2 **两窗同号**——两窗均 ≥ 0 → "CE 早停规则可疑，训满 checkpoint 为候选"（触发 §3 三种子确认）；两窗均 < 0 → "CE 早停正确，过拟合墙为真"；异号 → "不可判"，记录 |
| K3 显著 | 任一窗 \|E15 − E1\| > 26pp → 允许强措辞（"显著更好/更差"）；带内只许方向性描述 |
| K4 关闭 | E15 双基准 ≤ 0 且 backtest 配对差 < −26pp → "墙另一侧无 alpha，选择规则议题关闭" |
| E0 判读 | 仅描述：E0 backtest AER 与 E1 并列——E0 若接近 E1（差在噪声底内）→ "alpha 主要由 tokenizer 承载"记录为机制线索；若深负 → "predictor 的 2000 步适配不可或缺" |
| 纪律 | E5/E10 只画曲线；**禁止**事后选任何非 E15 的 epoch 当结论；禁止改 lr/步数 |

## 3. 条件后续（本轮不执行）

| 候选 | 触发条件 |
|---|---|
| 三种子确认（s101/102 全 epoch 重训 + E15 评估） | K2 "训满候选"触发 |
| 再训协议修订（选择规则由 val CE 早停改为训满 / 或双 checkpoint 并跑） | 三种子确认通过后，按协议 §5 公开修订流程 |
| 学习率/步数实验 | **永不**（属超参搜索） |

## 4. 步骤

- [ ] 4.1 建分支提交本计划；`g9_ckpt/` + 测试先 FAIL 后 PASS：`test_all_epoch_saver`
  （15 个 checkpoint 目录齐全、每个可 `from_pretrained` 装载）、`test_recipe_frozen`
  （除落盘策略外与 G1 配方逐字段一致）；
- [ ] 4.2 predictor 全 epoch 重训（~3-4h），贴逐 epoch val CE 表；
- [ ] 4.3 六次推理（E1×2 窗、E15×2 窗、E5/E10/E0×backtest）落盘 + DuckDB 追加对拍
  （~7h，断点续跑）；**避开 16:30 登记 cron 时段或确认 cron 已恢复后错峰**；
- [ ] 4.4 引擎全表 + epoch 曲线图（x=epoch，y=AER 等权，E1/E15 加粗，E0 虚线）；
  K0~K4 + E0 一次封盘判读；结果文档 `docs/G9实验结果_<日期>.md`；提交推送。

预算：训练 ~3-4h + 推理 ~7h ≈ **一夜**。

## 5. 纪律

- 唯一预声明对照臂 = E15；其余 epoch 描述性；K4 触发不做 epoch/超参搜索；
- 评估数字在六组信号全部落盘前不看；判读一次开封；
- 半污染窗声明沿用：K2 有利结论上限"候选，待三种子 + 前向确认"；
- 只新增 `g9_ckpt/`；G1 tokenizer/权重/信号只读；`finetune_suite/` 零改动；
  引擎零改动；forward 零判读；
- 每阶段贴实际输出，测试先失败后通过；提交规范同 CLAUDE.md，
  `Co-Authored-By: Hugo <shen.lan123@gmail.com>`。
