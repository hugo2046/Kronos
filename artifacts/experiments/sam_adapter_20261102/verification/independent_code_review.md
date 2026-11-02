# SAM 单变量分支独立代码审查

审查日期：2026-11-02。结论：**通过，Critical 0 / Important 0 / Minor 1**。最终已审代码提交为 **4575b98edca00f333e9111dc33b4c42bed48b436**，最新五文件SHA见文末增量复核；以下保留首轮审查记录。当前版本可继续冻结候选训练和推理；此结论是代码审查，不等于尚未完成的经济结果或最终图表验收。

## 版本与范围

- 远端工作树：`dev-debian:/home/user/workspace/Kronos-sam-adapter`。
- 比较基点：`9f5654507a5ba5afe88c8966956f6feecb7114ee`；协议提交 `8d54168`。
- 初次读取 HEAD：`694ab35656f35ca2ffb1ffcb8cb047ee42cef4ac`。
- 首轮已审 HEAD：`1916f7c1d527b2e360d044408d80e0e487cbcda1`。期间新增报告模块及记录；已核对核心四文件未变化，报告模块原文也已审阅。
- 已审：SAM 更新、诊断、runner、报告生成器、train_arm 最小差异、行为测试、冻结协议；按调用链审查既有 checkpoint/RNG、预算、推理、数据窗和纠偏账户回放。
- 未执行：GPU 训练、GPU 推理、真实行情回测、报告生成。没有修改远端工作树、Git refs/index，也没有启动子代理。CPU 测试显式屏蔽 CUDA，禁用字节码及 pytest cache，临时目录位于 `/tmp`。
- CodeGraph 已优先使用；索引未含 SAM 新模块，按允许的降级方式读取新文件原文。

| 文件 | SHA256 |
|---|---|
| state_adapter/sam_train.py | 17b7957971b2dc818ddb5d8130a04df9134a44d90c6d6ac679adff5ae8028cac |
| state_adapter/sam_diagnostics.py | 572e6ee7fa6de2c333334cb2a807c44a7edca168cef8e572ac23df70942cc816 |
| state_adapter/sam_run.py | ad19642ef76e8f6ac4c2e0d00381c3623fa6e04a83875b1fdbde3614bc86d1e8 |
| state_adapter/train.py | df88ce847ae41f8be174197375de75b32f84d0ec1917db0c3df2f350bc143284 |
| state_adapter/sam_report.py | e57085ea26f279b545532e1507641f568cf24a2dc8b4d24f578a2f12cccbe23e |

## 唯一非阻断发现

**Minor：恢复后的前向/反向次数字段是名义次数，不能当作实际累计计算量。**

位置：`state_adapter/sam_run.py:184-185`（同文件174-177的批次日志也没有step/attempt身份）。成功后固定写入 SAM 6000 次或控制 3000 次。如果在两次500步checkpoint之间崩溃，已完成但尚未落盘的批次会重算，最终字段仍为6000/3000；训练日志可能重复且没有step来区分。不会改变恢复后的最终权重和冻结更新数，GPU时间仍由累计预算单独记录，因此不阻断本次当前正常运行。

建议：将这两个字段明确命名或注明为“最终有效训练轨迹的名义次数”；如果报告要呈现实耗前向/反向次数，另外记录attempt及累计计数。报告中的同更新次数/非等计算预算判断本身正确，无需改研究配置。

## 验证证据

独立命令：

```text
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/user/miniconda3/envs/quant/bin/python -m pytest -q -p no:cacheprovider --basetemp=/tmp/kronos-sam-review-tests tests/test_sam_train.py tests/test_sam_diagnostics.py tests/test_sam_run.py tests/test_state_adapter_train.py tests/test_state_adapter_budget.py
27 passed in 1.68s
```

另用CPU直接加载本轮control及冻结MC40权重，独立检查key集合与每个tensor的torch.equal，结果：12 tensors、107712参数全部逐值相同，beststep均为3000，CE均为2.448015436530113；不是仅相信control_comparison.json的布尔值。

已读取真实GPU smoke产物：第一CE3.1437768936157227、第二CE3.163114070892334、第二梯度范数0.4150415360927582；共同RNG、step前精确还原、验证RNG/权重恢复、未来位置89..98和basehash均通过。该记录由实施者产生，审查者未另跑GPU。

## 主要核查结果

- 仅state_adapter命名参数可训练，optimizer参数集合严格一致；train_arm强制107712参数、last2/rank32。tokenize一次，未裁剪第一梯度构造全局L2 rho=.05扰动。
- 两次前向共用token batch及CPU/CUDA/Python/NumPy初始RNG，成功只保留第一次前向后的RNG；未额外调用训练采样器。finally以copy_精确还原原参数，第二梯度clip1后仅一次AdamW.step；第二前向异常及非有限/零梯度不更新并清理梯度。
- rho=0测试走通用双阶段路径，连续3步参数、AdamW状态、RNG与原train_batch逐值相同；checkpoint连续/恢复等价测试通过，协议/config/seed/base身份绑定完整。
- validation与诊断不改变训练随机状态和原参数。reference/SAM均使用同一冻结2025H1 cache、同一eval_seed103100、同一方向seed20261102。4独立方向及负向共8方向的解释已登记。报告明确小CE变化不外推收益，也没有声称证明尖锐最优点。
- 训练WindowCorpus目标末日2024-12-31，验证目标2025H1；数据与G1权重文件SHA由legacy.load_protocol重验。推理CensoredProvider与market_guard、账户replay_account内market_guard限制真实行情至2026-07-24。
- 输出绑定独立SAM OUT；逐日信号绑定算法/rho/配置/协议/adapterSHA/seed/window/executionID；两个infer流各20路径及等权池化，覆盖要求对齐G1完整finite mask。
- 回测只使用execution('new')上下文，pctchange/100对比±.095，finally恢复；没有调用会写旧OUT的limit_unit_audit.run。G1/MC40实际重放与冻结纠偏日报比较，候选相同口径。
- 6h累计GPU预算cumulative=True、8h固定deadline、文件锁及活动PID拒绝重复执行；watchdog覆盖长算子。报告区分收益门槛与风险兼容，主图颜色/共享轴/净累计、含初始NAV的MDD、Sharpe及bootstrap口径与研究规范一致。

## Considered-but-declined

- 不要求两次前向离散采样token相同：共同随机数不保证扰动后输出分布对应相同token，协议与报告已准确限定。
- 不要求SWA、不同rho、额外训练seed或6000步控制：这些会改变用户已冻结研究范围。
- 不把已有train_arm验证过程中设置的临时adapter state作为污染：每次训练前向都会重新set_state，权重及全局RNG不变。
- 诊断/smoke未接登记守卫、train_arm守卫期间不卸载GPU是通用调度局限；本轮21:27开始且8h截止05:27，冻结时段不可能触及16:25—16:45，故不作为本次阻断发现，也不建议当前运行中无关改造。
- require_control依赖已生成比较JSON，infer直接入口没有重算控制tensor；正常all/train路径有完成权重SHA检查，本次已独立复核实际tensor，未发现当前产物漂移，不将假设篡改列为重要缺陷。
- 不要求通用optimizer.step故障事务回滚：当前冻结AdamW调用路径未见此类故障；本次明确测试的是临时扰动/第二前向异常，不额外引入优化器状态复制开销。
- 不把报告生成所需review_gate尚未写入当作缺陷：这是主协调器完成本次独立审查后的预期门禁。

## 交付前仍需外部结果验收

最终SAM权重及诊断、520日期×流完整覆盖、六账户真实回放与对账、请求边界JSONL、全部图表端点以及最终报告尚需在完成后由主协调器核验。代码审查通过不能替代这些结果门禁。若上述五个源码SHA改变，应针对差异重新审查。

## 增量复核：1916f7c → 4575b98

复核范围仅本次新增的verify_candidate_inputs、已完成信号SHA绑定、独立backtest/report身份门禁及对应tests/test_sam_run.py增量。结论：通过，无新增Critical/Important/Minor；首轮名义计算次数Minor仍保留。没有重复整分支审查、没有运行GPU、没有改远端文件或停止进程。

新门禁在合并/回测之前检查训练记录的算法/rho/协议/config/权重SHA，检查选点checkpoint的seed/family/base/step，检查四条流身份、完整性、signal_sha256、日期集合、封存最大日期及G1 finite mask。恢复infer时拒绝将已有完成流的内容篡改重新绑定到新SHA。报告入口也调用同一只读检查。未完成流只有部分日期、尚无最终SHA是正常状态；完成后由infer在覆盖检查通过后写入SHA，独立backtest/report拒绝缺SHA的流。

独立增量验证：

```text
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/user/miniconda3/envs/quant/bin/python -m pytest -q -p no:cacheprovider --basetemp=/tmp/kronos-sam-review-delta-tests tests/test_sam_run.py
18 passed in 0.88s
```

运行中版本凭据：

- `4575b98edca00f333e9111dc33b4c42bed48b436`提交时间2026-11-02 21:51:54 +08:00。
- `ps`显示PID603798于21:52:10启动，命令为`python -m state_adapter.sam_run --stage all`。
- `execution_version.json`记录同一PID、stage=all、commit=4575b98，以及sam_run SHA为245ceb96…，与本次审查源码一致。因此没有发现“进程早于修订启动、内存中仍运行旧infer”的问题。
- 读取时首条W3/infer42流已完成12日、complete为空、尚无signal_sha256；这是运行中partial的预期状态，没有伪装成完整信号。
- 复核时远端HEAD已推进至`1cd98a0fe1300cea85e83682aa38a12ab7d00a70`。4575b98之后仅新增ledger、execution_versions记录和`verification/finalize_when_ready.py`，五个核心源码无变化。**新增finalize_when_ready.py不在本次指定增量审查范围，未给它签发审查通过结论。**

最终已审代码版本：`4575b98edca00f333e9111dc33b4c42bed48b436`，门禁应绑定以下五文件SHA：

| 文件 | 最终SHA256 |
|---|---|
| state_adapter/sam_train.py | 17b7957971b2dc818ddb5d8130a04df9134a44d90c6d6ac679adff5ae8028cac |
| state_adapter/sam_diagnostics.py | 572e6ee7fa6de2c333334cb2a807c44a7edca168cef8e572ac23df70942cc816 |
| state_adapter/sam_run.py | 245ceb96f81e9a6fd36ed989358cb36636ef7ac26ab6b563325dd6d17a86a288 |
| state_adapter/train.py | df88ce847ae41f8be174197375de75b32f84d0ec1917db0c3df2f350bc143284 |
| state_adapter/sam_report.py | 6a81111dc3374d1714b7de71c58c8776b797876946456d2553edfd40b498e7fc |
