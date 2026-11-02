# SDD ledger — plan: docs/SAM适配器单变量实验计划_20261102.md

2026-11-02 首轮，BASE 9f5654507a5ba5afe88c8966956f6feecb7114ee；现有隔离工作树 codex/sam-adapter-20261102，无实施代理。
Pre-flight: Task1协议/config SHA -> Task2检查点与Task3信号身份；SAM/AdamW分别绑定算法及rho，唯一候选rho=.05。
Pre-flight: Task2控制精确复现 -> Task3 SAM门禁；复用cache，原训练路径默认行为必须不变。
Pre-flight: Task3完整finite mask -> Task4纠偏账户重放；原G1/MC40日报仅来自CORRECT新口径。
Ruling: 按用户内嵌冻结定义作为可执行Spec；脚本不可执行，ledger手动保留相同task-start/done证据，不修改技能目录权限 — 不影响实验变量，成本是人工记录漏项风险。
Ruling: 8个随机方向按4个独立方向及其负方向组成8个方向，每rho共8次CE；零扰动重复两次 — 满足正负成对8方向，成本是与8对的解读不同，报告明确。
Task 1: started; 读取AGENTS/研究规范/复制论文笔记；CodeGraph定位并读取train/run/market_guard/limit_unit_audit等。
监督器587289、tmux kronos-sam-adapter-20261102存在；GPU5090可见（sandbox外核实），无本实验GPU worker；不终止其他任务。
Task 1: protocol frozen commit 8d54168，canonical fee5cb7971e53577da3c792fb03a6207d54b50c7ad5cbcf26ca32e523c1a6cf3；原模型/语料/cache/adapter SHA已核验。
Task 2: started；SAM 9行为测试 RED(缺实现)→GREEN；可选train_arm dispatch RED(TypeError)→GREEN；诊断2测试RED→GREEN；runner身份及分离门槛3测试RED→GREEN待全套。
Task 2: 默认训练及预算/数据/推理/MC40/纠偏既有46测试通过；新SAM+dispatch+诊断合计58通过。
Task 1 diagnostic: 原MC40固定cache已完成，零扰动一致、原权重/base/RNG恢复全部通过；文件diagnostic_reference.json。
Task 2 engineering: 63 passed in 2.86s，完整命令 `OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/user/miniconda3/envs/quant/bin/python -m pytest tests/test_sam*.py tests/test_state_adapter*.py tests/test_mc40.py tests/test_limit_unit_audit.py -q`；全部RED输出保留verification。
Task 2 GPU smoke: 真实RTX5090，batch2；第一CE3.1437768936，第二CE3.1631140709，第二grad_norm .4150415361，原点还原、共同RNG、验证不污染、positions89..98及basehash均通过。
Task 2/3 running: tmux kronos-sam-gpu-20261102，PID595758，命令 `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/user/miniconda3/envs/quant/bin/python -m state_adapter.sam_run --stage train`。
Task 4: 实现报告生成器，必须有独立review_gate且全部源码身份一致才能发布完整报告；准备一次独立全分支审查。
Task 2: control complete。3000步beststep3000、CE2.448015436530113；所有12个adapter tensor max_abs=0，control_comparison.json。训练150.7 GPU秒，原默认行为未变。
Task 4 reference replay: CPU实际回放四份冻结原G1/MC40信号，market_guard读前保护；四份日报与CORRECT最大绝对误差2.275957200481571e-15，全部账户收益/费用对账通过。命令PYTHONPATH=. quant/python artifacts/.../verification/replay_baselines.py，真实日志reference_replay.log。
Task 3: SAM固定rho.05/seed100已进入训练，无rho或seed搜索；独立review正在审查全分支，无实施代理。
Task 3 training complete: SAM3000 updates/6000 forward-backward，beststep3000，CE2.4830846562981606，adapterSHA9ed6b8d808276dc1b4d0de20646ed61f41c91638cac07700e36f23085a60ef28。训练271.17秒，峰值1.075GiB。固定选点；验证CE较控制高不构成停止经济诊断理由。
Task 1: complete，原MC40与SAM最终权重均完成固定方向诊断；zero重复、原点恢复、basehash、全局RNG全部通过。
Task 2: complete，控制重训及真实GPU smoke通过。
Final review: fresh context sam_final_review，BASE9f565450..1916f7c，1 Important和2 Minor；独立CPU63通过。
Final: fixed 独立backtest/report信号身份绕过 — 12身份/内容/权重测试RED→GREEN，另1已完成信号重绑定RED→GREEN；全相关套件76通过。未改变训练变量，不作废既有训练；修复时无候选经济产物。
Final: minor (deferred): execution_version自动追加历史尚未实现；本轮每阶段手动快照execution_versions保留。
Final: minor (deferred): 零波动Sharpe=None报告格式分支未修；本轮真实日报将核验是否触发。
Final: Ruling: 审查员因最终经济产物未齐未判断收益/风险/请求审计 — 协调器在产物齐全后独立复算，不以CE外推收益 — 成本是后续数值审计仍必须完成。
接力部署：唯一GPU all runner PID603798，tmux kronos-sam-complete-20261102，代码4575b98；训练阶段已跳过，完整520日期×流预计约4.5小时，当前进度按signals/*.meta.json核验。all依次推理→回测→报告，不会止于信号。
另部署无GPU finalizer，等待图表报告后独立重算所有日报指标/配对区间/账户损益费用/finite mask/封存审计/预算，再本地提交报告与小产物。此脚本不新读市场数据，不改研究配置，不合并不push。


## 主协调器独立代码审查门禁
新上下文sam_review完成1916f7c审查及4575b98增量审查：Critical0/Important0/Minor1；27项初审CPU测试及18项增量CPU测试通过。五个最终源码SHA匹配，执行记录确认PID603798加载4575b98。审查后源码变化门禁曾正确拒绝旧SHA，已按真实增量审查结果绑定新SHA。
Final: minor (deferred): 前后向次数为名义次数；未来中断恢复的额外开销以GPU累计耗时账报告。当前两次训练均正常完成，不影响本轮计数。
Final: Ruling: 不强制扰动前后离散token相同，只控制共同随机数；强制相同会改变SAM目标。
Final: Ruling: 不追加SWA/rho/seed/等计算控制，保持首轮范围；结论不能声称多seed或等计算优势。
Final: Ruling: 验证临时state会在下一训练批重新设置，无参数/RNG污染，保留。
Final: Ruling: 8h执行窗不含16:25登记时段，不改通用卸载调度；未来改变时段须另核验。
Final: Ruling: 控制JSON门禁由完成权重SHA及实际tensor独立复核支持，不为假设篡改扩展；最终交付仍核验身份。
Final: Ruling: 不扩展标准AdamW.step失败事务回滚；临时扰动异常已覆盖，若实际异常必须按持久化未扰动checkpoint恢复。
Final: Ruling: report门禁现在由协调器写入；经济产物和图表仍需完整输出后独立复核。
Final: Ruling: 新增finalizer仅等待当前runner完成、读取产物对账并本地提交，不启动GPU或读取新行情；协调器检查其结束流程，最终本机复算仍为独立验收。

## 第2轮接力核验
2026-11-02 22:02 上海时间：sandbox内不可见PID及GPU，已授权sandbox外ps确认587289/603798/606992均存活，RTX5090利用率100%；未重启任何任务，原固定截止与累计预算保持。all_runner日志持续生成逐日窗口；最终收益与图表尚未完成，状态continue。

## 第3轮接力核验
2026-11-02T22:04:39.566283+08:00：sandbox外ps确认监督器587289、唯一GPU推理603798、finalizer606992存活；5090利用率100%。已落盘23/520日期×流。控制逐tensor精确复现证据再次核验；CodeGraph查询sam_run未索引，按授权原文核对all串行推理→回测→报告及finalizer独立复算/本地提交链。未重复启动、未修改算法/研究变量/固定截止；bot_status记continue，详见verification/relay_round3.json。

## 第4轮接力核验
2026-11-02T22:06:22.793571+08:00：sandbox外核实587289/603798/606992存活，5090利用率100%；已保存26/520日期×流。CodeGraph查询sam_run未索引，原文复核all串行推理→回测→报告及finalizer独立验收链。累计GPU快照1290.9/21600秒，原8小时截止保持。未重启、未更改研究配置；status=continue。

## 第5轮接力核验
2026-11-02T22:09:21.147079+08:00：sandbox外确认587289/603798/606992存活，32/520日期×流，累计GPU快照1481.5秒。未重启或改变研究配置/截止；现有执行链继续推理→回测→报告及独立复算。status=continue。

## 第6轮接力核验
2026-11-02T22:11:05.623344+08:00：sandbox外确认587289/603798/606992存活，RTX5090利用率100%；逐日meta已保存35/520日期×流，累计GPU快照1576.4/21600秒。CodeGraph未索引sam_run，原文确认all后续串行回测→报告与finalizer独立复算链。未重复启动、未变更配置或固定截止；status=continue，verification/relay_round6.json保留核验。

## 第7轮接力核验
2026-11-02T22:12:52.039207+08:00：sandbox外确认监督器587289、GPU runner603798、finalizer606992存活，GPU runner显存13092MiB；已保存38/520日期×流，累计GPU实时估算1695.6/21600秒。CodeGraph查询sam_run未索引，原文核验既有all串行推理→回测→报告及finalizer独立复算/本地提交链。控制精确复现及SAM训练均已有完成证据，不重复训练；未启动新进程、不更改研究配置或截止，status=continue。

## 第8轮接力核验
2026-11-02T22:15:21.846916+08:00：sandbox外确认587289/603798/606992存活、RTX5090利用率100%；已落盘43/520日期×流，GPU累计快照1829.6/21600秒。CodeGraph查询sam_run未索引，按授权原文核验all串行推理→纠偏回测→报告及finalizer独立复算/本地提交。未重复启动、未变更研究配置或固定截止；status=continue，详见verification/relay_round8.json。

Task 3: complete，四流520日期全部完成，finite mask与原G1逐格一致。
Task 4: complete，六账户真实回测/持仓/成交对账、配对bootstrap、三张图与中文报告完成；独立finalizer从原始日报和请求记录重新计算指标/区间/端点/预算/封存全部通过，见independent_recalculation.json。
