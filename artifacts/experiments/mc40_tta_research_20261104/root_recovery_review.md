# MC40-TTA2 事故修复增量复核

范围：远端39f054a→f9743ed，代码提交0c1315952faeddf6c23b2f3e5bff9c12e2de88f0。只读取变更、恢复证据、离线测试源码及日志；没有执行测试、DDB/GPU，没有查看旧行情缓存，没有修改远端或签署gate。

## 决策建议

**学习侧可以接受显式证据迁移，由root绑定最终源码、原协议和evidence_migration.json精确SHA后启动已授权长回放。无需仅因身份和评价代码变化机械重跑24流GPU先导。** 此判断只覆盖学习先导/网格，不接受经济评价等值。永久事故限制保留，不能改称整轮封存未读。原墙钟截止1793791822.565467、GPU累计782.072秒未重置。

**解释或发布相对冻结原G1/MC40收益增量前，仍需要新隔离评价器的真实允许期参照账户复现门禁。** 该门禁不阻塞与经济评价隔离的学习长回放；可在其后按预算完成。不能仅用合成账户测试宣布新旧数据/表达式引擎等值。

## 原四项Important复核

1. 身份和评价隔离：已实质修复。source_hashes包含market_guard/provider、本地执行依赖、恢复工具和完整当前QLib py/dos/so及原adapter协议。isolated_evaluation拒绝已初始化/非空Qlib状态，明确关闭并读回核验expression/dataset/calendar cache，独立UUID namespace；D.features直接到EvaluationReader，不经DatasetD/ExpressionD或旧DDB FeatureEngine。EvaluationDBClient新建单会话不加载DDB函数/共享池；session只允许有日期谓词的原字段SQL，calendar/member表只暴露固定列；封禁备用行情接口及已加载QLib backend别名。嵌套market_guard传播严格标记，不能先静默截断再隐藏越界请求。正常受审调用链未发现旧缓存内容进入后续学习/评价的路径。Python私有属性不是恶意代码沙箱，此结论针对冻结受审代码；不宣称任意代码均不能自行另连数据库。

2. 账户对账：已修复。reporter校验reconciliation.exact，并观测真实TopkDropoutStrategy每步只请求上一交易日信号、成交配置open、调用数与步数一致。它还核查固定Top50/drop5/hold5/费率等参数、冻结G1逐格网格，保存账户产物hash与信号身份。安装版本没有delay_execution参数不再被误当必然失败；实际信号时点是更合适证据。无效open时沿用QLib close回退已披露。下述新旧真实参照等值仍是独立待办。

3. 证据绑定/迁移：已修复。replay验证pilot/grid protocol_sha，require_gate重新计算实际协议canonical SHA，require_evidence对旧完整artifact SHA、旧source_hashes、目标完整source_hashes、scope及root gate指定迁移文件SHA逐项绑定。旧证据原地未伪改且有archive；迁移文件approved=false不影响“root通过gate显式批准”的设计。逐文件变化为tta_runtime的身份/门禁，tta_run的证据入口，tta_report的评价，以及单测gate fixture；tta.py、windows、原model/inference、update/scoring算法未变。live_provider、Budget、worker_lock及pilot/grid/learner函数体AST保持。新增依赖历史SHA不存在这一限制有保留，不伪造历史身份。此范围可接受迁移；root须执行绑定核验，不能只以文件名批准。

4. CPU评价预算：已修复。先获worker_lock再构建Budget，防止错误结算活worker；watchdog覆盖整个evaluate，回测后及最终结果发布前check，finally停止watchdog。原budget未改；恢复及评价不得创建新的first_model_at来延期。

## 事故隔离与参数选择

isolation_audit保留实际返回最大日未知、原缓存内容未知、服务器孤立会话未知。原回归日志无价格/收益/max_diff，仅通过状态及warning；通过状态确实泄露请求区间一致性断言。测试源未保存G1/原adapter，旧测试HF基座与正式冻结本地G1不同；正式模型、参照信号、纠偏账户及协议hash均保持，adapter canonical协议匹配。后续没有读取旧pytest数组或缓存的代码通道。由此有依据继续隔离后的研究，但不能证明实施者未受任何未记录输出影响。协议/超参未变且没有新候选收益，支持未根据事故结果重新选参数；这不是对人类不可见认知状态的证明。

## 新评价表达式的等值边界

EvaluationReader仅白名单原字段和三种固定表达式；最终输出float32与旧LocalExpressionProvider边界一致。存在表达式时扩展observed codes×observed dates panel，Ref按panel日期shift(1)，缺失股票格不跨日填补；查询首行返回NaN，不额外取历史，成员spans在计算之后掩码。这与已读旧DDB panel/查询局部Ref设计一致。没有用未来价格填补首行，也未改窗口lookback。

但不能据静态近似宣称所有真实输入逐位相等：原DDB运算的中间数值类型、NaN/null比较、不同partition/panel边界、上市/退市spans及极限阈值附近浮点行为都可能影响成交。`account_probe.py`使用手写Pandas reference，不调用旧DDB表达式引擎；它确实跑真实QLib账户并验证report/positions/trades/balances/contributions逐位相同，但证明的是合成reference与新reader，不是旧真实引擎与新reader。缺失格和涨停日覆盖有价值，仍不能替代允许期基准复现。

最小经济验收：在新独立进程、同一isolated_evaluation和固定执行参数下，对W3/W4的原G1、状态适配MC40共四个冻结信号账户重放；仅SQL读<=2026-07-24，绝不调用旧表达式引擎/旧行情缓存。与冻结纠偏report逐日逐列比对，账户自身exact、实际上一日信号/open配置均需通过；若旧持仓成交产物存在也比较，不存在则明确可比范围。差异超过事前固定数值容差时停止“候选相对旧参照”的解释，保存差异并定位，不调候选参数、不覆写旧参照、不把不同引擎结果直接相减。通过后保存独立parity artifact，绑定四个信号/旧report/新report/源码/protocol哈希；最终报告消费该门禁或由root在发布前强制核验。

## 测试与剩余限制

已阅读offline_tests.py：显式两文件白名单，屏蔽dotenv/真实初始化、HF离线/CUDA隐藏，seccomp拒绝connect/sendto/sendmsg/sendmmsg并由子进程继承；日志32 passed in 11.01s。恢复测试包含fresh-child Qlib配置缓存陷阱、session/metadata旁路、嵌套守卫、协议迁移、对账/时点/网格、watchdog及锁先后。此轮审查未重新运行，采用源码核查和留存结果。旧完整测试成功不作为事故合规证明。

无新增阻塞学习的Critical/Important。剩余Important为真实参照账户等值尚未验证，限制经济比较发布。若后续为加入参照门禁修改评价源码，应在长回放前冻结最终方案，或以同样显式迁移记录处理后续仅评价变动，不能静默修改已提交日事务身份。当前结论不构成root签字，本审查未签gate。
