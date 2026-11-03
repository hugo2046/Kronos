# SDD ledger — plan: docs/SAM与AdamW等更新续训计划_20261103.md

用户授权续训；同额外更新次数已说明，两臂+3000，非等计算量。原G1/MC40/旧SAM3000固定。新协议543a2d3133152ab196f0f3ef84ee0d20bc5358aee92d40de123ba3fe65fd9d79。
Task 1: complete；checkpoint迁移/网格/预算RED→GREEN，真实两次3001步恢复逐值一致。两臂6000完成，均best6000。固定eval_seed诊断已完成。训练在dad997f代码完成；修复只影响后续异常恢复，不改训练算法/权重。
Task 2: in progress；冻结两新best，1040日期流、纠偏账户回测及报告接续。
Final review: 独立sam_continuation_review已完成，Critical0/Important3/Minor1。
Final: fixed I1 latest提交但快照丢失——从current state按原eval seed重评恢复，不从best假造；RED→GREEN。
Final: fixed I2 running状态但训练PID死亡——命令身份与/proc竞态处理，及时needs_recovery，活进程不重复启动；RED→GREEN。
Final: fixed I3 partial数值漂移——每日期SHA，prepared journal与原子替换保障中断恢复；缺SHA/内容漂移拒绝，合法文件已提交meta未提交可从同身份journal恢复；两项RED→GREEN。
相关回归54/54通过，tests_after_review.txt。未重新派审查员，遵循一次修复流程；主协调器审查修复并绑定source SHA。
Final: minor (deferred): 自动报告未给末1000步实际趋势判断；最终交付由协调器补齐5000/5500/6000值与训练批趋势，不能宣称收敛。
Ruling: 不运行可能直连封存行情的全仓集成测试，限定受影响测试；代价是不能宣称全仓测试通过。
Ruling: 尚无完整1040日期流/账户/图时不能作经济获益结论，最终由协调器独立验收；代价是训练结果不能马上采纳。
Ruling: GPU12h是否足够由实际吞吐决定，预算不重置，触限报告未完成部分；代价是可能无法在固定预算内完成。
Ruling: 单训练seed/历史反复查看/非等计算是已公开研究边界；代价是只能给有界开发期证据。

用户后续明确确认两臂各追加3000更新。两臂6000训练完成，固定seed验证CE控制2.448015→2.321385，SAM2.483085→2.331213；经济结果仍待完整推理。
数据库配置恢复：新worktree缺.env，链接原sam-adapter既有配置，未输出/修改凭证；旧失败状态保留recovery_database_config.json，预算未重置。
I3真实落盘验证：AdamW6000_infer42/W3首个日期文件与逐日meta SHA一致；实际推理PID951641、监督PID951592，后续必须重新检查。
Final minor M1：late_training_trend.json保存末1000步具体结果，最终报告交付需加入实际描述，不可宣称收敛。
