# G1 周期残差资源继承 R1 修复限定复审

2026-11-17。仅复核前轮R1及直接修改。审查脚本 `/private/tmp/timesblock_resource_inherit.py` 的实际SHA256为：

`ecb40cfec74913b71e1f55314d0924b0360af6d84a9467b2d0e8324d87973d81`

**R1：ADDRESSED。脚本限定复审：批准。** 未发现本次修复引入的新Important。可由root执行上述精确版本的一次性资源证据继承；本报告不是A_gate，仍须迁移后核验新身份链及实际14h GPU/16h墙钟可行性，再独立决定是否签A_gate。

核验结果：

- `REVIEWED_HEAD` 固定完整 `1f5b39170435e4243b24b16ec91923244d8f8bad`；锁内在任何archive或metadata变更之前调用 `verify_reviewed_checkout`。
- 预检使用 `git diff HEAD --name-only`，覆盖暂存区与工作区；HEAD错误或受跟踪文件变化立即拒绝，不再把未审暂存代码写成已审HEAD身份。
- `verify_evidence_sources` 对五个允许变化文件分别取得固定a7d264e模型来源及737aacd账户来源的 `git show` 原始bytes并计算SHA，与旧证据source字典逐项相等后，才继续已有数学兼容性检查。没有文本换行归一化，也没有仅凭文件名假定来源。
- 原预算未活动/未训练、协议唯一差异gpu_seconds、科学数学不变、资产SHA、旧记录归档、当前14/16实际预算可行性以及budget字节不变等断言均保留。
- manifest新增归档旧review_gate SHA和实际执行脚本SHA；新gate绑定本次最终复审文件 `timesblock_resource_revision_fix_review.md` 及迁移manifest。原始12h预算失败、旧证据SHA和未重新计算声明继续保留。

只读检查了纯fixture脚本：仅AST抽取两个预检函数到临时Git仓库，覆盖干净/错误HEAD、暂存与未暂存拒绝、五文件×两来源共10个SHA错误子项、固定HEAD及原断言保留和新增manifest字段。接受已报告4项通过结果，没有重复执行fixture或21项研究测试。

前轮资源代码审查结论继续有效：14h是root基于既有完整任务授权的显式执行规划修订，不能写成用户单独批准14h；16h墙钟和累计起点不变，不依据收益追胜。实际脚本若因时间流逝使预算可行性断言失败，应停止，不能改断言或重置计时。迁移后新metadata必须继续称为有原始来源的兼容性继承，不能称为新source重新计算出的先导证据。

本复审未执行迁移脚本本体，未改远端protocol/meta/gate/budget，未运行GPU或DB。批准仅针对上述脚本SHA、固定已审HEAD及其当前显式作用域。
