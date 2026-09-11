# SAE / PEER 与残差诊断证据归档

服务于[最新封盘记录](../../Kronos残差预测头研究结论与封盘记录_20260911.md)。22件文件共3,001,160字节，约2.86MiB，全部按固定提交原字节复制。每件文件的来源提交、原路径、字节数和 SHA256 见 [manifest.json](manifest.json)。

| 目录 | 主分支保存的内容 |
|---|---|
| sae | AE/SAE与G1的同口径结果汇总 |
| peer | 原判据/协议身份、三臂两窗结果汇总、六份逐日报告、两张主图 |
| diagnostic | G1零残差对照汇总、35,584行监督格小表、逐日诊断、纠偏摘要、复算表、图、时间及当时64项测试记录 |
| engineering | e8a19e3生产身份接线修复说明；75项测试状态另见最新封盘记录 |

这是只读证据，不是可执行实验目录。JSON里的原Debian路径仅用于溯源；不自动读取那些路径。旧`correction_summary.json`与测试输出保留当时字节，其接线完成声明须结合后续修复说明阅读。没有归档大隐状态缓存、底座权重、数据库凭据、mlruns或forward产物，也没有合入实验实现代码。

在仓库根目录以 Mac conda base 的 Python 复算 PEER 曲线末值：

```python
from pathlib import Path
import pandas as pd

root = Path('docs/实验档案/20260911')
for w in ('W3', 'W4'):
    ends = {}
    for a in ('G1_mean', 'PEER_OFF_s100', 'PEER_ON_s100'):
        report = pd.read_parquet(root / 'peer' / f'report_{w}_{a}.parquet')
        ends[a] = float((report['return'] - report['cost']).cumsum().iloc[-1])
    print(w, {a: 100 * (v - ends['G1_mean']) for a, v in ends.items()})
```

诊断小表包含 `y_mean/s_G1/r_hat_OFF/r_hat_ON/s_OFF/s_ON`，日期与股票键完整。收益量纲 MSE 可直接由 `(y_mean-s_arm)**2` 先日内均值再跨日等权均值得到；归一化 MSE 再除以 `0.04908841653707577**2`。RankIC 逐日计算分数与 y_mean 的 Spearman，不把所有日期股票拼成一个相关系数。

完整性校验：

```python
import hashlib
import json
from pathlib import Path

root = Path('docs/实验档案/20260911')
manifest = json.loads((root / 'manifest.json').read_text())
for item in manifest['files']:
    data = (root / item['path']).read_bytes()
    assert len(data) == item['bytes']
    assert hashlib.sha256(data).hexdigest() == item['sha256']
print('22件归档身份一致')
```

哈希证明归档文件完整，不证明旧权重到旧信号的生成关系，也不追认历史协议符合性。
