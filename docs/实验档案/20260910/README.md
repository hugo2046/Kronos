# 关键实验结果归档

本目录随主分支保存，服务于[研究总结](../../Kronos预测层优化研究总结_20260910.md)。30件证据文件按固定提交原字节复制，合计2,461,486字节（约2.35MiB）；完整来源见[manifest.json](manifest.json)。

| 目录 | 内容 |
|---|---|
| h1 | 既有H1的IC结果背景 |
| mh1 | 多期限监督的原始IC判据汇总 |
| g1_grid | FULL/CROPPED日期对照汇总 |
| g10 | 原始评价及撤回V后的历史审计；不能只读原始评价就声称V通过 |
| qlib_mean | 原Qlib mean比较manifest/summary、6份逐日报告、2张主图 |
| lora_mean | LoRA protocol/pilot判据/history、6份逐日报告、2张图、4份逐股信号 |

这是只读证据快照，不是执行目录。JSON中原Debian路径只用于溯源，不应在Mac按路径执行或自动加载。旧生成关系继续保留原证据等级；文件SHA验证完整性，不追认权重到信号的历史关系。

在仓库根目录用pandas复算主指标，无需模型或数据库：

```python
from pathlib import Path
import pandas as pd

root = Path('docs/实验档案/20260910')
for window in ('W3', 'W4'):
    g1 = pd.read_parquet(root / 'qlib_mean' / f'report_{window}_G1_mean.parquet')
    a1 = pd.read_parquet(root / 'qlib_mean' / f'report_{window}_A1_mean.parquet')
    assert g1.index.equals(a1.index) and g1['bench'].equals(a1['bench'])
    g1_curve = (g1['return'] - g1['cost']).cumsum()
    a1_curve = (a1['return'] - a1['cost']).cumsum()
    print(window, 'A1−G1（pp）=', 100 * (a1_curve.iloc[-1] - g1_curve.iloc[-1]))

    reports = {
        arm: pd.read_parquet(root / 'lora_mean' / f'report_{window}_{arm}.parquet')
        for arm in ('G1_original_mean', 'G1_paired_100', 'LoRA_100')
    }
    ends = {arm: (r['return'] - r['cost']).sum() for arm, r in reports.items()}
    print(window, 'LoRA−原G1（pp）=', 100 * (ends['LoRA_100'] - ends['G1_original_mean']))
    print(window, 'LoRA−配对G1（pp）=', 100 * (ends['LoRA_100'] - ends['G1_paired_100']))
```

验证文件完整性：

```python
import hashlib
import json
from pathlib import Path

root = Path('docs/实验档案/20260910')
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
for item in manifest['files']:
    content = (root / item['path']).read_bytes()
    assert len(content) == item['bytes']
    assert hashlib.sha256(content).hexdigest() == item['sha256'], item['path']
print('全部归档文件校验通过')
```

不要从旧年化/旧v2表重构此处Qlib图，也不要将cumsum日收益相加误标为复利净值。训练语料、checkpoint权重、原始市场快照不在本目录；因此这里不支持重训或完整信号再生成。
