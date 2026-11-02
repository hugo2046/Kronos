"""独立以共同日期块核验统计入口；合成四格仅检验算法，不是实验结论。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from state_adapter.factorial_report import bootstrap_effects, mechanism


def main() -> None:
    """向量化重抽四臂后计算交互，独立比对生产统计入口。

    :returns: 无，保存算法核验和已完成W3动态sampling区间。
    :raises AssertionError: 共同日期块统计与生产入口不一致。
    """
    out = Path(__file__).resolve().parent.parent
    result: dict[str, Any] = {'at': datetime.now(timezone.utc).isoformat(),
                              'synthetic_checks': {}}
    for n in [126, 134]:
        raw = np.random.default_rng(700 + n).normal(0, .01, (n, 4))
        frame = pd.DataFrame(raw, columns=list('ABCD'),
                             index=pd.bdate_range('2025-01-01', periods=n))
        actual = bootstrap_effects(mechanism(frame))['interaction']
        starts = np.random.default_rng(100).integers(0, n, (5000, (n + 19) // 20))
        indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(5000, -1)[:, :n]
        # 四臂共同重抽后才做交互，独立于生产代码的先作日差再重抽。
        sums = raw[indices].sum(axis=1)
        interaction = (sums[:, 3] - sums[:, 2]) - (sums[:, 1] - sums[:, 0])
        interval = np.quantile(interaction, [.025, .975]) * 100
        error = float(np.max(np.abs(interval - [actual['lower_95_pp'], actual['upper_95_pp']])))
        assert error < 1e-10
        result['synthetic_checks'][str(n)] = {'max_abs_pp': error, 'resamples': 5000}
    reports = {arm: pd.read_parquet(out / 'risk' / arm / 'report.parquet') for arm in ['B', 'D']}
    assert reports['B'].index.equals(reports['D'].index)
    difference = (reports['D']['return'] - reports['D']['cost']) - (reports['B']['return'] - reports['B']['cost'])
    result['W3_D_minus_B'] = bootstrap_effects(difference.to_frame('sampling_dynamic'))
    result['note'] = '仅动态sampling已有完整真实日报；合成四格不替代尚缺C推理的真实机制结果。'
    result['status'] = 'passed'
    (out / 'verification/round7_bootstrap_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    logger.info('共同日期块交互算法核验通过：{}', result['synthetic_checks'])


if __name__ == '__main__':
    main()
