"""基准探测与取数（计划 §2.3）。

**双基准设计（2026-08-10 实测后定稿）**：

1. **csi300 指数基准**（``000300.SH``，市值加权）：论文口径，用于 AER 锚点比较
   （规则 2）。DDB 实测命中，论文窗口 242 行。

2. **同池等权基准**（每日 point-in-time csi300 成分等权日收益）：用于**引擎门禁**
   （规则 1）。原因：等权 top-50 组合天然带约 +4.4% 的等权-beta 溢价（2024-07~
   2025-06 全池等权 +17.6% vs csi300 指数 +13.2%）。用指数基准判门禁时，随机信号
   也会"复现" +4.3% AER——这是基准结构差不是 alpha。同池等权基准剥离掉这个 beta
   差后，随机占位 AER≈0（实测 +0.02%），才是"引擎不制造 alpha"的干净判据。

3. 两基准的差（指数 AER − 等权 AER）= 等权-beta 溢价，作为结构性项并列报告。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# 沪深300 指数代码（探测命中，论文口径）
CSI300_INDEX = "000300.SH"


def probe_index_benchmark(
    provider, start: str, end: str
) -> pd.Series:
    """取沪深300 指数日收益（论文口径的市值加权基准）。

    :returns: 指数逐日收益（close-to-close）。DDB 无指数时抛异常。
    """
    from kronos_qlib import QlibProvider

    p = QlibProvider([CSI300_INDEX], start, end)
    df = p.fetch(["$close"], freq="day")
    if len(df) == 0:
        raise RuntimeError(f"沪深300 指数 {CSI300_INDEX} 在 {start}~{end} 无数据")
    if "instrument" in df.index.names:
        df = df.xs(CSI300_INDEX, level="instrument")
    close = df["close"].sort_index()
    ret = close.pct_change(fill_method=None).dropna()
    logger.info(f"指数基准 {CSI300_INDEX}：{len(ret)} 行，累计 {(1+ret).prod()-1:+.2%}")
    return ret


def build_pool_equal_weight_benchmark(
    px_wide: pd.DataFrame, tradeable: pd.DataFrame
) -> pd.Series:
    """构造同池等权基准（每日可交易股票的等权日收益）。

    与组合选股域一致：只用当日 ``tradeable==True`` 的股票，等权平均。
    这剥离掉等权-beta 溢价，使随机占位 AER≈0。

    :param px_wide: 后复权 close 宽表。
    :param tradeable: bool 宽表（可交易掩码）。
    :returns: 逐日等权基准收益。
    """
    rets = px_wide.pct_change(fill_method=None)
    # 仅对可交易股票等权
    masked = rets.where(tradeable)
    bench = masked.sum(axis=1) / tradeable.sum(axis=1).replace(0, np.nan)
    bench = bench.dropna()
    logger.info(f"同池等权基准：{len(bench)} 行，累计 {(1+bench).prod()-1:+.2%}")
    return bench
