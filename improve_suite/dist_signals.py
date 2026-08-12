"""路径分布统计信号族（计划 §4.2 阶段 2）。

对每个 (date, code)，由 N 条路径的 H 日平均收益
``r_i = mean(path_i) / close_t − 1`` 计算三条冻结信号：

    - S1 ``neg_std``     = ``-std(r_1..r_N)``——不确定度低的票更可信；
    - S2 ``sharpe_like`` = ``mean(r_i) / std(r_i)``——收益预期按不确定度折价；
    - S3 ``q10``          = ``quantile(r_i, 0.1)``——下界含风险信息。

**口径事实**：``mean(r_i)`` 等于 canonical mean 信号（``mean(所有路径所有步)/close−1``），
即分布信号与 canonical 共享同一中心，区别仅在分布形状。

三条信号分别独立过引擎，**禁止**与 mean 做加权组合搜索（§4.2 纪律）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# 冻结信号标签（顺序固定）
DIST_SIGNALS = ("neg_std", "sharpe_like", "q10")


def compute_dist_signals(path_close: np.ndarray, last_close: float) -> dict[str, float]:
    """从一条逐路径 close 矩阵算三条分布信号（计划 §4.2）。

    :param path_close: ``(N, H)`` 数组，第 i 行第 j 列 = 第 i 条路径第 j 步的预测 close。
    :param last_close: 决策日 t 的后复权 close。
    :returns: ``{"neg_std":..., "sharpe_like":..., "q10":..., "mean":...}``。
        ``mean`` = ``mean(r_i)`` = canonical mean 信号（对拍一致性）。
    """
    # 每条路径的 H 日平均收益
    r = path_close.mean(axis=1) / last_close - 1.0  # (N,)
    mu = float(r.mean())
    sd = float(r.std(ddof=0))  # 总体标准差（numpy 默认）
    return {
        "neg_std": -sd,
        "sharpe_like": mu / sd if sd > 0 else float("nan"),
        "q10": float(np.quantile(r, 0.1)),
        "mean": mu,
    }


def run_dist_signals(
    paths_long: pd.DataFrame,
    last_close_wide: pd.DataFrame,
    rebalances: pd.DatetimeIndex,
    *,
    progress_every: int = 20,
) -> dict[str, pd.DataFrame]:
    """从逐路径长表生成三条分布信号宽表。

    :param paths_long: 长表 ``date, code, path_id, step, pred_close``
        （由 :mod:`improve_suite.path_store` 读回）。
    :param last_close_wide: 决策日 t 后复权 close 宽表 ``index=date, columns=code``。
    :param rebalances: 决策日序列。
    :returns: ``{signal_tag: wide_df}``，三张宽表同 index 同 columns。
    """
    rows: dict[str, list[dict]] = {s: [] for s in DIST_SIGNALS}
    # 按 (date, code) 分组，便于取每只票的 (N, H) 矩阵
    grouped = paths_long.groupby(["date", "code"], sort=False)

    for d in rebalances:
        ds = pd.Timestamp(d)
        if ds not in last_close_wide.index:
            for s in DIST_SIGNALS:
                rows[s].append({})
            continue
        day_sigs: dict[str, dict[str, float]] = {s: {} for s in DIST_SIGNALS}
        for code, lc in last_close_wide.loc[ds].items():
            if pd.isna(lc) or lc <= 0:
                continue
            try:
                sub = grouped.get_group((ds, code))
            except KeyError:
                continue
            # (N, H) 矩阵：行=path_id，列=step
            mat = sub.pivot(index="path_id", columns="step", values="pred_close").sort_index()
            if mat.empty:
                continue
            sig = compute_dist_signals(mat.to_numpy(), float(lc))
            for s in DIST_SIGNALS:
                day_sigs[s][code] = sig[s]
        for s in DIST_SIGNALS:
            rows[s].append(day_sigs[s])

    wide = {}
    for s in DIST_SIGNALS:
        wide[s] = pd.DataFrame(rows[s], index=rebalances)
        logger.info(
            f"分布信号 {s}：{wide[s].shape[0]} 日 × 平均 {wide[s].notna().sum(axis=1).mean():.0f} 只/日"
        )
    return wide
