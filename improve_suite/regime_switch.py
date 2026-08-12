"""规则式状态切换（计划 §3 阶段 1）。

本模块是"零训练的假说证伪器"：用粗糙的可观测状态变量（指数趋势 / 路径离散度）
在两条信号间逐日切换。若观测规则都切不出增益，后续神经门控（阶段 5）取消立项。

预注册规则族（跑前冻结，共 4 条，禁止追加）：

    - R1：指数收盘 > MA200 时取 M 动量，否则取 canonical mean；
    - R1'：同门控，方向反转（True 取 mean，False 取 M）——反向对照；
    - R2：全期恒 True → 纯 M（对照基准）；
    - R3：路径离散度低分位 → canonical mean，否则退守同池等权（阶段 2 后回填）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def build_switch_signal(
    sig_a: pd.DataFrame, sig_b: pd.DataFrame, gate: pd.Series
) -> pd.DataFrame:
    """逐行选择：``gate[t]==True`` 取 ``sig_a.loc[t]``，否则取 ``sig_b.loc[t]``。

    :param sig_a / sig_b: 宽表 ``index=date, columns=code, values=signal``。
    :param gate: ``index=date, values=bool``。缺失日视为 ``False``（回退 sig_b）。
    :returns: 与并集日期 / 并集列同构的宽表。
    """
    dates = sig_a.index.union(sig_b.index)
    cols = sig_a.columns.union(sig_b.columns)
    a = sig_a.reindex(index=dates, columns=cols)
    b = sig_b.reindex(index=dates, columns=cols)
    g_vals = gate.reindex(dates).to_numpy()
    g = np.where(pd.isna(g_vals), False, g_vals).astype(bool)[:, None]
    # 广播到 (n_dates, n_cols)：pandas.where 要求 cond 与 self 同形
    g_full = np.broadcast_to(g, a.shape)
    # DataFrame.where(cond, other)：cond True 保留 a，False 替换为 b
    return a.where(g_full, b)


# ------------------------------------------------------------------
# gate_ma200：指数趋势门控（R1 / R1' 共用）
# ------------------------------------------------------------------


def ma200_gate_from_close(close: pd.Series, window: int = 200) -> pd.Series:
    """收盘 > MA(window) 的 bool 序列（前 window-1 日为 NaN）。

    :param close: 指数收盘价序列（升序）。
    :param window: 均线窗口（默认 200）。
    :returns: ``close > rolling(window).mean()``；不满窗口处为 NaN。
    """
    ma = close.rolling(window).mean()
    gate = close > ma
    # 不满窗口处 MA 为 NaN → 门控置 NaN（numpy 的 NaN 比较返回 False，会与"close≤MA"混；
    # 用 .where(ma.notna()) 显式标记"无 MA200"=未知，build_switch_signal 的 fillna(False) 回退）
    return gate.where(ma.notna())


def gate_ma200(
    provider, dates: pd.DatetimeIndex, index_code: str = "000300.SH", window: int = 200
) -> pd.Series:
    """000300.SH 决策日 t 收盘 > MA200(t) 的门控序列。

    取数向前多取 ``window*2`` 个日历日缓冲（≈ window 个交易日），保证窗口首日即有 MA200。

    :param dates: 决策日序列（窗口内交易日）。
    :returns: ``index=date, values=bool``（仅含 ``dates`` 内的日期）。
    """
    from kronos_qlib import QlibProvider

    fetch_start = (dates.min() - pd.Timedelta(days=window * 2)).strftime("%Y-%m-%d")
    end = dates.max().strftime("%Y-%m-%d")
    p = QlibProvider([index_code], fetch_start, end)
    df = p.fetch(["$close"], freq="day")
    if len(df) == 0:
        raise RuntimeError(f"指数 {index_code} 在 {fetch_start}~{end} 无数据")
    if "instrument" in df.index.names:
        df = df.xs(index_code, level="instrument")
    close = df["close"].sort_index()
    gate = ma200_gate_from_close(close, window=window)
    out = gate.reindex(dates)
    n_true = int(np.where(pd.isna(out.to_numpy()), False, out.to_numpy()).sum())
    logger.info(
        f"gate_ma200 [{index_code}, window={window}]：{len(dates)} 决策日，"
        f"True（趋势市）{n_true} 日（{n_true / len(dates):.1%}）"
    )
    return out


# ------------------------------------------------------------------
# gate_dispersion：路径离散度门控（R3，阶段 2 后回填）
# ------------------------------------------------------------------


def dispersion_gate_from_signals(
    path_close_wide: pd.DataFrame, dates: pd.DatetimeIndex, *, q: float = 0.8, lookback: int = 60
) -> pd.Series:
    """截面平均路径 std 的过去 ``lookback`` 日分位 < ``q`` → True（低心虚）。

    :param path_close_wide: 逐路径收益的长/宽表；此处接受"每日每股的路径 std"宽表
        ``index=date, columns=code, values=path_std``，由调用方从 path_store 聚合。
    :param dates: 决策日序列。
    :param q: 分位阈值（默认 0.8）——当日截面均值 std 的历史分位 < q 视为"低心虚"。
    :param lookback: 历史回看窗口（默认 60 日）。
    :returns: ``index=date, values=bool``。
    """
    # 截面每日的路径 std 均值（不确定度水平）
    daily_disp = path_close_wide.mean(axis=1).sort_index()
    # 过去 lookback 日的分位 rank
    roll_q = daily_disp.rolling(lookback, min_periods=max(lookback // 2, 10)).rank(pct=True)
    gate = (roll_q < q)
    return gate.reindex(dates)
