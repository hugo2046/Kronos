"""PEER 专用历史读取适配层（计划 20260910 §5：纠偏读取边界）。

已复现缺口：``build_inference_windows`` 的 fetch_end 是 t 之后第 10 个交易
日（纯工作日日历下 t=2026-07-24 会请求到 2026-08-07），随后才 ``loc[:t]``
切片——"零读取未来"的声明不成立（代码请求越界；实际返回内容范围未知，
无请求日志可证）。本适配层把**真正委托底层 fetch 的最后边界**钳制到当前
决策日 t：``_fetch_via`` 会临时改写 provider 的 ``_start_date/_end_date/
instruments_`` 再调 ``fetch``，因此钳制必须发生在 ``fetch`` 委托处，不能
只在 provider 构造时设 end_date。

约束：
- 任何传给底层 fetch 的 end ≤ 当前决策日 t（截止日 2026-07-24 当天允许）；
- t > 2026-07-24 在下层 fetch **前**拒绝（底层调用计数为 0）；
- 日历（``trading_days``）可提供未来时间戳，不随日历拉取未来价格；
- 返回观测日期再验证：底层若仍返回 > t 的行，截断并计数（不静默使用）。
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from peer_residual import config as C


class PeerHistoryProvider:
    """duck-typing 包装 QlibProvider：fetch 上界=当前决策日，其余透传。

    :param inner: 被包装的 provider（如 :class:`kronos_qlib.QlibProvider`）。
    :param cutoff: 决策日上界（默认 ``FORWARD_CUTOFF`` = 2026-07-24）。
    """

    def __init__(self, inner, cutoff: str = C.FORWARD_CUTOFF) -> None:
        self._inner = inner
        self._cutoff = pd.Timestamp(cutoff)
        self._bound: pd.Timestamp | None = None
        # _fetch_via 协议：临时改写这三个属性后调 fetch
        self._start_date = None
        self._end_date = None
        self.instruments_ = None
        self.truncated_rows = 0

    # ---- 决策日边界管理 ----

    def set_decision_bound(self, t: str) -> None:
        """声明当前决策日；超过 cutoff 立即拒绝（下层 fetch 前，计数 0）。"""
        ts = pd.Timestamp(t)
        if ts > self._cutoff:
            raise RuntimeError(
                f"决策日 {t} 超过读取上界 {self._cutoff.date()}——拒绝在任何"
                f"底层 fetch 前执行（本轮不读取 forward）")
        self._bound = ts

    # ---- fetch 钳制（_fetch_via 的最后委托边界） ----

    def fetch(self, fields, *, filter_pipe=None, freq="day") -> pd.DataFrame:
        if self._bound is None:
            raise RuntimeError("未声明决策日边界：先 set_decision_bound(t)")
        requested_end = (pd.Timestamp(self._end_date)
                         if self._end_date is not None else self._bound)
        effective_end = min(requested_end, self._bound)
        # 在 inner 上复刻 _fetch_via 协议（钳制后的 end 生效）
        orig = (self._inner._start_date, self._inner._end_date,
                self._inner.instruments_)
        try:
            self._inner._start_date = self._start_date
            self._inner._end_date = effective_end.strftime("%Y-%m-%d")
            self._inner.instruments_ = self.instruments_
            df = self._inner.fetch(fields, filter_pipe=filter_pipe, freq=freq)
        finally:
            self._inner._start_date, self._inner._end_date, \
                self._inner.instruments_ = orig
        # 返回观测日期再验证：底层仍返回 > t 的行 → 截断并计数（不静默用）
        if "datetime" in df.index.names:
            mask = df.index.get_level_values("datetime") > self._bound
            if mask.any():
                self.truncated_rows += int(mask.sum())
                logger.warning(
                    f"底层 fetch 返回 {int(mask.sum())} 行 > 决策日 "
                    f"{self._bound.date()} 的观测，已截断")
                df = df[~mask]
        return df

    # ---- 日历 / 成分：透传（未来日历仍可用，不拉未来价格） ----

    def trading_days(self, start=None, end=None) -> pd.DatetimeIndex:
        return self._inner.trading_days(start, end)

    def list_pool_at(self, pool: str, t: str) -> list[str]:
        return self._inner.list_pool_at(pool, t)

    def __getattr__(self, name):  # 其余属性透传 inner
        return getattr(self._inner, name)


__all__ = ["PeerHistoryProvider"]
