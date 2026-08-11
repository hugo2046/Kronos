"""webui 的数据入口——经 kronos_qlib 取单股日频 OHLCVA。

设计见 ``docs/webui接入qlib数据层计划_20260811.md`` §2.1。本模块是 webui
**唯一**的数据来源：把"选文件读 CSV"替换为"选股票经
:class:`kronos_qlib.QlibProvider` 直连 DolphinDB"。

核心修复（计划 §0）：六列 ``open/high/low/close/volume/amount`` **全取全传**，
避免旧 ``app.py`` 只传 OHLCV 导致
:func:`model.kronos.KronosPredictor.predict` 内部用
``volume × 均价`` 合成假 amount（model/kronos.py:531）。

语义规定（每条都是正确性要求，详见计划 §2.1）：

1. 六列全取全传，一列不剔；
2. 停牌日在 DDB 无行，**不前向填充**——窗口即"末 N 个有数据的交易日"；
   窗口内 ``tradestatuscode != -1``（非交易态）的行数挂在
   ``df.attrs["non_tradeable_rows"]`` 供前端展示；
3. 未来交易日走 :meth:`QlibProvider.trading_days`（日历延伸到 2040，取未来
   交易日正当——这是预测时间戳，不是评估边界），修掉旧 ``app.py`` 用
   ``pd.date_range`` 外推会推出周末/节假日的 bug；
4. provider 每请求新建实例即可（进程内 qlib init-once 已由
   :class:`QlibProvider` 保证）。

测试缝合：每个公开函数都接受可选 ``_provider`` 参数，传入则直接复用（duck-typing
等价 QlibProvider 的 FakeProvider），便于无 DDB 环境下注入式单测——风格沿用
``tests/test_kronos_qlib.py`` 的 FakeProvider 模式。
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from kronos_qlib import QlibProvider

# 与 KronosPredictor.price_cols + vol_col + amt_vol（model/kronos.py:489-491）
# 严格一致的六列顺序；predict 内部按 z-score + clip 5 自归一化，这里不预归一化。
OHLCVA_COLS: list[str] = ["open", "high", "low", "close", "volume", "amount"]

# fetch_ohlcva 输出列序：timestamps 置首（与计划 §2.1 签名一致）
OUTPUT_COLS: list[str] = ["timestamps"] + OHLCVA_COLS


def check_env() -> Optional[str]:
    """启动预检 ``DOLPHINDB_URI`` 是否配置（计划陷阱 2）。

    :returns: 缺失时返回可读错误串（供启动日志 / 页面提示）；已配置返回 None。
        webui 数据层**绝不静默降级回 CSV**（CSV 路径已整体移除）。
    """
    # provider.py 在 import 时已 load_dotenv；_qlib_config.database_uri 是
    # init_qlib_once 真正读取的字段，以它为准（与 os.environ 保持一致）。
    from kronos_qlib.provider import _qlib_config

    if not (_qlib_config.database_uri or os.environ.get("DOLPHINDB_URI")):
        return (
            "DOLPHINDB_URI 未配置：请在 Kronos 根目录的 .env 中设置 "
            "DOLPHINDB_URI（参照 .env.example），格式 "
            "dolphindb://<user>:<password>@<host>:<port>。"
            "webui 数据层已切换为 kronos_qlib 直连，绝不静默回退到 CSV。"
        )
    return None


def _scoped_fetch(_provider, instruments, start, end, fields):
    """在 [start, end] × instruments 上调一次 fetch，返回 MultiIndex df。

    ``_provider`` 为 None（生产路径）时新建 :class:`QlibProvider`（进程内
    qlib init-once）；非空（测试注入 FakeProvider）时临时改写其
    ``_start_date / _end_date / instruments_`` 并在 finally 还原——
    与 ``kronos_qlib.windows._fetch_via`` 同一手势，让现有 FakeProvider 直接复用。

    :returns: qlib 堆叠表，``MultiIndex(datetime, instrument)``，列名已去 ``$``。
    """
    if _provider is None:
        p = QlibProvider(instruments, start_date=start, end_date=end)
        return p.fetch(fields, freq="day")
    orig_start, orig_end, orig_inst = (
        _provider._start_date,
        _provider._end_date,
        _provider.instruments_,
    )
    try:
        _provider._start_date = start
        _provider._end_date = end
        _provider.instruments_ = instruments
        return _provider.fetch(fields, freq="day")
    finally:
        _provider._start_date = orig_start
        _provider._end_date = orig_end
        _provider.instruments_ = orig_inst


def fetch_ohlcva(
    code: str, *, end_date: Optional[str] = None, n_bars: int, _provider=None
) -> pd.DataFrame:
    """取单只股票末 ``n_bars`` 个交易日的 OHLCVA。

    :param code: 股票代码，如 ``"600000.SH"``。
    :param end_date: 末日 ``YYYY-MM-DD``；None 表示数据真实末日（最新）。
    :param n_bars: 取末 n_bars 个有数据的交易日。
    :returns: 列 = :data:`OUTPUT_COLS`（``[timestamps, open, high, low, close,
        volume, amount]``），``timestamps`` 为严格递增的真实交易日。行数可能
        < ``n_bars``（新股 / 长停牌），由调用方判断；**不填充、不报错**。
        窗口内非交易态（``tradestatuscode != -1``）行数挂在
        ``df.attrs["non_tradeable_rows"]``。
    """
    anchor = (
        pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.now().normalize()
    )
    # 保守回溯：n_bars 个交易日需更多日历日覆盖（周末 / 节假日 / 停牌都不出数据）。
    # 2.5 倍日历日 + 10 天冗余尽量凑足 n_bars；不足则如实返回少行（计划语义 2）。
    start = anchor - pd.Timedelta(days=int(n_bars * 2.5) + 10)

    # 六列全取 + tradestatuscode（仅用于非交易态计数，不进输出列）。
    fields = [f"${c}" for c in OHLCVA_COLS] + ["$tradestatuscode"]
    raw = _scoped_fetch(
        _provider,
        [code],
        start.strftime("%Y-%m-%d"),
        anchor.strftime("%Y-%m-%d"),
        fields,
    )
    if "instrument" not in raw.index.names:
        raise ValueError(
            f"fetch 返回的 DataFrame 缺少 instrument 索引层，实际 index.names="
            f"{raw.index.names}（应为 MultiIndex(datetime, instrument)）"
        )

    # 区间内该股无任何数据行 → 返回空表（列序固定），调用方按"行数不足"判断。
    if code not in raw.index.get_level_values("instrument"):
        empty = pd.DataFrame(columns=OUTPUT_COLS)
        empty.attrs["non_tradeable_rows"] = 0
        return empty

    # 单股 xs 出堆叠表 → datetime 升序 → 严格 ≤ anchor → 末 n_bars 根。
    sub = raw.xs(code, level="instrument").sort_index()
    sub = sub.loc[:anchor]
    window = sub.iloc[-n_bars:]

    non_tradeable = (
        int((window["tradestatuscode"] != -1).sum())
        if "tradestatuscode" in window.columns
        else 0
    )

    # timestamps 列来自窗口真实交易日（稀疏，已剔除无数据日），与行数严格一致。
    out = window[OHLCVA_COLS].copy()
    out["timestamps"] = window.index
    out = out[OUTPUT_COLS].reset_index(drop=True)
    out.attrs["non_tradeable_rows"] = non_tradeable
    return out


def future_trading_days(after, n: int, _provider=None) -> pd.Series:
    """``after``（不含）之后的 ``n`` 个交易日，来自 qlib 交易日历。

    日历延伸到 2040（含未来占位日）——取**未来**交易日作预测时间戳属正当用法
    （这是预测时间戳，不是评估边界，与计划陷阱 4 区分）。

    :param after: 截断时点（``pd.Timestamp`` 或可解析值），结果严格 > after。
    :param n: 取的交易日数。
    :returns: 长度 ``n`` 的 ``pd.Series``，值为 ``pd.Timestamp``，严格递增。
    """
    after = pd.Timestamp(after)
    # 交易日历与 instruments 无关；用合法市场名占位构造（init-once 已保证不重复）。
    p = _provider if _provider is not None else QlibProvider(
        "csi300", start_date="2010-01-01", end_date="2040-12-31"
    )
    cal = p.trading_days()
    future = cal[cal > after][:n]
    return pd.Series(future)


def validate_code(code: str, _provider=None) -> bool:
    """校验 ``code`` 在 ashares 池内存在性。

    生产路径：取近 60 天该股数据，有行即合法；无行 / 取数异常 → False。
    测试路径（``_provider`` 注入 FakeProvider）：直接看其 instruments_ 是否含 code。
    """
    if _provider is not None:
        insts = _provider.instruments_
        return code in (insts if isinstance(insts, list) else [insts])

    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=60)
    try:
        raw = _scoped_fetch(
            None, [code], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
            ["$close"],
        )
    except Exception:
        return False
    return code in raw.index.get_level_values("instrument")


def list_pool(pool: str = "csi300", _provider=None) -> list[str]:
    """返回 ``pool`` 当前时点的 point-in-time 成分股列表（前端下拉 / 搜索用）。"""
    t = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    if _provider is None:
        end = pd.Timestamp.now().normalize()
        p = QlibProvider(pool, start_date="2020-01-01", end_date=end.strftime("%Y-%m-%d"))
    else:
        p = _provider
    return p.list_pool_at(pool, t)
