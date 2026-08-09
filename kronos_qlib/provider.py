"""QlibProvider —— Kronos 的直连 qlib（DolphinDB 后端）取数封装。

设计照搬 AlphaFarmer ``factor_core/provider.py`` 三处已验证模式（读，不 import）：

1. ``load_dotenv`` 必须在 ``import qlib`` **之前**——``QlibConfig`` 的 dataclass
   默认值 ``os.environ.get("DOLPHINDB_URI", ...)`` 在类定义时求值，.env 加载若
   推迟会拿到错误的 fallback URI（AlphaFarmer 注释里的坑）。
2. 单一 ``_load_lock`` 同时串行 ``fetch``（``QlibDataLoader.load``）与
   ``trading_days``（``D.calendar``）——两者读同一份进程级缓存
   ``qlib.data.cache.H["c"]``，分开加锁等于没加。
3. 日志用 ``mask_uri`` 脱敏，禁止任何形式打印明文口令。

硬性约束（计划 §2.1.1）：``DOLPHINDB_URI`` 缺失时抛说明性异常，**绝不静默回退**
到 localhost 或文件后端——静默回退会让实验在错数据上跑完还看不出来。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# 必须先于 import qlib 加载 Kronos 根 .env。
# 用相对本文件的确定性路径定位仓库根 .env（provider.py → kronos_qlib → 仓库根），
# 不受 cwd / import 路径影响；仅当脱离仓库布局时回退 find_dotenv()。
_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ROOT_ENV if _ROOT_ENV.is_file() else find_dotenv())

import pandas as pd
import qlib
from loguru import logger
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.backend.utils import mask_uri
from qlib.data.dataset.loader import QlibDataLoader

# 预定义市场字符串白名单，集中维护便于扩展
_MARKET_WHITELIST: frozenset[str] = frozenset({"csi300", "csi500", "ashares", "cyb"})


@dataclass
class QlibConfig:
    """Qlib 初始化配置。

    ``database_uri`` 默认空串（不是 localhost）——缺失时由
    :meth:`QlibProvider.init_qlib_once` 抛说明性异常，杜绝静默回退。
    """

    database_uri: str = os.environ.get("DOLPHINDB_URI", "")
    region: str = REG_CN


_qlib_config = QlibConfig()


class QlibProvider:
    """基于 ``QlibDataLoader`` 的日频数据提供者。

    支持预定义市场名（``csi300`` / ``csi500`` / ``ashares`` / ``cyb``）或
    自定义股票代码列表。市场字符串透传给 qlib 内部解析。

    :param instruments: 市场字符串（白名单内）或股票代码列表。
    :param start_date: 起始日期，``YYYY-MM-DD``。
    :param end_date: 终止日期，``YYYY-MM-DD``。
    """

    _qlib_initialized: bool = False
    _init_lock: threading.Lock = threading.Lock()
    # qlib 全局数据层串行锁：覆盖一切触碰 qlib 进程级缓存的调用——
    # QlibDataLoader.load()（D.features 内部缓存）与 D.calendar()
    # （CalendarProvider._get_calendar 对模块级单例 H["c"] 做无锁
    # check-then-act）。两者读同一份共享状态，必须由**同一把**锁保护
    # （AlphaFarmer 曾拆双锁，对 H["c"] 毫无互斥效果，已撤销）。
    # 用类属性是因为共享状态是进程级而非实例级，跨实例也要互斥。
    # 与 _init_lock 仍分离：那是真正不相交的临界区（一次性初始化 vs 每次访问）。
    _load_lock: threading.Lock = threading.Lock()

    @classmethod
    def init_qlib_once(cls) -> None:
        """全局 Qlib 初始化（仅一次，线程安全）。

        双重检查锁：多线程并发首调时无锁的 check-then-act 会重复进入
        ``qlib.auto_init``，后到者在 qlib 内部抛 FileNotFoundError。

        :raises RuntimeError: ``DOLPHINDB_URI`` 未配置。**绝不静默回退**到
            localhost 或文件后端——静默回退会让实验在错数据上跑完看不出来。
        """
        if cls._qlib_initialized:
            return
        with cls._init_lock:
            if cls._qlib_initialized:  # 拿到锁时可能已被先行线程初始化
                return
            uri = _qlib_config.database_uri
            if not uri:
                raise RuntimeError(
                    "DOLPHINDB_URI 未配置：请在 Kronos 根目录的 .env 中设置 "
                    "DOLPHINDB_URI（参照 .env.example），格式 "
                    "dolphindb://<user>:<password>@<host>:<port>。"
                    "本数据层绝不静默回退到 localhost 或文件后端，"
                    "以避免在错误数据上跑完实验还无法察觉。"
                )
            qlib.auto_init(database_uri=uri, region=_qlib_config.region)
            cls._qlib_initialized = True
            # mask_uri 屏蔽密码 + 遮蔽公网主机（内网/localhost 保持可见），
            # 与 qlib 自身日志格式一致，避免日志泄露凭证与公网 IP。
            logger.info(f"Qlib 初始化成功 - URI: {mask_uri(uri)}")

    def __init__(
        self,
        instruments: str | list[str],
        start_date: str,
        end_date: str,
    ) -> None:
        # 市场字符串白名单校验：fail-fast，放在 init_qlib_once 之前，
        # 让非法 instruments 不触发 qlib 初始化副作用。
        if isinstance(instruments, str):
            if instruments not in _MARKET_WHITELIST:
                raise ValueError(
                    f"instruments 字符串仅支持 {sorted(_MARKET_WHITELIST)},"
                    f"收到 '{instruments}'"
                )
        elif not isinstance(instruments, list):
            raise TypeError("instruments 必须是 list[str] 或 str")

        self.init_qlib_once()
        self.instruments_ = instruments
        self._start_date = start_date
        self._end_date = end_date

    def fetch(
        self,
        fields: str | list[str] | tuple[list[str], list[str]],
        *,
        filter_pipe: list | None = None,
        freq: str = "day",
    ) -> pd.DataFrame:
        """通过 ``QlibDataLoader`` 拉数，列名自动去除 ``$`` 前缀。

        :param fields: 字段表达式。三种合法形态：

            * ``"$close"`` —— 单字符串
            * ``["$close", "$volume"]`` —— 表达式列表（列名为表达式本身）
            * ``(["$close/$preclose-1"], ["ret"])`` —— ``(exprs, names)``
              双列表，显式指定别名

        :param filter_pipe: 透传给 ``QlibDataLoader`` 的过滤器列表（如
            ``STFilter``）。**注意陷阱 5**：``instruments`` 传 **list** 时
            ``filter_pipe`` 只 warning 不生效；启用过滤器时请在构造
            provider 时传 str 市场名。
        :param freq: 频率（``"day"`` / ``"min"``），默认日频。
        :returns: ``MultiIndex(datetime, instrument)`` DataFrame，列名去 ``$``。
                  ⚠️ level 0 是 datetime（``swap_level=True``）——reshape
                  一律按 level **名** ``unstack("instrument")``，禁止按位置
                  ``unstack(level=0)``（会静默转置矩阵，不报错）。
        """
        qdl = QlibDataLoader(
            config=fields,
            filter_pipe=filter_pipe,
            swap_level=True,
            freq=freq,
        )
        # qlib 全局数据层非线程安全，仅锁 .load() 这一处真正不安全的调用：
        # 构造器只存配置、列名替换操作的是本次调用私有的返回 df，均无需进锁
        # 以保持临界区最小。
        with self._load_lock:
            df: pd.DataFrame = qdl.load(
                instruments=self.instruments_,
                start_time=self._start_date,
                end_time=self._end_date,
            )
        df.columns = df.columns.str.replace("$", "", regex=False)
        return df

    def trading_days(
        self, start: str | None = None, end: str | None = None
    ) -> pd.DatetimeIndex:
        """交易日历，走 ``D.calendar``（与 fetch 共用 ``_load_lock``）。

        ⚠️ 日历含未来占位日（到 2040-12-31），而真实数据止于更早日期。
        取未来调仓日 / 评估边界时必须以**数据实际末日**为界，否则会拿到
        一批永远取不到真实收益的"未来调仓日"（计划陷阱 3）。

        :param start: 起始日期 ``YYYY-MM-DD``，None 不限。
        :param end: 终止日期 ``YYYY-MM-DD``，None 不限。
        :returns: ``pd.DatetimeIndex``。
        """
        with self._load_lock:
            cal = D.calendar(
                start_time=start, end_time=end, freq="day"
            )
        return pd.DatetimeIndex(cal)

    def list_pool_at(self, pool: str, t: str) -> list[str]:
        """返回 ``t`` 时点 ``pool`` 的 point-in-time 成分股列表。

        逐个调仓日重取成分，避免"一次性取全区间成分当固定池"引入幸存者偏差。
        与 fetch / trading_days 共用 ``_load_lock``（读同一份 qlib 缓存）。

        :param pool: 市场字符串（``csi300`` 等）。
        :param t: 时点 ``YYYY-MM-DD``。
        :returns: 该时点成分股代码列表（去 ``instrument`` 层 key）。
        """
        with self._load_lock:
            members = D.list_instruments(
                D.instruments(pool),
                start_time=t,
                end_time=t,
                as_list=True,
            )
        return list(members)
