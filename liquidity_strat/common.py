"""配置与小工具（pathlib / loguru，遵循工程约定）。

与 ``baseline_suite/common.py`` 同构，但锚定**流动性分档**实验：全 A 母池，
按过去 60 交易日日均成交额的截面百分位每月末 PIT 重新分档（计划 §2.1）。

推理 / 采样口径**逐字**复用 ``paper_replication/config.yaml``（同一 Kronos-base
zero-shot canonical mean 管线，保证可与既有 baseline / 论文结果对拍）。

连接纪律（计划 §1.2）：``analyzer.auth(uri=...)`` 显式传 URI，禁止零参
``find_dotenv`` 自动发现；一切日志沿用 :func:`mask_uri` 脱敏，任何输出不得出现
明文凭据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

# liquidity_strat/ 目录自身
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
# 复用 paper_replication 的 config.yaml（同实验推理口径，禁止漂移）
PAPER_CONFIG_PATH = REPO_ROOT / "paper_replication" / "config.yaml"
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"

# Kronos 自带 .env（gitignored），DOLPHINDB_URI 指向本实例
KRONOS_ENV_PATH = REPO_ROOT / ".env"

# 窗口（计划 §2.2）：只用预训练截止 2024-06 之后——与论文时间隔离一致。
# 起点 2024-07-01，末日 2026-07-24（留 H=10 结算余量，与 baseline_suite 同口径）。
WINDOW_START = "2024-07-01"
WINDOW_END = "2026-07-24"
# 分档需回看 60 交易日成交额——窗口起点前需有足够历史。回看起算点留宽裕。
STRATIFY_LOOKBACK_TRADEDAYS = 60
# 数据末日（计划 §0：DDB 2026-08-07）
DATA_END = "2026-08-07"

# 流动性分档（计划 §2.1）：[0,5%]（帖子主角）、[5%,10%]、[45%,55%]（中位对照）
# 每个元组 = (low_pct, high_pct) 的截面百分位区间，左闭右闭。
# 高流动性对照 csi300 复用 baseline_suite 已有信号，不在本表。
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("lo5", 0.00, 0.05),
    ("lo5_10", 0.05, 0.10),
    ("mid45_55", 0.45, 0.55),
)
# csi300 对照档标签（复用 baseline_suite paper+oos mean，不重算）
HIGH_LIQ_BUCKET = "csi300"

# ST 处理双轨（计划 §2.1）：主口径 PIT 剔除 ST；附加组不剔除。两组分开报告。
ST_TRACKS: tuple[str, ...] = ("exst", "withst")
ST_TRACK_MAIN = "exst"  # 预注册判读用主口径

# 信号标签（计划 §2.3）：Kronos canonical mean、10日动量、10日反转、随机占位
SIGNAL_KRONOS = "K"
SIGNAL_MOM = "M"
SIGNAL_REV = "R"
SIGNAL_PLACEHOLDER = "P"
NEW_SIGNALS: tuple[str, ...] = (SIGNAL_KRONOS, SIGNAL_MOM, SIGNAL_REV, SIGNAL_PLACEHOLDER)


@dataclass(frozen=True)
class LiquidityConfig:
    """流动性分层检验全口径（不可变）。

    推理 / 采样参数从 ``paper_replication/config.yaml`` 加载（同实验，不漂移）；
    分档 / ST 轨 / 窗口由本模块常量固定。
    """

    # —— 数据层 ——
    pool: str  # 母池 = ashares
    lookback: int
    predict_len: int
    data_end: str
    filter_pipe: list | None
    # —— 推理 / 采样（与 paper_replication 逐字一致）——
    model_name: str
    tokenizer_name: str
    T: float
    top_p: float
    sample_top_k: int
    sample_count: int
    seed: int
    device: str
    max_context: int
    # —— 信号 ——
    signal_field: str
    # —— 分档（计划 §2.1）——
    buckets: tuple[tuple[str, float, float], ...]
    stratify_lookback: int
    st_tracks: tuple[str, ...]
    st_track_main: str
    # —— 窗口 ——
    window_start: str
    window_end: str

    @classmethod
    def load(cls) -> "LiquidityConfig":
        """从 paper_replication/config.yaml 加载推理口径，叠加分档常量。"""
        with open(PAPER_CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            pool=raw["data"]["pool"],
            lookback=raw["data"]["lookback"],
            predict_len=raw["data"]["predict_len"],
            data_end=raw["data"]["data_end"],
            filter_pipe=raw["data"].get("filter_pipe"),
            model_name=raw["inference"]["model_name"],
            tokenizer_name=raw["inference"]["tokenizer_name"],
            T=raw["inference"]["T"],
            top_p=raw["inference"]["top_p"],
            sample_top_k=raw["inference"]["top_k"],
            sample_count=raw["inference"]["sample_count"],
            seed=raw["inference"]["seed"],
            device=raw["inference"]["device"],
            max_context=raw["inference"]["max_context"],
            signal_field=raw["signal"]["field"],
            buckets=BUCKETS,
            stratify_lookback=STRATIFY_LOOKBACK_TRADEDAYS,
            st_tracks=ST_TRACKS,
            st_track_main=ST_TRACK_MAIN,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        )


def ensure_dirs() -> tuple[Path, Path]:
    """``data/`` 与 ``figures/`` 不入库（.gitignore 排 *.parquet），但需确保存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, FIG_DIR


def load_dolphindb_uri() -> str:
    """从 Kronos 自带 .env 显式读取 DOLPHINDB_URI（计划 §1.2）。

    禁止零参 ``find_dotenv`` 自动发现——那会拿到 AlphaFarmer/.env 的问题凭据。
    本函数只读 Kronos 仓库根的 .env，找不到则报错（不静默回退）。
    """
    from dotenv import load_dotenv

    if not KRONOS_ENV_PATH.is_file():
        raise FileNotFoundError(f"Kronos .env 不存在：{KRONOS_ENV_PATH}")
    load_dotenv(KRONOS_ENV_PATH, override=True)
    import os

    uri = os.environ.get("DOLPHINDB_URI", "").strip()
    if not uri:
        raise ValueError(f"{KRONOS_ENV_PATH} 中未配置 DOLPHINDB_URI")
    return uri


def mask_uri(uri: str) -> str:
    """脱敏 DDB URI（与 paper_replication.common.mask_uri 同口径）。"""
    if not uri or "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    cred, host = rest.split("@", 1)
    user = cred.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def init_analyzer_auth() -> str:
    """显式 URI 给 analyzer.auth（计划 §1.2），返回脱敏 URI 供日志。

    pydantic ValidationError 的 input_value 会带原始字符串——本函数捕获并脱敏，
    绝不让明文凭据进异常栈 / 日志。
    """
    import sys

    uri = load_dolphindb_uri()
    sys.path.insert(0, "/home/user/workspace/AlphaFarmer")
    import analyzer  # noqa: F401  —— 触发 import 副作用

    try:
        analyzer.auth(uri=uri)
    except Exception as exc:  # noqa: BLE001 —— 必须捕获以脱敏
        # 把任何 input_value 里的原始 URI 抹掉再抛
        masked = _scrub_exception(exc, uri)
        logger.error(f"analyzer.auth 失败（URI 已脱敏）：{mask_uri(uri)} -> {type(exc).__name__}")
        raise masked from None
    logger.info(f"analyzer.auth OK（URI={mask_uri(uri)}）")
    return uri


def _scrub_exception(exc: BaseException, uri: str) -> BaseException:
    """抹掉异常对象里残留的原始 URI 字符串，返回新异常（保留类型名与脱敏信息）。"""
    import re

    secret = uri
    try:
        for attr in ("input_value", "url", "value"):
            v = getattr(exc, attr, None)
            if isinstance(v, str) and secret in v:
                setattr(exc, attr, v.replace(secret, mask_uri(uri)))
    except Exception:  # noqa: BLE001
        pass
    # 再扫一遍 args / __str__
    try:
        msg = str(exc)
        if secret in msg:
            msg = re.sub(re.escape(secret), mask_uri(uri), msg)
            return type(exc)(msg)
    except Exception:  # noqa: BLE001
        pass
    return exc
