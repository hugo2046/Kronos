"""配置加载与小工具（pathlib / loguru，遵循工程约定）。

与 ``cross_section/common.py`` 同构，但口径锚定论文复现（每日调仓 + long-only top-k/drop-n），
非 cross_section 的每 10 日调仓 + 多空分组。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

# paper_replication/ 目录自身
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
CONFIG_PATH = PKG_DIR / "config.yaml"
DATA_DIR = PKG_DIR / "data"


@dataclass(frozen=True)
class ReplicationConfig:
    """复现实验全口径（不可变，防止阶段间漂移）。"""

    # —— 数据层 ——
    pool: str
    lookback: int
    predict_len: int
    backtest_start: str
    backtest_end: str
    data_end: str
    filter_pipe: list | None
    # —— 推理 / 采样 ——
    model_name: str
    tokenizer_name: str
    T: float
    top_p: float
    top_k: int
    sample_count: int
    seed: int
    device: str
    max_context: int
    # —— 信号 ——
    signal_field: str
    # —— 组合引擎（论文 top-k/drop-n）——
    top_k: int
    drop_n: int
    min_hold: int
    cost_bps: float
    baselines: tuple[str, ...]

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ReplicationConfig":
        """从 yaml 加载配置，展开为扁平字段。"""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            pool=raw["data"]["pool"],
            lookback=raw["data"]["lookback"],
            predict_len=raw["data"]["predict_len"],
            backtest_start=raw["data"]["backtest_start"],
            backtest_end=raw["data"]["backtest_end"],
            data_end=raw["data"]["data_end"],
            filter_pipe=raw["data"].get("filter_pipe"),
            model_name=raw["inference"]["model_name"],
            tokenizer_name=raw["inference"]["tokenizer_name"],
            T=raw["inference"]["T"],
            top_p=raw["inference"]["top_p"],
            top_k=raw["inference"]["top_k"],
            sample_count=raw["inference"]["sample_count"],
            seed=raw["inference"]["seed"],
            device=raw["inference"]["device"],
            max_context=raw["inference"]["max_context"],
            signal_field=raw["signal"]["field"],
            top_k=raw["portfolio"]["top_k"],
            drop_n=raw["portfolio"]["drop_n"],
            min_hold=raw["portfolio"]["min_hold"],
            cost_bps=raw["portfolio"]["cost_bps"],
            baselines=tuple(raw["portfolio"]["baselines"]),
        )


def ensure_data_dir() -> Path:
    """``paper_replication/data/`` 不入库，但需确保存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def mask_uri(uri: str) -> str:
    """脱敏 DDB URI（与 kronos_qlib 一致的口径，日志友好）。"""
    if not uri or "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    cred, host = rest.split("@", 1)
    user = cred.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"
