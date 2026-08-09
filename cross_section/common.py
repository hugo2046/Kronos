"""配置加载与小工具（pathlib / loguru，遵循工程约定）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

# cross_section/ 目录自身
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
CONFIG_PATH = PKG_DIR / "config.yaml"
DATA_DIR = PKG_DIR / "data"


@dataclass(frozen=True)
class ExperimentConfig:
    """实验全口径（不可变，防止阶段间漂移）。"""

    pool: str
    lookback: int
    predict_len: int
    rebalance_freq: int
    backtest_start: str
    backtest_end: str
    data_end: str
    filter_pipe: list | None
    model_name: str
    tokenizer_name: str
    T: float
    top_p: float
    top_k: int
    sample_count: int
    seed: int
    device: str
    max_context: int
    signal_field: str
    n_groups: int
    cost_bps: float

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ExperimentConfig":
        """从 yaml 加载配置，展开为扁平字段。"""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            pool=raw["data"]["pool"],
            lookback=raw["data"]["lookback"],
            predict_len=raw["data"]["predict_len"],
            rebalance_freq=raw["data"]["rebalance_freq"],
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
            n_groups=raw["evaluation"]["n_groups"],
            cost_bps=raw["evaluation"]["cost_bps"],
        )


def ensure_data_dir() -> Path:
    """``cross_section/data/`` 不入库，但需确保存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
