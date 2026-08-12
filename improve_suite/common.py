"""ImproveConfig——canonical 配置 + 网格覆盖字段（计划 §2 阶段 0）。

从 ``paper_replication/config.yaml`` 加载（与 baseline / paper_replication 同口径，
禁止漂移），叠加网格覆盖字段 ``lookback/predict_len/T/pool/top_k/drop_n``（阶段 3
网格、阶段 2 B3 跨池用）；窗口沿用 ``baseline_suite.common.WINDOWS``。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# 复用 baseline_suite 的窗口定义与数据末日（同实验口径）
from baseline_suite.common import DATA_END, WINDOWS

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
PAPER_CONFIG_PATH = REPO_ROOT / "paper_replication" / "config.yaml"
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"


@dataclass(frozen=True)
class ImproveConfig:
    """canonical 配置（不可变）+ 网格覆盖字段。

    未覆盖字段与 ``paper_replication/config.yaml`` 逐字一致；覆盖字段记录实际取值，
    便于下游落盘 / 复现。窗口通过 :data:`baseline_suite.common.WINDOWS` 的 key 指定。
    """

    # —— 数据层（可被网格覆盖）——
    pool: str
    lookback: int
    predict_len: int
    data_end: str
    filter_pipe: list | None
    # —— 推理 / 采样（T 可被覆盖）——
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
    # —— 组合引擎（top_k/drop_n 可被覆盖）——
    top_k: int
    drop_n: int
    min_hold: int
    cost_bps: float
    baselines: tuple[str, ...]
    # —— 窗口 ——
    window: str
    backtest_start: str
    backtest_end: str

    @classmethod
    def load(
        cls,
        *,
        window: str = "paper",
        lookback: int | None = None,
        predict_len: int | None = None,
        T: float | None = None,
        pool: str | None = None,
        top_k: int | None = None,
        drop_n: int | None = None,
    ) -> "ImproveConfig":
        """从 config.yaml 加载，叠加网格覆盖。

        :param window: ``paper``（2024-07-01~2025-06-30）或 ``oos``（2025-07-01~2026-07-24）。
        :param lookback/predict_len/T/pool/top_k/drop_n: 网格覆盖字段；
            ``None`` 表示沿用 canonical（计划 §5 三配置 C1/C2/C3 与 §4.3 B3 跨池用）。
        """
        if window not in WINDOWS:
            raise ValueError(f"未知窗口 {window!r}，可选 {list(WINDOWS)}")
        with open(PAPER_CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        start, end = WINDOWS[window]
        return cls(
            pool=pool if pool is not None else raw["data"]["pool"],
            lookback=lookback if lookback is not None else raw["data"]["lookback"],
            predict_len=predict_len if predict_len is not None else raw["data"]["predict_len"],
            data_end=raw["data"]["data_end"],
            filter_pipe=raw["data"].get("filter_pipe"),
            model_name=raw["inference"]["model_name"],
            tokenizer_name=raw["inference"]["tokenizer_name"],
            T=T if T is not None else raw["inference"]["T"],
            top_p=raw["inference"]["top_p"],
            sample_top_k=raw["inference"]["top_k"],
            sample_count=raw["inference"]["sample_count"],
            seed=raw["inference"]["seed"],
            device=raw["inference"]["device"],
            max_context=raw["inference"]["max_context"],
            signal_field=raw["signal"]["field"],
            top_k=top_k if top_k is not None else raw["portfolio"]["top_k"],
            drop_n=drop_n if drop_n is not None else raw["portfolio"]["drop_n"],
            min_hold=raw["portfolio"]["min_hold"],
            cost_bps=raw["portfolio"]["cost_bps"],
            baselines=tuple(raw["portfolio"]["baselines"]),
            window=window,
            backtest_start=start,
            backtest_end=end,
        )

    def canonical_label(self) -> str:
        """配置标签（落盘文件名用）：``L{lookback}_H{pred_len}_T{T}``。"""
        return f"L{self.lookback}_H{self.predict_len}_T{self.T}"

    def to_dict(self) -> dict:
        return {
            "pool": self.pool,
            "lookback": self.lookback,
            "predict_len": self.predict_len,
            "T": self.T,
            "top_p": self.top_p,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "top_k": self.top_k,
            "drop_n": self.drop_n,
            "min_hold": self.min_hold,
            "cost_bps": self.cost_bps,
            "window": self.window,
            "backtest_start": self.backtest_start,
            "backtest_end": self.backtest_end,
        }


def ensure_dirs() -> tuple[Path, Path]:
    """``data/`` 与 ``figures/`` 确保存在（parquet / png 落盘前调用）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, FIG_DIR
