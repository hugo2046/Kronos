"""配置与小工具（pathlib / loguru，遵循工程约定）。

与 ``paper_replication/common.py`` 同构，但锚定 baseline 四变体 + 样本外窗口。
推理 / 采样 / 引擎口径**逐字**复用 ``paper_replication.config.yaml``（同一实验，
不同呈现口径），故本模块的 :class:`BaselineConfig` 直接从那份 yaml 加载，
并把窗口扩成可变参数（论文窗口 / 样本外窗口两段切换）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

# baseline_suite/ 目录自身
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
# 复用 paper_replication 的 config.yaml（同实验口径，禁止漂移）
PAPER_CONFIG_PATH = REPO_ROOT / "paper_replication" / "config.yaml"
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"

# 窗口标签 → (start, end)（计划 §1 / §4）
# 论文窗口 = 2024-07-01~2025-06-30；样本外 = 2025-07-01~2026-07-24。
#
# **边界调整（2026-08-10 实测）**：计划 §4 原写样本外末日 2026-07-31，但该日之后
# 仅 5 个交易日（数据末日 2026-08-07），不足 H=10 的信号结算窗口——每个决策日 t
# 需 t+1..t+H 个交易日存在才能生成完整预测路径。最后一个完整决策日是 2026-07-24
# （其后恰好 10 个交易日 2026-07-27~2026-08-07）。窗口末日改为 2026-07-24，保留
# H=10 完整结算口径、与论文窗口同口径、无信号缺失。文档如实记录此边界调整。
WINDOWS: dict[str, tuple[str, str]] = {
    "paper": ("2024-07-01", "2025-06-30"),
    "oos": ("2025-07-01", "2026-07-24"),
}

# 数据末日（计划 §0：DDB 2026-08-07）。样本外窗口末日 + 结算余量 H=10
# 必须落在数据末日之前——运行前断言。
DATA_END = "2026-08-07"

# 四变体标签（计划 §1）——顺序固定，下游表格 / 绘图一致
VARIANTS: tuple[str, ...] = ("last", "mean", "max", "min")


@dataclass(frozen=True)
class BaselineConfig:
    """baseline 四变体全口径（不可变）。

    推理 / 引擎参数从 ``paper_replication/config.yaml`` 加载（同实验，不漂移）；
    窗口通过 :data:`WINDOWS` 的 key 指定（paper / oos）。
    """

    # —— 数据层 ——
    pool: str
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
    # —— 组合引擎（论文 top-k/drop-n）——
    top_k: int
    drop_n: int
    min_hold: int
    cost_bps: float
    baselines: tuple[str, ...]
    # —— 窗口（本包新增：可切换 paper / oos）——
    window: str
    backtest_start: str
    backtest_end: str

    @classmethod
    def load(cls, *, window: str = "paper") -> "BaselineConfig":
        """从 paper_replication/config.yaml 加载，叠加窗口。

        :param window: ``paper``（2024-07-01~2025-06-30）或 ``oos``
            （2025-07-01~2026-07-31）。
        """
        if window not in WINDOWS:
            raise ValueError(f"未知窗口 {window!r}，可选 {list(WINDOWS)}")
        with open(PAPER_CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        start, end = WINDOWS[window]
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
            top_k=raw["portfolio"]["top_k"],
            drop_n=raw["portfolio"]["drop_n"],
            min_hold=raw["portfolio"]["min_hold"],
            cost_bps=raw["portfolio"]["cost_bps"],
            baselines=tuple(raw["portfolio"]["baselines"]),
            window=window,
            backtest_start=start,
            backtest_end=end,
        )

    def assert_oos_within_data(self) -> None:
        """样本外窗口末日 + 结算余量 ≤ 数据末日（计划 §4 运行前断言）。

        窗口末交易日 + H 个交易日（结算）需落在数据末日 2026-08-07 前，
        否则样本外信号缺结算价、回测失真。
        """
        from kronos_qlib import QlibProvider

        p = QlibProvider(self.pool, self.backtest_start, self.data_end)
        cal = p.trading_days(self.backtest_start, self.data_end)
        win_end = pd.Timestamp(self.backtest_end)
        # 窗口末 + H 个交易日（结算窗口）
        settle_end_idx = cal.get_loc(win_end) + self.predict_len
        if settle_end_idx >= len(cal):
            raise AssertionError(
                f"样本外结算越界：{self.backtest_end}+{self.predict_len} 交易日 "
                f"超出数据末日 {self.data_end}"
            )
        settle_end = cal[settle_end_idx]
        logger.info(
            f"窗口断言 OK：{self.window} 末日 {win_end.date()} + H={self.predict_len} "
            f"结算日 {settle_end.date()} ≤ 数据末日 {self.data_end}"
        )


def ensure_dirs() -> tuple[Path, Path]:
    """``data/`` 与 ``figures/`` 不入库（.gitignore 排 *.parquet），但需确保存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, FIG_DIR
