"""finetune_suite 阶段 0 契约测试（计划 §2 步骤 0.2）。

- ``test_dataset_format``：3 股玩具行情走 DDB→pickle 适配器 → 断言 dict 结构、
  列名顺序 = ``feature_list``、datetime 索引单调、无 NaN；
- ``test_window_boundary``：train pkl 内 max(datetime) ≤ 2024-12-31 且
  val pkl 内日期 ∈ [2025-01-01, 2025-06-30]。

FakeProvider duck-type ``QlibProvider`` 的 fetch / list_pool_at（模式照搬
tests/test_kronos_qlib.py，无需 DolphinDB）。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from finetune_suite.build_dataset import build_pickles
from finetune_suite.config import Config

FETCH_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount"]


class FakeProvider:
    """内存 provider：仅实现 build_dataset 用到的 fetch / list_pool_at。"""

    def __init__(self, data: pd.DataFrame, members: list[str] | None = None):
        # data: MultiIndex(datetime, instrument)，列带 $ 前缀（模拟 qlib 原样）
        self._data = data
        if members is None:
            members = sorted(data.index.get_level_values("instrument").unique())
        self.instruments_ = members
        self._start_date = None
        self._end_date = None

    def fetch(self, fields, *, filter_pipe=None, freq="day"):
        df = self._data[
            self._data.index.get_level_values("instrument").isin(self.instruments_)
        ]
        df = df[list(fields)].copy()
        # 模拟 QlibProvider.fetch 的列名去 $
        df.columns = df.columns.str.replace("$", "", regex=False)
        return df

    def list_pool_at(self, pool, t):
        return list(self.instruments_)


def _toy_provider(codes: list[str]) -> FakeProvider:
    """3 股 × 2023-01-02~2025-06-30 工作日玩具行情（无 NaN）。"""
    dates = pd.bdate_range("2023-01-02", "2025-06-30")
    frames = []
    for ci, code in enumerate(codes):
        walk = 10.0 + ci + np.abs(
            np.cumsum(np.random.default_rng(ci).normal(0, 0.05, len(dates)))
        )
        idx = pd.MultiIndex.from_product(
            [dates, [code]], names=["datetime", "instrument"]
        )
        frames.append(
            pd.DataFrame(
                {
                    "$open": walk,
                    "$high": walk * 1.01,
                    "$low": walk * 0.99,
                    "$close": walk,
                    "$volume": 1e6,
                    "$amount": walk * 1e6,
                },
                index=idx,
            )
        )
    return FakeProvider(pd.concat(frames), members=codes)


def _build(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.dataset_path = str(tmp_path)
    stats = build_pickles(_toy_provider(["000001.SZ", "600000.SH", "300750.SZ"]), cfg)
    assert stats["n_universe"] == 3
    return cfg


def test_dataset_format(tmp_path):
    cfg = _build(tmp_path)

    for split in ("train", "val"):
        with open(Path(tmp_path, f"{split}_data.pkl"), "rb") as f:
            data = pickle.load(f)
        # dict 结构 + 每值 DataFrame
        assert isinstance(data, dict) and len(data) > 0
        assert all(isinstance(df, pd.DataFrame) for df in data.values())
        for sym, df in data.items():
            # 列名顺序 = feature_list（官方 QlibDataset 契约）
            assert list(df.columns) == cfg.feature_list
            # datetime 索引单调
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index.is_monotonic_increasing
            # 无 NaN
            assert not df.isna().any().any()
            # 整段 ≥ lookback+predict+1（官方 preprocess 同语义）
            assert len(df) >= cfg.lookback_window + cfg.predict_window + 1


def test_window_boundary(tmp_path):
    _build(tmp_path)

    with open(Path(tmp_path, "train_data.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(Path(tmp_path, "val_data.pkl"), "rb") as f:
        val = pickle.load(f)

    assert len(train) > 0 and len(val) > 0
    # train: max(datetime) ≤ 2024-12-31（且 ≥ train 窗始）
    for df in train.values():
        assert df.index.max() <= pd.Timestamp("2024-12-31")
        assert df.index.min() >= pd.Timestamp(cfg_lo("train"))
    # val: 日期 ∈ [2025-01-01, 2025-06-30]
    for df in val.values():
        assert df.index.min() >= pd.Timestamp("2025-01-01")
        assert df.index.max() <= pd.Timestamp("2025-06-30")


def cfg_lo(split: str) -> str:
    return Config().train_time_range[0] if split == "train" else Config().val_time_range[0]
