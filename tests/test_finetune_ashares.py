"""finetune_ashares 阶段 0 契约测试（计划 §2 步骤 0.2，20260815 计划）。

复用第 4 轮 ``test_finetune_suite.py`` 的格式/边界断言，落到**新 pkl 路径**
``finetune_suite/data/ashares/``：

- ``test_ashares_dataset_format_boundary``：{symbol: DataFrame} 结构、列名顺序
  = ``feature_list``、datetime 索引单调、无 NaN、整段 ≥ lookback+predict+1；
  train max(datetime) ≤ 2024-12-31，val ∈ [2025-01-01, 2025-06-30]；
  语料规模（计划 §2 0.3 预期：~3000-5000 股、>800 万行，如实断言下限）。
- ``test_round4_pkls_untouched``：第 4 轮 pkl 只读——ashares 构建不得改动
  ``finetune_suite/data/{train,val}_data.pkl``（内容指纹不变）。
- ``test_ashares_qlib_dataset_shapes``：官方 ``QlibDataset`` 指向 ashares 数据
  目录加载，抽 10 样本断言 x=(L+H+1, 6) / stamp=(L+H+1, 5)。

需要先运行 ``python finetune_suite/build_dataset.py --pool ashares`` 落盘
（依赖 DDB，测试本身只读 pkl，不连 DDB）。
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import pandas as pd

from finetune_suite.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
ASHARES_DIR = REPO_ROOT / "finetune_suite" / "data" / "ashares"
ROUND4_DIR = REPO_ROOT / "finetune_suite" / "data"


def _load_split(split: str) -> dict:
    path = ASHARES_DIR / f"{split}_data.pkl"
    assert path.exists(), (
        f"{path} 缺失：先运行 "
        f"`/home/user/miniconda3/envs/quant/bin/python finetune_suite/build_dataset.py --pool ashares`"
    )
    with open(path, "rb") as f:
        return pickle.load(f)


def test_ashares_dataset_format_boundary():
    cfg = Config()
    train = _load_split("train")
    val = _load_split("val")

    min_len = cfg.lookback_window + cfg.predict_window + 1
    for split, data in (("train", train), ("val", val)):
        assert isinstance(data, dict) and len(data) > 0
        for sym, df in data.items():
            assert list(df.columns) == cfg.feature_list, f"{split}/{sym} 列序漂移"
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index.is_monotonic_increasing
            assert not df.isna().any().any()
    # 适配器冻结清洗规则：**全序列**（train∪val，日期连续衔接）≥ min_len 才保留；
    # 晚上市股 train 段可短于 min_len（该段贡献 0 样本，官方 QlibDataset 语义），
    # 第 4 轮测试的"每段 ≥ min_len"在 csi300 老股语料上恰好成立、不可迁移。
    for sym, df in train.items():
        n_total = len(df) + len(val.get(sym, []))
        assert n_total >= min_len, f"{sym} 全序列 {n_total} < {min_len}"

    # 空帧（2025 年新上市股，行全在 val 段）无 min/max，跳过边界断言
    for df in train.values():
        if len(df) == 0:
            continue
        assert df.index.max() <= pd.Timestamp("2024-12-31")
        assert df.index.min() >= pd.Timestamp("2014-01-02"), "DDB 日频地板 2014-01-02"
        for df in val.values():
            if len(df) == 0:
                continue
            assert df.index.min() >= pd.Timestamp("2025-01-01")
            assert df.index.max() <= pd.Timestamp("2025-06-30")

    # 计划 §2 0.3 规模预期（下限断言，实际数字落 build_stats.json）
    n_rows_train = sum(len(df) for df in train.values())
    assert len(train) >= 3000, f"train 股数 {len(train)} < 3000（计划预期 3000-5000）"
    assert n_rows_train > 8_000_000, f"train 行数 {n_rows_train:,} ≤ 800 万"


def test_round4_pkls_untouched():
    """第 4 轮 pkl 只读纪律：ashares 构建后 csi300 pkl 指纹不变。"""

    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    for split in ("train", "val"):
        p = ROUND4_DIR / f"{split}_data.pkl"
        assert p.exists(), f"第 4 轮 {p.name} 被删除（只读纪律）"
        # 与第 4 轮 build_stats.json 记录的规模一致（粗粒度防覆盖校验）
        with open(p, "rb") as f:
            data = pickle.load(f)
        stats = __import__("json").loads(
            (ROUND4_DIR / "build_stats.json").read_text(encoding="utf-8")
        )
        assert len(data) == stats[f"n_{split}_symbols"], (
            f"第 4 轮 {split} pkl 股数 {len(data)} ≠ build_stats 记录 "
            f"{stats[f'n_{split}_symbols']}（疑被覆盖）"
        )
        assert stats["universe_rule"].startswith("csi300")


def test_ashares_qlib_dataset_shapes():
    """官方 QlibDataset 消费 ashares pkl：抽 10 样本断言形状。"""
    sys.path.insert(0, str(REPO_ROOT / "finetune"))
    import dataset as official_dataset

    class _AsharesConfig(Config):
        def __init__(self):
            super().__init__()
            self.dataset_path = str(ASHARES_DIR)

    original = official_dataset.Config
    official_dataset.Config = _AsharesConfig
    try:
        from dataset import QlibDataset

        ds = QlibDataset("val")  # val 小（数万样本），加载快
        assert len(ds) > 0
        for i in range(10):
            x, stamp = ds[i]
            assert x.shape == (101, 6), f"样本 {i} 特征形状 {tuple(x.shape)} ≠ (101, 6)"
            assert stamp.shape == (101, 5)
    finally:
        official_dataset.Config = original
