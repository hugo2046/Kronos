"""G8 语料新鲜度契约测试（计划 §1/§3.2，20260820 G8+E1 计划）。

- 配置唯一变量：G8DataConfig/G8Config 相对 G1 的差异**仅限**两处冻结日期
  （train 终点 +6 个月 / 早停窗 2025H2）+ 派生 dataset_end_time/dataset_path
  + 输出目录/seed 落点——其余协议字段逐字一致；
- 格式/边界（先 FAIL 后 PASS：需先运行 build_g8_dataset 落盘 pkl）：
  {symbol: DataFrame} 结构、feature_list 列序、datetime 单调、无 NaN、
  全序列 ≥ lookback+predict+1；train ≤ 2025-06-30 且 ≥ DDB 地板 2014-01-02、
  val ∈ [2025-07-01, 2025-12-31]（无重叠=泄漏检查）；规模 ≥ G1 ashares
  （train 语料只增不减——多了 2025H1 六个月）；
- G1 家族 pkl 只读：ashares/{train,val}_data.pkl 规模与 build_stats 一致。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from finetune_suite.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
G8_DIR = REPO_ROOT / "finetune_suite" / "data" / "g8"
ASHARES_DIR = REPO_ROOT / "finetune_suite" / "data" / "ashares"

# 协议字段（G8 必须与 G1 逐字一致；数据/训练配置共用同一清单）
_PROTOCOL_FIELDS = [
    "lookback_window", "predict_window", "max_context",
    "feature_list", "time_feature_list",
    "dataset_begin_time",
    "clip", "epochs", "log_interval", "batch_size",
    "n_train_iter", "n_val_iter",
    "tokenizer_learning_rate", "predictor_learning_rate", "accumulation_steps",
    "adam_beta1", "adam_beta2", "adam_weight_decay",
    "pretrained_tokenizer_path", "pretrained_predictor_path",
]


def test_g8_data_config_single_variable():
    from finetune_suite.build_g8_dataset import G8DataConfig
    from finetune_suite.train_g1 import G1Config

    g1, g8 = G1Config(), G8DataConfig()
    drift = [f for f in _PROTOCOL_FIELDS if getattr(g1, f) != getattr(g8, f)]
    assert not drift, f"协议字段漂移：{drift}"
    # 两处冻结日期
    assert g8.train_time_range == ["2011-01-01", "2025-06-30"]
    assert g8.val_time_range == ["2025-07-01", "2025-12-31"]
    # 派生/落点
    assert g8.dataset_end_time == "2025-12-31"
    assert g8.dataset_path.endswith("finetune_suite/data/g8")
    assert g8.instrument == "ashares"
    # G1 的 ashares 语料路径不被触碰
    assert g1.dataset_path.endswith("finetune_suite/data/ashares")


def test_g8_train_config_single_variable():
    from finetune_suite.train_g1 import G1Config
    from finetune_suite.train_g8 import G8Config

    g1 = G1Config()
    for seed in (100, 101, 102):
        g8 = G8Config(seed=seed)
        drift = [
            f for f in _PROTOCOL_FIELDS
            if getattr(g1, f) != getattr(g8, f)
        ]
        assert not drift, f"seed={seed} 协议字段漂移：{drift}"
        assert g8.seed == seed
        assert g8.train_time_range == ["2011-01-01", "2025-06-30"]
        assert g8.val_time_range == ["2025-07-01", "2025-12-31"]
        assert g8.dataset_path.endswith("finetune_suite/data/g8")
        # 三种子共享 G8 自有 tokenizer（G2 共享条款同款），目录互不重叠
        assert g8.tokenizer_save_folder_name == "finetune_tokenizer_g8"
        assert g8.predictor_save_folder_name == f"finetune_predictor_g8_s{seed}"
    toks = {G8Config(s).finetuned_tokenizer_path for s in (100, 101, 102)}
    assert len(toks) == 1, "三种子必须共享同一 G8 tokenizer"
    # G1/G2 权重目录不被触碰
    assert G8Config(100).finetuned_tokenizer_path != g1.finetuned_tokenizer_path
    assert G8Config(100).finetuned_predictor_path != g1.finetuned_predictor_path


def _load_g8_split(split: str) -> dict:
    path = G8_DIR / f"{split}_data.pkl"
    assert path.exists(), (
        f"{path} 缺失：先运行 "
        "`/home/user/miniconda3/envs/quant/bin/python -m finetune_suite.build_g8_dataset`"
    )
    with open(path, "rb") as f:
        return pickle.load(f)


def test_g8_dataset_format_boundary():
    """格式/边界/泄漏：train ≤ 2025-06-30、val ∈ 2025H2、规模 ≥ G1 ashares。"""
    cfg = Config()
    train = _load_g8_split("train")
    val = _load_g8_split("val")

    min_len = cfg.lookback_window + cfg.predict_window + 1
    for split, data in (("train", train), ("val", val)):
        assert isinstance(data, dict) and len(data) > 0
        for sym, df in data.items():
            assert list(df.columns) == cfg.feature_list, f"{split}/{sym} 列序漂移"
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index.is_monotonic_increasing
            assert not df.isna().any().any()
    # 清洗规则沿用：全序列（train∪val）≥ min_len
    for sym, df in train.items():
        n_total = len(df) + len(val.get(sym, []))
        assert n_total >= min_len, f"{sym} 全序列 {n_total} < {min_len}"

    for df in train.values():
        if len(df) == 0:
            continue
        assert df.index.max() <= pd.Timestamp("2025-06-30"), "train 越过冻结终点"
        assert df.index.min() >= pd.Timestamp("2014-01-02"), "DDB 日频地板 2014-01-02"
    for df in val.values():
        if len(df) == 0:
            continue
        assert df.index.min() >= pd.Timestamp("2025-07-01"), "val 早于冻结起点"
        assert df.index.max() <= pd.Timestamp("2025-12-31"), "val 越过冻结终点"

    # 规模 ≥ G1 ashares（同池同清洗 + 6 个月语料，只增不减）
    g1_stats = json.loads((ASHARES_DIR / "build_stats.json").read_text(encoding="utf-8"))
    n_rows_train = sum(len(df) for df in train.values())
    assert len(train) >= g1_stats["n_train_symbols"], (
        f"G8 train 股数 {len(train)} < G1 {g1_stats['n_train_symbols']}"
    )
    assert n_rows_train >= g1_stats["n_rows_train"], (
        f"G8 train 行数 {n_rows_train:,} < G1 {g1_stats['n_rows_train']:,}"
    )


def test_g1_family_pkls_untouched():
    """G1 家族 pkl 只读纪律：G8 构建后 ashares pkl 规模与 build_stats 一致。"""
    stats = json.loads((ASHARES_DIR / "build_stats.json").read_text(encoding="utf-8"))
    for split in ("train", "val"):
        p = ASHARES_DIR / f"{split}_data.pkl"
        assert p.exists(), f"ashares {p.name} 被删除（只读纪律）"
        with open(p, "rb") as f:
            data = pickle.load(f)
        assert len(data) == stats[f"n_{split}_symbols"], (
            f"ashares {split} pkl 股数 {len(data)} ≠ build_stats 记录 "
            f"{stats[f'n_{split}_symbols']}（疑被覆盖）"
        )


def test_g8_qlib_dataset_shapes():
    """官方 QlibDataset 消费 G8 pkl：抽 10 样本断言形状（第 4 轮同款）。"""
    sys.path.insert(0, str(REPO_ROOT / "finetune"))
    import dataset as official_dataset

    from finetune_suite.train_g8 import G8Config

    original = official_dataset.Config
    official_dataset.Config = G8Config
    try:
        ds = official_dataset.QlibDataset("val")  # val 小，加载快
        assert len(ds) > 0
        for i in range(10):
            x, stamp = ds[i]
            assert x.shape == (101, 6), f"样本 {i} 特征形状 {tuple(x.shape)} ≠ (101, 6)"
            assert stamp.shape == (101, 5)
    finally:
        official_dataset.Config = original
