"""finetune_ashares 阶段 2 契约测试（计划 §4，20260815 计划）。

- ``test_g1_config_single_variable``：G1Config 相对第 4 轮 Config 的差异**仅限**
  语料路径 / 输出目录 / 实验名——一切训练超参与协议逐字一致（单一变量纪律）；
- ``test_g1_dataset_injection``：G1Config 注入官方 dataset 模块后，QlibDataset
  读 ``data/ashares/`` 的 pkl；
- ``test_g1_signals_backtest_alignment``：G1 四变体 backtest 窗宽表存在、四表
  索引一致、且与第 4 轮 F0/M backtest 信号索引逐日一致（同窗可比）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from finetune_suite.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = REPO_ROOT / "finetune_suite" / "data" / "g1"
ROUND4_DATA = REPO_ROOT / "finetune_suite" / "data"

# 协议字段（纪律 §8：跑前冻结，G1 必须逐字复用第 4 轮）
_PROTOCOL_FIELDS = [
    "lookback_window", "predict_window", "max_context", "feature_list",
    "time_feature_list", "train_time_range", "val_time_range",
    "dataset_begin_time", "dataset_end_time",
    "clip", "epochs", "log_interval", "batch_size",
    "n_train_iter", "n_val_iter",
    "tokenizer_learning_rate", "predictor_learning_rate", "accumulation_steps",
    "adam_beta1", "adam_beta2", "adam_weight_decay", "seed",
    "pretrained_tokenizer_path", "pretrained_predictor_path",
]


def test_g1_config_single_variable():
    from finetune_suite.train_g1 import G1Config

    base, g1 = Config(), G1Config()
    drift = [f for f in _PROTOCOL_FIELDS if getattr(base, f) != getattr(g1, f)]
    assert not drift, f"协议字段漂移（G1 必须逐字复用第 4 轮）：{drift}"

    # 唯一变量的落点：语料池 → 数据路径 + 输出目录 + 实验名
    assert g1.dataset_path.endswith("data/ashares")
    assert g1.tokenizer_save_folder_name == "finetune_tokenizer_g1"
    assert g1.predictor_save_folder_name == "finetune_predictor_g1"
    assert g1.finetuned_tokenizer_path.endswith("finetune_tokenizer_g1/checkpoints/best_model")
    assert g1.finetuned_predictor_path.endswith("finetune_predictor_g1/checkpoints/best_model")
    # 第 4 轮 F1 输出目录不被 G1 触碰
    assert g1.tokenizer_save_folder_name != base.tokenizer_save_folder_name
    assert g1.predictor_save_folder_name != base.predictor_save_folder_name


def test_g1_dataset_injection():
    sys.path.insert(0, str(REPO_ROOT / "finetune"))
    import dataset as official_dataset

    # 镜像 train_g1.main 的真实顺序：先 import trainer（其模块级注入默认
    # Config），再重绑 G1Config——顺序错了会被 trainer 的默认注入覆盖
    import finetune_suite.train_tokenizer  # noqa: F401 触发其模块级注入
    from finetune_suite.train_g1 import G1Config, _inject_g1_dataset_config

    original = official_dataset.Config
    assert official_dataset.Config is not G1Config  # trainer 已注入默认 Config
    _inject_g1_dataset_config()
    try:
        assert official_dataset.Config is G1Config
        ds = official_dataset.QlibDataset("val")
        assert ds.data_path.endswith("data/ashares/val_data.pkl")
        x, stamp = ds[0]
        assert x.shape == (101, 6) and stamp.shape == (101, 5)
    finally:
        official_dataset.Config = original


def test_g1_signals_backtest_alignment():
    from baseline_suite.common import VARIANTS

    paths = {v: G1_DIR / f"daily_signals_backtest_G1_{v}.parquet" for v in VARIANTS}
    missing = [p.name for p in paths.values() if not p.exists()]
    assert not missing, f"G1 信号缺失 {missing}：先运行 run_g1_signals.py（断点续跑）"

    ref_idx = pd.read_parquet(paths["mean"]).index
    assert len(ref_idx) > 0, "G1_mean 信号为空"
    for v in VARIANTS:
        assert pd.read_parquet(paths[v]).index.equals(ref_idx), f"G1 {v} 索引不一致"
    # 同窗可比：与第 4 轮 F0/M backtest 信号（只读复用）逐日一致
    f0m_idx = pd.read_parquet(ROUND4_DATA / "daily_signals_backtest_M.parquet").index
    assert ref_idx.equals(f0m_idx), "G1 调仓日与第 4 轮 backtest 窗索引不一致"
