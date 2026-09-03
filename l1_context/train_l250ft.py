"""L250-ft 训练（计划 §1 臂表 / §4.3）：lookback=250 数据窗重建 + predictor 微调。

- 数据窗重建：复用 ``finetune_suite.build_dataset``（只读 import）——全 A 语料
  （--pool ashares 同款 PIT 并集采样、同款清洗规则），min_len 剔除阈值随
  lookback=250 重算（250+10+1=261 行）；产物落 ``l1_context/data/ashares_lb250/``；
- predictor 微调：G1 配方逐字（全 A 语料 / epochs=15 / batch 50 / 每 epoch 封顶
  2000 步 / AdamW 4e-5 OneCycle / seed 100），唯一变量 = lookback_window=250；
  tokenizer 冻结共享 G1 s100；**CE 早停** = 官方 train_predictor 每 epoch 验证集
  CE、最优 checkpoint 落 best_model（G1 同款，不改一字）；
- 端口 29521 与 f1(29517)/g1(29519) 错开。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m l1_context.train_l250ft dataset
    /home/user/miniconda3/envs/quant/bin/python -m l1_context.train_l250ft train
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from l1_context.config import L250FT_DATASET_DIR, _make_l250ft_config

L250FTConfig = _make_l250ft_config()

# 产物路径（模块常量；与 L250FTConfig().dataset_path 指向同一目录）
_TRAIN_PKL = L250FT_DATASET_DIR / "train_data.pkl"
_BUILD_STATS = L250FT_DATASET_DIR / "build_stats.json"


def run_dataset() -> None:
    """lookback=250 数据窗重建（DDB → 官方 pkl 格式，产物门禁 test_l250ft_dataset）。"""
    from finetune_suite.build_dataset import build_pickles, sample_pool_universe
    from kronos_qlib.provider import QlibProvider

    cfg = L250FTConfig()
    cfg.instrument = "ashares"  # 与 build_dataset.main(--pool ashares) 同款（构建期专用）
    assert cfg.lookback_window == 250 and cfg.dataset_path == str(L250FT_DATASET_DIR)
    print(f"[l250ft] dataset 重建：pool={cfg.instrument} "
          f"lookback={cfg.lookback_window} → {L250FT_DATASET_DIR}")

    pool_provider = QlibProvider(cfg.instrument, cfg.dataset_begin_time, cfg.dataset_end_time)
    universe = sample_pool_universe(pool_provider, cfg.instrument)
    fetch_provider = QlibProvider(universe, cfg.dataset_begin_time, cfg.dataset_end_time)
    stats = build_pickles(fetch_provider, cfg, universe)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    _BUILD_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def run_train() -> None:
    """predictor 微调：官方 train_predictor + L250FTConfig 注入（G1 同款机制）。"""
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29521")  # 与 f1/g1 训练端口错开

    cfg = L250FTConfig()
    assert _TRAIN_PKL.exists(), (
        f"l250ft 数据窗缺失 {_TRAIN_PKL}（先跑 `python -m l1_context.train_l250ft dataset`）")
    print(f"[l250ft] train：lookback={cfg.lookback_window} dataset={L250FT_DATASET_DIR}")
    print(f"[l250ft] epochs={cfg.epochs} batch={cfg.batch_size} n_train_iter={cfg.n_train_iter} "
          f"seed={cfg.seed} lr={cfg.predictor_learning_rate}")
    print(f"[l250ft] tokenizer（冻结共享 G1 s100）：{cfg.finetuned_tokenizer_path}")
    print(f"[l250ft] predictor 起点（官方底座）：{cfg.pretrained_predictor_path}")

    import dataset as official_dataset
    import finetune_suite.train_predictor as trainer

    assert official_dataset.Config is not L250FTConfig
    official_dataset.Config = L250FTConfig  # 注入（G1 同款，先 import 后重绑）
    trainer.main(cfg.__dict__)


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    if phase == "dataset":
        run_dataset()
    elif phase == "train":
        run_train()
    else:
        print(f"未知 phase={phase!r}，可选 dataset / train", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
