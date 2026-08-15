"""G2.1：G1 predictor 种子重训入口（计划 §1，20260816 计划）。

种子诊断臂（冻结）：G1 predictor 以 seed=101 / seed=102 各重训一次——
协议/超参/epochs=15/语料 pkl（ashares）/**冻结 G1 tokenizer（共享条款复用
不重训）**全部逐字复用，**唯一变量 = predictor 训练种子**。输出目录
``finetune_predictor_g2_s{seed}`` 与 G1（seed=100）隔离。

用法::

    python finetune_suite/train_g2.py --seed 101
    python finetune_suite/train_g2.py --seed 102
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "finetune"))
sys.path.insert(0, str(_REPO_ROOT))

from finetune_suite.train_g1 import G1Config


class G2Config(G1Config):
    """G2 配置：在 G1 基础上仅改 seed 与 predictor 输出目录（唯一变量的落点）。

    tokenizer 路径继承 G1Config（finetune_tokenizer_g1，共享条款）；
    语料/超参/epochs/数据窗口零改动。
    """

    def __init__(self, seed: int = 101):
        super().__init__()
        assert seed in (101, 102), "G2 计划冻结：仅 seed=101/102 两个新种子"
        self.seed = seed
        self.predictor_save_folder_name = f"finetune_predictor_g2_s{seed}"
        self.comet_tag = self.comet_name = f"finetune_suite_g2_s{seed}"
        # 派生路径按新目录名重算（G1Config.__init__ 按 g1 目录名拼接）
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )


def main(seed: int) -> None:
    # 单卡 DDP 回退环境变量（同 train_g1.py；端口按种子错开，支持双种子并行）
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29521 + seed - 101))

    cfg = G2Config(seed=seed)
    print(f"[G2] seed={seed} dataset_path={cfg.dataset_path}")
    print(f"[G2] tokenizer（共享复用）：{cfg.finetuned_tokenizer_path}")
    print(f"[G2] epochs={cfg.epochs} batch={cfg.batch_size} "
          f"n_train_iter={cfg.n_train_iter} n_val_iter={cfg.n_val_iter} seed={cfg.seed}")

    # 先 import trainer（其模块级注入默认 Config），再重绑 G2Config——顺序不可换
    # （与 train_g1 同款机制，G1 tokenizer 路径经 G2Config.finetuned_tokenizer_path 复用）
    import finetune_suite.train_predictor as trainer
    import dataset as official_dataset

    official_dataset.Config = G2Config
    trainer.main(cfg.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G2 predictor 种子重训（101/102）")
    parser.add_argument("--seed", type=int, choices=[101, 102], required=True)
    main(parser.parse_args().seed)
