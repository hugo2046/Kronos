"""阶段 2.1/2.2：G1 全 A 微调入口（计划 §4，20260815）。

G1 = 全 A 语料（``data/ashares/``）重跑两阶段微调，**协议/超参/epochs 逐字
复用第 4 轮**（epochs=15 / batch 50 / 每 epoch 封顶 2000 步 / lr 2e-4、4e-5 /
seed 100 / OneCycle / AdamW——全部继承 ``finetune_suite.config.Config`` 不改）；
唯一变量 = 训练语料池（csi300 并集 755 股 → ashares 并集 5599 股）。

本模块不改 ``train_tokenizer.py`` / ``train_predictor.py`` 一字（纪律 §8：
既有文件只读），仅以同款 Config 注入机制把 :class:`G1Config` 喂给官方
``QlibDataset``，并调用两阶段 ``main(config_dict)``。G1 权重目录
``finetune_tokenizer_g1`` / ``finetune_predictor_g1`` 与第 4 轮 F1 隔离。

用法::

    python finetune_suite/train_g1.py --stage tokenizer    # 2.1
    python finetune_suite/train_g1.py --stage predictor    # 2.2（加载 2.1 冻结 tokenizer）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "finetune"))
sys.path.insert(0, str(_REPO_ROOT))

from finetune_suite.config import Config


class G1Config(Config):
    """G1 配置：改数据路径与输出目录名（语料池变量的落点），超参零改动。"""

    def __init__(self):
        super().__init__()
        self.dataset_path = str(Path(self.dataset_path) / "ashares")
        self.tokenizer_save_folder_name = "finetune_tokenizer_g1"
        self.predictor_save_folder_name = "finetune_predictor_g1"
        self.comet_tag = self.comet_name = "finetune_suite_g1"
        # 派生路径按新目录名重算（Config.__init__ 按 f1 目录名拼接）
        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )


def _inject_g1_dataset_config() -> None:
    """把 G1Config 注入官方 dataset 模块（QlibDataset 构造时实例化 Config）。

    与 train_tokenizer.py / train_predictor.py 顶层的注入机制同款。**必须在该
    trainer 模块 import 之后调用**——其模块级代码会把默认 Config 绑进 dataset
    命名空间，后到者覆盖先到者。
    """
    import dataset as official_dataset

    assert official_dataset.Config is not G1Config
    official_dataset.Config = G1Config


def main(stage: str) -> None:
    # 单卡 DDP 回退环境变量（同 train_tokenizer.py 声明改动②的语义）
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29519")  # 与 f1 训练端口错开

    cfg = G1Config()
    print(f"[G1] stage={stage} dataset_path={cfg.dataset_path}")
    print(f"[G1] epochs={cfg.epochs} batch={cfg.batch_size} "
          f"n_train_iter={cfg.n_train_iter} n_val_iter={cfg.n_val_iter} seed={cfg.seed}")

    # 先 import trainer（其模块级注入默认 Config），再重绑为 G1Config——顺序不可换
    if stage == "tokenizer":
        import finetune_suite.train_tokenizer as trainer
    elif stage == "predictor":
        import finetune_suite.train_predictor as trainer
    else:  # pragma: no cover - argparse choices 已限定
        raise ValueError(stage)

    _inject_g1_dataset_config()
    trainer.main(cfg.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 全 A 微调（tokenizer→predictor）")
    parser.add_argument("--stage", choices=["tokenizer", "predictor"], required=True)
    main(parser.parse_args().stage)
