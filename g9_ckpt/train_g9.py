"""G9：predictor 全 epoch 重训入口（计划 §1/§4.2，20260821 G9 计划）。

G9 = checkpoint 选择规则预注册实验：加载 **G1 s100 tokenizer（冻结只读）**，
predictor 按 G1 配方逐字重训 seed 100，唯一差异 = 每 epoch 落盘全部 15 个
checkpoint（``g9_ckpt/train_predictor_all_epochs.py`` =
``finetune_suite/train_predictor.py`` 逐字复制 + 唯一落盘改动，复制保真由
``tests/test_g9_ckpt.py`` 字节级对拍）。

G9Config 相对 G1Config 的差异**仅限输出目录落点**（``g9_ckpt/`` 下与 G1 隔离，
不动 ``finetune_suite/outputs/``）：语料（ashares 并集）、窗口、全部超参、
seed=100、G1 tokenizer 路径逐字段继承（``tests/test_g9_ckpt.py::test_recipe_frozen``）。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m g9_ckpt.train_g9 --stage predictor
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from finetune_suite.train_g1 import G1Config


class G9Config(G1Config):
    """G9 配置：只改输出目录落点（g9_ckpt/ 下隔离），配方零改动。

    继承链 G1Config → 本类：dataset_path（ashares 并集）、时间窗、epochs=15、
    lr 4e-5、batch 50、每 epoch 2000 步、seed=100、G1 tokenizer 路径全部原样；
    save_path 指向 ``g9_ckpt/outputs/models``，predictor 权重目录
    ``finetune_predictor_g9``。
    """

    def __init__(self):
        super().__init__()
        self.save_path = str(Path(__file__).resolve().parent / "outputs" / "models")
        self.predictor_save_folder_name = "finetune_predictor_g9"
        self.comet_tag = self.comet_name = "finetune_suite_g9"
        # 派生 predictor 路径按新 save_path 重算；finetuned_tokenizer_path
        # 继承 G1Config 不动——G1 tokenizer 冻结只读，本轮不重训。
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )


def _inject_g9_dataset_config() -> None:
    """把 G9Config 注入官方 dataset 模块（QlibDataset 构造时实例化 Config）。

    与 train_g1.py 的注入机制同款：先 import trainer（其模块级把默认 Config
    绑进 dataset 命名空间），再重绑为 G9Config——顺序不可换，后到者覆盖先到者。
    """
    import dataset as official_dataset

    assert official_dataset.Config is not G9Config
    official_dataset.Config = G9Config


def main(stage: str) -> None:
    # 单卡 DDP 回退环境变量（同 train_g1.py 语义；端口与 f1/g1 训练错开）
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29521")

    cfg = G9Config()
    print(f"[G9] stage={stage} dataset_path={cfg.dataset_path}")
    print(f"[G9] epochs={cfg.epochs} batch={cfg.batch_size} "
          f"n_train_iter={cfg.n_train_iter} n_val_iter={cfg.n_val_iter} seed={cfg.seed}")
    print(f"[G9] tokenizer（G1 冻结只读）={cfg.finetuned_tokenizer_path}")
    print(f"[G9] save_dir={cfg.save_path}/{cfg.predictor_save_folder_name}")

    # 先 import trainer（其模块级注入默认 Config），再重绑为 G9Config——顺序不可换
    if stage == "predictor":
        import g9_ckpt.train_predictor_all_epochs as trainer
    else:  # pragma: no cover - argparse choices 已限定
        raise ValueError(stage)

    _inject_g9_dataset_config()
    trainer.main(cfg.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G9 predictor 全 epoch 重训（G1 配方逐字 + 每 epoch 落盘）")
    parser.add_argument("--stage", choices=["predictor"], required=True)
    main(parser.parse_args().stage)
