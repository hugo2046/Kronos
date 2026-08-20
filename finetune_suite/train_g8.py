"""G8 全 A 语料新鲜度微调入口（计划 §1/§3.3，20260820 G8+E1 计划）。

G8 = G1 两阶段微调**逐字一致**，唯一变量 = 语料终点（train 2024-12-31 →
2025-06-30；val/早停窗 2025H1 → 2025H2）。语料池（ashares 并集）、两阶段、
epochs=15 / batch 50 / 每 epoch 封顶 2000 步 / lr 2e-4、4e-5 / OneCycle / AdamW
全部继承不改；种子 100/101/102 镜像 G1 种子族结构：

    - s100 训 tokenizer + predictor（G8 自有 tokenizer：finetune_tokenizer_g8）；
    - s101/s102 只训 predictor，共享 G8 tokenizer（G2 共享条款同款）。

G1 权重目录（finetune_*_g1 / finetune_*_g2_*）只读不触碰（纪律 §4）。

用法::

    python finetune_suite/train_g8.py --stage tokenizer            # s100 两阶段之一
    python finetune_suite/train_g8.py --stage predictor --seed 100  # 100/101/102
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


class G8Config(G1Config):
    """G8 训练配置：G1 之上仅改两处日期 + 派生边界 + 语料/输出目录 + seed。

    日期字段与 :class:`finetune_suite.build_g8_dataset.G8DataConfig` 逐字一致
    （同一冻结表落点）；其余训练超参零改动。
    """

    def __init__(self, seed: int = 100):
        super().__init__()
        assert seed in (100, 101, 102), "G8 冻结：seed ∈ {100,101,102}（镜像 G1 种子族）"
        self.seed = seed
        # —— 冻结的两处日期（计划 §1 表）——
        self.train_time_range = ["2011-01-01", "2025-06-30"]
        self.val_time_range = ["2025-07-01", "2025-12-31"]
        # —— 派生（声明式对齐，非自由度）——
        self.dataset_end_time = "2025-12-31"  # 对齐 val 末
        self.dataset_path = str(Path(self.dataset_path).parent / "g8")  # G8 语料 pkl
        # —— 输出目录（s101/102 共享 G8 tokenizer = G2 共享条款同款）——
        self.tokenizer_save_folder_name = "finetune_tokenizer_g8"
        self.predictor_save_folder_name = f"finetune_predictor_g8_s{seed}"
        self.comet_tag = self.comet_name = f"finetune_suite_g8_s{seed}"
        # 派生路径按新目录名重算
        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )


def main(stage: str, seed: int) -> None:
    # 单卡 DDP 回退（同 train_g1；端口与 g1/g2 家族错开）
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29531 + seed - 100))

    cfg = G8Config(seed=seed)
    print(f"[G8] stage={stage} seed={seed} dataset_path={cfg.dataset_path}")
    print(f"[G8] train={cfg.train_time_range} val={cfg.val_time_range} "
          f"dataset_end={cfg.dataset_end_time}")
    print(f"[G8] epochs={cfg.epochs} batch={cfg.batch_size} "
          f"n_train_iter={cfg.n_train_iter} n_val_iter={cfg.n_val_iter}")

    # 先 import trainer（其模块级注入默认 Config），再重绑 G8Config——顺序不可换
    if stage == "tokenizer":
        import finetune_suite.train_tokenizer as trainer
    elif stage == "predictor":
        import finetune_suite.train_predictor as trainer
    else:  # pragma: no cover - argparse choices 已限定
        raise ValueError(stage)

    import dataset as official_dataset

    assert official_dataset.Config is not G8Config
    official_dataset.Config = G8Config
    # tokenizer 阶段恒以 seed=100 落 G8 家族共享工件（G1 家族同构）
    trainer.main(G8Config(seed=100 if stage == "tokenizer" else seed).__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G8 语料新鲜度微调（tokenizer→predictor）")
    parser.add_argument("--stage", choices=["tokenizer", "predictor"], required=True)
    parser.add_argument("--seed", type=int, choices=[100, 101, 102], default=100,
                        help="predictor 种子（100/101/102）；tokenizer 恒 seed=100")
    args = parser.parse_args()
    main(args.stage, args.seed)
