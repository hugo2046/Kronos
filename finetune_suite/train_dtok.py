"""G2 增补臂 D-tok：tokenizer + predictor 全管线换 seed=101（跑前增补 37aba7d）。

补计划 §3 已声明的洞——"tokenizer 种子敏感性未检验"：tokenizer 与 predictor
**都**以 seed=101 重训（语料/协议/超参/epochs 逐字复用），与共享 G1 tokenizer
的同种子 predictor（s101）构成对照，D-tok − s101 的差值即 tokenizer 种子效应。

用法::

    python finetune_suite/train_dtok.py --stage tokenizer   # tokenizer seed=101
    python finetune_suite/train_dtok.py --stage predictor   # 加载 dtok tokenizer
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


class DtokConfig(G1Config):
    """D-tok 配置：seed=101 贯穿 tokenizer 与 predictor；输出目录 *_dtok 隔离。"""

    def __init__(self):
        super().__init__()
        self.seed = 101
        self.tokenizer_save_folder_name = "finetune_tokenizer_dtok"
        self.predictor_save_folder_name = "finetune_predictor_dtok"
        self.comet_tag = self.comet_name = "finetune_suite_g2_dtok"
        # 派生路径按 dtok 目录名重算（G1Config.__init__ 按 g1 目录名拼接）
        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )


def main(stage: str) -> None:
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29525")

    cfg = DtokConfig()
    print(f"[D-tok] stage={stage} seed={cfg.seed} dataset_path={cfg.dataset_path}")
    print(f"[D-tok] tokenizer_dir={cfg.tokenizer_save_folder_name} "
          f"predictor_dir={cfg.predictor_save_folder_name}")

    # 先 import trainer（模块级注入默认 Config），再重绑 DtokConfig——顺序不可换
    if stage == "tokenizer":
        import finetune_suite.train_tokenizer as trainer
    else:
        import finetune_suite.train_predictor as trainer
    import dataset as official_dataset

    official_dataset.Config = DtokConfig
    trainer.main(cfg.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G2 增补臂 D-tok 全管线 seed=101")
    parser.add_argument("--stage", choices=["tokenizer", "predictor"], required=True)
    main(parser.parse_args().stage)
