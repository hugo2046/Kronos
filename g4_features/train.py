"""G4 两阶段训练入口（G4 计划 §3 1.1）。

每种子（100/101/102，冻结）两阶段：

- **tokenizer 阶段**：G1 tokenizer（``finetune_tokenizer_g1``，只读）经
  6→9 列零初始化手术（:func:`g4_features.surgery.expand_tokenizer_6to9`，
  等价门禁已由 ``tests/test_g4_features.py::TestWarmstartEquivalence`` 钉死）
  后，按官方两阶段协议续训；
- **predictor 阶段**：冻结本种子 G4 tokenizer + **G1 族 predictor 原形装载**
  （s100=``finetune_predictor_g1``，s101/s102=``finetune_predictor_g2_s{seed}``，
  架构与 token 词表与 d_in 无关，权重零手术），同协议续训。

训练循环**逐字复用** ``finetune_suite/train_{tokenizer,predictor}.py`` 的
``train_model``（协议/超参/epochs=15/batch=50/每 epoch 2000 步/lr 2e-4、4e-5
全部继承 ``Config`` 零改动）；本入口只替换模型构造（手术/warm-start）与
配置注入（``G4Config``：9 列 feature_list + 本包数据路径），并复刻
trainer ``main`` 的 DDP/seed/summary 骨架。

用法::

    python g4_features/train.py --seed 100 --stage tokenizer
    python g4_features/train.py --seed 100 --stage predictor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import gmtime, strftime

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "finetune"))
sys.path.insert(0, str(_REPO_ROOT))

from g4_features.config import G1_PREDICTOR_DIRS, G1_TOKENIZER_DIR, G4Config

_CONFIG_CACHE: dict[int, type] = {}


def _bind_config_class(seed: int) -> type:
    """生成无参构造返回 G4Config(seed) 的轻量子类（注入 QlibDataset 用）。"""
    if seed not in _CONFIG_CACHE:
        _bound = type(f"G4ConfigS{seed}", (G4Config,), {"__init__": lambda self, s=seed: super(_bound, self).__init__(s)})  # noqa: E501
        _CONFIG_CACHE[seed] = _bound
    return _CONFIG_CACHE[seed]


def main(seed: int, stage: str) -> None:
    # 单卡 DDP 回退环境变量（端口按种子错开，同 train_g1/train_g2 惯例）
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29531 + seed - 100))

    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from utils.training_utils import cleanup_ddp, get_model_size, set_seed, setup_ddp

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(seed, rank)

    cfg = G4Config(seed=seed)
    print(f"[G4] seed={seed} stage={stage} dataset_path={cfg.dataset_path}")
    print(f"[G4] feature_list={cfg.feature_list}")
    print(f"[G4] epochs={cfg.epochs} batch={cfg.batch_size} "
          f"n_train_iter={cfg.n_train_iter} n_val_iter={cfg.n_val_iter} "
          f"tok_lr={cfg.tokenizer_learning_rate} pred_lr={cfg.predictor_learning_rate}")

    # 先 import trainer（其模块级注入默认 Config），再重绑为 G4Config——顺序不可换
    if stage == "tokenizer":
        import finetune_suite.train_tokenizer as trainer
    elif stage == "predictor":
        import finetune_suite.train_predictor as trainer
    else:  # pragma: no cover - argparse choices 已限定
        raise ValueError(stage)
    _bind_config_class(seed)  # 确保注入类已生成
    import dataset as official_dataset

    official_dataset.Config = _bind_config_class(seed)

    save_dir = os.path.join(
        cfg.save_path,
        cfg.tokenizer_save_folder_name if stage == "tokenizer" else cfg.predictor_save_folder_name,
    )
    master_summary: dict = {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
        master_summary = {
            "start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            "save_directory": save_dir,
            "world_size": world_size,
            "seed": seed,
            "stage": stage,
            "warm_start": {
                "tokenizer": str(G1_TOKENIZER_DIR),
                "predictor": str(G1_PREDICTOR_DIRS[seed]),
                "surgery": "expand_tokenizer_6to9（tokenizer 阶段；predictor 原形装载零手术）",
            },
        }
    import torch.distributed as dist

    dist.barrier()

    if stage == "tokenizer":
        from model.kronos import KronosTokenizer

        from g4_features.surgery import expand_tokenizer_6to9

        base = KronosTokenizer.from_pretrained(str(G1_TOKENIZER_DIR))  # G1 权重只读
        model = expand_tokenizer_6to9(base)
        model.to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if rank == 0:
            print(f"Tokenizer Model Size: {get_model_size(model.module)}")
            print(f"warm-start：{G1_TOKENIZER_DIR} → 6→9 列零初始化手术")
        _, dt_result = trainer.train_model(
            model, device, cfg.__dict__, save_dir, None, rank, world_size
        )
    else:
        from model.kronos import Kronos, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(cfg.finetuned_tokenizer_path)
        tokenizer.eval().to(device)
        model = Kronos.from_pretrained(str(G1_PREDICTOR_DIRS[seed]))  # 原形装载
        model.to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if rank == 0:
            print(f"Predictor Model Size: {get_model_size(model.module)}")
            print(f"warm-start：tokenizer={cfg.finetuned_tokenizer_path}")
            print(f"             predictor={G1_PREDICTOR_DIRS[seed]}（原形装载）")
        dt_result = trainer.train_model(
            model, tokenizer, device, cfg.__dict__, save_dir, None, rank, world_size
        )

    if rank == 0:
        master_summary["final_result"] = dt_result
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(master_summary, f, indent=4, ensure_ascii=False)
        print("Training finished. Summary file saved.")
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G4 两阶段微调（9 列 warm-start）")
    parser.add_argument("--seed", type=int, choices=[100, 101, 102], required=True)
    parser.add_argument("--stage", choices=["tokenizer", "predictor"], required=True)
    args = parser.parse_args()
    main(args.seed, args.stage)
