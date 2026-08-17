"""G1 微调底座装载（G5 计划 §2：G1 底座冻结，只读）。

G1 权重（seed=100，finetune_suite 阶段 2.1/2.2 产物）以
``KronosTokenizer.from_pretrained`` / ``Kronos.from_pretrained`` 装载后包进
``cross_section_kda.backbone.KronosFrozenBackbone``（吃实例）——与第 3 轮
``improve_suite.run_mamba_head._load_backbone`` 同构，唯一差别是权重来源
（HF 官方 → 本地 G1 微调目录）。G1 权重只读：本模块及下游绝不写回。
"""
from __future__ import annotations

from loguru import logger


def load_g1_backbone(device: str):
    """装载冻结 G1 底座（tokenizer + predictor → KronosFrozenBackbone）。

    :param device: 目标设备（实例先 ``.to(device)`` 再入包装，与契约一致）。
    """
    from cross_section_kda import KronosFrozenBackbone
    from finetune_suite.train_g1 import G1Config
    from model import Kronos, KronosTokenizer

    g1 = G1Config()
    logger.info(f"加载 G1 tokenizer：{g1.finetuned_tokenizer_path}")
    tokenizer = KronosTokenizer.from_pretrained(g1.finetuned_tokenizer_path).to(device)
    logger.info(f"加载 G1 predictor：{g1.finetuned_predictor_path}")
    kronos = Kronos.from_pretrained(g1.finetuned_predictor_path).to(device)
    return KronosFrozenBackbone(tokenizer=tokenizer, kronos=kronos, device=device)


__all__ = ["load_g1_backbone"]
