"""H1 日截面采样器（计划 §1：batch 128 = 同一交易日截面随机抽 128 股）。

每步：均匀抽一个训练决策日 → 该日截面内均匀抽 128 个成员（不足 128 整日全取）
→ 在线物化 (x_norm, stamp, y_z)。独立 ``random.Random(seed)``（官方
``finetune/dataset.py.py_rng`` 同款——不干扰模型初始化的随机流）。
"""
from __future__ import annotations

import random

import numpy as np
import torch

from h1_readout.corpus import Corpus, TrainDay, build_train_batch


class DailyBatchSampler:
    """同日截面随机批采样（IC 在批内计算——R1 损失的批语义）。"""

    def __init__(self, corpus: Corpus, *, seed: int, batch_size: int = 128):
        self.corpus = corpus
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TrainDay]:
        """抽一批：返回 (x_norm [B,90,6], stamp [B,90,5], y_z [B], 当日对象)。"""
        day = self.rng.choice(self.corpus.train_days)
        n = len(day.codes)
        k = min(self.batch_size, n)
        idx = np.array(self.rng.sample(range(n), k), dtype=np.int64)
        x, stamp, y = build_train_batch(self.corpus, day, idx)
        return x, stamp, y, day

    def steps(self, steps_per_epoch: int):
        """epoch 迭代器（步数封顶——G1 式 2000 步/epoch）。"""
        for _ in range(steps_per_epoch):
            yield self.sample()


__all__ = ["DailyBatchSampler"]
