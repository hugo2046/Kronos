"""冻结 Kronos 主干 · 逐步隐状态在线提取（计划 §4.1）。

语义与 ``feature/direction-classifier:model/kronos_classifier.py`` 的
``KronosProbeClassifier.forward`` 主干前半段逐字一致，但本模块**返回逐步隐状态**
``[B, T, d_model]``（B3 KDA 头需要序列输入），而非仅末步——调用方自行 ``[:, -1]``
即可得到与 linear probe 相同的末步表征。

主干恒为 ``eval`` + ``no_grad``：token_drop / dropout 不生效，故同一输入确定可复现，
允许在训练吞吐受限时安全缓存（计划 §4.1：特征内容不许变）。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class KronosFrozenBackbone(nn.Module):
    """冻结的 Kronos tokenizer + 主干，提供 ``extract`` 逐步隐状态。

    :param tokenizer: 冻结的 :class:`model.KronosTokenizer` 实例。
    :param kronos: 冻结的 :class:`model.Kronos` 主干实例。
    :param device: 计算设备（与 tokenizer/kronos 一致）。

    前置不变式：tokenizer / kronos 已 ``move`` 到 ``device``，且本对象构造后
    它们恒为 ``eval`` 模式（参考 ``KronosProbeClassifier.train`` 重写的理由）。
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        kronos: nn.Module,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.kronos = kronos
        # 冻结全部参数（双保险：train(mode) 重写已保证 eval，requires_grad 再兜一层）
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        for p in self.kronos.parameters():
            p.requires_grad_(False)
        self.d_model: int = kronos.d_model
        self.device = device

    def train(self, mode: bool = True) -> "KronosFrozenBackbone":
        """重写 train：主干恒为 eval，避免 token_drop / dropout 影响。

        与 ``KronosProbeClassifier.train`` 同一理由：probe/头对照实验中主干表征
        必须稳定。
        """
        super().train(mode)
        self.tokenizer.eval()
        self.kronos.eval()
        return self

    @torch.no_grad()
    def extract(
        self,
        features: torch.Tensor,
        stamp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """在线提取逐步隐状态。

        :param features: 已归一化的输入 ``[B, T, d_in]``（d_in=6，OHLCVA）。
            **调用方负责窗口 z-score + clip5**（与 ``KronosPredictor.predict_batch``
            同口径），本方法不自做归一化——保证 B1「原始特征 vs 预训练表示」唯一变量。
        :param stamp: 时间特征 ``[B, T, 5]``（minute/hour/weekday/day/month，
            同 ``calc_time_stamps``）；``None`` 时不加时间嵌入。
        :returns: 主干逐步隐状态 ``[B, T, d_model]``（d_model=832）。
        """
        s1_ids, s2_ids = self.tokenizer.encode(features, half=True)
        # 复刻 Kronos.forward 的主干前半段（不走 DualHead / DependencyAwareLayer）
        x = self.kronos.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.kronos.time_emb(stamp)
        x = self.kronos.token_drop(x)
        for layer in self.kronos.transformer:
            x = layer(x)
        x = self.kronos.norm(x)
        return x


__all__ = ["KronosFrozenBackbone"]
