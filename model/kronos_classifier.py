"""判别式方向分类模型。

本模块实现两种方向分类头，均输出 ``forward(features, stamp) -> logits[B, n_classes]``，
便于训练/评估代码在同一脚本内按 config 的 ``backbone`` 字段切换：

- :class:`KronosClassifier`（方案 A）：连续特征线性投影 + 时间嵌入 + 从头训练的
  Transformer 主干 + 分类头；
- :class:`KronosProbeClassifier`（方案 B）：冻结预训练 Tokenizer + Kronos 主干，
  仅在其最后时间步隐状态上训练一个 linear probe。

设计参考见 ``docs/Kronos方向分类头改造方案_20260809.md``。不修改
``model/kronos.py`` 现有生成式类，保证原链路不受影响。
"""

import torch
import torch.nn as nn

from model.module import RMSNorm, TemporalEmbedding, TransformerBlock


class KronosClassifier(nn.Module):
    """判别式方向分类模型（方案 A，从头训练）。

    连续 OHLCV+amount 经线性投影后与时间嵌入相加，送入 N 层 TransformerBlock，
    取最后一个时间步的表征经线性层输出 ``n_classes`` 维 logits。

    :param d_in: 输入特征维度（默认 6：OHLCV + amount）。
    :param d_model: 模型隐层维度。
    :param n_heads: 注意力头数。
    :param ff_dim: FFN 维度。
    :param n_layers: TransformerBlock 层数。
    :param n_classes: 分类数（默认 3：跌/平/涨）。
    :param learn_te: 是否使用可学习时间嵌入。
    :param dropout: 输入嵌入后的 dropout 概率。
    """

    def __init__(
        self,
        d_in: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        n_layers: int = 4,
        n_classes: int = 3,
        learn_te: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_classes = n_classes

        # 连续输入直接线性投影，写法参考 KronosTokenizer.embed
        self.embed = nn.Linear(d_in, d_model)
        self.time_emb = TemporalEmbedding(d_model, learn_te)
        self.input_dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_dim, dropout, 0.0, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        # 取最后时间步表征接分类头
        self.classifier = nn.Linear(d_model, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier 从头初始化，写法参考 ``Kronos._init_weights``。

        :param module: 待初始化的子模块。
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self, features: torch.Tensor, stamp: torch.Tensor
    ) -> torch.Tensor:
        """前向计算。

        :param features: 输入特征张量，形状 ``[B, L, d_in]``。
        :param stamp: 时间特征张量，形状 ``[B, L, 5]``（minute/hour/weekday/day/month）。
        :return: 分类 logits，形状 ``[B, n_classes]``。
        """
        x = self.embed(features)
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.input_dropout(x)

        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x)
        # 取最后一个时间步用于分类
        x = x[:, -1]
        return self.classifier(x)


class KronosProbeClassifier(nn.Module):
    """冻结预训练主干 + linear probe 的方向分类模型（方案 B）。

    内部持有冻结的 :class:`KronosTokenizer` 与 :class:`Kronos` 主干，前向时先用
    Tokenizer 将连续特征编码为离散 token，再过 Kronos 主干取最后时间步隐状态，
    最后由唯一可训练的 :attr:`probe` 输出 logits。用以对照验证“预训练红利不可丢”。

    :param tokenizer: 冻结的 KronosTokenizer 实例。
    :param kronos: 冻结的 Kronos 主干实例。
    :param n_classes: 分类数（默认 3：跌/平/涨）。
    """

    def __init__(self, tokenizer: nn.Module, kronos: nn.Module, n_classes: int = 3):
        super().__init__()
        self.tokenizer = tokenizer
        self.kronos = kronos
        self.d_model = kronos.d_model
        self.n_classes = n_classes

        # 冻结整个预训练主干（含 Tokenizer），仅 probe 可训练
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        for p in self.kronos.parameters():
            p.requires_grad_(False)

        self.probe = nn.Linear(self.d_model, n_classes)

    def train(self, mode: bool = True) -> "KronosProbeClassifier":
        """重写 train，确保冻结主干恒为 eval 模式。

        避免主干内的 token_drop / dropout 受 mode 影响，保证 probe 对照实验中
        主干表征稳定。

        :param mode: probe 是否进入训练模式。
        :return: self。
        """
        super().train(mode)
        # 主干始终为 eval，仅 probe 遵循 mode
        self.tokenizer.eval()
        self.kronos.eval()
        return self

    def forward(
        self, features: torch.Tensor, stamp: torch.Tensor
    ) -> torch.Tensor:
        """前向计算：冻结主干取隐状态 + 可训练 probe。

        :param features: 输入特征张量，形状 ``[B, L, d_in]``（已归一化）。
        :param stamp: 时间特征张量，形状 ``[B, L, 5]``。
        :return: 分类 logits，形状 ``[B, n_classes]``。
        """
        # 主干冻结，推理阶段不计算梯度
        with torch.no_grad():
            s1_ids, s2_ids = self.tokenizer.encode(features, half=True)
            # 复刻 Kronos.forward 的主干前半段（不走 DualHead / DependencyAwareLayer）
            x = self.kronos.embedding([s1_ids, s2_ids])
            if stamp is not None:
                x = x + self.kronos.time_emb(stamp)
            x = self.kronos.token_drop(x)
            for layer in self.kronos.transformer:
                x = layer(x)
            x = self.kronos.norm(x)
            # 取最后一个时间步的隐状态
            x = x[:, -1]

        # probe 在 no_grad 之外，保留梯度用于反向传播
        return self.probe(x)
