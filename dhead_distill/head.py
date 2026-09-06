"""多期限预测头（方案 §4.1）。

结构（冻结 G1 最后 RMSNorm 后隐状态 ``[B,90,832]`` → ``[B,10]`` 收益预测）：

1. ``Linear(832,128)`` 投影历史；
2. 10 个期限 embedding 为 query，加上未来日历 5 项 embedding 之和
   （minute/hour/weekday/day/month，各 128 维）——假期位置可表达；
3. 单层 ``MultiheadAttention(128,4,batch_first=True,dropout=0)``，K/V 来自
   历史投影；
4. query 残差 + LayerNorm，共享 ``Linear(128,1)`` 输出 10 个期限收益。

期限 query 只看截至 t 的历史；第一版不加 query 间 self-attention / MLP。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MultiHorizonHead(nn.Module):
    """单层交叉注意力的多期限收益预测头。

    :param d_model: 底座隐状态维度（真实 G1 = 832，构造时校验由接线方负责）。
    :param head_dim: 投影/attention 维度（默认 128）。
    :param n_heads: attention 头数（默认 4）。
    :param n_horizons: 期限数（默认 10）。
    :param calendar_cardinalities: 未来日历 5 项基数，默认 ``(60,24,7,32,13)``。
    """

    def __init__(
        self,
        *,
        d_model: int = 832,
        head_dim: int = 128,
        n_heads: int = 4,
        n_horizons: int = 10,
        calendar_cardinalities: Sequence[int] = (60, 24, 7, 32, 13),
        output_space: str = "raw_return",
    ) -> None:
        super().__init__()
        if len(calendar_cardinalities) != 5:
            raise ValueError("future_stamp 须为 5 列（minute/hour/weekday/day/month）")
        if d_model <= 0 or head_dim <= 0:
            raise ValueError("d_model / head_dim 须为正")
        if output_space not in ("raw_return", "normalized_close_affine_return"):
            raise ValueError(f"未知 output_space：{output_space}")
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_horizons = n_horizons
        self.cardinalities = tuple(int(c) for c in calendar_cardinalities)
        self.output_space = output_space

        self.proj = nn.Linear(d_model, head_dim)
        self.calendar_embeddings = nn.ModuleList(
            nn.Embedding(c, head_dim) for c in self.cardinalities
        )
        self.horizon_embedding = nn.Embedding(n_horizons, head_dim)
        self.attn = nn.MultiheadAttention(
            head_dim, n_heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(head_dim)
        self.out = nn.Linear(head_dim, 1)

    def validate_future_stamp(self, future_stamp: torch.Tensor) -> None:
        """future_stamp 合法范围校验：``[B,H,5]``，整数值、各列在基数内。

        接受整数或浮点存储（浮点须与 long 转换逐位一致——日历特征本就是整数）。
        """
        if future_stamp.dim() != 3 or future_stamp.shape[-1] != 5:
            raise ValueError(
                f"future_stamp 须为 [B,{self.n_horizons},5]，得到 {tuple(future_stamp.shape)}"
            )
        if future_stamp.shape[1] != self.n_horizons:
            raise ValueError(
                f"future_stamp 期限维须为 {self.n_horizons}，得到 {future_stamp.shape[1]}"
            )
        fs = future_stamp.long()
        if not torch.equal(fs, future_stamp) and future_stamp.dtype.is_floating_point:
            # 浮点存储但含非整数值 → 非法
            if not torch.equal(fs.float(), future_stamp.float()):
                raise ValueError("future_stamp 含非整数值（日历特征必须整数）")
        for j, card in enumerate(self.cardinalities):
            col = fs[..., j]
            if (col < 0).any() or (col >= card).any():
                raise ValueError(
                    f"future_stamp 第 {j} 列越界：值域 [0,{card})，"
                    f"实测 [{int(col.min())},{int(col.max())}]"
                )

    def forward(
        self, hidden: torch.Tensor, future_stamp: torch.Tensor,
        a: torch.Tensor | None = None, b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向：历史隐状态 + 未来日历 → ``[B,n_horizons]`` 收益预测。

        两种输出语义（R2，由 ``output_space`` 冻结于构造/协议身份）：

        - ``raw_return``（v1）：直接输出原始收益，禁止传 a/b；
        - ``normalized_close_affine_return``：输出标准化未来 close z，
          经逐样本仿射还原 ``r_hat = a_i·z + b_i``（a/b 只由历史 90 日原始
          close 算出，见 :func:`dhead_distill.data.affine_restore_params`；
          a/b 必须显式传入，缺失报错——不靠隐藏全局状态）。

        :param hidden: 冻结底座隐状态 ``[B,T,d_model]``。
        :param future_stamp: 未来日历 ``[B,H,5]``。
        :param a/b: ``[B]`` 仿射还原系数（仅 affine 语义）。
        :returns: ``[B,H]`` 各期限收益预测（原始小数单位）。
        """
        self.validate_future_stamp(future_stamp)
        fs = future_stamp.long()
        memory = self.proj(hidden)
        calendar = sum(
            embedding(fs[..., j].long())
            for j, embedding in enumerate(self.calendar_embeddings)
        )
        query = self.horizon_embedding.weight.unsqueeze(0) + calendar
        pooled, _ = self.attn(query, memory, memory, need_weights=False)
        z = self.out(self.norm(query + pooled)).squeeze(-1)
        if self.output_space == "raw_return":
            if a is not None or b is not None:
                raise ValueError("output_space=raw_return 不接受仿射系数 a/b")
            return z
        if a is None or b is None:
            raise ValueError(
                "output_space=normalized_close_affine_return 必须传入逐样本 a/b"
            )
        if a.shape != z.shape[:1] or b.shape != z.shape[:1]:
            raise ValueError(
                f"a/b 形状 {tuple(a.shape)}/{tuple(b.shape)} 须为 [B]={tuple(z.shape[:1])}"
            )
        return a.unsqueeze(-1) * z + b.unsqueeze(-1)

    def parameter_summary(self) -> dict:
        """实测参数统计（§4.1：记录实测可训练参数数目，不依赖估计文字）。"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable_params": trainable,
            "total_params": total,
            "d_model": self.d_model,
            "head_dim": self.head_dim,
            "n_horizons": self.n_horizons,
            "calendar_cardinalities": list(self.cardinalities),
            "output_space": self.output_space,
        }


__all__ = ["MultiHorizonHead"]
