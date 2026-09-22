"""TAS1 两遍条件生成模型（计划 §3）。

**架构**（冻结官方 Kronos tokenizer + 主干 + DualHead，新增小型状态编码器）：

1. 第一遍（``eval + no_grad``）：历史 90 行 → tokenizer → 冻结主干 → ``H∈R^{90×d}``；
2. 段摘要 ``Z_j = concat(mean(H[seg_j]), H[89])`` → :class:`StateEncoder` →
   四个连续状态 token ``S`` + 可训练查询 token ``q``；
3. 第二遍（梯度可传回 φ）：按 PRE / POST 布局重排 ``[S]`` 与历史，仍以原
   粗细 token 生成头预测未来十日。

**关键约束**：

- 底座（Kronos 权重 ffn/resid dropout=0.2）恒 ``eval``——冻结即关闭底座
  dropout；可训练参数仅 :class:`StateEncoder`（公式 ``202d+64``）；
- 原 cross-attention 的 ``is_causal`` 与 ``self.training`` 联动
  （model/module.py:387），本 wrapper **显式**实现因果 s2 attention，
  不依赖模块 training 标志（计划 §3.4）；
- ``S``、``q`` 不加日历 embedding、不经行情 tokenizer；tokenizer.decode
  只接收真实历史与生成的粗细 token（计划 §3.3）；
- baseline 旁路（``layout=None``）：不拼 ``S``/``q``，嵌入序列与原版
  ``decode_s1`` 逐位一致，供 P0 对拍。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kronos import Kronos, KronosTokenizer, sample_from_logits
from tas_state.config import TASConfig

Layout = Literal["pre", "post", "baseline"]


class StateEncoder(nn.Module):
    """共享 MLP 状态编码器 + 槽位 embedding + 查询 token（计划 §3.2）。

    ``S_j = W2·GELU(W1·LayerNorm(Z_j)+b1)+b2+e_j``；参数量 ``202d+64``
    （LayerNorm 2×2d，W1 2d×64+64，W2 64×d+d，e 4d，q d）。全部 dropout=0。

    :param d_model: 底座隐维 d（从 checkpoint 读取，不可硬编码）。
    :param hidden: MLP 隐层宽度（计划固定 64）。
    :param n_states: 状态槽位数（= 段数 4）。
    """

    def __init__(self, d_model: int, hidden: int = 64, n_states: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2 * d_model)
        self.w1 = nn.Linear(2 * d_model, hidden, bias=True)
        self.w2 = nn.Linear(hidden, d_model, bias=True)
        self.slot_emb = nn.Parameter(torch.zeros(n_states, d_model))
        self.query = nn.Parameter(torch.zeros(d_model))
        # 小尺度初始化：初始 S≈e、q≈0，训练初期对冻结底座扰动最小
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w2.bias)
        nn.init.normal_(self.slot_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """``Z=[B, n_states, 2d]`` → ``S=[B, n_states, d]``。"""
        h = self.w1(F.gelu(self.norm(Z)))
        return self.w2(h) + self.slot_emb.unsqueeze(0)

    def trainable_parameter_count(self) -> int:
        """可训练参数实测（计划 §3.2：最终必须用 numel 实测）。"""
        return sum(p.numel() for p in self.parameters())


@dataclass(frozen=True)
class SecondPassOutput:
    """第二遍 teacher-forcing 输出（未来 10 步，计划 §3.4 损失输入）。

    ``s1_logits`` / ``s2_logits``：``[B, 10, vocab]``，第 k 行 = 第 k 日的
    预测位置（PRE/POST 下 q 位置起、随后第 k−1 个未来 token 位置）。
    """

    s1_logits: torch.Tensor
    s2_logits: torch.Tensor
    pred_positions: torch.Tensor  # 读出位置（相对第二遍序列）


class ConditionalKronos(nn.Module):
    """两遍条件生成 wrapper（冻结底座 + 可训练状态接口）。

    :param kronos: 冻结 Kronos 主干（官方权重）。
    :param tokenizer: 冻结 Kronos tokenizer。
    :param encoder: 可训练 :class:`StateEncoder`。
    :param cfg: 冻结实验配置。
    """

    def __init__(
        self,
        kronos: Kronos,
        tokenizer: KronosTokenizer,
        encoder: StateEncoder,
        cfg: TASConfig,
    ) -> None:
        super().__init__()
        self.kronos = kronos
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.cfg = cfg
        self.freeze_base()

    # —— 冻结与模式 ——
    def freeze_base(self) -> None:
        """底座与 tokenizer 全冻结（requires_grad=False + eval）。"""
        for p in self.kronos.parameters():
            p.requires_grad_(False)
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        self.kronos.eval()
        self.tokenizer.eval()

    def train(self, mode: bool = True) -> "ConditionalKronos":
        """训练模式仅作用于新增模块；底座恒 eval（dropout=0.2 不得打开）。

        这同时保证原 cross-attention 的 ``is_causal=self.training`` 联动被
        排除：因果性由本 wrapper 的显式实现控制（计划 §3.4）。
        """
        super().train(mode)
        self.kronos.eval()
        self.tokenizer.eval()
        return self

    # —— 第一遍：历史 → H → Z → S ——
    def first_pass(
        self, x_norm: torch.Tensor, stamp: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """第一遍编码（eval + no_grad，H detach；不含任何未来信息）。

        :param x_norm: ``[B, 90, 6]`` 归一化历史。
        :param stamp: ``[B, 90, 5]`` 历史时间特征。
        :returns: ``(H [B,90,d] detach, (s1_ids, s2_ids) 历史粗细 token)``。
        """
        with torch.no_grad():
            s1_ids, s2_ids = self.tokenizer.encode(x_norm, half=True)
            H = self._backbone(s1_ids, s2_ids, stamp)
        return H.detach(), (s1_ids.detach(), s2_ids.detach())

    def _backbone(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: torch.Tensor | None,
    ) -> torch.Tensor:
        """冻结主干（与原版 ``decode_s1`` 前半逐字一致的调用序）。"""
        x = self.kronos.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.kronos.time_emb(stamp)
        x = self.kronos.token_drop(x)  # 底座 eval 下恒等；保留调用序以对拍
        for layer in self.kronos.transformer:
            x = layer(x)
        return self.kronos.norm(x)

    def summarize(self, H: torch.Tensor) -> torch.Tensor:
        """段摘要 ``Z``（计划 §3.1：四段均值 + 最后位置拼接）。"""
        means = [H[:, lo:hi].mean(dim=1) for lo, hi in self.cfg.segments]
        last = H[:, H.shape[1] - 1]
        return torch.stack(
            [torch.cat([m, last], dim=-1) for m in means], dim=1
        )  # [B, 4, 2d]

    def encode_state(
        self, x_norm: torch.Tensor, stamp: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """第一遍 + 摘要 + 状态编码（便捷组合；S 固定于决策日，采样间复用）。"""
        H, hist_tokens = self.first_pass(x_norm, stamp)
        Z = self.summarize(H)
        S = self.encoder(Z)
        return S, hist_tokens

    # —— 第二遍：布局拼接与 teacher forcing ——
    def _embed_hist(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: torch.Tensor | None,
    ) -> torch.Tensor:
        """行情 token 嵌入（+ 日历 embedding），与原版调用序一致。"""
        x = self.kronos.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.kronos.time_emb(stamp)
        return self.kronos.token_drop(x)

    def _prefix_embed(
        self,
        prefix_s1: torch.Tensor,
        prefix_s2: torch.Tensor,
        prefix_stamp: torch.Tensor | None,
    ) -> torch.Tensor:
        """teacher-forced / 已生成未来 token 的嵌入（加日历 embedding）。"""
        return self._embed_hist(prefix_s1, prefix_s2, prefix_stamp)

    def second_pass_input(
        self,
        hist_tokens: tuple[torch.Tensor, torch.Tensor],
        hist_stamp: torch.Tensor,
        *,
        S: torch.Tensor | None,
        q: torch.Tensor | None,
        layout: Layout,
        prefix_s1: torch.Tensor | None = None,
        prefix_s2: torch.Tensor | None = None,
        prefix_stamp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """构造第二遍输入嵌入序列（计划 §3.3 布局）。

        ``baseline``：不拼 S/q，输出与原版主干输入逐位一致（P0 对拍旁路）；
        ``pre``：``[S][历史]``；``post``：``[历史][S]``；q 恒在历史之后、
        未来前缀之前（两布局中 q 位置相同 = 94）。S、q 不加日历 embedding。
        """
        hist = self._embed_hist(hist_tokens[0], hist_tokens[1], hist_stamp)
        if layout == "baseline":
            if S is not None or q is not None:
                raise ValueError("baseline 旁路不接受 S/q")
            parts = [hist]
        elif layout == "pre":
            parts = [S, hist]
        elif layout == "post":
            parts = [hist, S]
        else:
            raise ValueError(f"未知布局 {layout!r}")
        if q is not None:
            parts.append(q.unsqueeze(1))  # [B,1,d]
        if prefix_s1 is not None:
            parts.append(self._prefix_embed(prefix_s1, prefix_s2, prefix_stamp))
        return torch.cat(parts, dim=1)

    def query_pred_positions(self, n_future: int, device) -> torch.Tensor:
        """未来读出位置：q 位置（=4+90）起连续 n_future 个（两布局相同）。"""
        start = self.cfg.n_states + self.cfg.lookback
        return torch.arange(start, start + n_future, device=device)

    def teacher_forced_forward(
        self,
        x_norm: torch.Tensor,
        hist_stamp: torch.Tensor,
        y_s1: torch.Tensor,
        y_s2: torch.Tensor,
        y_stamp: torch.Tensor,
        *,
        S: torch.Tensor | None = None,
        layout: Layout = "pre",
    ) -> SecondPassOutput:
        """批量 teacher forcing（训练 / teacher-forced 验证共用）。

        输入序列 = ``[S][历史90][q][y_1..y_{H-1}]``（104 token；y_H 只作目标），
        第 k 日 logits 取第 k−1 个未来 token 位置。s2 使用显式因果
        cross-attention，sibling = 第 k 日真实粗 token（原层级分解）。

        :param y_s1: ``[B, H]`` 未来真实粗 token（标签）。
        :param y_s2: ``[B, H]`` 未来真实细 token（标签）。
        :param S: 预计算状态（None 时内部现算——仅诊断用途，训练循环
            应传入预计算 S 以保证 20 路采样复用语义）。
        """
        if S is None:
            S, hist_tokens = self.encode_state(x_norm, hist_stamp)
        else:
            _, hist_tokens = self.first_pass(x_norm, hist_stamp)
        B, H_len = y_s1.shape
        q = self.encoder.query.unsqueeze(0).expand(B, -1)
        seq = self.second_pass_input(
            hist_tokens,
            hist_stamp,
            S=S,
            q=q,
            layout=layout,
            prefix_s1=y_s1[:, :-1],
            prefix_s2=y_s2[:, :-1],
            prefix_stamp=y_stamp[:, :-1],
        )
        # 冻结主干（requires_grad=False 但不 no_grad：φ 梯度经 S / q 传回）
        h = seq
        for layer in self.kronos.transformer:
            h = layer(h)
        h = self.kronos.norm(h)
        s1_logits_all = self.kronos.head(h)

        pred_pos = self.query_pred_positions(H_len, seq.device)
        s1_logits = s1_logits_all[:, pred_pos]  # [B, H, vocab_s1]
        # 显式因果 s2：sibling = 同日真实粗 token；query 位置 = pred_pos
        s2_logits = self.causal_s2_logits(h, pred_pos, y_s1)
        return SecondPassOutput(s1_logits, s2_logits, pred_pos)

    # —— 显式因果 s2（原冻结投影权重，计划 §3.4）——
    def causal_s2_logits(
        self,
        context: torch.Tensor,
        pred_positions: torch.Tensor,
        s1_targets: torch.Tensor,
    ) -> torch.Tensor:
        """因果 s2 cross-attention（复用冻结 ``dep_layer`` 投影，显式掩码）。

        语义与原版 ``decode_s2`` 在 ``training=True`` 下逐位一致：位置 p 的
        s2 条件于 sibling（该位置预测的粗 token）与 ``context[≤p]``；与底座
        eval/ training 状态解耦。RoPE 按显式位置索引计算（子集 query 不得
        丢失原始位置）。
        """
        ca = self.kronos.dep_layer.cross_attn
        B, L, d = context.shape
        P = pred_positions.shape[0]
        n_h, hd = ca.n_heads, ca.head_dim

        sib = self.kronos.embedding.emb_s1(s1_targets)  # [B, P, d]
        q = ca.q_proj(sib).view(B, P, n_h, hd).transpose(1, 2)
        k = ca.k_proj(context).view(B, L, n_h, hd).transpose(1, 2)
        v = ca.v_proj(context).view(B, L, n_h, hd).transpose(1, 2)
        q = _apply_rope_at(ca.rotary, q, pred_positions, L)
        k = _apply_rope_at(ca.rotary, k, None, L)

        # 因果掩码：query i（位置 p_i）只看 key j ≤ p_i
        # （SDPA 布尔掩码语义：True=允许参与，False=屏蔽）
        ar = torch.arange(L, device=context.device)
        attn_mask = ar[None, :] <= pred_positions[:, None]  # [P, L]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, P, d)
        attn_out = ca.resid_dropout(ca.out_proj(out))  # 底座 eval 下恒等
        x2 = self.kronos.dep_layer.norm(
            context[:, pred_positions] + attn_out
        )
        return self.kronos.head.cond_forward(x2)

    # —— 自由生成（推理）——
    @torch.no_grad()
    def generate(
        self,
        x_norm: torch.Tensor,
        hist_stamp: torch.Tensor,
        y_stamp: torch.Tensor,
        *,
        S: torch.Tensor | None = None,
        layout: Layout = "pre",
        sample_count: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """N 路自由生成（S 一次计算后复用；不重算状态）。

        :param x_norm: ``[B, 90, 6]``；:param y_stamp: ``[B, 10, 5]``。
        :returns: ``(gen_s1, gen_s2)`` 各 ``[B, sample_count, H]``。
            baseline 布局下与原版 ``auto_regressive_inference`` 同构
            （同前缀递进、最后位置读出）。
        """
        cfg = self.cfg
        B = x_norm.shape[0]
        H_len = cfg.predict_len
        dev = x_norm.device

        if layout == "baseline":
            _, hist_tokens0 = self.first_pass(x_norm, hist_stamp)
            S_use = None  # baseline 旁路不构造状态
        elif S is None:
            S_use, hist_tokens0 = self.encode_state(x_norm, hist_stamp)
        else:
            S_use = S
            _, hist_tokens0 = self.first_pass(x_norm, hist_stamp)

        # N 路采样：历史与状态沿 sample 维复制（原版 auto_regressive 同构）
        rep = lambda t: t.unsqueeze(1).repeat(1, sample_count, *([1] * (t.dim() - 1))).reshape(B * sample_count, *t.shape[1:])
        s1_hist, s2_hist = rep(hist_tokens0[0]), rep(hist_tokens0[1])
        stamp_rep = rep(hist_stamp)
        S_rep = rep(S_use) if S_use is not None else None
        q = self.encoder.query.unsqueeze(0).expand(B * sample_count, -1)

        gen_s1 = torch.empty(B * sample_count, H_len, dtype=torch.long, device=dev)
        gen_s2 = torch.empty(B * sample_count, H_len, dtype=torch.long, device=dev)
        ys_rep = rep(y_stamp)  # [B*N, 10, 5]，循环外一次
        pass_q = None if layout == "baseline" else q
        for k in range(H_len):
            prefix_s1 = gen_s1[:, :k] if k > 0 else None
            prefix_s2 = gen_s2[:, :k] if k > 0 else None
            prefix_st = ys_rep[:, :k] if k > 0 else None
            seq = self.second_pass_input(
                (s1_hist, s2_hist),
                stamp_rep,
                S=S_rep,
                q=pass_q,
                layout=layout,
                prefix_s1=prefix_s1,
                prefix_s2=prefix_s2,
                prefix_stamp=prefix_st,
            )
            h = seq
            for layer in self.kronos.transformer:
                h = layer(h)
            h = self.kronos.norm(h)
            s1_logits = self.kronos.head(h)[:, -1, :]  # 最后位置（k=0 → q）
            sample_s1 = sample_from_logits(
                s1_logits, temperature=temperature, top_k=top_k, top_p=top_p,
                sample_logits=True,
            )
            # s2：直接复用原版 decode_s2（eval 下 is_causal=False、q_len=1，
            # RoPE cache 取 q_len——query 与全部 key 均按位置 0 旋转）。
            # 官方训练（方形 q=k=L）与生成（q_len=1）的这一 RoPE 不一致是
            # 原设计的一部分，本 wrapper 原封保留，保证与 B0 生成路径同数值。
            s2_logits = self.kronos.decode_s2(h, sample_s1)[:, -1, :]
            sample_s2 = sample_from_logits(
                s2_logits, temperature=temperature, top_k=top_k, top_p=top_p,
                sample_logits=True,
            )
            gen_s1[:, k] = sample_s1.squeeze(-1)
            gen_s2[:, k] = sample_s2.squeeze(-1)

        return (
            gen_s1.view(B, sample_count, H_len),
            gen_s2.view(B, sample_count, H_len),
        )

    # —— tokenizer 解码（S/q 永不进入）——
    def decode_tokens(
        self,
        hist_tokens: tuple[torch.Tensor, torch.Tensor],
        gen_s1: torch.Tensor,
        gen_s2: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
    ) -> torch.Tensor:
        """粗细 token → 预测行情 ``[B, N, H, 6]``（真实量纲）。

        只拼真实历史 token 与生成 token（spy 测试锁住本契约）；S、q 是
        连续向量，从不进入 ``tokenizer.decode``。
        """
        B, N, H_len = gen_s1.shape
        s1 = torch.cat(
            [hist_tokens[0].unsqueeze(1).expand(B, N, -1), gen_s1], dim=2
        ).reshape(B * N, -1)
        s2 = torch.cat(
            [hist_tokens[1].unsqueeze(1).expand(B, N, -1), gen_s2], dim=2
        ).reshape(B * N, -1)
        z = self.tokenizer.decode([s1, s2], half=True)  # [B*N, 90+H, 6]
        z = z[:, -H_len:, :].reshape(B, N, H_len, -1)
        # 反归一化（与 KronosPredictor 同式：×(std+1e-5)+mean）
        return z * (stds[:, None, None, :] + 1e-5) + means[:, None, None, :]


def _apply_rope_at(
    rotary,
    x: torch.Tensor,
    positions: torch.Tensor | None,
    max_len: int,
) -> torch.Tensor:
    """按显式位置索引施加 RoPE（与 ``RotaryPositionalEmbedding`` 同式）。

    :param x: ``[B, H, P, hd]``。
    :param positions: ``[P]`` 位置索引；None 时 = ``arange(max_len)``。
    """
    if positions is None:
        t = torch.arange(max_len, device=x.device).type_as(rotary.inv_freq)
    else:
        t = positions.type_as(rotary.inv_freq)
    freqs = torch.einsum("i,j->ij", t, rotary.inv_freq)  # [P, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)[None, None]  # [1,1,P,hd]
    x1, x2 = x.chunk(2, dim=-1)
    rotate_half = torch.cat((-x2, x1), dim=-1)
    return (x * emb.cos()) + (rotate_half * emb.sin())
