"""tokenizer 6→9 列 warm-start 手术（G4 计划 §1 冻结）。

手术面（model/kronos.py:56-57，只读引用不改）：仅 ``embed = nn.Linear(d_in,
d_model)`` 入口与 ``head = nn.Linear(d_model, d_in)`` 出口两处形状变化——

- ``embed.weight[:, :6]`` ← G1；``embed.weight[:, 6:]`` = **零初始化**；
  ``embed.bias`` ← G1 原样；
- ``head.weight[:6, :]`` ← G1；``head.weight[6:, :]`` = **零初始化**（重构目标
  同扩 9 列，计划 §5.2 声明的最小化代价）；``head.bias[:6]`` ← G1、
  ``head.bias[6:]`` = 0；
- 其余全部参数（encoder / decoder / quant_embed / post_quant_embed* /
  BSQuantizer）逐位复制——token 位宽（s1+s2 bits）不依赖 d_in，
  predictor 架构与词表完全不变。

零初始化保证：训练第 0 步的 G4 在 6 列输入上与 G1 逐位等价
（``tests/test_g4_features.py::TestWarmstartEquivalence`` 冻结门禁）。
"""
from __future__ import annotations

import copy

import torch

from model.kronos import KronosTokenizer

NEW_D_IN = 9
OLD_D_IN = 6


def expand_tokenizer_6to9(g1: KronosTokenizer) -> KronosTokenizer:
    """把 d_in=6 的 G1 tokenizer 扩成 d_in=9 的 G4 tokenizer（零初始化新列）。

    :param g1: 从 ``finetune_tokenizer_g1/checkpoints/best_model`` 装载的实例
        （只读——本函数不写回、不落地任何 G1 权重）。
    :returns: 新构造的 d_in=9 实例；embed/head 旧位逐位继承、新位为零，
        其余参数逐位复制。``g1`` 本身保持不动。
    """
    if g1.d_in != OLD_D_IN:
        raise ValueError(f"手术要求源 d_in=6，实际 {g1.d_in}")

    # from_pretrained 用的 config dict（含全部超参），仅改 d_in。
    # 本环境 hub_mixin 不在实例上留 .config → 从实例属性 + BSQ 超参完整重建
    # （与 checkpoint config.json 的 16 键一一对应，KronosTokenizer 构造签名同构）。
    bsq = g1.tokenizer.bsq
    cfg = {
        "d_in": g1.d_in, "d_model": g1.d_model, "n_heads": g1.n_heads,
        "ff_dim": g1.ff_dim, "n_enc_layers": g1.enc_layers,
        "n_dec_layers": g1.dec_layers, "ffn_dropout_p": g1.ffn_dropout_p,
        "attn_dropout_p": g1.attn_dropout_p,
        "resid_dropout_p": g1.resid_dropout_p,
        "s1_bits": g1.s1_bits, "s2_bits": g1.s2_bits,
        "beta": bsq.beta, "gamma0": bsq.gamma0,
        "gamma": bsq.gamma, "zeta": bsq.zeta,
        "group_size": bsq.group_size,
    }
    cfg = {k: copy.deepcopy(v) for k, v in cfg.items()}
    cfg["d_in"] = NEW_D_IN

    g4 = KronosTokenizer(**cfg)
    sd1 = g1.state_dict()
    sd4 = g4.state_dict()
    assert set(sd1) == set(sd4), "6→9 手术不应改变参数名集合"

    for k in sd1:
        t1, t4 = sd1[k], sd4[k]
        if k.startswith("embed."):
            if t1.dim() == 2:  # weight (d_model, d_in)
                t4[:, :OLD_D_IN] = t1
                t4[:, OLD_D_IN:] = 0.0
            else:  # bias (d_model,)
                t4.copy_(t1)
        elif k.startswith("head."):
            if t1.dim() == 2:  # weight (d_in, d_model)
                t4[:OLD_D_IN, :] = t1
                t4[OLD_D_IN:, :] = 0.0
            else:  # bias (d_in,)
                t4[:OLD_D_IN] = t1
                t4[OLD_D_IN:] = 0.0
        else:
            t4.copy_(t1)
    g4.load_state_dict(sd4, strict=True)
    return g4
