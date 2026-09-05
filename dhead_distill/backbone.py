"""学生底座：冻结提取与（任务4 扩展的）末两层 LoRA 路径（方案 §5 关键实现）。

与 ``cross_section_kda.backbone.KronosFrozenBackbone`` 的差异（§5）：

- 本类是**学生**底座，需要为 D2 预留「前 10 层 no_grad + 末 2 层可训」的
  分段提取（``extract_trainable``，任务4 接 LoRA 后启用）——
  ``KronosFrozenBackbone.extract`` 整段 ``@torch.no_grad`` 且构造即冻结，
  不能直接作为 LoRA 学生；
- 冻结臂（D0/S/D1）用 :meth:`StudentBackbone.extract`：整段 no_grad，
  语义与 G1 ``decode_s1`` 主干前半段逐字一致（embedding + time_emb +
  token_drop + transformer + RMSNorm），返回逐步隐状态 ``[B,T,d_model]``。

教师与学生必须是不同实例，不共享可变模块（§5）——本类不做单例缓存，
调用方各自装载。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StudentBackbone(nn.Module):
    """G1 tokenizer + 主干的学生侧封装（冻结为主，末两层可按需开启梯度）。

    :param tokenizer: G1（或离线测试 stub）tokenizer，冻结。
    :param kronos: G1（或 stub）主干，冻结；D2 时仅末 ``n_trainable_layers``
        层在 :meth:`extract_trainable` 中参与梯度计算。
    :param d_model_expected: 配置要求的 d_model（真实 G1 = 832），构造时校验。
    :param n_trainable_layers: 末尾可训层数（D2 = 2；冻结臂不使用）。
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        kronos: nn.Module,
        *,
        d_model_expected: int = 832,
        n_trainable_layers: int = 2,
    ) -> None:
        super().__init__()
        d_model = getattr(kronos, "d_model", None)
        if d_model is None:
            raise ValueError("kronos 缺少 d_model 属性，不是 Kronos 主干（或 stub）")
        if d_model != d_model_expected:
            raise ValueError(
                f"G1 d_model 校验失败：主干 {d_model} ≠ 配置 {d_model_expected}（§4.1）"
            )
        n_layers = len(kronos.transformer)
        if not 0 < n_trainable_layers < n_layers:
            raise ValueError(
                f"n_trainable_layers={n_trainable_layers} 须在 (0,{n_layers}) 内"
            )
        self.tokenizer = tokenizer
        self.kronos = kronos
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_trainable_layers = n_trainable_layers
        self._freeze_all()

    def _freeze_all(self) -> None:
        """冻结全部参数（双保险：requires_grad + eval，参考 KronosFrozenBackbone）。"""
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        for p in self.kronos.parameters():
            p.requires_grad_(False)
        self.tokenizer.eval()
        self.kronos.eval()

    def train(self, mode: bool = True) -> "StudentBackbone":
        """重写 train：底座恒 eval，避免 token_drop/dropout 成为额外变量（§5）。"""
        super().train(mode)
        self.tokenizer.eval()
        self.kronos.eval()
        return self

    def _embed(self, features: torch.Tensor, stamp: Optional[torch.Tensor]) -> torch.Tensor:
        """token 化 + 嵌入 + 时间嵌入（复刻 Kronos.decode_s1 前半段）。"""
        s1_ids, s2_ids = self.tokenizer.encode(features, half=True)
        x = self.kronos.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.kronos.time_emb(stamp)
        x = self.kronos.token_drop(x)
        return x

    @torch.no_grad()
    def extract(
        self, features: torch.Tensor, stamp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """冻结提取逐步隐状态（D0/S/D1 与缓存路径）。

        :param features: 已窗口 z-score + clip 的输入 ``[B,T,6]``。
        :param stamp: ``[B,T,5]`` 时间特征；None 时不加时间嵌入。
        :returns: 末层 RMSNorm 后隐状态 ``[B,T,d_model]``。
        """
        x = self._embed(features, stamp)
        for layer in self.kronos.transformer:
            x = layer(x)
        return self.kronos.norm(x)

    def extract_trainable(
        self, features: torch.Tensor, stamp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """D2 可训路径：前缀 no_grad，末 ``n_trainable_layers`` 层参与梯度。

        实现 §5 关键实现的分段语义——tokenizer 与前 N-2 层在 no_grad 下计算
        （脱离计算图、显存只读），末 2 层在启用梯度的环境运行（LoRA 可训）；
        **不要把整段包进 no_grad**。末两层保持 eval（dropout 关闭）不妨碍梯度。
        """
        with torch.no_grad():
            x = self._embed(features, stamp)
            for layer in self.kronos.transformer[: self.n_layers - self.n_trainable_layers]:
                x = layer(x)
            x = x.detach()  # 截断前缀图：即使外部在 grad 环境也不回传冻结段
        for layer in self.kronos.transformer[self.n_layers - self.n_trainable_layers:]:
            x = layer(x)
        return self.kronos.norm(x)

    def trainable_parameters(self):
        """可训参数迭代器（冻结臂为空；D2 注入 LoRA 后返回 LoRA 参数）。"""
        return (p for p in self.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    """nn.Linear 的低秩适配包装：``base(x) + scale * B(A(dropout(x)))``。

    :param base: 被包装的原始 Linear（权重冻结、逐字节不变）。
    :param rank: LoRA 秩（D2 = 8）。
    :param alpha: 缩放系数（D2 = 8，即 scale = alpha/rank = 1）。
    :param dropout: LoRA dropout（D2 = 0）。

    A Kaiming 初始化、B 零初始化——零初始化保证注入当刻前向与原模型
    逐位一致（§5 对拍前提）。
    """

    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_normal_(self.lora_A.weight, nonlinearity="linear")
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale


def inject_lora(
    backbone: "StudentBackbone",
    *,
    rank: int = 8,
    alpha: int = 8,
    dropout: float = 0.0,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
) -> list[tuple[str, nn.Parameter]]:
    """把末 ``n_trainable_layers`` 层 self_attn 的 q/v 投影包装为 LoRALinear。

    :returns: [(参数名, Parameter)] 全部 LoRA 可训参数（A/B），注册名含
        ``lora``——供优化器与梯度隔离断言使用。
    """
    named: list[tuple[str, nn.Parameter]] = []
    start = backbone.n_layers - backbone.n_trainable_layers
    for li in range(start, backbone.n_layers):
        block = backbone.kronos.transformer[li]
        attn = block.self_attn
        for tname in targets:
            base = getattr(attn, tname)
            if isinstance(base, LoRALinear):
                continue
            wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
            setattr(attn, tname, wrapped)
            for pname, p in wrapped.named_parameters():
                if pname.startswith("lora_"):
                    p.requires_grad_(True)
                    named.append((f"transformer.{li}.self_attn.{tname}.{pname}", p))
    # 末两层整体保持 eval（dropout 关闭），但 LoRA 参数可训
    backbone.kronos.eval()
    return named


def load_g1_student(env_paths, device: str) -> "StudentBackbone":
    """从基仓只读装载 G1 权重为学生底座（真实运行用；离线测试用 stub）。

    权重路径来自 :class:`dhead_distill.config.EnvPaths`（DHEAD_BASE_REPO 映射），
    绝不写回 G1 目录（§10.2）。
    """
    from model import Kronos, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(env_paths.g1_tokenizer)).to(device)
    kronos = Kronos.from_pretrained(str(env_paths.g1_predictor)).to(device)
    return StudentBackbone(tokenizer, kronos, d_model_expected=832)


__all__ = ["StudentBackbone", "load_g1_student", "LoRALinear", "inject_lora"]
