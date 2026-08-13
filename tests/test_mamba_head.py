"""improve_suite Mamba 时序头单测（计划 §3 阶段 1）。

五条覆盖（纯张量 / _StubBackbone，不触发 DDB / GPU / HF 权重）：

    1. ``test_mamba_head_contract``：随机 [B=4,90,6] + stamp → 输出 shape (4,)；
    2. ``test_mamba_head_params``：``count_trainable(head) ∈ [0.8e6, 1.3e6]``（不含冻结主干）；
    3. ``test_backbone_no_grad``：backward 后主干参数 grad 全 None、head 参数 grad 非 None；
    4. ``test_seed_determinism``：同 seed 两次构造+前向逐位一致；
    5. ``test_protocol_constants_match``：run_mamba_head 的 lr/wd/batch/epochs/patience/窗口/purge
       与 ``cross_section_kda`` 源常量逐字相等（防协议漂移）。

先于实现写成：模块未建时 import 失败 → FAIL；实现后 → PASS。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _StubBackbone(nn.Module):
    """桩主干：固定投影 [B,T,6]→[B,T,d_model]，params requires_grad=False，
    ``extract`` 在 ``no_grad`` 下——逐字复刻 :class:`KronosFrozenBackbone` 的不变式
    （冻结 + eval + no_grad），但不触发 qlib / HF 权重，纯张量测试用。

    固定投影用**局部 generator**（不沾全局 RNG），保证 ``test_seed_determinism``
    里头构造的随机性完全由 ``set_seed`` 决定。
    """

    def __init__(self, d_model: int = 832, d_in: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        g = torch.Generator().manual_seed(12345)
        self._proj = nn.Parameter(torch.randn(d_in, d_model, generator=g), requires_grad=False)

    def train(self, mode: bool = True) -> "_StubBackbone":
        """主干恒 eval（与 KronosFrozenBackbone.train 同理由）。"""
        super().train(mode)
        return self

    @torch.no_grad()
    def extract(self, features: torch.Tensor, stamp: torch.Tensor | None = None) -> torch.Tensor:
        """[B,T,d_in] @ [d_in,d_model] → [B,T,d_model]（no_grad，与真主干一致）。"""
        return features @ self._proj


# ============================================================
# §3 1.2 test_mamba_head_contract / params / no_grad / determinism
# ============================================================


class TestMambaHeadContract:
    """头契约：形状、参数量、骨干冻结、种子确定性（计划 §3 步骤 1.2）。"""

    def test_mamba_head_contract(self) -> None:
        """随机 [B=4,90,6] + stamp → 输出 shape (4,)。"""
        from improve_suite.mamba_head import MambaTemporalHead

        head = MambaTemporalHead(_StubBackbone())
        x = torch.randn(4, 90, 6)
        stamp = torch.randn(4, 90, 5)
        out = head(x, stamp)
        assert out.shape == (4,), f"输出形状错：{out.shape}，期望 (4,)"

    def test_mamba_head_params(self) -> None:
        """可训练参数 ∈ [0.8e6, 1.3e6]（B3 ~1M 的 ±30%，容量对齐断言）。"""
        from cross_section_kda.models import count_trainable

        from improve_suite.mamba_head import MambaTemporalHead

        head = MambaTemporalHead(_StubBackbone())
        n = count_trainable(head)
        assert 0.8e6 <= n <= 1.3e6, f"可训练参数 {n:,} 不在 [0.8M, 1.3M]"
        # 冻结主干不计入可训练
        assert all(not p.requires_grad for p in head.backbone.parameters()), "主干参数必须冻结"

    def test_backbone_no_grad(self) -> None:
        """backward 后主干参数 grad 全 None、head 参数 grad 非 None。"""
        from improve_suite.mamba_head import MambaTemporalHead

        backbone = _StubBackbone()
        head = MambaTemporalHead(backbone).train()
        x = torch.randn(4, 90, 6)
        stamp = torch.randn(4, 90, 5)
        score = head(x, stamp)
        score.sum().backward()
        for p in backbone.parameters():
            assert p.grad is None, "冻结主干参数不应积累梯度"
        head_grads = [p.grad for p in head.head.parameters()]
        assert all(g is not None for g in head_grads), "head 线性层应有梯度"

    def test_seed_determinism(self) -> None:
        """同 seed 两次构造+前向逐位一致。"""
        from cross_section_kda.train import set_seed

        from improve_suite.mamba_head import MambaTemporalHead

        x = torch.randn(4, 90, 6)
        stamp = torch.randn(4, 90, 5)
        backbone = _StubBackbone()

        set_seed(42)
        head1 = MambaTemporalHead(backbone)
        out1 = head1(x, stamp)

        set_seed(42)
        head2 = MambaTemporalHead(backbone)
        out2 = head2(x, stamp)

        assert torch.allclose(out1, out2), (
            f"同 seed 两次构造+前向不一致：max|Δ|={(out1 - out2).abs().max():.3e}"
        )


# ============================================================
# §3 1.3 test_protocol_constants_match
# ============================================================


class TestProtocolConstants:
    """训练协议常量与 cross_section_kda 源逐字相等（防协议漂移，计划 §3 步骤 1.3）。"""

    def test_protocol_constants_match(self) -> None:
        from cross_section_kda import data as D
        from cross_section_kda import train as T

        from improve_suite import run_mamba_head as R

        # 优化器 / 训练超参
        assert R.T1_LR == T.LR["B3"], f"lr 漂移：{R.T1_LR} vs {T.LR['B3']}"
        assert R.T1_WD == T.WEIGHT_DECAY, f"wd 漂移：{R.T1_WD} vs {T.WEIGHT_DECAY}"
        assert R.T1_BATCH == T.BATCH_SIZE, f"batch 漂移：{R.T1_BATCH} vs {T.BATCH_SIZE}"
        assert R.T1_EPOCHS == T.EPOCHS, f"epochs 漂移：{R.T1_EPOCHS} vs {T.EPOCHS}"
        assert R.T1_PATIENCE == T.PATIENCE, f"patience 漂移：{R.T1_PATIENCE} vs {T.PATIENCE}"
        # 切分 / purge
        assert R.T1_TRAIN_START == D.TRAIN_START
        assert R.T1_TRAIN_END == D.TRAIN_END
        assert R.T1_ES_START == D.EARLY_STOP_START
        assert R.T1_ES_END == D.EARLY_STOP_END
        assert R.T1_PURGE == D.PURGE
