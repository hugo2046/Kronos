"""cross_section_kda 测试（计划 §4.5）。

覆盖：
    1. KDA 组件形状单测；
    2. 因果性注入式验证——篡改 t 之后输入，t 时刻输出必须不变；
       并构造「去掉卷积 padding 裁剪」的破损变体，断言其会泄漏（负向断言，
       证明因果测试有能力捕获泄漏而非恒真）；
    3. 隐状态提取与 feature/direction-classifier 的 KronosProbeClassifier
       主干前半段语义一致（对拍，golden 逻辑内联进测试，不 import 那个分支）；
    4. 切分 purge 间隔断言（fake 日历，不连 DolphinDB）。

需要真实 DolphinDB / Kronos 权重的对拍测试标 ``integration``——
无权重时自动跳过（保持单测可在离线环境跑）。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# ---------- 公共 fixture ----------

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture(scope="module")
def small_kda_block() -> "KimiLinearBlock":
    """小尺寸 KDA block，单测用（真尺寸 d=832 留给 integration）。"""
    from cross_section_kda import KimiLinearBlock

    # d=32, nhead=4 保证可整除
    return KimiLinearBlock(
        d_model=32,
        nhead=4,
        ffn_dim=64,
        dropout=0.0,
        conv_kernel=4,
        gate_rank=8,
    )


# ============================================================
# 1. 形状单测
# ============================================================


class TestShapes:
    def test_causal_conv_preserves_length(self) -> None:
        """CausalDepthwiseConv1d 输出时间长度 == 输入时间长度。"""
        from cross_section_kda import CausalDepthwiseConv1d

        set_seed()
        conv = CausalDepthwiseConv1d(dim=16, kernel_size=4)
        x = torch.randn(4, 90, 16)
        y = conv(x)
        assert y.shape == x.shape, f"长度应保持不变，got {tuple(y.shape)}"

    def test_kda_block_preserves_shape(self, small_kda_block) -> None:
        """KimiLinearBlock [B, T, d] -> [B, T, d]。"""
        set_seed()
        x = torch.randn(4, 90, 32)
        y = small_kda_block(x)
        assert y.shape == x.shape, f"形状应不变，got {tuple(y.shape)}"


# ============================================================
# 2. 因果性注入式验证（§4.5 必做）
# ============================================================


def _perturb_future(x: torch.Tensor, t: int) -> torch.Tensor:
    """复制 x 并对第 t+1 步之后的行施加扰动（模拟篡改未来）。"""
    x2 = x.clone()
    # 加一个明显大于数值噪声的扰动，确保非恒等
    x2[:, t + 1 :, :] += torch.randn_like(x2[:, t + 1 :, :]) * 10.0
    return x2


class TestCausality:
    def test_causal_conv_is_causal(self) -> None:
        """CausalDepthwiseConv1d：篡改 t 之后输入，<=t 输出逐位不变。"""
        from cross_section_kda import CausalDepthwiseConv1d

        set_seed()
        conv = CausalDepthwiseConv1d(dim=16, kernel_size=4).eval()
        x = torch.randn(3, 90, 16)
        t = 20
        x2 = _perturb_future(x, t)
        with torch.no_grad():
            y, y2 = conv(x), conv(x2)
        # <=t 各步输出必须逐位一致
        torch.testing.assert_close(
            y[:, : t + 1], y2[:, : t + 1], rtol=0, atol=0
        )
        # 被篡改的未来步本身应有变化（确认扰动确实作用到了输入）
        assert not torch.allclose(y[:, t + 1 :], y2[:, t + 1 :])

    def test_causal_conv_broken_variant_leaks(self) -> None:
        """注入式负向断言：右对齐裁剪的反因果卷积会泄漏，证明因果测试非恒真。

        真实的 :class:`CausalDepthwiseConv1d` 用 ``padding=kernel-1``（对称填充，
        输出长 ``T+kernel-1``）后取**左对齐** ``[..., :T]``：输出位置 t 仅依赖
        输入 ``[t-kernel+1 .. t]``，全为过去 → 因果。

        破损变体故意改成**右对齐** ``[..., -T:]``：输出位置 t 平移到全长的
        后段，依赖输入 ``[t .. t+kernel-1]``，含未来 → 反因果。对同一篡改用例，
        破损版的 ``<=t`` 输出**应当**变化；若不变，说明因果测试恒真、无检测力。
        """
        set_seed()
        leak_dim, kernel = 16, 4
        leak_conv = torch.nn.Conv1d(
            leak_dim, leak_dim, kernel_size=kernel, padding=kernel - 1, groups=leak_dim, bias=True
        ).eval()
        x = torch.randn(3, 90, leak_dim)
        t = 20
        x2 = _perturb_future(x, t)
        with torch.no_grad():
            full = leak_conv(x.transpose(1, 2))   # [B, dim, T + kernel - 1]
            full2 = leak_conv(x2.transpose(1, 2))
            # 破损：右对齐裁剪 → 反因果（输出 t 依赖 [t .. t+kernel-1]）
            y = full[..., -x.size(1):].transpose(1, 2)
            y2 = full2[..., -x.size(1):].transpose(1, 2)
        # 破损版：篡改未来后 <=t 的输出应当变化（证明因果约束在本测试里可被违反）
        diff = (y[:, : t + 1] - y2[:, : t + 1]).abs().max()
        assert diff > 1e-4, (
            "破损（右对齐裁剪）卷积本应泄漏未来信息（<=t 输出变化），但未检测到"
            "变化——因果测试可能恒真，请检查测试设计"
        )

    def test_kda_block_is_causal(self, small_kda_block) -> None:
        """KimiLinearBlock（含 KDA 递归 + 因果卷积）：篡改未来，<=t 输出不变。"""
        set_seed()
        block = small_kda_block.eval()
        x = torch.randn(3, 90, 32)
        t = 20
        x2 = _perturb_future(x, t)
        with torch.no_grad():
            y, y2 = block(x), block(x2)
        torch.testing.assert_close(
            y[:, : t + 1], y2[:, : t + 1], rtol=1e-5, atol=1e-5
        )

    def test_kda_attention_is_causal(self) -> None:
        """单独测 KimiDeltaAttention 递归的因果性。"""
        from cross_section_kda import KimiDeltaAttention

        set_seed()
        attn = KimiDeltaAttention(d_model=32, nhead=4, conv_kernel=4, gate_rank=8).eval()
        x = torch.randn(3, 90, 32)
        t = 20
        x2 = _perturb_future(x, t)
        with torch.no_grad():
            y, y2 = attn(x), attn(x2)
        torch.testing.assert_close(
            y[:, : t + 1], y2[:, : t + 1], rtol=1e-5, atol=1e-5
        )


# ============================================================
# 3. 隐状态提取对拍（integration：需 Kronos 权重）
# ============================================================


def _golden_backbone_last_hidden(
    tokenizer: torch.nn.Module,
    kronos: torch.nn.Module,
    features: torch.Tensor,
    stamp: torch.Tensor,
) -> torch.Tensor:
    """Golden 参考：逐字复刻 feature/direction-classifier 的 KronosProbeClassifier
    主干前半段，取末步隐状态。

    与 KronosFrozenBackbone.extract 对拍用。不 import 那个分支，逻辑内联在此。
    """
    with torch.no_grad():
        tokenizer.eval()
        kronos.eval()
        s1_ids, s2_ids = tokenizer.encode(features, half=True)
        x = kronos.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + kronos.time_emb(stamp)
        x = kronos.token_drop(x)
        for layer in kronos.transformer:
            x = layer(x)
        x = kronos.norm(x)
        return x[:, -1]


@pytest.mark.integration
class TestBackboneAlignment:
    """需要 Kronos 权重 +（可选）GPU。无权重时自动跳过。"""

    def test_extract_last_hidden_matches_probe(self) -> None:
        from model import Kronos, KronosTokenizer

        from cross_section_kda.backbone import KronosFrozenBackbone

        try:
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            kronos = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        except Exception as e:  # 权重不可得（离线 / 无 HF token）→ 跳过，不当失败
            pytest.skip(f"无法加载 Kronos 权重，跳过对拍：{type(e).__name__}: {e}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = tokenizer.to(device)
        kronos = kronos.to(device)

        bb = KronosFrozenBackbone(tokenizer=tokenizer, kronos=kronos, device=device)

        set_seed()
        B, T = 2, 90
        features = torch.randn(B, T, 6, device=device)
        stamp = torch.zeros(B, T, 5, device=device)

        got_full = bb.extract(features, stamp)  # [B, T, 832]
        got_last = got_full[:, -1]
        want_last = _golden_backbone_last_hidden(tokenizer, kronos, features, stamp)

        torch.testing.assert_close(got_last, want_last, rtol=1e-5, atol=1e-5)


# ============================================================
# 4. 切分 purge 间隔断言（fake 日历，不连 DolphinDB）
# ============================================================


def _fake_calendar(start: str, end: str, *, step: int = 1) -> pd.DatetimeIndex:
    """构造 fake 交易日历：工作日按 step 抽样近似（仅用于 purge 间隔断言）。"""
    days = pd.bdate_range(start, end)
    return pd.DatetimeIndex(days[::step])


class TestSplits:
    def test_split_purge_intervals(self) -> None:
        """train/early-stop/final 边界两两间隔 >= 10 个交易日（含 purge）。"""
        from cross_section_kda.data import Splits, assert_purge_intervals

        cal = _fake_calendar("2021-06-01", "2026-08-31")
        splits = Splits(
            calendar=cal,
            train=("2022-01-04", "2023-12-15"),
            early_stop=("2024-01-02", "2024-06-14"),
            final_start="2024-07-01",
            purge=10,
        )
        # 正常切分应通过
        assert_purge_intervals(splits)

    def test_split_detects_too_close(self) -> None:
        """train 与 early-stop 间隔不足 10 个交易日时必须报错。"""
        from cross_section_kda.data import Splits, assert_purge_intervals

        cal = _fake_calendar("2021-06-01", "2026-08-31")
        # early_stop 起点 2023-12-18 紧挨 train 末 2023-12-15，不足 purge
        bad = Splits(
            calendar=cal,
            train=("2022-01-04", "2023-12-15"),
            early_stop=("2023-12-18", "2024-06-14"),
            final_start="2024-07-01",
            purge=10,
        )
        with pytest.raises(AssertionError):
            assert_purge_intervals(bad)

    def test_final_grid_matches_b0(self) -> None:
        """final 段调仓日网格必须与 B0 的 signals_with_baselines.parquet
        在 2024-07-01 之后的 50 期完全一致（保证五臂同日期网格）。"""
        from cross_section_kda.data import build_final_grid

        b0_path = REPO_ROOT / "cross_section" / "data" / "signals_with_baselines.parquet"
        if not b0_path.exists():
            pytest.skip("B0 signals parquet 不存在，跳过 final 网格一致性测试")
        b0 = pd.read_parquet(b0_path)
        b0_final = sorted(
            pd.Timestamp(d) for d in b0["date"].unique()
            if pd.Timestamp(d) >= pd.Timestamp("2024-07-01")
        )
        got = list(build_final_grid(b0_path, final_start="2024-07-01"))
        assert got == b0_final, "final 网格与 B0 的 50 期不一致"
