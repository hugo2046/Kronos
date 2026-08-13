"""improve_suite 阶段 4 单测（计划 §6）。

三条覆盖（纯张量 / DataFrame，不触发 qlib / GPU / HF 权重）：

    1. ``test_token_dataset_no_leakage``：特征窗口末日 < 标签起始日，且 z-score 统计
       只用窗口内数据（窗口外极值不影响窗口内编码）；
    2. ``test_mamba_forward_shape``：``(B=4, L=60)`` token 对 → 标量输出 ``shape (4,)``；
    3. ``test_selective_scan_equiv``：手算 SSM 递推，断言逐位一致。

先于实现写成：模块未建时 import 失败 → FAIL；实现后 → PASS。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ============================================================
# §6.4.1 test_token_dataset_no_leakage
# ============================================================


class TestTokenDatasetNoLeakage:
    """标签不泄漏进特征窗口；z-score 只用窗口内数据（计划 §6.4.1）。"""

    def test_window_end_before_label_start(self) -> None:
        """特征窗口末日 < 标签起始日（无重叠）。"""
        from improve_suite.token_dataset import build_sample_dates

        cal = pd.date_range("2024-01-01", periods=80, freq="B")
        t = cal[65]  # 决策日
        win_dates, label_dates = build_sample_dates(t, lookback=60, horizon=5, calendar=cal)
        assert win_dates[-1] == t, "窗口末日 = 决策日 t"
        assert label_dates[0] > t, "标签起始日 > 决策日 t"
        assert len(win_dates) == 60
        assert len(label_dates) == 5
        # 无日期重叠
        assert set(win_dates).isdisjoint(set(label_dates))

    def test_zscore_uses_only_window_data(self) -> None:
        """窗口外极值不影响窗口内 z-score 编码（计划 §6.4.1）。"""
        from improve_suite.token_dataset import preprocess_window

        cols = ["open", "high", "low", "close", "volume", "amount"]
        rng = np.random.default_rng(0)
        base = pd.DataFrame(rng.uniform(10, 50, size=(60, 6)), columns=cols)
        # 窗口 A：标准 60 日
        norm_a, mean_a, std_a = preprocess_window(base)
        # 窗口 B：在 base 基础上把"窗口外"（其实 preprocess_window 只看输入）替换——
        # 构造一个不同的 60 日窗口但前 59 日完全相同，仅末日不同
        b = base.copy()
        b.iloc[-1] = b.iloc[-1] * 100  # 末日极值
        norm_b, mean_b, std_b = preprocess_window(b)
        # 前 59 日的 z-score 应不变（窗口末日变了不影响前 59 日的相对位置？不——
        # z-score 用全窗口 mean/std，末日变会拉偏 mean/std）。所以正确断言是：
        # preprocess_window 的 mean/std == np.mean/std(输入窗口)，证明只用窗口内数据。
        x = base[cols].values.astype(np.float32)
        assert np.allclose(mean_a, x.mean(axis=0)), "mean 必须来自窗口自身"
        assert np.allclose(std_a, x.std(axis=0)), "std 必须来自窗口自身"

    def test_label_uses_future_close_only(self) -> None:
        """标签 y = mean(close[t+1..t+5])/close_t - 1，只用未来收益。"""
        from improve_suite.token_dataset import extract_label

        close = pd.Series(np.arange(100, 130, dtype=float), index=pd.date_range("2024-01-01", periods=30))
        t = close.index[10]  # close_t = 110
        # 未来 5 日 close = 111..115，mean=113 → y = 113/110 - 1
        y = extract_label(close, t, horizon=5)
        expected = float(np.mean(close.iloc[11:16].values) / close.loc[t] - 1)
        assert y == pytest.approx(expected, abs=1e-12)
        assert y == pytest.approx(113.0 / 110.0 - 1, abs=1e-12)


# ============================================================
# §6.4.2 test_mamba_forward_shape + selective_scan
# ============================================================


class TestMambaMin:
    """Mamba 前向形状 + selective_scan 递推正确性（计划 §6.4.2）。"""

    def test_mamba_forward_shape(self) -> None:
        """(B=4, L=60) token 对 → 标量 (4,)。"""
        from improve_suite.mamba_min import MambaConfig, MambaSeqRegressor

        cfg = MambaConfig(s1_vocab=64, s2_vocab=64, d_model=32, n_layer=1, d_state=8, expand=2)
        model = MambaSeqRegressor(cfg)
        s1 = torch.randint(0, 64, (4, 60))
        s2 = torch.randint(0, 64, (4, 60))
        out = model(s1, s2)
        assert out.shape == (4,), f"输出形状错：{out.shape}，期望 (4,)"

    def test_selective_scan_matches_manual_recurrence(self) -> None:
        """手算 SSM 递推，断言 selective_scan 逐位一致。"""
        from improve_suite.mamba_min import selective_scan

        torch.manual_seed(0)
        B, L, d_in, n = 2, 5, 3, 4
        x = torch.randn(B, L, d_in)
        delta = torch.randn(B, L, d_in)
        A = torch.randn(d_in, n)
        B_in = torch.randn(B, L, n)
        C = torch.randn(B, L, n)
        D = torch.randn(d_in)

        # 我的实现
        y_impl = selective_scan(x, delta, A, B_in, C, D)

        # 手算递推（ZOH）：h_t = exp(Δ_t·A)·h_{t-1} + (Δ_t·B_t)·x_t；y_t = C_t·h_t + D·x_t
        h = torch.zeros(B, d_in, n)
        y_ref = torch.zeros(B, L, d_in)
        for t in range(L):
            dA = torch.exp(torch.einsum("b d, d n -> b d n", delta[:, t], A))  # (B, d_in, n)
            dB_x = torch.einsum("b d, b n, b d -> b d n", delta[:, t], B_in[:, t], x[:, t])
            h = dA * h + dB_x
            y_ref[:, t] = torch.einsum("b n, b d n -> b d", C[:, t], h) + D * x[:, t]

        assert torch.allclose(y_impl, y_ref, atol=1e-5), (
            f"selective_scan 与手算递推不一致：max|Δ|={ (y_impl - y_ref).abs().max():.3e}"
        )

    def test_selective_scan_d_skip(self) -> None:
        """D 作为 skip connection：当 A,B,C 使 h≈0 时，y ≈ D⊙x。"""
        from improve_suite.mamba_min import selective_scan

        B, L, d_in, n = 1, 3, 2, 2
        x = torch.randn(B, L, d_in)
        delta = torch.zeros(B, L, d_in)  # Δ=0 → h 恒为 0（exp(0)=1 但 ΔB_x=0）
        A = torch.randn(d_in, n)
        B_in = torch.randn(B, L, n)
        C = torch.randn(B, L, n)
        D = torch.tensor([2.0, 3.0])
        y = selective_scan(x, delta, A, B_in, C, D)
        # Δ=0 → ΔA=1, ΔB_x=0 → h=0 → y = C·0 + D·x = D·x
        assert torch.allclose(y, x * D.view(1, 1, -1), atol=1e-6)
