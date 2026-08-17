"""g5_head 单测（G5 计划 §1/§2，20260817）。

阶段 0（归因）：
    - ``TestHoldingsReplay``：``replay_holdings`` 与 canonical 引擎
      ``run_portfolio`` 的换手日志逐日逐位一致（合成数据，纯张量/纯 pandas，
      不触发 DDB / GPU）——归因的持仓重放忠实性门禁。

阶段 1（换头臂，计划 §2 步骤 1.1 指定三测试）：
    - ``test_backbone_loads_g1``：G1 目录装载且 ``extract`` 输出 [B,90,832]；
    - ``test_capacity_range``：H-kda 数值核算 ∈ [0.8M, 1.3M]；
    - ``test_backbone_frozen``：backward 后底座梯度全 None。

先于实现写成：模块未建时 import 失败 → FAIL；实现后 → PASS。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _StubBackbone(nn.Module):
    """桩主干（与 tests/test_mamba_head.py 同构）：冻结 + eval + no_grad extract。"""

    def __init__(self, d_model: int = 832, d_in: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        g = torch.Generator().manual_seed(12345)
        self._proj = nn.Parameter(torch.randn(d_in, d_model, generator=g), requires_grad=False)

    def train(self, mode: bool = True) -> "_StubBackbone":
        super().train(mode)
        return self

    @torch.no_grad()
    def extract(self, features: torch.Tensor, stamp: torch.Tensor | None = None) -> torch.Tensor:
        return features @ self._proj



# ============================================================
# 阶段 0：持仓重放忠实性
# ============================================================


class TestHoldingsReplay:
    """replay_holdings 必须与 paper_replication.engine 引擎逐日一致。"""

    def _synthetic(self, n_days: int = 60, n_codes: int = 80, seed: int = 7):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2026-01-01", periods=n_days)
        codes = [f"C{i:03d}" for i in range(n_codes)]
        signal = pd.DataFrame(rng.standard_normal((n_days, n_codes)), index=dates, columns=codes)
        # 随机信号挖 NaN（模拟池外/停牌缺失）
        signal = signal.mask(rng.random((n_days, n_codes)) < 0.05)
        px = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.standard_normal((n_days, n_codes)) * 0.02, axis=0)),
            index=dates, columns=codes,
        )
        tradeable = pd.DataFrame(
            rng.random((n_days, n_codes)) > 0.05, index=dates, columns=codes
        )
        return signal, px, tradeable

    def test_replay_matches_engine_trades(self) -> None:
        """换手日志逐日逐位一致：sold/bought 列表与引擎 TradeLog 完全相同。"""
        from paper_replication.engine import EngineConfig, run_portfolio

        from g5_head.holdings import replay_holdings

        signal, px, tradeable = self._synthetic()
        cfg = EngineConfig(top_k=10, drop_n=3, min_hold=2, cost_bps=15.0)

        _, _, trades = run_portfolio(signal, px, tradeable, cfg=cfg)
        holdings, events = replay_holdings(signal, px, tradeable, cfg=cfg)

        assert len(events) == len(trades.rows), "日数不一致"
        for row, (d, sold, bought) in zip(trades.rows, events):
            assert row["date"] == d
            assert row["sold"] == sold, f"{d.date()} sold 不一致：{row['sold']} vs {sold}"
            assert row["bought"] == bought, f"{d.date()} bought 不一致：{row['bought']} vs {bought}"
        # 持仓集合 = 首日 top-k，之后每日 −sold +bought，规模自检
        prev: set[str] = set()
        for i, (d, sold, bought) in enumerate(events):
            cur = holdings[d]
            if i == 0:
                assert len(cur) == cfg.top_k
            else:
                assert cur == (prev - set(sold)) | set(bought), f"{d.date()} 持仓演化不一致"
                assert len(cur) == len(prev), f"{d.date()} 持仓数漂移"
            prev = set(cur)

    def test_daily_overlap_bounds(self) -> None:
        """重合度：同信号 = 1.0；不相交信号 = 0.0。"""
        from g5_head.holdings import daily_overlap

        dates = pd.bdate_range("2026-01-01", periods=5)
        a = {d: frozenset({f"C{i}" for i in range(10)}) for d in dates}
        b_same = {d: frozenset({f"C{i}" for i in range(10)}) for d in dates}
        b_disjoint = {d: frozenset({f"C{i}" for i in range(10, 20)}) for d in dates}
        assert daily_overlap(a, b_same, k=10) == 1.0
        assert daily_overlap(a, b_disjoint, k=10) == 0.0


# ============================================================
# 阶段 1：换头臂（计划 §2 步骤 1.1 指定三测试）
# ============================================================


class TestG5Heads:
    """H-kda / H-mamba / H-lin 三头契约（G1 底座冻结）。"""

    def test_backbone_loads_g1(self) -> None:
        """G1 目录装载且 extract 输出 [B,90,832]（CPU 装载，不占 GPU）。"""
        from g5_head.backbone_g1 import load_g1_backbone

        backbone = load_g1_backbone(device="cpu")
        assert backbone.d_model == 832, f"d_model={backbone.d_model}，期望 832"
        x = torch.randn(2, 90, 6)
        stamp = torch.zeros(2, 90, 5)
        with torch.no_grad():
            hidden = backbone.extract(x, stamp)
        assert hidden.shape == (2, 90, 832), f"extract 输出 {tuple(hidden.shape)}，期望 (2,90,832)"

    def test_capacity_range(self) -> None:
        """H-kda 数值核算 ∈ [0.8M, 1.3M]（计划 §0 教训：以数值核算为准）。"""
        from cross_section_kda.models import count_trainable

        from g5_head.heads import G5KdaHead

        head = G5KdaHead(_StubBackbone())
        n = count_trainable(head)
        assert 0.8e6 <= n <= 1.3e6, f"H-kda 可训练参数 {n:,} 不在 [0.8M, 1.3M]"
        assert all(not p.requires_grad for p in head.backbone.parameters()), "主干参数必须冻结"

    def test_backbone_frozen(self) -> None:
        """backward 后底座梯度全 None、头参数梯度非 None（三头全覆盖）。"""
        from cross_section_kda.models import B2LinearProbe
        from improve_suite.mamba_head import MambaTemporalHead

        from g5_head.heads import G5KdaHead

        for make in (
            lambda bb: G5KdaHead(bb),
            lambda bb: MambaTemporalHead(bb),
            lambda bb: B2LinearProbe(bb),
        ):
            backbone = _StubBackbone()
            head = make(backbone).train()
            x = torch.randn(4, 90, 6)
            stamp = torch.randn(4, 90, 5)
            score = head(x, stamp)
            score.sum().backward()
            for p in backbone.parameters():
                assert p.grad is None, "冻结主干参数不应积累梯度"
            trainable = [p for p in head.parameters() if p.requires_grad]
            assert trainable and all(p.grad is not None for p in trainable), "头参数应有梯度"

    def test_head_shapes(self) -> None:
        """三头 forward 输出 shape (B,)；G5KdaHead._decode 与 forward 数学一致。"""
        from cross_section_kda.models import B2LinearProbe
        from improve_suite.mamba_head import MambaTemporalHead

        from g5_head.heads import G5KdaHead, decode_score

        x = torch.randn(4, 90, 6)
        stamp = torch.randn(4, 90, 5)
        for make in (
            lambda bb: G5KdaHead(bb),
            lambda bb: MambaTemporalHead(bb),
            lambda bb: B2LinearProbe(bb),
        ):
            head = make(_StubBackbone())
            with torch.no_grad():
                out = head(x, stamp)
                hidden = head.backbone.extract(x, stamp)
                dec = decode_score(head, hidden)
            assert out.shape == (4,), f"forward 输出 {tuple(out.shape)}，期望 (4,)"
            assert dec.shape == (4,)
            assert torch.allclose(out, dec, atol=1e-5), "forward 与 _decode/缓存路径不一致"

    def test_protocol_constants_match(self) -> None:
        """训练协议常量与 cross_section_kda 源逐字相等（防协议漂移）。"""
        from cross_section_kda import data as D
        from cross_section_kda import train as T

        from g5_head import run_g5_head as R

        assert R.G5_LR == T.LR["B3"]
        assert R.G5_WD == T.WEIGHT_DECAY
        assert R.G5_BATCH == T.BATCH_SIZE
        assert R.G5_EPOCHS == T.EPOCHS
        assert R.G5_PATIENCE == T.PATIENCE
        assert R.G5_TRAIN_START == D.TRAIN_START
        assert R.G5_TRAIN_END == D.TRAIN_END
        assert R.G5_ES_START == D.EARLY_STOP_START
        assert R.G5_ES_END == D.EARLY_STOP_END
        assert R.G5_PURGE == D.PURGE
        assert R.G5_SEEDS == (42, 43, 44)
        assert R.HLIN_SEED == 42
