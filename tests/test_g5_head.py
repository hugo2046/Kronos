"""g5_head 单测（G5 计划 §1/§2，20260817）。

阶段 0（归因）：
    - ``TestHoldingsReplay``：``replay_holdings`` 与 canonical 引擎
      ``run_portfolio`` 的换手日志逐日逐位一致（合成数据，纯张量/纯 pandas，
      不触发 DDB / GPU）——归因的持仓重放忠实性门禁。

阶段 1（换头臂，计划 §2 步骤 1.1 指定三测试）将随实现增补：
    ``test_backbone_loads_g1`` / ``test_capacity_range`` / ``test_backbone_frozen``。

先于实现写成：模块未建时 import 失败 → FAIL；实现后 → PASS。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
