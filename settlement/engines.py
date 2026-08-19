"""引擎适配层：结算执行器只依赖 ``evaluate(signals_wide) -> perf`` 一个接口。

- ``SyntheticEngine``：演习专用玩具引擎（top-k 等权、合成收益、AER=252×日均
  超额）——**非 canonical 口径**，仅用于走通判据代入/分支路由/文档管线；
- ``CanonicalEngine``：真实结算引擎（2026-11 启用），逐字复用
  baseline_suite.pipeline 双基准口径（与 G4/G7/N50 封盘同跑法）。演习与
  测试绝不实例化 CanonicalEngine（真实价格读取只在结算分支发生）。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SyntheticEngine:
    """合成玩具引擎：date×code 信号宽表 → {"aer_ew", "aer_idx"}（确定性）。

    口径（演习专用，非 canonical）：每日取信号 top-10 等权持有一日；
    AER = 252 × 日均超额（组合 − 基准）；等权基准 = 全池等权收益，
    指数基准 = world.index_ret（合成"市值加权"代理）。NaN 信号不参与选股。
    """

    world: object
    top_k: int = 10

    def evaluate(self, signals: pd.DataFrame) -> dict:
        rets: pd.DataFrame = self.world.returns
        idx_ret: pd.Series = self.world.index_ret
        excess_ew, excess_idx = [], []
        for d in signals.index:
            if d not in rets.index:
                continue
            s = signals.loc[d].dropna()
            if len(s) < self.top_k:
                excess_ew.append(0.0)
                excess_idx.append(0.0)
                continue
            picks = s.nlargest(self.top_k).index
            port = float(rets.loc[d, picks].mean())
            excess_ew.append(port - float(rets.loc[d].mean()))
            excess_idx.append(port - float(idx_ret.loc[d]))
        n = max(len(excess_ew), 1)
        return {
            "aer_ew": 252.0 * sum(excess_ew) / n,
            "aer_idx": 252.0 * sum(excess_idx) / n,
        }


class CanonicalEngine:  # pragma: no cover - 2026-11 结算分支启用；演习零实例化
    """canonical 双基准引擎（csi300/k=50/n=5/min_hold=5/15bp，含成本）。

    与 g7_shortwindow/n50_amplify 封盘跑法同构：冻结宇宙列集上建
    px/tradeable + 双基准，逐臂 run_group。构造时才触碰 qlib/DDB（真实结算）。
    """

    def __init__(self, *, backtest_start: str, backtest_end: str,
                 universe_cols: list[str]) -> None:
        from dataclasses import replace

        from baseline_suite.common import BaselineConfig

        self.cfg = replace(
            BaselineConfig.load(window="oos"),
            window="forward_settlement",
            backtest_start=backtest_start,
            backtest_end=backtest_end,
        )
        self.universe_cols = list(universe_cols)
        self._ctx: dict | None = None

    def _ensure_context(self, rebalances: pd.DatetimeIndex) -> dict:
        from kronos_qlib import QlibProvider

        from baseline_suite.pipeline import build_dual_benchmarks
        from baseline_suite.signal import build_px_tradeable

        if self._ctx is None:
            provider = QlibProvider(self.cfg.pool, self.cfg.backtest_start,
                                    self.cfg.backtest_end)
            px, trd = build_px_tradeable(provider, self.cfg, rebalances,
                                         self.universe_cols)
            bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, self.cfg,
                                                                  px, trd)
            self._ctx = {"px": px, "trd": trd, "bench_idx": bench_idx,
                         "bench_ew": bench_ew, "beta_gap": beta_gap}
        return self._ctx

    def evaluate(self, signals: pd.DataFrame) -> dict:
        from baseline_suite.pipeline import run_group

        ctx = self._ensure_context(pd.DatetimeIndex(signals.index))
        pi, pe, _, _, _ = run_group(signals, ctx["px"], ctx["trd"],
                                    ctx["bench_idx"], ctx["bench_ew"],
                                    cfg=self.cfg, name="arm")
        return {"aer_ew": float(pe["aer"]), "aer_idx": float(pi["aer"]),
                "perf_ew": pe.to_dict(), "perf_idx": pi.to_dict()}
