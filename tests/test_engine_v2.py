"""engine_v2 六项修正的单测（先 FAIL 后 PASS，反例锚定旧引擎错误行为）。

覆盖复核报告 §2-B/§2-C 的六项修正，每项一个反例：

    (1) 双边成本：一次换手 X（卖 X 买 X）扣 2×15bp·X（旧引擎扣 15bp·X）；
    (2) delay=1：信号日 t、成交日 t+1、首个收益日 t+2——
        反例 "B 当日先涨 100%、次日涨 10%" 必须得 5.0%（旧引擎 6.667%）；
    (3) 涨跌停/一字板不可成交（DDB uls 字段 + high==low 回退）；
    (4) 新买入腿权重不参与当日收益漂移（与 (2) 反例联合验证 + 独立隔离验证）；
    (5) 等权基准掩码 = tradeable ∧ signal.notna()；
    (6) 年化常量 252（AER 与 IR 的 √N）。

另含 TestLegacyEquivalence：六开关全关时 v2 与旧引擎在随机数据上逐日等价
（保证 v2 是旧引擎的严格推广，重放差值全部来自六项修正本身）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper_replication.engine import EngineConfig, run_portfolio
from paper_replication.engine_v2 import (
    EngineConfigV2,
    build_limit_masks,
    build_pool_equal_weight_benchmark_v2,
    compute_perf_v2,
    run_portfolio_v2,
)


def _frames(px: dict[str, list[float]], dates: pd.DatetimeIndex):
    """价格字典 → (signal, px, tradeable)，信号按列名排序 B>A 保证确定性。"""
    px_wide = pd.DataFrame(px, index=dates)
    sig = pd.DataFrame(
        {c: [float(i)] * len(dates) for i, c in enumerate(sorted(px, reverse=True))},
        index=dates,
    )
    trd = pd.DataFrame(True, index=dates, columns=px_wide.columns)
    return sig, px_wide, trd


# ====================================================================
# (2)+(4) 核心反例：B 当日先涨 100%、次日涨 10% → v2 首个收益日必须 5.0%
# ====================================================================

class TestCounterexampleB:
    """信号日 d0 选出 A、B；B 在 d1 +100%、d2 +10%。

    - 旧引擎：d0 收盘建仓（吃不到 d0 之前）、d1 吃 +100%（权重漂移到 2/3），
      d2 组合收益 = 2/3 × 10% = 6.667%。
    - v2：d0 决策、d1 收盘成交（错过 +100%）、首个收益日 d2 = 0.5 × 10% = 5.0%。
      5.0% 同时锁定修正 (2)（若仍当日成交，d2 前权重已漂移到 2/3）与修正 (4)
      （若新腿仍被当日漂移，d1 +100% 会把 B 权重虚推到 2/3）。
    """

    DATES = pd.date_range("2025-01-01", periods=4)  # d0..d3

    def _run_v2(self, **kw):
        sig, px, trd = _frames(
            {"A": [10.0, 10, 10, 10], "B": [10.0, 20, 22, 22]}, self.DATES
        )
        cfg = EngineConfigV2(top_k=2, drop_n=1, min_hold=5, cost_bps=0.0, **kw)
        rets, trades = run_portfolio_v2(sig, px, trd, cfg=cfg)
        return rets, trades

    def test_v2_first_return_day_is_5pct(self):
        rets, trades = self._run_v2()
        # 成交在 d1 收盘：d0/d1 无持仓收益（=0），首个收益日 d2
        assert rets.iloc[0] == 0.0 and rets.iloc[1] == 0.0
        assert rets.iloc[2] == pytest.approx(0.05, abs=1e-12)  # 5.0%

    def test_old_engine_gives_6_667pct(self):
        sig, px, trd = _frames(
            {"A": [10.0, 10, 10, 10], "B": [10.0, 20, 22, 22]}, self.DATES
        )
        old, _, _ = run_portfolio(
            sig, px, trd, cfg=EngineConfig(top_k=2, drop_n=1, min_hold=5, cost_bps=0.0)
        )
        assert old.iloc[1] == pytest.approx(0.50, abs=1e-12)     # 同日收益 lookahead
        assert old.iloc[2] == pytest.approx(0.066667, abs=1e-5)  # 6.667%（漂移权重 2/3）

    def test_delay_off_returns_to_same_day_execution(self):
        # 关掉 (2)：回到旧时序（d0 成交、d1 起收收益），d2 = 2/3×10%（fix4 仍开，
        # 但 B 在 d0 买入后 d1 的漂移是合法持仓漂移）
        rets, _ = self._run_v2(fix_delay_1=False)
        assert rets.iloc[1] == pytest.approx(0.50, abs=1e-12)
        assert rets.iloc[2] == pytest.approx(0.066667, abs=1e-5)


# ====================================================================
# (1) 双边成本：一次换手 X 扣 2×15bp·X
# ====================================================================

class TestDoubleSidedCost:
    """d0 决策持 B（top_k=1）；d1 收盘建仓 B；d2 决策换 A（min_hold=1 已满）；
    d3 收盘执行换手：卖出腿 X=1.0、买入腿 1.0 → 成本 2×15bp×1.0 = 30bp。
    旧引擎同换手只扣 15bp（(freed+bought)/2 单边近似）。
    """

    DATES = pd.date_range("2025-01-01", periods=6)

    def _sig(self):
        # 信号：d0 B 最好；d1 起 A 最好
        rows = []
        for i, d in enumerate(self.DATES):
            rows.append({"A": 1.0, "B": 2.0} if i == 0 else {"A": 2.0, "B": 1.0})
        return pd.DataFrame(rows, index=self.DATES)

    def _px_trd(self):
        px = pd.DataFrame({"A": [10.0] * 6, "B": [10.0] * 6}, index=self.DATES)
        trd = pd.DataFrame(True, index=self.DATES, columns=px.columns)
        return px, trd

    def test_v2_costs_double_side_30bp(self):
        cfg = EngineConfigV2(top_k=1, drop_n=1, min_hold=1, cost_bps=15.0)
        rets, trades = run_portfolio_v2(self._sig(), *self._px_trd(), cfg=cfg)
        tdf = trades.to_frame()
        rot = tdf[tdf["sold"].apply(len) > 0].iloc[0]
        assert rot["freed"] == pytest.approx(1.0)
        assert rot["bought_amt"] == pytest.approx(1.0)
        assert rot["cost"] == pytest.approx(2 * 15.0 / 1e4)  # 2×15bp·X
        # 执行日组合收益 = 0（全平价股）− 30bp 成本
        assert rets.loc[rot["date"]] == pytest.approx(-0.003, abs=1e-12)

    def test_fix_off_costs_half(self):
        cfg = EngineConfigV2(
            top_k=1, drop_n=1, min_hold=1, cost_bps=15.0, fix_double_sided_cost=False
        )
        rets, trades = run_portfolio_v2(self._sig(), *self._px_trd(), cfg=cfg)
        tdf = trades.to_frame()
        rot = tdf[tdf["sold"].apply(len) > 0].iloc[0]
        assert rot["cost"] == pytest.approx(15.0 / 1e4)  # 旧口径 (1+1)/2×15bp

    def test_old_engine_costs_half(self):
        old, _, old_trades = run_portfolio(
            self._sig(), *self._px_trd(),
            cfg=EngineConfig(top_k=1, drop_n=1, min_hold=1, cost_bps=15.0),
        )
        rot = old_trades.to_frame()
        rot = rot[rot["sold"].apply(len) > 0].iloc[0]
        assert rot["turnover_ratio"] == pytest.approx(1.0)  # (1+1)/2
        assert old.loc[rot["date"]] == pytest.approx(-0.0015, abs=1e-12)


# ====================================================================
# (3) 涨跌停/一字板不可成交
# ====================================================================

class TestLimitBlock:
    DATES = pd.date_range("2025-01-01", periods=5)

    def _base(self):
        px = pd.DataFrame(
            {"B": [10.0] * 5, "C": [10.0] * 5, "A": [10.0] * 5}, index=self.DATES
        )
        # 信号排序恒定：B > C > A
        sig = pd.DataFrame(
            {"B": 3.0, "C": 2.0, "A": 1.0}, index=self.DATES
        )
        trd = pd.DataFrame(True, index=self.DATES, columns=px.columns)
        return sig, px, trd

    def test_buy_skips_limit_up_stock(self):
        sig, px, trd = self._base()
        # B 在成交日 d1 收盘涨停 → 建仓顺延到次优 C
        uls = pd.DataFrame(0.0, index=self.DATES, columns=px.columns)
        uls.loc[self.DATES[1], "B"] = 1.0
        buy_block, sell_block = build_limit_masks(uls_wide=uls)
        cfg = EngineConfigV2(top_k=1, drop_n=1, min_hold=5, cost_bps=0.0)
        rets, trades = run_portfolio_v2(
            sig, px, trd, cfg=cfg, buy_blocked=buy_block, sell_blocked=sell_block
        )
        tdf = trades.to_frame()
        build = tdf[tdf["bought"].apply(len) > 0].iloc[0]  # delay 下建仓成交在 d1
        assert build["bought"] == ["C"]  # B 被涨停挡下，买 C

    def test_sell_defers_at_limit_down(self):
        sig, px, trd = self._base()
        # d0 决策持 B；d1 成交（holding_since=d1）。d3 决策卖出 B（min_hold=2：
        # between(d1,d3)=2 已满），d4 执行日 B 收盘跌停 → 顺延不卖
        sig2 = sig.copy()
        sig2.loc[self.DATES[3]] = {"B": 1.0, "C": 3.0, "A": 2.0}  # d3 信号翻成 C 最好
        uls = pd.DataFrame(0.0, index=self.DATES, columns=px.columns)
        uls.loc[self.DATES[4], "B"] = -1.0  # d4 B 收盘跌停
        buy_block, sell_block = build_limit_masks(uls_wide=uls)
        cfg = EngineConfigV2(top_k=1, drop_n=1, min_hold=2, cost_bps=0.0)
        rets, trades = run_portfolio_v2(
            sig2, px, trd, cfg=cfg, buy_blocked=buy_block, sell_blocked=sell_block
        )
        tdf = trades.to_frame()
        assert all(tdf["sold"].apply(lambda x: "B" not in x))  # 全程未卖出 B

    def test_fallback_one_line_board_high_eq_low(self):
        sig, px, trd = self._base()
        # uls 缺失 → 一字板回退：d1 B 一字涨停（high==low 且 > preclose）
        high = pd.DataFrame(11.0, index=self.DATES, columns=px.columns)
        low = pd.DataFrame(10.0, index=self.DATES, columns=px.columns)
        high.loc[self.DATES[1], "B"] = 11.0
        low.loc[self.DATES[1], "B"] = 11.0  # high == low 一字
        pre = pd.DataFrame(10.0, index=self.DATES, columns=px.columns)
        buy_block, sell_block = build_limit_masks(
            high_wide=high, low_wide=low, preclose_wide=pre
        )
        assert bool(buy_block.loc[self.DATES[1], "B"])
        assert not bool(sell_block.loc[self.DATES[1], "B"])
        cfg = EngineConfigV2(top_k=1, drop_n=1, min_hold=5, cost_bps=0.0)
        rets, trades = run_portfolio_v2(
            sig, px, trd, cfg=cfg, buy_blocked=buy_block, sell_blocked=sell_block
        )
        assert trades.to_frame().iloc[1]["bought"] == ["C"]

    def test_fix_off_ignores_limit(self):
        sig, px, trd = self._base()
        uls = pd.DataFrame(0.0, index=self.DATES, columns=px.columns)
        uls.loc[self.DATES[1], "B"] = 1.0
        buy_block, _ = build_limit_masks(uls_wide=uls)
        cfg = EngineConfigV2(
            top_k=1, drop_n=1, min_hold=5, cost_bps=0.0, fix_limit_block=False
        )
        rets, trades = run_portfolio_v2(
            sig, px, trd, cfg=cfg, buy_blocked=buy_block, sell_blocked=None
        )
        assert trades.to_frame().iloc[1]["bought"] == ["B"]


# ====================================================================
# (4) 新买入腿权重不参与当日收益漂移（delay 关闭下隔离验证）
# ====================================================================

class TestNewLegDrift:
    """d1 决策换仓 C→B（delay 关 → d1 收盘成交），B 当日 +100%：
    B 成交价即 d1 收盘价，d1 的 +100% 不属于本组合，其权重不得漂移。
    d2 B +10% → 组合收益 = 0.5×10% = 5.0%（漂移 bug 下会是 6.667%）。
    """

    DATES = pd.date_range("2025-01-01", periods=4)

    def test_new_leg_weight_undrifted(self):
        px = pd.DataFrame(
            {"A": [10.0] * 4, "B": [10.0, 20, 22, 22], "C": [10.0] * 4},
            index=self.DATES,
        )
        # d0: A、C 入选（top_k=2）；d1 起 C 信号变差、B 最好 → d1 收盘换 C→B
        rows = []
        for i in range(4):
            if i == 0:
                rows.append({"A": 3.0, "B": 0.5, "C": 2.0})
            else:
                rows.append({"A": 3.0, "B": 2.0, "C": 0.5})
        sig = pd.DataFrame(rows, index=self.DATES)
        trd = pd.DataFrame(True, index=self.DATES, columns=px.columns)
        cfg = EngineConfigV2(
            top_k=2, drop_n=1, min_hold=1, cost_bps=0.0,
            fix_delay_1=False,  # 隔离 (4)：当日决策当日成交
        )
        rets, trades = run_portfolio_v2(sig, px, trd, cfg=cfg)
        tdf = trades.to_frame()
        rot = tdf[tdf["sold"].apply(len) > 0].iloc[0]
        assert rot["date"] == self.DATES[1] and rot["sold"] == ["C"] and rot["bought"] == ["B"]
        # d1：换仓前持仓 A、C 全平价 → 0 收益；d2：A 0% + B 0.5×10% = 5.0%
        assert rets.loc[self.DATES[1]] == pytest.approx(0.0, abs=1e-12)
        assert rets.loc[self.DATES[2]] == pytest.approx(0.05, abs=1e-12)
        # 对照：旧引擎（漂移 bug）d2 = 2/3 × 10%
        old, _, _ = run_portfolio(
            sig, px, trd, cfg=EngineConfig(top_k=2, drop_n=1, min_hold=1, cost_bps=0.0)
        )
        assert old.loc[self.DATES[2]] == pytest.approx(0.066667, abs=1e-5)


# ====================================================================
# (6) 年化常量 252
# ====================================================================

class TestAnnualization252:
    def test_aer_and_ir_use_252(self):
        n = 252
        # AER：常数日超额 μ=0.1% → 连乘净值 = (1+μ)^252，年化 = 252 口径自映
        ex = pd.Series(0.001, index=pd.date_range("2025-01-01", periods=n))
        perf = compute_perf_v2(ex, trades=None, name="t", fix_annualization_252=True)
        assert perf.aer == pytest.approx(1.001 ** 252 - 1, rel=1e-9)
        # IR：随机序列 μ/σ×√252
        rng = np.random.default_rng(7)
        ex2 = pd.Series(rng.normal(0.001, 0.01, n), index=pd.date_range("2025-01-01", periods=n))
        mu, sd = ex2.mean(), ex2.std(ddof=1)
        perf2 = compute_perf_v2(ex2, trades=None, name="t", fix_annualization_252=True)
        assert perf2.ir == pytest.approx(mu / sd * np.sqrt(252), rel=1e-9)

    def test_legacy_uses_244(self):
        ex = pd.Series(0.001, index=pd.date_range("2025-01-01", periods=252))
        perf = compute_perf_v2(ex, trades=None, name="t", fix_annualization_252=False)
        # n=252、旧口径 244：nav=(1+μ)^252，年化指数 244/252 → 净效果 (1+μ)^244
        assert perf.aer == pytest.approx(1.001 ** 244 - 1, rel=1e-9)
        rng = np.random.default_rng(7)
        ex2 = pd.Series(rng.normal(0.001, 0.01, 252), index=pd.date_range("2025-01-01", periods=252))
        mu, sd = ex2.mean(), ex2.std(ddof=1)
        perf2 = compute_perf_v2(ex2, trades=None, name="t", fix_annualization_252=False)
        assert perf2.ir == pytest.approx(mu / sd * np.sqrt(244), rel=1e-9)


# ====================================================================
# (5) 等权基准掩码 = tradeable ∧ signal.notna()
# ====================================================================

class TestBenchmarkMask:
    DATES = pd.date_range("2025-01-01", periods=4)

    def test_mask_excludes_nan_signal_stocks(self):
        px = pd.DataFrame(
            {"A": [10.0, 10, 11, 11], "B": [10.0, 12, 14, 14]}, index=self.DATES
        )
        trd = pd.DataFrame(True, index=self.DATES, columns=px.columns)
        sig = pd.DataFrame(  # B 信号全缺失（如停牌窗口外/新上市）
            {"A": 1.0, "B": np.nan}, index=self.DATES
        )
        bench = build_pool_equal_weight_benchmark_v2(px, trd, sig, fix_mask=True)
        rets = px.pct_change(fill_method=None)
        # 掩码后基准 = A 单票等权
        assert bench.loc[self.DATES[2]] == pytest.approx(rets.loc[self.DATES[2], "A"])

    def test_legacy_mask_counts_all_tradeable(self):
        px = pd.DataFrame(
            {"A": [10.0, 10, 11, 11], "B": [10.0, 12, 14, 14]}, index=self.DATES
        )
        trd = pd.DataFrame(True, index=self.DATES, columns=px.columns)
        sig = pd.DataFrame({"A": 1.0, "B": np.nan}, index=self.DATES)
        bench = build_pool_equal_weight_benchmark_v2(px, trd, sig, fix_mask=False)
        rets = px.pct_change(fill_method=None)
        assert bench.loc[self.DATES[2]] == pytest.approx(
            (rets.loc[self.DATES[2], "A"] + rets.loc[self.DATES[2], "B"]) / 2
        )


# ====================================================================
# 六开关全关 == 旧引擎（随机数据逐日等价）
# ====================================================================

class TestLegacyEquivalence:
    def test_all_off_matches_old_engine(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=80)
        cols = [f"S{i:02d}" for i in range(30)]
        px = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0, 0.02, (80, 30)), axis=0),
            index=dates, columns=cols,
        )
        # 随机缺失价格（模拟停牌/退市）+ 随机信号缺失
        px = px.mask(rng.random((80, 30)) < 0.03)
        sig = pd.DataFrame(
            rng.normal(0, 1, (80, 30)), index=dates, columns=cols
        ).mask(rng.random((80, 30)) < 0.1)
        trd = pd.DataFrame(True, index=dates, columns=cols)
        trd.iloc[10:13, 3] = False  # 一段停牌

        old_cfg = EngineConfig(top_k=10, drop_n=3, min_hold=4, cost_bps=15.0)
        old_ret, _, old_trades = run_portfolio(sig, px, trd, cfg=old_cfg)

        v2_cfg = EngineConfigV2(
            top_k=10, drop_n=3, min_hold=4, cost_bps=15.0,
            fix_double_sided_cost=False,
            fix_delay_1=False,
            fix_limit_block=False,
            fix_new_leg_drift=False,
            fix_annualization_252=False,
        )
        v2_ret, v2_trades = run_portfolio_v2(sig, px, trd, cfg=v2_cfg)

        pd.testing.assert_series_equal(v2_ret, old_ret, check_names=False)
        odf, vdf = old_trades.to_frame(), v2_trades.to_frame()
        assert len(odf) == len(vdf)
        for (_, orow), (_, vrow) in zip(odf.iterrows(), vdf.iterrows()):
            assert orow["sold"] == vrow["sold"]
            assert orow["bought"] == vrow["bought"]
