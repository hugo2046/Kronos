"""paper_replication 引擎单测（计划 §2.2）。

先于任何真实信号；5 条门禁逐条覆盖：

    1. 最少持有期约束：构造信号翻转用例，持有 <5 日的股票必须没被卖出；
    2. drop-n 约束：单日换手不超过 n；
    3. 等权再平衡口径（选定口径：新腿按 1/k，存量随行就市）；
    4. 成本扣除：给定换手额，净值扣减精确可算；
    5. 占位信号回归：seed 固定的随机信号跑引擎，AER 应≈0
       （|AER|<3%）——证明引擎本身不制造 alpha（最重要的一道门禁）。

**门禁口径注记（2026-08-10）**：计划 §2.2.5 / §4 规则1 原文写的是
"占位 |AER|<3%"（对称阈值）。但同一计划 §0 锁定了单边 0.15% 成本 + 每日调仓 +
n=5/k=50——这套成本结构下随机占位的稳态换手约 10%/日，纯成本拖折年化约 -7%，
即占位 AER 必然在 -7% 附近（负值，来自成本，不是 bug）。故本测试按**门禁的意图**
（证明引擎不制造 alpha）落实为：**占位不得有显著正 AER**（AER < +3%）。
负的成本拖是预期内的，如实记录于结果文档。此偏差在跑 K 组前冻结。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 公共：构造合成数据的小工具
# ============================================================


def make_dates(n: int = 30) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-07-01", periods=n)


def flat_px(dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    """全 1.0 的价格（收益恒 0），便于隔离换手/成本逻辑。"""
    return pd.DataFrame(1.0, index=dates, columns=codes)


def all_tradeable(dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=codes)


# ============================================================
# §2.2.1 最少持有期约束
# ============================================================


class TestMinHold:
    def test_held_under_5days_not_sold(self) -> None:
        """持有 <5 日的股票即便信号变最差，也不得被卖出。

        构造：k=3, n=2, min_hold=5。首日建仓 top-3 = [A,B,C]。
        次日起把 A/B/C 的信号砸到最低（理应被卖），但持有 <5 日 → 不得卖。
        前 5 日换手应为 0（首日建仓除外）。
        """
        from paper_replication.engine import EngineConfig, run_portfolio

        set_seed(42)
        dates = make_dates(12)
        codes = ["A", "B", "C", "D", "E", "F"]
        # 首日信号：A>B>C>D>E>F → 建 A,B,C
        sig0 = pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], index=codes)
        # 之后信号翻转：A,B,C 降到最低，D,E,F 升到最高（引诱换仓）
        sig_flip = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=codes)
        signal_wide = pd.DataFrame([sig_flip] * len(dates), index=dates, columns=codes)
        signal_wide.iloc[0] = sig0  # 首日用原信号建仓

        px = flat_px(dates, codes)
        trd = all_tradeable(dates, codes)
        cfg = EngineConfig(top_k=3, drop_n=2, min_hold=5, cost_bps=15.0)

        _, _, trades = run_portfolio(signal_wide, px, trd, cfg=cfg)
        tdf = trades.to_frame()
        # 首日（i=0）建仓；i=1..4 持有 <5 日不得卖 → sold 必须为空
        for i in range(1, 5):
            assert tdf.iloc[i]["sold"] == [], (
                f"第{i}日卖出了持有<5日的股票: {tdf.iloc[i]['sold']}（min_hold 未生效）"
            )
        # 到了 i=5（持有满 5 日），A/B/C 信号最差应被卖，D/E/F 信号最好应被买
        assert len(tdf.iloc[5]["sold"]) > 0, "持有满 5 日后应开始换手"


# ============================================================
# §2.2.2 drop-n 约束
# ============================================================


class TestDropN:
    def test_daily_turnover_capped_at_n(self) -> None:
        """单日卖出数 / 买入数均不超过 n。

        构造：k=3, n=2, min_hold=0（放松持有期以隔离 drop-n）。
        信号每期完全洗牌，理应有大量换手候选，但单日实际换手 = min(n, ...) ≤ n。
        """
        from paper_replication.engine import EngineConfig, run_portfolio

        set_seed(0)
        dates = make_dates(20)
        codes = [f"C{i}" for i in range(10)]
        rng = np.random.default_rng(0)
        signal_wide = pd.DataFrame(rng.standard_normal((len(dates), len(codes))),
                                   index=dates, columns=codes)
        px = flat_px(dates, codes)
        trd = all_tradeable(dates, codes)
        cfg = EngineConfig(top_k=3, drop_n=2, min_hold=0, cost_bps=15.0)

        _, _, trades = run_portfolio(signal_wide, px, trd, cfg=cfg)
        tdf = trades.to_frame()
        # 首日建仓可一次性建满（不受 drop_n 限制），从 i=1 起检查
        for i in range(1, len(tdf)):
            row = tdf.iloc[i]
            assert len(row["sold"]) <= cfg.drop_n, (
                f"第{i}日卖出 {len(row['sold'])} > n={cfg.drop_n}"
            )
            assert len(row["bought"]) <= cfg.drop_n, (
                f"第{i}日买入 {len(row['bought'])} > n={cfg.drop_n}"
            )


# ============================================================
# §2.2.3 等权再平衡口径
# ============================================================


class TestEqualWeight:
    def test_new_legs_get_one_over_k(self) -> None:
        """新买入腿按 1/k 配仓（选定口径，§2.2.3）。

        构造：k=4, 首日建仓 A,B,C,D，价格恒定（无漂移），
        则每只权重应 = 1/4 = 0.25。
        """
        from paper_replication.engine import EngineConfig, run_portfolio

        set_seed(7)
        dates = make_dates(3)
        codes = ["A", "B", "C", "D", "E", "F", "G", "H"]
        sig = pd.Series([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], index=codes)
        signal_wide = pd.DataFrame([sig] * len(dates), index=dates, columns=codes)
        px = flat_px(dates, codes)
        trd = all_tradeable(dates, codes)
        cfg = EngineConfig(top_k=4, drop_n=2, min_hold=5, cost_bps=0.0)

        _, _, trades = run_portfolio(signal_wide, px, trd, cfg=cfg)
        # 首日建仓后，A,B,C,D 权重应各 0.25（成本为 0，无价格漂移）
        # 通过换手日志间接验证：turnover_ratio 应 = 1.0（全额配仓）
        tdf = trades.to_frame()
        assert abs(tdf.iloc[0]["turnover_ratio"] - 1.0) < 1e-9, (
            f"首日建仓换手应=1.0（全额配仓），实际 {tdf.iloc[0]['turnover_ratio']}"
        )


# ============================================================
# §2.2.4 成本扣除
# ============================================================


class TestCostDeduction:
    def test_cost_exact_on_known_turnover(self) -> None:
        """给定换手额，净值扣减精确可算。

        构造：k=2, 首日建仓 A,B（turnover=1.0）。cost_bps=15 → 单边成本 = 1.0*15/1e4。
        首日组合收益（毛）= 0（价格恒定）→ 净收益 = -cost。
        """
        from paper_replication.engine import EngineConfig, run_portfolio

        dates = make_dates(2)
        codes = ["A", "B", "C", "D"]
        sig = pd.Series([4.0, 3.0, 2.0, 1.0], index=codes)
        signal_wide = pd.DataFrame([sig, sig], index=dates, columns=codes)
        px = flat_px(dates, codes)  # 价格恒定 → 毛收益 0
        trd = all_tradeable(dates, codes)
        cfg = EngineConfig(top_k=2, drop_n=2, min_hold=5, cost_bps=15.0)

        daily_ret, _, trades = run_portfolio(signal_wide, px, trd, cfg=cfg)
        expected_cost = 1.0 * 15.0 / 1e4  # turnover=1.0 × 15bp
        assert abs(daily_ret.iloc[0] - (-expected_cost)) < 1e-12, (
            f"首日净收益应为 -{expected_cost}（毛0 - 成本），实际 {daily_ret.iloc[0]}"
        )


# ============================================================
# §2.2.5 占位信号回归（引擎零点门禁，最重要）
# ============================================================


class TestPlaceholderGate:
    def test_random_signal_no_positive_alpha(self) -> None:
        """seed 固定的随机信号跑引擎，AER 应不显著为正（< +3%）。

        这是计划 §2.2.5 的引擎零点门禁——证明引擎本身不制造 alpha。
        用合成几何布朗运动价格 + 随机信号，组合相对池等权基准不应有正超额。

        注：原计划写 |AER|<3%（对称），但每日调仓 + 0.15% 成本下随机信号的稳态
        换手会产生显著**负**成本拖（约 -7% 年化），那是成本不是 bug。
        故门禁按意图落实为"不得有显著正 AER"（AER < +3%）。
        """
        from paper_replication.engine import (
            EngineConfig,
            attach_benchmark,
            compute_perf,
            run_portfolio,
        )

        set_seed(42)
        n_days = 252
        dates = make_dates(n_days)
        codes = [f"S{i:03d}" for i in range(60)]  # 60 只，k=50

        rng = np.random.default_rng(42)
        # 几何布朗运动价格，全部同分布（无任何信号-收益关系）
        mu_d, sig_d = 0.0 / 252, 0.02
        rets = rng.normal(mu_d, sig_d, (n_days, len(codes)))
        px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=codes)
        # 随机信号（与收益独立）
        signal_wide = pd.DataFrame(rng.standard_normal((n_days, len(codes))),
                                   index=dates, columns=codes)
        trd = all_tradeable(dates, codes)
        cfg = EngineConfig(top_k=50, drop_n=5, min_hold=5, cost_bps=15.0)

        daily_ret, _, trades = run_portfolio(signal_wide, px, trd, cfg=cfg)
        # 基准：池等权日收益（与组合同区间）
        bench_ret = px.pct_change().mean(axis=1).loc[daily_ret.index]
        excess = attach_benchmark(daily_ret, bench_ret)
        perf = compute_perf(excess, trades, name="placeholder")

        # 门禁：占位不得有显著正 AER（引擎不应制造 alpha）
        assert perf.aer < 0.03, (
            f"占位 AER={perf.aer:.2%} ≥ +3%：引擎在随机信号上制造了正 alpha，有 bug"
        )
        # 成本拖应为负（换手产生的预期负值），且量级合理（个位数 ~ 十几个百分点）
        assert perf.aer > -0.30, (
            f"占位 AER={perf.aer:.2%} 异常过负（<-30%）：换手或成本逻辑可能有 bug"
        )
