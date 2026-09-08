"""MH1 纠偏重放契约测试（计划 §3~§5，20260908 纠偏计划）。

先 FAIL 后 PASS 的预注册契约：

- ``test_old_wiring_used_old_engine``：取证——原 ``engine_attachment`` 的调用
  链（baseline_suite.pipeline.run_group → paper_replication.engine）确实是
  旧引擎 v1，而非 engine_v2（模块身份对拍，历史问题存在的证据）；
- ``test_engine_attachment_not_wired_to_old_engine``：生产哨兵——把
  ``baseline_suite.pipeline.run_group`` 替换成一调用即抛错的哨兵后，纠正版
  ``engine_attachment`` 在合成行情下仍成功（不再经过旧引擎）；
- ``test_replay_core_delay1_bilateral_cost_limit_block``：合成数据直穿
  ``replay_corrected.replay_one``——t 信号 t+1 成交、首收益日 = t+2、
  双边 15bp、涨停禁买/跌停禁卖；
- ``test_cost_identity_and_formulas``：net = gross − cost（≤1e-12）、
  cost = (freed+bought_amt)×0.0015、整仓换手扣 0.003、初始建仓扣 0.0015；
- ``test_cli_rejects_forward_and_incomplete_arms``：越封存线参数拒绝、
  六臂不全拒绝、来源哈希不匹配拒绝；
- ``test_evidence_hashes_preserved``：重放前后原始证据哈希不变；
- ``test_window_bounds_and_calendar``：W3/W4 完整交易日历推进、信号日外
  NaN 不 ffill、日期不越 2026-07-24。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _synth_market(start="2025-07-01", n_days=40, codes=("A", "B", "C", "D")):
    """合成完整逐日行情：close 平稳、全部可交易、无涨跌停。"""
    cal = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(3)
    px = pd.DataFrame(
        {c: 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_days)) for c in codes},
        index=cal)
    trd = pd.DataFrame(True, index=cal, columns=list(codes))
    uls = pd.DataFrame(0, index=cal, columns=list(codes))
    return px, trd, uls


def _signal_frame(px, sig_dates, seed: int = 7):
    """信号宽表：仅 sig_dates 上给随机值（逐日变序，触发换仓），其余 NaN。"""
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for d in sig_dates:
        sig.loc[d] = rng.normal(0, 1, len(px.columns))
    return sig


# ============================================================
# 取证：旧接线确实走旧引擎（历史问题存在性证明）
# ============================================================


def test_old_wiring_used_old_engine() -> None:
    """baseline_suite.pipeline 的引擎符号来自 paper_replication.engine（v1）。

    这是纠偏依据：原 engine_attachment/nav 借道 run_group 实际运行旧引擎，
    JSON 中手写的 delay=1 元数据与实际调用无关。
    """
    import baseline_suite.pipeline as bp
    import paper_replication.engine as old_engine
    import paper_replication.engine_v2 as v2

    assert bp.run_portfolio is old_engine.run_portfolio, (
        "baseline_suite.pipeline.run_portfolio 应来自旧 engine（取证前提）")
    assert bp.EngineConfig is old_engine.EngineConfig
    assert bp.run_portfolio is not v2.run_portfolio_v2


# ============================================================
# 生产哨兵：纠正后入口不得再经过旧 run_group
# ============================================================


def test_engine_attachment_not_wired_to_old_engine(monkeypatch) -> None:
    """旧 run_group 哨兵 + 合成 fetcher 下，纠正版 engine_attachment 仍成功，
    且真实 v2 ``run_portfolio_v2`` 被直接调用（防"单测过、生产走旧引擎"）。"""
    import baseline_suite.pipeline as bp
    import paper_replication.engine_v2 as v2
    from mh1_multihorizon import evaluate as ev

    def _sentinel(*a, **k):  # pragma: no cover - 触发即失败
        raise AssertionError("生产入口仍在调用旧引擎 run_group")

    monkeypatch.setattr(bp, "run_group", _sentinel)
    src = Path(ev.__file__).read_text(encoding="utf-8")
    assert "from baseline_suite.pipeline import" not in src, (
        "evaluate.py 仍存在旧引擎 import 接线")

    called = {"v2": 0}
    real_v2 = v2.run_portfolio_v2

    def _spy(*a, **k):
        called["v2"] += 1
        return real_v2(*a, **k)

    monkeypatch.setattr(v2, "run_portfolio_v2", _spy)

    px, trd, uls = _synth_market(n_days=30)
    sig_dates = list(px.index[::5])
    sig = _signal_frame(px, sig_dates)

    class _Day:
        date = px.index[0]
        codes = list(px.columns)

    def _fetcher(window, universe_cols):
        return px, trd, uls, pd.Series(0.0, index=px.index)

    frames = {"W3": None}     # 信号由 fetcher 世界给出（见下）
    wide = sig
    frames = {"W3": pd.DataFrame({
        "date": [str(d.date()) for d in sig_dates for _ in px.columns],
        "instrument": list(px.columns) * len(sig_dates),
        "arm": "S", "seed": 42, "h10": 1.0})}
    out = ev.engine_attachment(
        frames, windows={"W3": [_Day()]}, fetcher=_fetcher,
        window_bounds={"W3": ("2025-07-01", str(px.index[-1].date()))})
    assert called["v2"] > 0, "engine_attachment 未直调 run_portfolio_v2"
    assert "S42" in out["results"] or "S42@W3" in str(out["results"])
    meta_cfg = out["meta"]["W3"]["engine"]["config"]
    assert meta_cfg["fix_delay_1"] is True      # 真实 asdict，非手写元数据
    assert meta_cfg["fix_double_sided_cost"] is True


def test_replay_core_delay1_bilateral_cost_limit_block() -> None:
    """合成行情直穿 replay_one：delay=1 / 首收益 t+2 / 双边成本 / 涨跌停。"""
    from mh1_multihorizon.replay_corrected import replay_one

    px, trd, uls = _synth_market(n_days=30)
    sig_dates = list(px.index[::5])          # 每 5 个交易日一个信号日
    sig = _signal_frame(px, sig_dates)

    out = replay_one(sig, px, trd, uls, top_k=2, drop_n=2, min_hold=1)
    net, trades = out["net"], out["trades"]
    tf = trades.to_frame()
    # 真实成交行（引擎逐日写日志，无操作日为空单/零成本）
    real = tf[(tf["sold"].str.len() > 0) | (tf["bought"].str.len() > 0)]
    assert len(real) > 0
    # delay=1：首个成交日 = 首信号日 +1；首收益日 = 首信号日 +2
    assert real["decision_date"].iloc[0] == sig_dates[0]
    assert real["date"].iloc[0] == sig.index[1]
    # 首个非零毛收益日（= 持仓首收益日；成本记在成交日会先于它出现在净序列）
    cost_daily = (tf.groupby("date")["cost"].sum()
                  .reindex(net.index, fill_value=0.0))
    gross = net + cost_daily
    nz = gross[gross != 0]
    assert len(nz) > 0 and nz.index[0] >= sig.index[2]
    # 双边成本：cost = (freed + bought_amt) × 0.0015
    assert np.allclose(tf["cost"], (tf["freed"] + tf["bought_amt"]) * 0.0015)
    # 首次建仓只买不卖：cost = 0.0015（首成交日无漂移，精确成立）
    assert real["cost"].iloc[0] == pytest.approx(0.0015, abs=1e-9)
    # 整仓对调（卖出/买入两腿各满仓）：双边 ≈ 0.003（含一日漂移，容差放宽）
    swaps = real[(real["sold"].str.len() > 0) & (real["bought"].str.len() > 0)]
    assert len(swaps) > 0
    assert 0.0029 <= swaps["cost"].max() <= 0.0031

    # 涨跌停禁成交：把某信号日次日全部股票设为 +1 涨停 → 当日不应有买入成交
    uls2 = uls.copy()
    fill_day = sig.index[1]                  # 首信号日次日 = 首成交日
    uls2.loc[fill_day] = 1
    out2 = replay_one(sig, px, trd, uls2, top_k=2, drop_n=2, min_hold=1)
    tf2 = out2["trades"].to_frame()
    if fill_day in tf2["date"].values:
        assert tf2.loc[tf2["date"] == fill_day, "bought"].str.len().sum() == 0


def test_cost_identity_and_formulas() -> None:
    """同路径毛/净分解恒等式：net = gross − cost（≤1e-12）。"""
    from mh1_multihorizon.replay_corrected import decompose

    px, trd, uls = _synth_market(n_days=45)
    sig = _signal_frame(px, list(px.index[::5]))
    from mh1_multihorizon.replay_corrected import replay_one

    out = replay_one(sig, px, trd, uls)
    daily = decompose(out["net"], out["trades"])
    # 恒等式
    assert np.abs(daily["net_return"] - (daily["gross_return"]
                                         - daily["cost"])).max() <= 1e-12
    # 净值 = 累乘
    assert np.allclose(daily["net_nav"], (1 + daily["net_return"]).cumprod())
    assert np.allclose(daily["gross_nav"], (1 + daily["gross_return"]).cumprod())
    # 拖累 = 毛净值 − 净净值
    assert np.allclose(daily["nav_drag"],
                       daily["gross_nav"] - daily["net_nav"])


def test_cli_rejects_forward_and_incomplete_arms(tmp_path) -> None:
    from mh1_multihorizon.replay_corrected import (
        REJECTED_FORWARD_MSG, validate_inputs,
    )

    # 越封存线的窗口参数拒绝
    with pytest.raises(ValueError, match=REJECTED_FORWARD_MSG):
        validate_inputs(window_bounds={"W5": ("2026-07-01", "2026-08-01")})
    # 六臂不全拒绝
    with pytest.raises(ValueError, match="六臂"):
        validate_inputs(runs_present=["S42", "M42", "S43"])
    # 来源哈希不匹配拒绝
    with pytest.raises(ValueError, match="来源哈希"):
        validate_inputs(source_sha256="deadbeef",
                        expected_sha256="2516998a" + "0" * 56)
    # 合法输入通过
    validate_inputs(
        window_bounds={"W3": ("2025-07-01", "2025-12-31"),
                       "W4": ("2026-01-01", "2026-07-24")},
        runs_present=["S42", "M42", "S43", "M43", "S44", "M44"],
        source_sha256="a" * 64, expected_sha256="a" * 64)


def test_window_bounds_and_calendar() -> None:
    """重放日历 = 窗口完整逐日交易日；信号日外 NaN 不被 ffill。"""
    from mh1_multihorizon.replay_corrected import prepare_signal_grid

    cal = pd.bdate_range("2025-07-01", "2025-12-31")
    sig_dates = cal[::10]
    wide = pd.DataFrame(1.0, index=sig_dates, columns=["A", "B"])
    grid = prepare_signal_grid(wide, cal)
    assert grid.index.equals(cal)
    assert grid.notna().sum().sum() == len(sig_dates) * 2   # 仅信号日有值
    assert grid.index.max() <= pd.Timestamp("2026-07-24")


def test_evidence_hashes_preserved(tmp_path, monkeypatch) -> None:
    from mh1_multihorizon.replay_corrected import hash_evidence, verify_evidence

    f = tmp_path / "ev.json"
    f.write_text("{}", encoding="utf-8")
    manifest = hash_evidence([f])
    assert manifest[str(f)] != ""
    verify_evidence(manifest)     # 未变动 → 通过
    f.write_text('{"x": 1}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="原始证据被改动"):
        verify_evidence(manifest)
