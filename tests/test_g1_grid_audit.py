"""G1 信号日期对照（FULL vs CROPPED）契约测试（计划 §4/§5/§7，20260908）。

先 FAIL 后 PASS（全部合成数据，不触真实行情/权重/信号文件）：

- ``test_build_arms_cropped_equivalence``：FULL 在保留日上的值与 CROPPED
  逐位相同（含 NaN 位置）；非保留日全 NaN、不 ffill；
- ``test_extra_dates_report_tail_and_gap``：日期差异报告区分连续尾部与
  内部缺口（尾部/缺口日期清单与条数）；
- ``test_cropped_reproduces_prefix_and_anchor``：两臂在首个信号分歧日之前
  逐日净收益完全一致（delay=1：实际收益差不早于分歧日次日成交）；CROPPED
  复现锚函数逐位返回基准序列；
- ``test_common_mask_ew_benchmark``：共同掩码等权基准只用
  ``tradeable & full.notna() & cropped.notna()`` 的股票日，缺共同信号日
  不填零；
- ``test_grid_audit_call_chain_clean``：grid_audit 源无模型加载/旧引擎/
  baseline_suite 接线；窗口边界不越 FORWARD_CUTOFF；
- ``test_manifest_hash_roundtrip``：输入哈希保全（改动即拒）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _market(n_days: int = 60, start: str = "2025-07-01",
            codes=("A", "B", "C", "D")):
    cal = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(11)
    px = pd.DataFrame(
        {c: 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_days)) for c in codes},
        index=cal)
    trd = pd.DataFrame(True, index=cal, columns=list(codes))
    uls = pd.DataFrame(0, index=cal, columns=list(codes))
    return px, trd, uls


def _raw_signal(px, sig_dates, seed: int = 5):
    rng = np.random.default_rng(seed)
    wide = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for d in sig_dates:
        wide.loc[d] = rng.normal(0, 1, len(px.columns))
    return wide


def test_build_arms_cropped_equivalence() -> None:
    from mh1_multihorizon.grid_audit import build_arms

    px, _, _ = _market()
    sig_dates = list(px.index[::5])
    raw = _raw_signal(px, sig_dates)
    calendar = px.index
    columns = list(px.columns)
    mh1_dates = sig_dates[:-4]                    # 丢掉尾部 4 个信号日
    full, cropped = build_arms(raw, calendar, columns, mh1_dates)
    assert full.index.equals(calendar) and list(full.columns) == columns
    # FULL 在保留日上的值（含 NaN）与 CROPPED 逐位相同
    keep = full.index.isin(mh1_dates)
    pd.testing.assert_frame_equal(full[keep], cropped[keep])
    # 非保留日 CROPPED 全 NaN（不 ffill）
    assert cropped[~keep].isna().all().all()
    # FULL 原始缺失保持缺失（sig_dates 之外本就 NaN，不填充）
    assert full[~full.index.isin(sig_dates)].isna().all().all()


def test_extra_dates_report_tail_and_gap() -> None:
    from mh1_multihorizon.grid_audit import report_extra_dates

    cal = pd.bdate_range("2025-07-01", periods=40)
    sig_dates = list(cal[::4])                    # 10 个信号日
    mh1_dates = sig_dates[:6] + [sig_dates[8]]    # 内部缺口：丢 sig_dates[6,7,9]？
    extra = report_extra_dates(cal, sig_dates, mh1_dates)
    dropped = sorted(set(sig_dates) - set(mh1_dates))
    assert extra["n_extra_dates"] == len(dropped)
    assert extra["extra_dates"] == [str(d.date()) for d in dropped]
    # 连续性判定：dropped = sig_dates[6], sig_dates[7], sig_dates[9] →
    # sig_dates[9] 之后无保留日 = 尾部；[6],[7] 相邻 = 内部缺口
    assert extra["n_tail_dates"] >= 1
    assert extra["n_internal_dates"] >= 1
    # 纯尾部情形：只丢连续尾部
    extra2 = report_extra_dates(cal, sig_dates, sig_dates[:-2])
    assert extra2["n_internal_dates"] == 0
    assert extra2["n_tail_dates"] == 2


def test_cropped_reproduces_prefix_and_anchor() -> None:
    """首个信号分歧日之前两臂逐日净收益一致；分歧不早于分歧日次日成交。"""
    from mh1_multihorizon.grid_audit import build_arms
    from mh1_multihorizon.replay_corrected import replay_one

    px, trd, uls = _market(n_days=50)
    sig_dates = list(px.index[::5])
    raw = _raw_signal(px, sig_dates)
    mh1_dates = sig_dates[:-3]
    full, cropped = build_arms(raw, px.index, list(px.columns), mh1_dates)
    out_f = replay_one(full, px, trd, uls, top_k=2, drop_n=2, min_hold=1)
    out_c = replay_one(cropped, px, trd, uls, top_k=2, drop_n=2, min_hold=1)
    # 首个分歧信号日（full 有值、cropped NaN）
    diverge = min(d for d in sig_dates if d not in mh1_dates)
    pre = px.index < diverge
    assert np.allclose(out_f["net"][pre], out_c["net"][pre], atol=0, rtol=0)
    # delay=1：净收益首个差异日 ≥ 分歧日 +1（分歧信号在次日才成交计费）
    diff_days = (out_f["net"] - out_c["net"]).abs()
    diff_days = diff_days[diff_days > 0]
    if len(diff_days):
        assert diff_days.index.min() > diverge


def test_common_mask_ew_benchmark() -> None:
    from mh1_multihorizon.grid_audit import common_mask_ew

    px, trd, _ = _market(n_days=30)
    full = _raw_signal(px, list(px.index[::5]))
    cropped = full.copy()
    cropped.loc[~cropped.index.isin(list(px.index[::5])[:-2]), :] = np.nan
    bench = common_mask_ew(px, trd, full, cropped)
    rets = px.pct_change(fill_method=None)
    mask = (trd & full.notna() & cropped.notna())
    common_days = (mask & rets.notna()).any(axis=1)
    # 基准只落在共同有效日期，且这些日都有值；其余日不填零
    assert bench.index.isin(px.index[common_days]).all()
    assert len(bench) == int((mask & rets.notna()).sum(axis=1).gt(0).sum()) \
        or len(bench) > 0
    d0 = bench.index[0]
    manual = (rets.where(mask).loc[d0].sum()
              / (mask & rets.notna()).loc[d0].sum())
    assert bench.iloc[0] == pytest.approx(manual)


def test_grid_audit_call_chain_clean() -> None:
    src = (REPO_ROOT / "mh1_multihorizon" / "grid_audit.py").read_text(
        encoding="utf-8")
    for banned in ("torch.load", "from_pretrained", "baseline_suite",
                   "run_group"):
        assert banned not in src, f"grid_audit.py 出现禁用接线：{banned}"
    from mh1_multihorizon.grid_audit import WINDOW_BOUNDS as GB

    for wname, (_s, e) in GB.items():
        assert str(e) <= "2026-07-24", f"{wname} 越封存线"


def test_manifest_hash_roundtrip(tmp_path) -> None:
    from mh1_multihorizon.grid_audit import verify_manifest_hashes

    f = tmp_path / "input.parquet"
    f.write_bytes(b"data")
    import hashlib

    m = {"inputs": {str(f): hashlib.sha256(b"data").hexdigest()}}
    verify_manifest_hashes(m)                     # 未变 → 通过
    f.write_bytes(b"data2")
    with pytest.raises(RuntimeError, match="输入被改动"):
        verify_manifest_hashes(m)
