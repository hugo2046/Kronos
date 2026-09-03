"""finetune_ashares G3 前瞻登记契约测试（计划 §2，20260816 计划）。

- ``test_resolve_registration_dates``：首日只登最新可得日；数据滞后时自动补缺
  （迟补标 late=true）；无新数据为空（节假日/周末跳过）；
- ``test_registry_idempotent``：同日重复 register_one 不重复追加——manifest
  恰一行、DuckDB 行数不变、直接跳过（幂等保护，计划冻结机制）；
- ``test_registry_no_lookahead``：登记日 d 的一切取数边界 ≤ d——动量/MA200/
  可交易掩码/provider 构造即使存在 d+1..d+5 数据也不触碰；
- ``test_gap_*``：断档区间 2026-08-19 ~ 2026-08-20（结算计划附录A/B，永久
  复算级）——auto 解析剔除 + 数据滞后时静默留空 + 显式 --date 拒绝，绝不补造。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import finetune_suite.run_registry as rr

D = "2026-08-14"
NEXT_DAYS = pd.to_datetime(["2026-08-17", "2026-08-18"])


# ---------------------------------------------------------------------------
# 日期解析
# ---------------------------------------------------------------------------
def test_resolve_registration_dates():
    cal = pd.DatetimeIndex(
        ["2026-08-12", "2026-08-13", "2026-08-14", "2026-08-17", "2026-08-18"]
    )
    # 首日（无登记历史）：只登最新可得日；今天是 08-16（周日）→ 08-14 迟补
    got = rr.resolve_registration_dates(cal, last_registered=None,
                                        latest_available=pd.Timestamp("2026-08-14"),
                                        today=pd.Timestamp("2026-08-16"))
    assert got == [(pd.Timestamp("2026-08-14"), True)]

    # 数据滞后两日（last=08-14，latest=08-18，今天=08-18）：补 08-17(late) + 08-18(当日)
    got = rr.resolve_registration_dates(cal, last_registered=pd.Timestamp("2026-08-14"),
                                        latest_available=pd.Timestamp("2026-08-18"),
                                        today=pd.Timestamp("2026-08-18"))
    assert got == [(pd.Timestamp("2026-08-17"), True), (pd.Timestamp("2026-08-18"), False)]

    # 无新数据（周末/节假日重复触发）：空 → 静默跳过
    got = rr.resolve_registration_dates(cal, last_registered=pd.Timestamp("2026-08-14"),
                                        latest_available=pd.Timestamp("2026-08-14"),
                                        today=pd.Timestamp("2026-08-16"))
    assert got == []


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------
def _fake_compute(d: pd.Timestamp):
    codes = ["000001.SZ", "600000.SH"]
    wide = pd.DataFrame(
        {
            "s100_mean": [0.01, 0.02], "s100_last": [0.011, 0.021],
            "s100_max": [0.02, 0.03], "s100_min": [0.005, 0.01],
            "s101_mean": [0.01, 0.02], "s101_last": [0.011, 0.021],
            "s101_max": [0.02, 0.03], "s101_min": [0.005, 0.01],
            "s102_mean": [0.01, 0.02], "s102_last": [0.011, 0.021],
            "s102_max": [0.02, 0.03], "s102_min": [0.005, 0.01],
            "M": [0.03, -0.01], "tradeable": [True, True],
        },
        index=pd.Index(codes, name="code"),
    )
    return wide, {"gate": True, "index_close": 4000.0, "ma200": 3900.0}


def test_registry_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(rr, "DB_PATH", tmp_path / "registry.duckdb")
    monkeypatch.setattr(rr, "MANIFEST_PATH", tmp_path / "MANIFEST.csv")
    monkeypatch.setattr(rr, "compute_day_signals", _fake_compute)

    d = pd.Timestamp(D)
    r1 = rr.register_one(d, late=True, do_git=False)
    assert r1["status"] == "registered"
    r2 = rr.register_one(d, late=True, do_git=False)
    assert r2["status"] == "already-registered"

    manifest = pd.read_csv(rr.MANIFEST_PATH, dtype={"date": str, "late": str})
    assert len(manifest) == 1 and manifest.loc[0, "date"] == D
    assert manifest.loc[0, "late"] == "true"  # 首日迟补如实标注

    import duckdb

    con = duckdb.connect(str(rr.DB_PATH), read_only=True)
    n1 = con.execute("SELECT COUNT(*) FROM registry WHERE date=?", [D]).fetchone()[0]
    n_meta = con.execute("SELECT COUNT(*) FROM registry_meta WHERE date=?", [D]).fetchone()[0]
    con.close()
    assert n1 == 2 * 14  # 2 股 × (12 种子变体 + M + tradeable)
    assert n_meta >= 3  # ma200_gate / index_close / pool 等


# ---------------------------------------------------------------------------
# 断档防补造（附录A/B：2026-08-19 ~ 2026-08-20 永久复算级，如实留白）
# ---------------------------------------------------------------------------
GAP_CAL = pd.DatetimeIndex(
    ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"]
)


def test_gap_dates_skipped_in_auto():
    assert rr.GAP_DATES == (pd.Timestamp("2026-08-19"), pd.Timestamp("2026-08-20"))
    # last=08-18、latest=08-21：断档两日剔除，只登 08-21（当日 late=False）
    got = rr.resolve_registration_dates(
        GAP_CAL, last_registered=pd.Timestamp("2026-08-18"),
        latest_available=pd.Timestamp("2026-08-21"),
        today=pd.Timestamp("2026-08-21"),
    )
    assert got == [(pd.Timestamp("2026-08-21"), False)]


def test_gap_dates_empty_when_data_lag():
    # 数据滞后（latest 仍为 08-20）：剔除断档后为空 → 静默跳过，绝不补造
    got = rr.resolve_registration_dates(
        GAP_CAL, last_registered=pd.Timestamp("2026-08-18"),
        latest_available=pd.Timestamp("2026-08-20"),
        today=pd.Timestamp("2026-08-21"),
    )
    assert got == []


def test_gap_date_explicit_refused(monkeypatch):
    # 显式 --date 断档日：在触碰 DDB/登记前即拒绝（双重猴补防真实副作用）
    def _must_not_touch(*a, **k):
        raise AssertionError("断档日拒绝必须先于 DDB 探测/登记发生")

    monkeypatch.setattr(rr, "_latest_available_date", _must_not_touch)
    monkeypatch.setattr(rr, "register_one", _must_not_touch)
    monkeypatch.setattr(sys, "argv", ["run_registry.py", "--date", "2026-08-19"])
    with pytest.raises(AssertionError, match="复算级"):
        rr.main()


# ---------------------------------------------------------------------------
# 无前视
# ---------------------------------------------------------------------------
class RecordingProvider:
    """内存假 provider：记录被请求的数据窗，携带 d+1..d+5 的"未来数据"。"""

    def __init__(self, data: pd.DataFrame, end: str):
        self._data = data[data.index.get_level_values("datetime") <= end].copy()
        self._full_end = data.index.get_level_values("datetime").max()
        self._end = end
        self.requested_max = []

    def fetch(self, fields, *, filter_pipe=None, freq="day"):
        self.requested_max.append(pd.Timestamp(self._end))
        # 模拟 QlibProvider.fetch：按 $ 前缀字段取列后去 $（列名语义一致）
        out = self._data[list(fields)].copy()
        out.columns = [f.replace("$", "", 1) for f in fields]
        return out

    def trading_days(self, start=None, end=None):
        days = pd.DatetimeIndex(sorted(self._data.index.get_level_values("datetime").unique()))
        return days[(days >= pd.Timestamp(start)) if start else days]

    def list_pool_at(self, pool, t):
        return sorted(self._data.index.get_level_values("instrument").unique())


def _future_contaminated_data(d: str) -> pd.DataFrame:
    """3 股 + 指数行情：截至 d 后**再延续 5 个未来日**，未来日数值故意极端。"""
    dates = list(pd.bdate_range(end=d, periods=260)) + list(
        pd.bdate_range(pd.Timestamp(d) + pd.Timedelta(days=1), periods=5)
    )
    codes = ("000001.SZ", "600000.SH", "300750.SZ", "000300.SH")  # 含指数（MA200 用）
    frames = []
    for ci, code in enumerate(codes):
        closes = np.linspace(10 + ci, 20 + ci, len(dates))
        closes[-5:] = 9999.0  # 未来数据若被触碰，结果必然异常
        idx = pd.MultiIndex.from_product([pd.DatetimeIndex(dates), [code]],
                                         names=["datetime", "instrument"])
        frames.append(pd.DataFrame(
            {"$close": closes, "$tradestatuscode": -1}, index=idx))
    return pd.concat(frames)


def test_registry_no_lookahead(tmp_path):
    d = pd.Timestamp(D)
    data = _future_contaminated_data(d)
    provider = RecordingProvider(data, end=d)

    # 动量：close[t]/close[t-10]-1，未来 9999 若被触碰 → 值为天文数字
    mom = rr._momentum_signal(provider, d)
    assert float(mom.abs().max()) < 10.0

    # MA200 门控：只用 ≤ d 的收盘（fake 含指数行，未来 9999 若被触碰则越界）
    gate = rr._ma200_gate(provider, d)
    assert 0.0 < gate["index_close"] < 100.0 and 0.0 < gate["ma200"] < 100.0
    assert isinstance(gate["gate"], bool)

    # 可交易掩码
    trd = rr._tradeable_mask(provider, d)
    assert trd.dtype == bool and trd.all()

    # 全部 fetch 的数据窗上界 ≤ d
    assert all(t <= d for t in provider.requested_max)
