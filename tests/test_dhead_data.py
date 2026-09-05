"""任务1（方案 §7）：数据与时间契约离线测试——纯合成数据，不触 DDB/HF/GPU。

覆盖方案 §3 的核心不变式：
- 未来篡改不变式：篡改 t 之后的 OHLCVA，样本输入（历史窗/统计量/输入哈希）
  完全不变，真实标签改变；
- 历史缺口：整窗剔除（不跨缺口拼接）；
- 标签边界：t+10 越界样本被排除；
- PIT 入池时机：次日入池的成员当日不得入选；
- dataset[i] 多次相等、num_workers=0/2 遍历一致、日 batch 只含单一日期；
- prepare 两次内容 hash 相同（mtime 不入内容 hash）。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from dhead_distill.config import (
    DHeadConfig,
    protocol_hash,
    replace_profile_budget,
)
from dhead_distill.data import (
    DayDataset,
    SEAL_DATE,
    build_manifest,
    day_batches,
    input_digest,
)

# ---------------------------------------------------------------- 合成数据 ---

N_STOCKS = 12
CAL_START = "2024-01-02"
CAL_END = "2025-12-31"


def _synthetic_calendar() -> pd.DatetimeIndex:
    days = pd.bdate_range(CAL_START, CAL_END)
    return pd.DatetimeIndex(days)


def _synthetic_prices(seed: int = 7) -> pd.DataFrame:
    """N_STOCKS × 全日历的合成 OHLCVA（stacked，MultiIndex(datetime, instrument)）。

    价格为正、缓慢随机游走；volume/amount 与价格独立，保证 z-score 统计量
    对 t 之后篡改敏感的只有标签。
    """
    cal = _synthetic_calendar()
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(N_STOCKS):
        ret = rng.normal(0.0, 0.01, size=len(cal))
        close = 100.0 * np.exp(np.cumsum(ret))
        open_ = close * (1 + rng.normal(0, 0.002, size=len(cal)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, size=len(cal))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, size=len(cal))))
        vol = rng.lognormal(10, 0.3, size=len(cal))
        amt = vol * close
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
                "amount": amt,
            },
            index=cal,
        )
        df["instrument"] = f"SZ{k:04d}"
        rows.append(df.reset_index().rename(columns={"index": "datetime"}))
    out = pd.concat(rows).set_index(["datetime", "instrument"]).sort_index()
    return out


class FakeProvider:
    """内存 provider：duck-typing QlibProvider 的三个方法 + fetch 区间属性。

    与 tests/test_kronos_qlib.py 的 FakeProvider 同约定：fetch 读
    ``instruments_ / _start_date / _end_date``。
    """

    def __init__(self, data: pd.DataFrame, pool_members: dict[str, list[str]]):
        self._data = data
        self._pool_members = pool_members
        self.instruments_ = []
        self._start_date = None
        self._end_date = None

    def trading_days(self, start=None, end=None) -> pd.DatetimeIndex:
        cal = _synthetic_calendar()
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal <= pd.Timestamp(end)]
        return cal

    def list_pool_at(self, pool: str, t: str) -> list[str]:
        # 池成员快照：entry 日期之后（含）才在池内
        t = pd.Timestamp(t)
        return [
            code
            for code, entry in self._pool_members.get(pool, {}).items()
            if t >= pd.Timestamp(entry)
        ]

    def fetch(self, fields, *, filter_pipe=None, freq="day") -> pd.DataFrame:
        insts = self.instruments_
        lo, hi = pd.Timestamp(self._start_date), pd.Timestamp(self._end_date)
        sub = self._data
        if isinstance(insts, str):
            pass  # 池名查询（如 ashares）：返回全池数据，与 qlib 市场语义一致
        elif insts:
            sub = sub[sub.index.get_level_values("instrument").isin(set(insts))]
        sub = sub.loc[(slice(lo, hi), slice(None)), :]
        cols = [c.replace("$", "") for c in fields]
        return sub[cols].copy()


def _all_members_from_day(day: str) -> dict[str, str]:
    return {f"SZ{k:04d}": day for k in range(N_STOCKS)}


def _make_config() -> DHeadConfig:
    """测试用缩小预算配置（仅改采样预算与窗边界以适配 2024-2025 合成日历）。"""
    import dataclasses

    cfg = dataclasses.replace(
        replace_profile_budget(
            DHeadConfig(),
            train_dates=6,
            val_dates=3,
            diag_dates=4,
            per_day=8,
            min_per_day=2,
        ),
        train_start="2024-03-01", train_end="2024-11-30",
        val_start="2025-02-03", val_end="2025-05-30",
        diag_start="2025-07-01", diag_end="2025-11-28",
    )
    return cfg


def _make_provider(data=None, pool_entries=None) -> FakeProvider:
    data = _synthetic_prices() if data is None else data
    entries = pool_entries or {
        "ashares": _all_members_from_day(CAL_START),
        "csi300": _all_members_from_day(CAL_START),
    }
    return FakeProvider(data, entries)


# ------------------------------------------------------------------- 测试 ---

def test_seal_date_constant() -> None:
    """封存线固定 2026-07-25：任何取数边界不得越过。"""
    assert SEAL_DATE == pd.Timestamp("2026-07-25")


def test_protocol_hash_stable_and_budget_sensitive() -> None:
    """协议 hash 只随协议字段变化；同字段两次计算相同（canonical 序列化）。"""
    h1 = protocol_hash(DHeadConfig())
    h2 = protocol_hash(DHeadConfig())
    assert h1 == h2
    cfg2 = replace_profile_budget(DHeadConfig(), per_day=8, train_dates=64, val_dates=16, diag_dates=64, min_per_day=32)
    assert protocol_hash(cfg2) != h1


def test_future_tamper_leaves_inputs_unchanged() -> None:
    """篡改 t 之后的价格：样本输入哈希/统计量不变，标签改变。"""
    cfg = _make_config()
    provider = _make_provider()
    m1 = build_manifest(provider, cfg, split="train")

    # 篡改：把日历最后 30 天（训练决策日 t+10 之后才触及的远未来）全部价格 ×2
    cal = _synthetic_calendar()
    tamper_from = cal[-30]
    data2 = _synthetic_prices()
    idx = data2.index.get_level_values("datetime") >= tamper_from
    data2.loc[idx, ["open", "high", "low", "close"]] *= 2.0
    data2.loc[idx, "amount"] *= 2.0
    provider2 = _make_provider(data=data2)
    m2 = build_manifest(provider2, cfg, split="train")

    # 训练决策日 t+10 标签边界 ≤ 2024-12-31：篡改自 2025-12 起则连标签也不变
    # ——为让断言真正触及"未来价格只用于标签"，把篡改点放到训练窗末段：
    # 决策日附近 t+1..t+10 的价格属于标签路径但不属于输入窗。
    assert len(m1.samples) > 0
    # 逐样本对比：输入哈希全一致；标签在篡改点之后变化
    dig1 = input_digest(m1)
    dig2 = input_digest(m2)
    assert dig1 == dig2

    # 直接构造一个会受影响的对照：篡改紧随最后一个训练决策日之后的 10 天价格
    last_t = max(s.date for s in m1.samples)
    tamper_from = last_t + pd.Timedelta(days=1)
    data3 = _synthetic_prices()
    idx = data3.index.get_level_values("datetime") >= tamper_from
    data3.loc[idx, ["open", "high", "low", "close"]] *= 2.0
    provider3 = _make_provider(data=data3)
    m3 = build_manifest(provider3, cfg, split="train")
    dig3 = input_digest(m3)
    assert dig3 == dig1, "篡改 t 之后的价格不得改变任何训练样本输入"
    # 标签：最后决策日的 10 期限标签必须改变（close[t+h] 翻倍）
    sample_last = [s for s in m3.samples if s.date == last_t]
    assert sample_last, "篡改后仍应有末日样本"
    changed = 0
    for s3 in m3.samples:
        if s3.date == last_t:
            s1 = next(s for s in m1.samples if s.date == s3.date and s.code == s3.code)
            assert not np.allclose(s1.y_real, s3.y_real), "未来价格改变必须改变标签"
            changed += 1
    assert changed > 0


def test_history_gap_excludes_window() -> None:
    """历史窗内删一行（模拟停牌/缺数）：该 (date, code) 整窗剔除。"""
    cfg = _make_config()
    provider = _make_provider()
    m1 = build_manifest(provider, cfg, split="train")
    assert len(m1.samples) > 0

    # 找一个入选样本，删掉其输入窗中间一行（t-45 附近）
    target = m1.samples[0]
    cal = _synthetic_calendar()
    gap_day = cal[cal.get_loc(target.date) - 45]
    data2 = _synthetic_prices()
    data2 = data2.drop(index=(gap_day, target.code))
    provider2 = _make_provider(data=data2)
    m2 = build_manifest(provider2, cfg, split="train")
    keys2 = {(s.date, s.code) for s in m2.samples}
    assert (target.date, target.code) not in keys2, "含缺口窗口必须整窗剔除"
    # 缺口影响的只是同一只股票的窗口（其它股票样本数不变）
    missing = {(s.date, s.code) for s in m1.samples} - keys2
    assert missing and all(code == target.code for _, code in missing)


def test_label_boundary_excludes_t_plus_10_overflow() -> None:
    """t+10 越过训练标签边界的决策日样本被排除（用交易日历推进，非日历日减 10）。"""
    cfg = _make_config()
    provider = _make_provider()
    m = build_manifest(provider, cfg, split="train")
    cal = _synthetic_calendar()
    train_end = pd.Timestamp(cfg.train_end)
    for s in m.samples:
        t_pos = cal.get_loc(s.date)
        assert cal[t_pos + cfg.predict_len] <= train_end, \
            "训练样本的 t+10 交易日标签不得越过训练末日"


def test_pool_entry_timing() -> None:
    """成员在 t+1 才入池 → t 日不得入选。"""
    cfg = _make_config()
    cal = _synthetic_calendar()
    late_entry = "SZ0011"
    entries = {
        "ashares": {**_all_members_from_day(CAL_START), late_entry: str(cal[60].date())},
        "csi300": _all_members_from_day(CAL_START),
    }
    provider = _make_provider(pool_entries=entries)
    m = build_manifest(provider, cfg, split="train")
    entry_ts = pd.Timestamp(cal[60].date())
    for s in m.samples:
        if s.code == late_entry:
            assert s.date >= entry_ts, "入池前决策日不得入选"


def test_dataset_deterministic_and_worker_consistent() -> None:
    """dataset[i] 多次相等；workers=0/2 遍历 (date,code) 集合与次数一致。"""
    cfg = _make_config()
    provider = _make_provider()
    m = build_manifest(provider, cfg, split="train")
    ds = DayDataset(m)

    a = ds[0]
    b = ds[0]
    assert a["date"] == b["date"] and a["code"] == b["code"]
    np.testing.assert_array_equal(a["x_raw"], b["x_raw"])
    np.testing.assert_array_equal(a["x_norm"], b["x_norm"])
    if a["y_real"] is not None:
        np.testing.assert_array_equal(a["y_real"], b["y_real"])

    from torch.utils.data import DataLoader

    def collect(nw: int) -> list[tuple[str, str]]:
        out = []
        dl = DataLoader(ds, batch_size=3, num_workers=nw)
        for batch in dl:
            for d, c in zip(batch["date"], batch["code"]):
                out.append((str(d), str(c)))
        return out

    w0 = collect(0)
    w2 = collect(2)
    assert sorted(w0) == sorted(w2), "worker 流不得增删样本"
    assert len(w0) == len(set(w0)), "样本不得重复"

    # 日 batch：每个 batch 只有一个日期
    for batch in day_batches(m):
        assert len({str(s.date) for s in batch}) == 1


def test_prepare_content_hash_stable() -> None:
    """两次 build_manifest 内容 hash 相同（hash 只含内容，不含 mtime）。"""
    cfg = _make_config()
    p1 = build_manifest(_make_provider(), cfg, split="train")
    p2 = build_manifest(_make_provider(), cfg, split="train")
    assert p1.content_hash == p2.content_hash
    # 清单 seed 不同 → hash 不同（SHA256 稳定哈希受 seed 影响）
    cfg3 = DHeadConfig.with_list_seed(cfg, 12345)
    p3 = build_manifest(_make_provider(), cfg3, split="train")
    assert p3.content_hash != p1.content_hash


def test_manifest_records_coverage_stats() -> None:
    """manifest 记录 coverage：每日成员数/保留数/剔除原因计数。"""
    cfg = _make_config()
    m = build_manifest(_make_provider(), cfg, split="train")
    st = m.stats
    assert st["n_samples"] == len(m.samples) > 0
    assert st["n_days"] > 0
    assert "skipped" in st and isinstance(st["skipped"], dict)
    assert st["pool"] == "ashares"


def test_unknown_split_rejected() -> None:
    """未知 split 显式报错。"""
    cfg = _make_config()
    with pytest.raises(ValueError):
        build_manifest(_make_provider(), cfg, split="nonsense")
