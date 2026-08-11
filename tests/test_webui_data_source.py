"""webui 数据层（webui.data_source）测试 + app 全链路集成测试。

单测（无需 DolphinDB）：FakeProvider 注入模式，覆盖计划 §3 验收 1-5。
集成测试（marker=integration，无 DOLPHINDB_URI 时 skipif 跳过）：覆盖 §3 验收 6-8。
人工冒烟（验收 9）由 zcode 起 ``python run.py`` 单独执行，不在本文件内。
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from webui import data_source
from webui.data_source import OHLCVA_COLS, OUTPUT_COLS, fetch_ohlcva, future_trading_days, validate_code

# 与 test_kronos_qlib.py 同口径的 DDB 可用性探测
from kronos_qlib import provider as provider_module

_HAS_DDB = bool(os.environ.get("DOLPHINDB_URI")) or bool(
    provider_module._qlib_config.database_uri
)
integration = pytest.mark.integration
integration_skip = pytest.mark.skipif(not _HAS_DDB, reason="未配置 DOLPHINDB_URI，跳过集成测试")


# ----------------------------- FakeProvider ----------------------------------
class FakeProvider:
    """单测用的内存 provider：duck-typing 等价 QlibProvider。

    只实现 data_source 用到的 fetch / trading_days / list_pool_at，以及
    _scoped_fetch 临时改写的 _start_date / _end_date / instruments_ 三个属性。
    风格沿用 tests/test_kronos_qlib.py 的 FakeProvider。
    """

    def __init__(self, data: pd.DataFrame, members: list[str]):
        # data: MultiIndex(datetime, instrument)，列含 $open 等（模拟 qlib 原样）
        self._data = data
        self.instruments_ = members
        self._start_date = None
        self._end_date = None

    def fetch(self, fields, *, filter_pipe=None, freq="day"):
        df = self._data.copy()
        if self._start_date is not None:
            df = df[df.index.get_level_values("datetime") >= pd.Timestamp(self._start_date)]
        if self._end_date is not None:
            df = df[df.index.get_level_values("datetime") <= pd.Timestamp(self._end_date)]
        df = df[list(fields)]
        df.columns = df.columns.str.replace("$", "", regex=False)  # 模拟 provider.fetch 去 $
        return df

    def trading_days(self, start=None, end=None):
        cal = pd.DatetimeIndex(
            sorted(self._data.index.get_level_values("datetime").unique())
        )
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal <= pd.Timestamp(end)]
        return cal

    def list_pool_at(self, pool, t):
        return list(self.instruments_) if isinstance(self.instruments_, list) else []


def _build_fake_data(codes, dates, amount_fn=None):
    """构造 FakeProvider 用的 MultiIndex DataFrame，模拟 qlib 原样输出。

    :param amount_fn: 可选，``f(i) -> amount``；默认 amount = 10000.0（常数）。
        验收 5 用它注入可辨识的 amount 值，证明直传非合成。
    """
    field_cols = [
        "$open", "$high", "$low", "$close", "$volume", "$amount", "$tradestatuscode",
    ]
    records, idx = [], []
    for code in codes:
        for i, d in enumerate(dates):
            base = 10.0 + i * 0.01
            amt = amount_fn(i) if amount_fn else 10000.0
            records.append([base, base + 0.1, base - 0.1, base, 1000.0, amt, -1])
            idx.append((pd.Timestamp(d), code))
    return pd.DataFrame(
        records, columns=field_cols,
        index=pd.MultiIndex.from_tuples(idx, names=["datetime", "instrument"]),
    )


# ===========================================================================
# 单元测试（无需 DDB）
# ===========================================================================

def test_fetch_ohlcva_columns_and_order():
    """验收 1：fetch_ohlcva 返回恰好七列、列序固定、timestamps 严格递增。"""
    dates = pd.date_range("2024-01-01", periods=40, freq="B").strftime("%Y-%m-%d")
    fake = FakeProvider(_build_fake_data(["A"], dates), ["A"])

    # 显式锚到 fake 数据末尾（end_date=None 会锚到 now()，2024 数据被滤掉）
    df = fetch_ohlcva("A", end_date=dates[-1], n_bars=20, _provider=fake)

    assert list(df.columns) == OUTPUT_COLS == ["timestamps"] + OHLCVA_COLS
    assert df.shape == (20, 7)
    assert df["timestamps"].is_monotonic_increasing


def test_fetch_ohlcva_short_history_returned_honestly():
    """验收 2：行数不足 n_bars → 如实返回（不填充、不报错）。"""
    dates = pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d")
    fake = FakeProvider(_build_fake_data(["B"], dates), ["B"])

    df = fetch_ohlcva("B", end_date=dates[-1], n_bars=20, _provider=fake)

    assert len(df) == 5  # 只有 5 行，如实返回，不填充到 20
    assert list(df.columns) == OUTPUT_COLS


def test_future_trading_days_skips_weekend_and_holiday():
    """验收 3：future_trading_days 跨周末 / 假期正确（fake 日历含非交易日间隙）。

    日历：Thu 01-04, Fri 01-05, [跳过周末], Mon 01-08, [跳过 Tue 当假期], Wed 01-10。
    after=Thu 01-04 取 3 个 → 应为 [Fri, Mon, Wed]，不含任何周末 / 假期。
    """
    dates = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-10"]  # Thu, Fri, Mon, Wed
    fake = FakeProvider(_build_fake_data(["A"], dates), ["A"])

    fut = future_trading_days("2024-01-04", 3, _provider=fake)

    assert len(fut) == 3
    assert fut.iloc[0] == pd.Timestamp("2024-01-05")  # Fri
    assert fut.iloc[1] == pd.Timestamp("2024-01-08")  # Mon（跳过周末）
    assert fut.iloc[2] == pd.Timestamp("2024-01-10")  # Wed（跳过 Tue 假期）
    # 严格递增且无周末
    assert fut.is_monotonic_increasing
    assert all(d.weekday() < 5 for d in fut)


def test_validate_code_out_of_pool():
    """验收 4：池外代码 → False；池内代码 → True。"""
    fake = FakeProvider(_build_fake_data(["A", "B"], ["2024-01-04"]), ["A", "B"])

    assert validate_code("A", _provider=fake) is True
    assert validate_code("XXX.SH", _provider=fake) is False


def test_amount_passed_through_to_predictor():
    """验收 5（核心）：amount 列直传 predictor，值来自数据层（非 volume×均价合成）。

    注入式验证：把 amount 设为可辨识值（远异于 volume×均价），fetch_ohlcva 取出后
    经 ``app.run_prediction`` 喂给 mock predictor，断言收到的 df 含 amount 列且值与
    数据层一致；并断言该值 ≠ volume×均价（合成分支的产出），证明本断言有判别力。
    """
    from webui.app import run_prediction

    # 25 个交易日：前 15 个供窗口，后留 10 个供 future_trading_days 取"未来"。
    dates = pd.date_range("2024-01-01", periods=25, freq="B").strftime("%Y-%m-%d")
    # 可辨识 amount：100000 + i，远异于 volume(1000) × 均价(~10.x) ≈ 10500
    fake = FakeProvider(
        _build_fake_data(["A"], dates, amount_fn=lambda i: 100000.0 + i), ["A"]
    )

    # 窗口锚到 dates[14]（留 dates[15..] 作未来），取末 10 根。
    window = fetch_ohlcva("A", end_date=dates[14], n_bars=10, _provider=fake)
    assert "amount" in window.columns

    x_ts = window["timestamps"].reset_index(drop=True)
    y_ts = future_trading_days(x_ts.iloc[-1], 5, _provider=fake)

    mock_pred = MagicMock()
    mock_pred.predict.return_value = pd.DataFrame(
        np.full((5, 6), 10.0), columns=OHLCVA_COLS, index=y_ts,
    )

    run_prediction(
        window, mock_pred, x_ts, y_ts,
        pred_len=5, T=1.0, top_p=0.9, sample_count=1,
    )

    received = mock_pred.predict.call_args.kwargs["df"]
    # ① 收到的 df 含 amount 列
    assert "amount" in received.columns
    # ② amount 值来自数据层（逐行一致）
    assert np.array_equal(
        received["amount"].values, window["amount"].values
    )
    # ③ 判别力：收到的 amount ≠ volume × 均价（kronos.py:531 合成分支的产出）
    synthetic = received["volume"] * received[["open", "high", "low", "close"]].mean(axis=1)
    assert not np.allclose(received["amount"].values, synthetic.values), (
        "amount 与 volume×均价重合 → 断言无判别力（合成与直传不可区分）"
    )


# ===========================================================================
# 集成测试（需真实 DolphinDB）
# ===========================================================================

@integration
@integration_skip
def test_fetch_ohlcva_real_600000():
    """验收 6：fetch_ohlcva("600000.SH", n_bars=90) → 90 行、无 NaN、后复权价数百元。"""
    df = fetch_ohlcva("600000.SH", n_bars=90)
    assert df.shape[0] == 90
    assert list(df.columns) == OUTPUT_COLS
    assert not bool(df[OHLCVA_COLS].isna().any().any())
    # 后复权口径（陷阱 3）：600000.SH 现价 ~13 元，后复权 ~150-200 元
    assert df["close"].median() > 50, f"close 中位 {df['close'].median():.2f} 与后复权量级不符"
    assert df["timestamps"].is_monotonic_increasing


@pytest.fixture(scope="module")
def loaded_client():
    """模块级：加载一次 kronos-base 模型，供验收 7/8 复用。"""
    import torch
    from webui import app as app_module

    client = app_module.app.test_client()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    r = client.post("/api/load-model", json={"model_key": "kronos-base", "device": device})
    assert r.get_json()["success"], f"模型加载失败：{r.get_json()}"
    return client


@integration
@integration_skip
def test_predict_latest_mode_full_chain(loaded_client):
    """验收 7：Flask test client 全链路 load-data → predict（最新模式）。

    断言：prediction_results 恰 pred_len 条、close 有限正数且与窗口末值同数量级、
    时间戳全为交易日（无周末）。
    """
    import torch
    torch.manual_seed(42)

    client = loaded_client
    # ① load-data
    r = client.post("/api/load-data", json={"code": "600000.SH"})
    assert r.get_json()["success"], r.get_json()
    info = r.get_json()["data_info"]
    assert info["code"] == "600000.SH"
    assert info["frequency"] == "1 day"

    # ② predict（最新模式，无 anchor_date）
    r = client.post("/api/predict", json={
        "code": "600000.SH", "lookback": 90, "pred_len": 10,
        "temperature": 1.0, "top_p": 0.9, "sample_count": 1,
    })
    body = r.get_json()
    assert body["success"], body

    preds = body["prediction_results"]
    assert len(preds) == 10
    closes = [p["close"] for p in preds]
    assert np.all(np.isfinite(closes))
    assert all(c > 0 for c in closes)

    # 与窗口末值同数量级（反归一化正确性）：600000.SH 后复权 ~150-200
    last_window_close = info["price_range"]["max"]  # 粗量级参考
    for c in closes:
        assert 10 < c < 1000, f"close={c} 量级异常（窗口后复权价数百元）"
    assert abs(np.median(closes) - last_window_close) < last_window_close, (
        "预测中位与窗口末值差超 1 倍 → 反归一化可能异常"
    )

    # 时间戳全为交易日（无周末）——修 bug ① 的直接验证
    for p in preds:
        ts = pd.Timestamp(p["timestamp"])
        assert ts.weekday() < 5, f"预测时间戳 {ts} 落在周末"

    # 最新模式无对比段
    assert body["has_comparison"] is False


@integration
@integration_skip
def test_predict_historical_mode_with_comparison(loaded_client):
    """验收 8：历史回看模式 anchor_date=2025-06-16 → 有对比段，长度 = pred_len。"""
    import torch
    torch.manual_seed(42)

    client = loaded_client
    r = client.post("/api/predict", json={
        "code": "600000.SH", "anchor_date": "2025-06-16",
        "lookback": 90, "pred_len": 10,
        "temperature": 1.0, "top_p": 0.9, "sample_count": 1,
    })
    body = r.get_json()
    assert body["success"], body

    assert body["has_comparison"] is True
    actuals = body["actual_data"]
    assert len(actuals) == 10, f"对比段应满 10 根，实际 {len(actuals)}"
    assert len(body["prediction_results"]) == 10
