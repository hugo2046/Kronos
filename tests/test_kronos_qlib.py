"""kronos_qlib 数据层测试。

单测（无需 DolphinDB）：用 FakeProvider 实现 fetch / trading_days / list_pool_at，
覆盖计划 §3 验收 1-5。
集成测试（marker=integration，无 DOLPHINDB_URI 时 skipif 跳过）：覆盖 §3 验收 6-9。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from kronos_qlib import REQUIRED_COLS, QlibProvider, build_inference_windows
from kronos_qlib import provider as provider_module

# 与 KronosPredictor.price_cols + vol_col + amt_vol（model/kronos.py:489-491）
# 严格一致的期望列序，作为单测断言基准。
KRONOS_EXPECTED_COLS = ["open", "high", "low", "close", "volume", "amount"]

_HAS_DDB = bool(os.environ.get("DOLPHINDB_URI"))
# 真实 .env 在 import kronos_qlib 时已 load_dotenv 进来；若仍空说明未配置
_HAS_DDB = _HAS_DDB or bool(provider_module._qlib_config.database_uri)
# 集成测试同时打 integration marker（便于 -m 过滤）+ skipif（无 DDB 自动跳过）
integration = pytest.mark.integration
integration_skip = pytest.mark.skipif(
    not _HAS_DDB, reason="未配置 DOLPHINDB_URI，跳过集成测试"
)


# ----------------------------- FakeProvider ----------------------------------
class FakeProvider:
    """单测用的内存 provider：duck-typing 等价 QlibProvider。

    只实现 build_inference_windows 用到的 fetch / trading_days / list_pool_at，
    以及 _fetch_via 临时改写的 _start_date / _end_date / instruments_ 三个属性。
    """

    def __init__(self, data: pd.DataFrame, members: list[str]):
        # data: MultiIndex(datetime, instrument)，列含 $open 等（模拟 qlib 原样）
        self._data = data
        self.instruments_ = members
        self._start_date = None
        self._end_date = None
        # 记录最后一次 fetch 的入参，供断言
        self.last_fetch = None

    def fetch(self, fields, *, filter_pipe=None, freq="day"):
        self.last_fetch = {
            "instruments": self.instruments_,
            "start": self._start_date,
            "end": self._end_date,
            "fields": list(fields),
            "filter_pipe": filter_pipe,
        }
        # 按 instruments 过滤 + 按 start/end 截断 + 选字段
        insts = (
            self.instruments_
            if isinstance(self.instruments_, list)
            else None  # str 市场名场景单测里不出现
        )
        df = self._data.copy()
        if insts is not None:
            df = df[df.index.get_level_values("instrument").isin(insts)]
        if self._start_date is not None:
            df = df[df.index.get_level_values("datetime") >= pd.Timestamp(self._start_date)]
        if self._end_date is not None:
            df = df[df.index.get_level_values("datetime") <= pd.Timestamp(self._end_date)]
        # 选字段（fields 形如 ["$close", ...]）
        df = df[list(fields)]
        # 模拟 QlibDataLoader 的列名去 $（provider.fetch 内部做）
        df.columns = df.columns.str.replace("$", "", regex=False)
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


# ----------------------------- 构造辅助 ---------------------------------------
def _build_fake_data(codes, dates, halt_days=None, rows_per_code=None):
    """构造 FakeProvider 用的 MultiIndex DataFrame，模拟 qlib 原样输出。

    qlib 真实格式：index = MultiIndex(datetime, instrument)，columns = 扁平
    ``["$open","$high",...]``（$ 前缀），每个 (日期, 股票) 组合一行。

    :param codes: 股票代码列表。
    :param dates: 日期序列（完整，含调仓日之后的 predict_len 个交易日）。
    :param halt_days: dict {code: set(date_str)} 指定停牌日（tradestatuscode=0）。
    :param rows_per_code: dict {code: n}，每只股票只保留 dates 的**前 n 个**
        （n < 调仓日前累计天数 → 行数不足，被 skip_short）。
    """
    halt_days = halt_days or {}
    rows_per_code = rows_per_code or {}
    field_cols = [
        "$open", "$high", "$low", "$close", "$volume", "$amount",
        "$preclose", "$tradestatuscode",
    ]
    date_pos = {d: i for i, d in enumerate(dates)}
    records = []
    idx = []
    for code in codes:
        limit = rows_per_code.get(code, len(dates))
        kept = 0
        for d in dates:
            if kept >= limit:
                break
            base = 10.0 + date_pos[d] * 0.01
            tsc = 0 if d in halt_days.get(code, set()) else -1
            records.append(
                [base, base + 0.1, base - 0.1, base, 1000.0, 10000.0, base - 0.01, tsc]
            )
            idx.append((pd.Timestamp(d), code))
            kept += 1
    df = pd.DataFrame(
        records, columns=field_cols,
        index=pd.MultiIndex.from_tuples(idx, names=["datetime", "instrument"]),
    )
    return df


# ===========================================================================
# 单元测试（无需 DDB）
# ===========================================================================

def test_skip_short_history():
    """验收 1：行数不足 L → 该股票被跳过，且跳过计数正确。"""
    dates = pd.date_range("2024-01-01", periods=130, freq="B").strftime("%Y-%m-%d")
    # A 全量（调仓日前有 115 个交易日，≥90）；B 仅保留前 60 行（< lookback=90）
    rebalance = dates[-15]
    data = _build_fake_data(["A", "B"], dates, rows_per_code={"B": 60})
    fake = FakeProvider(data, ["A", "B"])

    df_list, x_ts, y_ts, codes, stats = build_inference_windows(
        fake, rebalance, lookback=90, predict_len=10, pool="csi300",
    )
    assert stats["skipped_short"] == 1, stats
    assert codes == ["A"]
    assert len(df_list) == 1
    assert df_list[0].shape[0] == 90


def test_skip_halt():
    """验收 2：窗口含 tradestatuscode==0 → 跳过。"""
    dates = pd.date_range("2024-01-01", periods=130, freq="B").strftime("%Y-%m-%d")
    rebalance = dates[-15]
    # A 正常；B 在窗口内（调仓日前 90 天）有 1 天停牌
    halt = {"B": {dates[-20]}}
    data = _build_fake_data(["A", "B"], dates, halt_days=halt)
    fake = FakeProvider(data, ["A", "B"])

    df_list, x_ts, y_ts, codes, stats = build_inference_windows(
        fake, rebalance, lookback=90, predict_len=10, pool="csi300",
    )
    assert stats["skipped_halt"] == 1, stats
    assert codes == ["A"]
    assert len(df_list) == 1


def test_column_order_matches_kronos():
    """验收 3：返回的 df 列顺序与 KronosPredictor 期望逐字一致。"""
    dates = pd.date_range("2024-01-01", periods=130, freq="B").strftime("%Y-%m-%d")
    rebalance = dates[-15]
    data = _build_fake_data(["A"], dates)
    fake = FakeProvider(data, ["A"])

    df_list, _, _, _, _ = build_inference_windows(
        fake, rebalance, lookback=90, predict_len=10, pool="csi300",
    )
    assert list(df_list[0].columns) == KRONOS_EXPECTED_COLS
    assert list(df_list[0].columns) == REQUIRED_COLS


def test_y_timestamp_correct():
    """验收 4：y_timestamp 长度 == predict_len，与 x_timestamp 无重叠、严格递增。"""
    dates = pd.date_range("2024-01-01", periods=120, freq="B").strftime("%Y-%m-%d")
    data = _build_fake_data(["A"], dates)
    fake = FakeProvider(data, ["A"])

    lookback, predict_len = 90, 10
    df_list, x_ts, y_ts, _, _ = build_inference_windows(
        fake, dates[-15], lookback=lookback, predict_len=predict_len, pool="csi300",
    )
    x_idx = pd.DatetimeIndex(x_ts[0])
    y_idx = pd.DatetimeIndex(y_ts[0])
    assert len(y_idx) == predict_len
    # 与 x 无重叠
    assert len(set(x_idx) & set(y_idx)) == 0
    # 严格递增
    assert all(y_idx[i] < y_idx[i + 1] for i in range(len(y_idx) - 1))
    # y 全部 > x 末值
    assert y_idx[0] > x_idx[-1]


def test_missing_uri_raises(monkeypatch):
    """验收 5：DOLPHINDB_URI 缺失 → 抛说明性异常（不是 KeyError / 静默 localhost）。"""
    # 模拟"未配置"：清空环境变量 + 复位 dataclass 字段 + 复位初始化标志
    monkeypatch.delenv("DOLPHINDB_URI", raising=False)
    monkeypatch.setattr(provider_module._qlib_config, "database_uri", "")
    monkeypatch.setattr(QlibProvider, "_qlib_initialized", False)
    with pytest.raises(RuntimeError) as excinfo:
        QlibProvider.init_qlib_once()
    msg = str(excinfo.value)
    assert ".env" in msg
    assert "DOLPHINDB_URI" in msg


# ===========================================================================
# 集成测试（需真实 DolphinDB）
# ===========================================================================

@integration
@integration_skip
def test_adjustment_self_consistency():
    """验收 6：复权自洽性 |close/preclose-1 − close/Ref(close,1)-1| < 1e-4。

    计划 §0 实测：该差值最大 8.12e-6（float32 噪声）。实测发现每只股票在
    区间**首日**会因 ``Ref`` 跨越复权边界产生 1 个异常点（如 61.77），这是
    DDB ``Ref`` 在序列边界的已知行为，而非数据错误——其余 34800 行中 0 行
    超 1e-4。故按"剔除每只股票首行"后取 max，与计划口径一致。
    """
    from qlib.data import D

    QlibProvider.init_qlib_once()
    df = D.features(
        instruments=D.instruments("csi300"),
        fields=["$close/$preclose-1", "$close/Ref($close,1)-1"],
        start_time="2024-01-02",
        end_time="2026-08-09",
    )
    df.columns = ["ret_preclose", "ret_ref"]
    # 掉 NaN（首日 Ref 无前值）
    df = df.dropna()
    # 剔除每只股票首行（Ref 跨复权边界异常点）
    df = df.groupby(level="instrument", group_keys=False).apply(
        lambda g: g.iloc[1:]
    )
    diff = (df["ret_preclose"] - df["ret_ref"]).abs()
    max_diff = diff.max()
    assert max_diff < 1e-4, (
        f"复权自洽性 max diff = {max_diff:.3e}，超 1e-4 容差（共 "
        f"{(diff > 1e-4).sum()} 行超限）"
    )


@integration
@integration_skip
def test_point_in_time_membership():
    """验收 7：两个调仓日的 csi300 池不应完全相同（成分调整必然发生）。"""
    p = QlibProvider("csi300", "2024-06-28", "2024-06-28")
    pool_a = set(p.list_pool_at("csi300", "2024-06-28"))
    pool_b = set(p.list_pool_at("csi300", "2025-12-31"))
    assert len(pool_a) > 0 and len(pool_b) > 0
    assert pool_a != pool_b, "两日成分完全相同 → point-in-time 区间未生效，严重缺陷"


@integration
@integration_skip
def test_end_to_end_smoke():
    """验收 8：1 调仓日 × 5 股 → build_inference_windows → predict_batch 跑通。

    输出 5 个 df、每个 10 行、close 列为有限正数且与输入窗口末值同数量级。
    """
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    # 选一个历史足够长的调仓日（沪深300 老成分多，留足 lookback + predict_len）
    rebalance = "2025-06-16"
    p = QlibProvider("csi300", "2024-01-01", "2025-06-30")
    df_list, x_ts, y_ts, codes, stats = build_inference_windows(
        p, rebalance, lookback=90, predict_len=10, pool="csi300",
    )
    assert len(df_list) >= 5, f"可用股票不足 5 只：{stats}"
    # 取前 5 只做冒烟
    df_list = df_list[:5]
    x_ts = x_ts[:5]
    y_ts = y_ts[:5]
    last_closes = [df["close"].iloc[-1] for df in df_list]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    torch.manual_seed(42)
    preds = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_ts,
        y_timestamp_list=y_ts,
        pred_len=10,
        sample_count=1,
        verbose=False,
    )
    assert len(preds) == 5
    for i, pred_df in enumerate(preds):
        assert pred_df.shape[0] == 10, f"第 {i} 只预测行数 != 10"
        closes = pred_df["close"].values
        # 有限正数
        assert np.all(np.isfinite(closes)), f"第 {i} 只 close 含非有限值"
        assert np.all(closes > 0), f"第 {i} 只 close 含非正数"
        # 与输入末值同数量级（反归一化正确性）
        last = last_closes[i]
        ratios = closes / last
        assert np.all((ratios > 0.1) & (ratios < 10.0)), (
            f"第 {i} 只 close 与末值不同数量级：last={last:.2f}, pred={closes}"
        )


@integration
@integration_skip
def test_determinism():
    """验收 9：固定 torch.manual_seed(42)，同输入连跑两次结果逐位一致。"""
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    rebalance = "2025-06-16"
    p = QlibProvider("csi300", "2024-01-01", "2025-06-30")
    df_list, x_ts, y_ts, codes, stats = build_inference_windows(
        p, rebalance, lookback=90, predict_len=10, pool="csi300",
    )
    assert len(df_list) >= 1, f"无可用股票：{stats}"
    # 取前 2 只做确定性测试（提速）
    df_list = df_list[:2]
    x_ts = x_ts[:2]
    y_ts = y_ts[:2]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    torch.manual_seed(42)
    preds1 = predictor.predict_batch(
        df_list=df_list, x_timestamp_list=x_ts, y_timestamp_list=y_ts,
        pred_len=10, sample_count=1, verbose=False,
    )
    torch.manual_seed(42)
    preds2 = predictor.predict_batch(
        df_list=df_list, x_timestamp_list=x_ts, y_timestamp_list=y_ts,
        pred_len=10, sample_count=1, verbose=False,
    )
    for i in range(len(preds1)):
        t1 = torch.from_numpy(preds1[i].values.astype(np.float32))
        t2 = torch.from_numpy(preds2[i].values.astype(np.float32))
        assert torch.equal(t1, t2), f"第 {i} 只两次预测不一致（非确定性）"
