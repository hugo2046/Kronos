"""市场上下文三列的因果计算（G4 计划 §1，冻结定义）。

| 列 | 定义 | 说明 |
|---|---|---|
| ``idx_ret`` | 000300.SH 当日收益率（close-to-close） | 每日全截面同值 |
| ``mkt_vol`` | 指数收益率 20 日滚动 std × √252（年化） | 波动 regime |
| ``ma200_gate`` | 指数收盘 > MA200 的 0/1 | 与 G3 登记的 MA200 门控同口径
  （``run_registry._ma200_gate``：close > mean(最近 200 个收盘)） |

全部为 trailing 滚动（只依赖 ≤d 的指数数据，无前视；单测钉死因果性）。

预热（执行说明，跑前定案）：DDB 日频地板 = 2014-01-02（**含指数**，2026-08-17
实测 000300.SH/000001.SH/399001.SZ 在 2013 全空），计划"MA200 预热用 2013 年起
指数数据"无法从 DDB 满足 → 用腾讯公开接口拉 2013 段指数收盘，**与 DDB
2014-01-02~2014-06-30 重叠段逐日对拍验证口径后**仅取 2014 前部分作预热，
落盘 ``g4_features/data/index_warmup_2013.csv``（入库工件，网络只需一次）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MARKET_COLS = ["idx_ret", "mkt_vol", "ma200_gate"]

# 训练语料首日（G1 ashares pkl 实测下界；ma200_gate 在该日必须可算）
TRAIN_FIRST_DAY = pd.Timestamp("2014-01-02")

CSI300_INDEX = "000300.SH"

# DDB 日频地板（实测）：预热必须覆盖到该日之前
DDB_DAILY_FLOOR = pd.Timestamp("2014-01-02")

_WARMUP_CSV = Path(__file__).resolve().parent / "data" / "index_warmup_2013.csv"


def compute_market_context(index_close: pd.Series) -> pd.DataFrame:
    """由指数收盘序列计算市场三列（纯函数，trailing 无前视）。

    :param index_close: DatetimeIndex 索引的指数收盘价（升序）。
    :returns: 三列 DataFrame，头部 NaN 保留（idx_ret 首 1 日 / mkt_vol 首 19 日
        / ma200_gate 首 199 日不可算）——由调用方断言所需日期已覆盖。
    """
    close = index_close.sort_index().astype(float)
    ret = close.pct_change(fill_method=None)
    mkt_vol = ret.rolling(20).std() * np_sqrt252()
    ma200 = close.rolling(200).mean()
    gate = (close > ma200).astype(float)
    gate[ma200.isna()] = float("nan")
    return pd.DataFrame(
        {"idx_ret": ret, "mkt_vol": mkt_vol, "ma200_gate": gate}
    )


def np_sqrt252() -> float:
    import math

    return math.sqrt(252.0)


def load_warmup_csv(path: Path | None = None) -> pd.Series:
    """读预热 CSV（date,close 两列）→ 升序收盘序列。"""
    p = path or _WARMUP_CSV
    df = pd.read_csv(p, parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def build_index_series(provider, end_date: str, warmup_csv: Path | None = None) -> pd.Series:
    """预热 CSV（2013 段）+ DDB（2014-01-02 起）拼成完整指数收盘序列。

    拼接处做连续性断言：无重复日期、预热末日 < DDB 首日、首日可算 MA200
    （预热起点距 TRAIN_FIRST_DAY ≥ 200 个交易日）。
    """
    warm = load_warmup_csv(warmup_csv)
    from kronos_qlib import QlibProvider

    p = QlibProvider([CSI300_INDEX], str(DDB_DAILY_FLOOR.date()), end_date)
    raw = p.fetch(["$close"])
    if len(raw) == 0:
        raise RuntimeError(f"{CSI300_INDEX} 在 DDB {DDB_DAILY_FLOOR.date()}~{end_date} 无数据")
    ddb = raw.xs(CSI300_INDEX, level="instrument")["close"].sort_index()

    assert warm.index.max() < ddb.index.min(), (
        f"预热段（末 {warm.index.max().date()}）须整体早于 DDB 段（首 {ddb.index.min().date()}）"
    )
    full = pd.concat([warm, ddb])
    assert not full.index.duplicated().any(), "指数序列存在重复日期"
    n_before_train = int((full.index < TRAIN_FIRST_DAY).sum())
    assert n_before_train >= 200, (
        f"train 首日前仅 {n_before_train} 个指数交易日，不足以算 MA200"
    )
    return full
