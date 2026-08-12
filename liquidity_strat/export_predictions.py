"""把 Kronos 预测结果整合为单张整洁长表 parquet，便于后续查验（计划外交付，用户要求）。

输入（union 宽表，index=date, columns=code）：
    daily_signals_{K,M,R,P}_union.parquet
    strat_membership.parquet（月末 date/bucket/st_track/code）
    baseline_suite paper+oos mean（csi300 对照 K）

输出 ``liquidity_strat/data/kronos_predictions.parquet``（长表，一行一个 日期-股票-轨）：
    datetime | instrument | st_track | bucket | signal_K | signal_M | signal_R | signal_P

- 信号值是 (日期, 股票) 内禀的，与档无关；同一 (日期, 股票) 在 exst/withst 两轨下
  各一行（档标签可能不同，因 ST 剔除改变百分位排名）。
- 成员按"最近月末"前向填充到每个交易日（PIT）。
- csi300 对照档 signal_M/R/P 为 NaN（本轮未对 csi300 生成对照信号）。

注：parquet 受 .gitignore 排除（计划 §4"信号 parquet 不入库"），仅落盘本机供查验。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from liquidity_strat.common import DATA_DIR, HIGH_LIQ_BUCKET, ST_TRACK_MAIN

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = DATA_DIR / "kronos_predictions.parquet"

UNION = {
    "K": DATA_DIR / "daily_signals_K_union.parquet",
    "M": DATA_DIR / "daily_signals_M_union.parquet",
    "R": DATA_DIR / "daily_signals_R_union.parquet",
    "P": DATA_DIR / "daily_signals_P_union.parquet",
}


def _melt_wide(path: Path, name: str) -> pd.DataFrame:
    wide = pd.read_parquet(path)
    wide.index = pd.to_datetime(wide.index)
    long = wide.stack(dropna=True).rename(name).reset_index()
    # 宽表 index 无名 → 命名
    long.columns = ["datetime", "instrument", name]
    return long


def _daily_membership(strat: pd.DataFrame) -> pd.DataFrame:
    """月末分档 → 每个交易日（前向填充到最近月末）。返回 (datetime, instrument, st_track, bucket)。"""
    reb_ts = pd.DatetimeIndex(sorted(strat["date"].unique()))

    def nearest_reb(d: pd.Timestamp) -> pd.Timestamp:
        le = reb_ts[reb_ts <= d]
        return le[-1] if len(le) else reb_ts[0]

    # 每个月末的 (track, bucket, code) 展开到该月所有交易日
    month_span: dict[pd.Timestamp, pd.DatetimeIndex] = {}
    for i, r in enumerate(reb_ts):
        nxt = reb_ts[i + 1] if i + 1 < len(reb_ts) else r + pd.Timedelta(days=45)
        month_span[r] = pd.date_range(r, nxt, freq="B")  # 工作日粗粒度，后面按信号日交集
    # 用信号表的实际交易日作为日历，避免引入非交易日
    K = pd.read_parquet(UNION["K"])
    K.index = pd.to_datetime(K.index)
    cal = pd.DatetimeIndex(K.index)

    parts = []
    for d in cal:
        src = nearest_reb(d)
        sub = strat[strat["date"] == src][["code", "st_track", "bucket"]].copy()
        sub["datetime"] = d
        parts.append(sub.rename(columns={"code": "instrument"}))
    return pd.concat(parts, ignore_index=True)


def build() -> Path:
    strat = pd.read_parquet(DATA_DIR / "strat_membership.parquet")
    strat["date"] = pd.to_datetime(strat["date"])

    # 1) 四信号长表合并
    sig_long = None
    for tag, p in UNION.items():
        if not p.exists():
            raise FileNotFoundError(f"union 信号缺失：{p}")
        s = _melt_wide(p, f"signal_{tag}")
        sig_long = s if sig_long is None else sig_long.merge(s, on=["datetime", "instrument"], how="outer")
    logger.info(f"四信号合并长表：{len(sig_long)} 行")

    # 2) 日频成员（3 档 × 2 轨）
    memb = _daily_membership(strat)
    logger.info(f"日频成员表：{len(memb)} 行")

    # 3) 信号 × 成员（left join 成员 → 每个信号行带上其档/轨标签）
    out = memb.merge(sig_long, on=["datetime", "instrument"], how="left")
    # 丢弃完全无信号的成员行（非 union 成员）
    out = out.dropna(subset=["signal_K"], how="all")

    # 4) 追加 csi300 对照 K（复用 baseline_suite mean）
    csi_p = REPO_ROOT / "baseline_suite" / "data" / "daily_signals_paper_mean.parquet"
    csi_o = REPO_ROOT / "baseline_suite" / "data" / "daily_signals_oos_mean.parquet"
    if csi_p.exists() and csi_o.exists():
        csi = pd.read_parquet(csi_p)
        csio = pd.read_parquet(csi_o)
        csi.index = pd.to_datetime(csi.index)
        csio.index = pd.to_datetime(csio.index)
        cw = pd.concat([csi, csio])
        cw = cw[~cw.index.duplicated(keep="last")]
        cl = cw.stack(dropna=True).rename("signal_K").reset_index()
        cl.columns = ["datetime", "instrument", "signal_K"]
        cl["st_track"] = ST_TRACK_MAIN
        cl["bucket"] = HIGH_LIQ_BUCKET
        for c in ("signal_M", "signal_R", "signal_P"):
            cl[c] = pd.NA
        out = pd.concat([out, cl], ignore_index=True)
        logger.info(f"追加 csi300 对照 K：{len(cl)} 行")

    # 列序 + 排序
    out = out[["datetime", "instrument", "st_track", "bucket", "signal_K", "signal_M", "signal_R", "signal_P"]]
    out = out.sort_values(["datetime", "st_track", "bucket", "instrument"]).reset_index(drop=True)

    out.to_parquet(OUT, index=False)
    logger.info(f"落盘：{OUT}（{len(out)} 行）")
    return out


if __name__ == "__main__":
    df = build()
    print("\n=== schema ===")
    print(df.dtypes)
    print("\n=== head ===")
    print(df.head(6).to_string())
    print("\n=== 抽样查验：lo5/exst 某日 signal_K 前5 ===")
    d = df["datetime"].min()
    print(df[(df.bucket == "lo5") & (df.st_track == "exst") & (df.datetime == d)].head(5).to_string())
    print(f"\n维度：{df.shape} | 日期 {df.datetime.min().date()}..{df.datetime.max().date()}")
    print(f"档/轨分布（行数）：\n{df.groupby(['st_track','bucket']).size()}")
