"""把 Kronos 预测结果整合为单张整洁长表 parquet，便于后续查验（计划外交付，用户要求）。

输入（union 宽表，index=date, columns=code）：
    daily_signals_K_{last,mean,max,min}_union.parquet（4 变体）
    daily_signals_{M,R,P}_union.parquet
    strat_membership.parquet（月末 date/bucket/st_track/code）
    baseline_suite paper+oos mean（csi300 对照 K_mean）
    pred_close_path_union.parquet（原始 close 路径，推理直接落盘，本脚本不重复导出）

输出 ``liquidity_strat/data/kronos_predictions.parquet``（长表，一行一个 日期-股票-轨）：
    datetime | instrument | st_track | bucket |
    signal_K_last | signal_K_mean | signal_K_max | signal_K_min |
    signal_M | signal_R | signal_P

- 4 个 K 变体由原始预测 close 路径聚合（除以现价 close_t）：
  last=末步、mean=均值（canonical）、max、min。
- 原始 close 路径（H=10 步）见 ``pred_close_path_union.parquet``（datetime/instrument/horizon/pred_close）。
- 信号值是 (日期, 股票) 内禀的，与档无关；同一 (日期, 股票) 在 exst/withst 两轨下
  各一行（档标签可能不同，因 ST 剔除改变百分位排名）。
- 成员按"最近月末"前向填充到每个交易日（PIT）。
- csi300 对照档只有 K_mean（复用 baseline_suite），其余变体/对照为 NaN。

注：parquet 受 .gitignore 排除（计划 §4"信号 parquet 不入库"），仅落盘本机供查验。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from liquidity_strat.common import DATA_DIR, HIGH_LIQ_BUCKET, ST_TRACK_MAIN

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = DATA_DIR / "kronos_predictions.parquet"

# K 四变体 + M/R/P 对照。csi300 仅 K_mean（复用 baseline_suite）。
UNION = {
    "K_last": DATA_DIR / "daily_signals_K_last_union.parquet",
    "K_mean": DATA_DIR / "daily_signals_K_mean_union.parquet",
    "K_max": DATA_DIR / "daily_signals_K_max_union.parquet",
    "K_min": DATA_DIR / "daily_signals_K_min_union.parquet",
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

    # 1) K 四变体 + M/R/P 长表合并
    sig_long = None
    for tag, p in UNION.items():
        if not p.exists():
            raise FileNotFoundError(f"union 信号缺失：{p}（先跑 run_signals kronos/baselines）")
        s = _melt_wide(p, f"signal_{tag}")
        sig_long = s if sig_long is None else sig_long.merge(s, on=["datetime", "instrument"], how="outer")
    logger.info(f"K 四变体 + M/R/P 合并长表：{len(sig_long)} 行")

    # 2) 日频成员（3 档 × 2 轨）
    memb = _daily_membership(strat)
    logger.info(f"日频成员表：{len(memb)} 行")

    # 3) 信号 × 成员（left join 成员 → 每个信号行带上其档/轨标签）
    out = memb.merge(sig_long, on=["datetime", "instrument"], how="left")
    # 丢弃 K 四变体全空的成员行（非 union 成员）
    k_cols = ["signal_K_last", "signal_K_mean", "signal_K_max", "signal_K_min"]
    out = out.dropna(subset=k_cols, how="all")

    # 4) 追加 csi300 对照 K_mean（复用 baseline_suite mean）
    csi_p = REPO_ROOT / "baseline_suite" / "data" / "daily_signals_paper_mean.parquet"
    csi_o = REPO_ROOT / "baseline_suite" / "data" / "daily_signals_oos_mean.parquet"
    if csi_p.exists() and csi_o.exists():
        csi = pd.read_parquet(csi_p)
        csio = pd.read_parquet(csi_o)
        csi.index = pd.to_datetime(csi.index)
        csio.index = pd.to_datetime(csio.index)
        cw = pd.concat([csi, csio])
        cw = cw[~cw.index.duplicated(keep="last")]
        cl = cw.stack(dropna=True).rename("signal_K_mean").reset_index()
        cl.columns = ["datetime", "instrument", "signal_K_mean"]
        cl["st_track"] = ST_TRACK_MAIN
        cl["bucket"] = HIGH_LIQ_BUCKET
        # csi300 仅 mean；其余变体与对照为空
        for c in ["signal_K_last", "signal_K_max", "signal_K_min", "signal_M", "signal_R", "signal_P"]:
            cl[c] = pd.NA
        out = pd.concat([out, cl], ignore_index=True)
        logger.info(f"追加 csi300 对照 K_mean：{len(cl)} 行")

    # 列序 + 排序
    cols = ["datetime", "instrument", "st_track", "bucket",
            "signal_K_last", "signal_K_mean", "signal_K_max", "signal_K_min",
            "signal_M", "signal_R", "signal_P"]
    out = out[cols]
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
    print("\n=== 抽样查验：lo5/exst 某日 K 四变体 + 对照 前5 ===")
    d = df["datetime"].min()
    print(df[(df.bucket == "lo5") & (df.st_track == "exst") & (df.datetime == d)].head(5).to_string())
    print(f"\n维度：{df.shape} | 日期 {df.datetime.min().date()}..{df.datetime.max().date()}")
    print(f"档/轨分布（行数）：\n{df.groupby(['st_track','bucket']).size()}")
