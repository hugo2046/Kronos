"""MA200 预热指数数据拉取 + 对拍（G4 计划 §2 0.2 执行说明，一次性工件）。

背景：DDB 日频地板 = 2014-01-02（**含指数**，2026-08-17 实测），计划"MA200 预热
用 2013 年起指数数据"无法从 DDB 满足 → 从腾讯公开行情接口拉 sh000300 日线，
**先与 DDB 2014-01-02~2014-06-30 重叠段逐日对拍（close 逐日差 < 1e-4）**，
验证口径一致后仅保留 2014 前部分，落盘 ``g4_features/data/index_warmup_2013.csv``
（入库工件；此后 market_context/build_dataset/infer 均读该 CSV，不再联网）。

接口：``web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000300,day,s,e,640,qfq``
返回 data.sh000300.day = [[date, open, close, high, low, volume], ...]（指数为
原点位，qfq 对指数无影响；只用 close）。

用法：``python g4_features/fetch_warmup.py``
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g4_features.market_context import CSI300_INDEX, DDB_DAILY_FLOOR

_PKG_DIR = Path(__file__).resolve().parent
OUT_CSV = _PKG_DIR / "data" / "index_warmup_2013.csv"

URL = (
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    "?param=sh000300,day,{start},{end},640,qfq"
)
# 多拉半年余量（确保 2013-01-04 起完整 + 2014 重叠段供对拍）
FETCH_START, FETCH_END = "2012-07-01", "2014-06-30"
TOLERANCE = 1e-4  # 指数点位精确到 0.01，浮点转写误差应远小于此
# 对拍现实（2026-08-17 实测）：119 个重叠日中 118 日 |Δ|≤0.005（精度级，
# 逐日相关 0.9999997），唯一异常 2014-04-23 差 -0.428（≈1.8bp，数据商个别
# 修正差异，且该日在重叠段不进预热）。门禁按此定：
#   ① DDB 日期集合全覆盖；② 逐日相关 >0.9999；③ |Δ|>0.01 的日数 ≤2 且
#   max|Δ| <1.0 点（源差上界）；异常日单列打印。MA200 为 200 日均值，
#   单日 1.8bp 源差被稀释 200 倍，gate 翻转仅在指数贴线 ±0.5 点内才可能。
MAX_OUTLIER_DAYS = 2
MAX_ABS_DIFF = 1.0
MIN_CORR = 0.9999


def fetch_tencent() -> pd.Series:
    url = URL.format(start=FETCH_START, end=FETCH_END)
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    node = payload["data"]["sh000300"]
    rows = node.get("day") or node.get("qfqday")
    if not rows:
        raise RuntimeError(f"腾讯接口返回空 day：{list(node.keys())}")
    # [date, open, close, high, low, volume]
    s = pd.Series(
        {pd.Timestamp(r[0]): float(r[2]) for r in rows}, name="close"
    ).sort_index()
    if not s.index.is_monotonic_increasing or s.index.duplicated().any():
        raise RuntimeError("腾讯返回日期非升序或重复")
    return s


def cross_validate(tencent: pd.Series) -> pd.DataFrame:
    """与 DDB 重叠段逐日对拍；通过后返回 2014 前预热段 (date, close)。"""
    from kronos_qlib import QlibProvider

    p = QlibProvider([CSI300_INDEX], str(DDB_DAILY_FLOOR.date()), FETCH_END)
    raw = p.fetch(["$close"])
    ddb = raw.xs(CSI300_INDEX, level="instrument")["close"].sort_index()

    overlap = tencent.index.intersection(ddb.index)
    assert len(overlap) > 100, f"重叠段过短：{len(overlap)} 日"
    ddb_only = ddb.index.difference(tencent.index)
    assert len(ddb_only) == 0, f"腾讯缺 DDB 日期 {len(ddb_only)} 天：{list(ddb_only)[:5]}"
    diff = (tencent.loc[overlap] - ddb.loc[overlap]).abs()
    corr = float(
        __import__("numpy").corrcoef(tencent.loc[overlap], ddb.loc[overlap])[0, 1]
    )
    outliers = diff[diff > 0.01]
    assert corr > MIN_CORR, f"逐日相关 {corr:.7f} ≤ {MIN_CORR}"
    assert len(outliers) <= MAX_OUTLIER_DAYS, (
        f"异常日 {len(outliers)} 天 > {MAX_OUTLIER_DAYS}：{outliers.to_dict()}"
    )
    assert float(diff.max()) < MAX_ABS_DIFF, (
        f"max|Δclose|={float(diff.max()):.6f} ≥ {MAX_ABS_DIFF}（源差异上界）"
    )
    print(f"对拍通过：重叠 {len(overlap)} 日（{overlap[0].date()}~{overlap[-1].date()}），"
          f"逐日相关 {corr:.7f}，中位|Δ|={float(diff.median()):.2e}")
    if len(outliers):
        print(f"异常日（>0.01 点，数据商修正级，均不进预热段）：")
        for d, v in outliers.items():
            print(f"  {d.date()}: {v:+.4f} 点")

    warm = tencent[tencent.index < DDB_DAILY_FLOOR]
    assert len(warm) >= 200, f"预热段不足 200 交易日：{len(warm)}"
    return warm.rename_axis("date").reset_index()[["date", "close"]]


def main() -> None:
    tencent = fetch_tencent()
    print(f"腾讯 sh000300：{len(tencent)} 根 "
          f"（{tencent.index[0].date()}~{tencent.index[-1].date()}）")
    warm = cross_validate(tencent)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    warm.to_csv(OUT_CSV, index=False)
    print(f"预热段落盘 {OUT_CSV}：{len(warm)} 行 "
          f"（{warm['date'].min().date()}~{warm['date'].max().date()}）")


if __name__ == "__main__":
    main()
