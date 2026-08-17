"""9 列语料构建（G4 计划 §2 0.2/0.4）。

**数据来源（执行说明，跑前定案）**：不重拉 DDB 个股数据，而是把既有
G1 ashares 语料 pkl（``finetune_suite/data/ashares/{train,val}_data.pkl``，
只读）右连接 3 列市场上下文——前 6 列与 G1 训练语料逐位一致，
"唯一变量 = 输入特征集"在数据层成立（重拉会受 DDB 活库更新影响引入漂移）。

清洗规则沿用 G1 冻结口径（不新增清洗）：市场列若在源行日期不可得 → 显式
报错（fail-fast），不静默产出 NaN。

落盘（g4_features/data/，不覆盖任何既有 pkl）：
    train_data.pkl / val_data.pkl / build_stats.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune_suite.config import Config
from g4_features.market_context import (
    MARKET_COLS,
    TRAIN_FIRST_DAY,
    build_index_series,
    compute_market_context,
)

BASE_COLS = ["open", "high", "low", "close", "vol", "amt"]
NINE_COLS = BASE_COLS + MARKET_COLS

_PKG_DIR = Path(__file__).resolve().parent
G1_CORPUS_DIR = _PKG_DIR.parent / "finetune_suite" / "data" / "ashares"
OUT_DIR = _PKG_DIR / "data"


def attach_market_context(
    data: dict[str, pd.DataFrame], mkt: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """每只股票的 6 列 DataFrame 右连接市场三列（冻结列序 = 前 6 + 后 3）。

    :raises ValueError: 任一源行日期的市场列不可得（NaN）——语料首日 2014-01-02
        起预热已保证可算，出现 NaN 说明预热/对齐破坏，禁止静默继续。
    """
    mkt = mkt[MARKET_COLS].sort_index()
    out: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        joined = df.join(mkt, how="left")
        na_mask = joined[MARKET_COLS].isna().any(axis=1)
        if na_mask.any():
            bad = joined.index[na_mask]
            raise ValueError(
                f"市场上下文列存在 NaN：{sym} 有 {len(bad)} 行未覆盖"
                f"（首 {bad[0].date()}，末 {bad[-1].date()}）——检查预热覆盖"
            )
        out[sym] = joined[NINE_COLS]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="G4 9 列语料构建（G1 pkl 只读 + 市场列右连接）")
    parser.add_argument(
        "--end", default=None,
        help="指数序列末 日（默认 val 末日 2025-06-30；推理窗另由 infer 模块自建）",
    )
    args = parser.parse_args()

    cfg = Config()  # 只为取 train/val 时间窗声明（与 G1 冻结窗口一致）
    end = args.end or cfg.val_time_range[1]

    index_close = build_index_series(None, end)  # provider 在函数内按需构造
    mkt = compute_market_context(index_close)
    # 训练首日三列全部可算（预热断言；测试亦钉死）
    assert mkt.loc[TRAIN_FIRST_DAY:, MARKET_COLS].notna().all().all(), (
        f"{TRAIN_FIRST_DAY.date()} 起市场列存在 NaN"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {
        "source_corpus": str(G1_CORPUS_DIR),
        "market_cols": MARKET_COLS,
        "nine_cols": NINE_COLS,
        "index_series": {
            "first": str(index_close.index[0].date()),
            "last": str(index_close.index[-1].date()),
            "n_days": len(index_close),
        },
        "warmup_from": "tencent ifzq.gtimg.cn sh000300（2013 段，对拍 DDB 2014 重叠段后采用）",
        "splits": {},
    }
    for split in ("train", "val"):
        src_path = G1_CORPUS_DIR / f"{split}_data.pkl"
        with open(src_path, "rb") as f:
            data = pickle.load(f)
        out_data = attach_market_context(data, mkt)

        out_path = OUT_DIR / f"{split}_data.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(out_data, f)

        rows = sum(len(df) for df in out_data.values())
        idx = pd.DatetimeIndex([]) if not out_data else pd.concat(
            [df.index.to_series() for df in out_data.values()]
        )
        stats["splits"][split] = {
            "n_symbols": len(out_data),
            "n_rows": rows,
            "date_range": [str(idx.min().date()), str(idx.max().date())],
            "market_join_coverage": 1.0,  # NaN 即 raise，到达即 100%
            "pkl_bytes": out_path.stat().st_size,
            "source_pkl_bytes": src_path.stat().st_size,
        }
        print(f"[{split}] {len(out_data)} symbols, {rows} rows → {out_path}")

    with open(OUT_DIR / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
