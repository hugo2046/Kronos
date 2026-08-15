"""DDB → 官方微调格式 pickle 适配器（阶段 0，计划 §2）。

- 股票族 = 2011~2025 每年 01-01 与 07-01 的 csi300 point-in-time 成分**并集**
  （``provider.list_pool_at`` 采样；退市成员数据以 DDB 可得为准，覆盖率如实
  写入 stats）；
- 每股窗口内可得的全部日频后复权 OHLCV + **真实 amount**（DDB 有真值，不用
  官方 (O+H+L+C)/4×vol 合成 amt——与官方口径的唯一数据差异，计划 §2 已声明；
  tokenizer 窗口 z-score 下量纲差异被归一化吸收）；
- train/val 按日期切分：train ≤ 2024-12-31，val ∈ [2025-01-01, 2025-06-30]；
- 输出 ``{dataset_path}/train_data.pkl`` / ``val_data.pkl``，格式 =
  ``{symbol: DataFrame}``（DatetimeIndex + feature_list 列），与官方
  ``finetune/qlib_data_preprocess.py`` 产物同构。

清洗规则（冻结）：每股丢弃任一特征列 NaN 的行（停牌/缺数）；整段不足
lookback+predict+1 行的股票从两个 pkl 中剔除（同官方 preprocess 语义）。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# 以脚本身份运行时（python finetune_suite/build_dataset.py）把仓库根加入
# sys.path，保证 finetune_suite 包可导入；pytest / -m 场景下无副作用。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from finetune_suite.config import Config

# qlib/DDB 字段 → 官方 feature_list 列名（volume→vol；amount 用 DDB 真值）
FETCH_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
_RENAME = {"volume": "vol", "amount": "amt"}

# 股票族采样：2011~2025 每年 01-01 与 07-01（计划 §2 冻结口径）
UNIVERSE_YEARS = range(2011, 2026)


def sample_pool_universe(provider, pool: str = "csi300", years=UNIVERSE_YEARS) -> list[str]:
    """csi300 PIT 成分并集：每年 01-01 与 07-01 各取一次 point-in-time 成员。"""
    dates = sorted(
        [f"{y}-01-01" for y in years] + [f"{y}-07-01" for y in years]
    )
    union: set[str] = set()
    for t in dates:
        union.update(provider.list_pool_at(pool, t))
    return sorted(union)


def build_pickles(provider, cfg: Config, universe: list[str] | None = None) -> dict:
    """fetch → 清洗 → train/val 切分 → 落盘，返回统计 dict。

    provider 需按 (universe, dataset_begin_time, dataset_end_time) 构造完毕
    （``QlibProvider(instruments=list, start_date, end_date)``）；本函数只做
    转换与落盘，便于测试注入 FakeProvider。
    """
    if universe is None:
        universe = sorted(getattr(provider, "instruments_", []))
    raw = provider.fetch(FETCH_FIELDS)

    # MultiIndex(datetime, instrument) → {symbol: DataFrame}
    grouped: dict[str, pd.DataFrame] = {
        sym: df.droplevel("instrument")
        for sym, df in raw.groupby(level="instrument")
    }
    min_len = cfg.lookback_window + cfg.predict_window + 1
    train_lo, train_hi = (pd.Timestamp(x) for x in cfg.train_time_range)
    val_lo, val_hi = (pd.Timestamp(x) for x in cfg.val_time_range)

    train_data: dict[str, pd.DataFrame] = {}
    val_data: dict[str, pd.DataFrame] = {}
    n_dropped_rows = 0
    n_short_symbols = 0
    for sym in universe:
        if sym not in grouped:
            continue  # 退市成员 DDB 无数据 → 覆盖率统计里如实体现
        df = grouped[sym].rename(columns=_RENAME)[cfg.feature_list]
        n_before = len(df)
        df = df.dropna()
        n_dropped_rows += n_before - len(df)
        if len(df) < min_len:
            n_short_symbols += 1
            continue
        df = df.sort_index()
        train_data[sym] = df.loc[(df.index >= train_lo) & (df.index <= train_hi)]
        val_data[sym] = df.loc[(df.index >= val_lo) & (df.index <= val_hi)]

    out_dir = Path(cfg.dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data in (("train", train_data), ("val", val_data)):
        with open(out_dir / f"{split}_data.pkl", "wb") as f:
            pickle.dump(data, f)

    def _range(data: dict[str, pd.DataFrame]) -> tuple[str, str]:
        idx = pd.DatetimeIndex([]) if not data else pd.concat(
            [df.index.to_series() for df in data.values()]
        )
        return (str(idx.min().date()) if len(idx) else "-",
                str(idx.max().date()) if len(idx) else "-")

    stats = {
        "n_universe": len(universe),
        "n_fetched": len(grouped),
        "coverage": round(len(grouped) / len(universe), 4) if universe else 0.0,
        "n_train_symbols": len(train_data),
        "n_val_symbols": len(val_data),
        "n_rows_train": int(sum(len(df) for df in train_data.values())),
        "n_rows_val": int(sum(len(df) for df in val_data.values())),
        "train_range": _range(train_data),
        "val_range": _range(val_data),
        "n_dropped_nan_rows": n_dropped_rows,
        "n_short_symbols": n_short_symbols,
        "dataset_path": str(out_dir),
        "fields": FETCH_FIELDS,
        "universe_rule": (
            f"{getattr(cfg, 'instrument', 'csi300')} PIT union at each "
            "01-01/07-01, 2011~2025"
        ),
    }
    return stats


def main() -> None:
    # === finetune_ashares 声明改动（计划 §2 0.2：diff 仅限 pool 传参）===
    # --pool ashares：唯一变量 = 训练语料池；采样年份/清洗规则/窗口逐字不动。
    # 注：计划 §1 写"(2014~2025 采样)"——DDB 日频地板 2014-01-02，2011~2013 采样
    # 独有的 12 只成分股在 DDB 零数据、被既有"无数据跳过"规则剔除，两种采样
    # 产生完全等价语料（2026-08-15 实测），故代码保持 2011~2025 不变。
    parser = argparse.ArgumentParser(description="DDB → 官方微调格式 pickle 适配器")
    parser.add_argument(
        "--pool",
        default="csi300",
        choices=["csi300", "ashares"],
        help="股票池（20260815 计划 §1：G1 唯一变量=csi300→ashares）",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.pool != "csi300":
        cfg.instrument = args.pool
        cfg.dataset_path = str(Path(cfg.dataset_path) / args.pool)  # 不覆盖第 4 轮 pkl
    # 延迟 import：单测无需 DolphinDB
    from kronos_qlib.provider import QlibProvider

    pool_provider = QlibProvider(cfg.instrument, cfg.dataset_begin_time, cfg.dataset_end_time)
    universe = sample_pool_universe(pool_provider, cfg.instrument)
    fetch_provider = QlibProvider(universe, cfg.dataset_begin_time, cfg.dataset_end_time)
    stats = build_pickles(fetch_provider, cfg, universe)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    with open(Path(cfg.dataset_path) / "build_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
