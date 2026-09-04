"""v2 vs qlib 官方回测对拍（integration 分支 QlibBacktest 思路的 DDB 版）。

integration 分支 ``finetune/qlib_test.py::QlibBacktest`` 用 qlib 官方
``TopkDropoutStrategy`` + ``qlib.backtest.backtest``（文件后端）。本脚本把
同一套配置跑到本仓库的 DDB 后端上，抽 3 臂与 v2 引擎对拍（目标 AER 差
<1pp），数字进 ``docs/引擎v2重放对照_20260905.md``。

口径对齐（与 v2 的已知差异在文档中如实列出）：

    - 策略：TopkDropoutStrategy(topk=50, n_drop=5, hold_thresh=5,
      only_tradable=True)——与 v2 的 top-k/drop-n/min_hold 对应；
    - 成交价：``deal_price="close"``，信号**前移 1 个交易日**后再喂入
      （qlib 在信号日当根 bar 成交；前移后 = t 日信号 t+1 收盘成交，
      对齐 v2 的 delay=1）；
    - 成本：open_cost=close_cost=0.0015（双边 15bp，对齐修正 (1)），min_cost=0；
    - 涨跌停：qlib ``limit_threshold=0.095`` 启发式（近似修正 (3)；与 v2 的
      DDB uls 精确判定不同——20cm 板会漏判，文档披露）；
    - 基准：000300.SH 指数（对齐 v2 的 idx 口径），AER 年化 252。

用法::

    python -m paper_replication.qlib_crosscheck_v2 --arms G1m_bt,M_bt,K_paper
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from loguru import logger

from paper_replication.common import DATA_DIR, REPO_ROOT

TRADING_DAYS = 252

ARMS = {
    # 臂名 → (parquet 相对路径, 窗口起, 窗口止)
    "G1m_bt": ("finetune_suite/data/g1/daily_signals_backtest_G1_mean.parquet",
               "2026-01-01", "2026-07-24"),
    "M_bt": ("finetune_suite/data/daily_signals_backtest_M.parquet",
             "2026-01-01", "2026-07-24"),
    "K_paper": ("paper_replication/data/daily_signals_K.parquet",
                "2024-07-01", "2025-06-30"),
}


def run_qlib_arm(name: str, relpath: str, start: str, end: str) -> dict:
    """单臂 qlib 官方回测 → 与 v2 同口径的 AER(idx, 含成本, 年化 252)。"""
    from qlib.backtest import executor as ql_executor
    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy

    sig = pd.read_parquet(REPO_ROOT / relpath)
    # 信号前移 1 个交易日：t 日信号值落到 t+1 的标签上，qlib 在标签日当根
    # bar close 成交 = t 日决策、t+1 收盘成交（v2 delay=1）
    dates = sig.index
    shifted = sig.iloc[:-1].copy()
    shifted.index = dates[1:]
    pred = shifted.stack()
    pred.index.names = ["datetime", "instrument"]
    pred = pred.swaplevel().sort_index()

    strategy = TopkDropoutStrategy(
        topk=50, n_drop=5, hold_thresh=5, only_tradable=True, signal=pred
    )
    pm, _ = backtest(
        start_time=start, end_time=end, account=100_000_000,
        benchmark="000300.SH", strategy=strategy,
        executor=ql_executor.SimulatorExecutor(
            time_per_step="day", generate_portfolio_metrics=True, verbose=False
        ),
        exchange_kwargs=dict(
            freq="day", limit_threshold=0.095, deal_price="close",
            open_cost=0.0015, close_cost=0.0015, min_cost=0,
            codes="csi300",  # DDB instruments 表无 "all"，显式预载 csi300
        ),
    )
    report, _ = pm.get("1day")
    # report 列：return / bench / cost（逐日，简单收益）
    excess = report["return"] - report["bench"] - report["cost"]
    nav = (1 + excess).cumprod()
    n = len(excess)
    aer = float(nav.iloc[-1] ** (TRADING_DAYS / n) - 1)
    mu, sd = float(excess.mean()), float(excess.std(ddof=1))
    ir = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    logger.info(f"[qlib:{name}] n={n} AER(idx,net)={aer:+.2%} IR={ir:+.3f}")
    return {
        "arm": name, "window": [start, end], "n_days": n,
        "aer_idx_net": aer, "ir_idx": ir,
        "turnover": float(report["cost"].mean() / 0.0015) if len(report) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="G1m_bt,M_bt,K_paper")
    args = ap.parse_args()

    # 触发 qlib DDB 初始化（复用仓库 .env，绝不静默回退）
    from kronos_qlib import QlibProvider

    QlibProvider("csi300", "2024-07-01", "2026-07-24")

    out = {}
    for name in args.arms.split(","):
        relpath, start, end = ARMS[name]
        out[name] = run_qlib_arm(name, relpath, start, end)

    out_path = DATA_DIR / "qlib_crosscheck_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    logger.info(f"qlib 对拍落盘 {out_path}")


if __name__ == "__main__":
    main()
