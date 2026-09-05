"""DuckDB 归档追加 H1 臂（计划 §3.3，20260905 H1 计划）。

复用 ``build_duckdb._long_from_wide``（纪律：append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，逐臂幂等：

- arm=H1a-lin/H1a-kda/H1b-lin/H1b-kda（_s{seed} 后缀并入 arm 字符串）：
  **读出头为确定性打分（单前向 decode，无 AR 采样）→ 每臂每窗 1 组，
  variant 固定 "mean"**（不落 last/max/min 重复副本；与
  ``run_h1_signals`` 的单文件产物口径一致——2026-09-05 修复：原稿按
  g5 式 4 变体循环找 ``_{v}.parquet``，与打分器产物互相矛盾，两轮夜跑
  均在此 FileNotFoundError）；
- ``runs`` 元数据按臂追加（语料/协议/在线前向 JSON）；
- 写入后每臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值对拍（贴输出）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m h1_readout.append_duckdb_h1 --seed 42``
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune_suite.build_duckdb import _long_from_wide
from h1_readout.corpus import ES_END, ES_START, TRAIN_LABEL_END, TRAIN_START
from h1_readout.run_h1_signals import WINDOW_BOUNDS
from h1_readout.train_h1 import ARM_POOL

PKG_DIR = Path(__file__).resolve().parent
H1_DATA_DIR = PKG_DIR / "data"
DB_PATH = PKG_DIR.parent / "finetune_suite" / "data" / "signals.duckdb"


def _runs_row(arm: str, seed: int) -> tuple[str, str, str, str]:
    desc = {
        "weights": (
            f"H1 读出头（G1 s100 底座冻结 + {'B2LinearProbe 833' if arm.endswith('lin') else 'G5KdaHead 1,209,937'} 参数头，"
            f"IC 损失，seed {seed}）"),
        "corpus": (
            f"{ARM_POOL[arm]} PIT pkl {TRAIN_START}~{TRAIN_LABEL_END}（G1 同源，"
            f"在线前向 2000 步/epoch × ≤15，早停 {ES_START}~{ES_END} csi300 日均 RankIC）"),
        "inference": "L=90/H=10 单前向 decode（读出头打分，无 AR 采样）",
        "note": "唯一变量=训练数据广度（R1 其余协议逐字）",
    }
    windows = "; ".join(f"{w} {b[0]}~{b[1]}" for w, b in WINDOW_BOUNDS.items())
    return (f"{arm}_s{seed}", windows, json.dumps(desc, ensure_ascii=False),
            datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _pick_backtest_day(con, arm_tag: str) -> tuple[str, dict[str, float], pd.Series]:
    wide = pd.read_parquet(
        H1_DATA_DIR / arm_tag.rsplit("_s", 1)[0] / f"daily_signals_backtest_{arm_tag}.parquet")
    d = wide.index[len(wide) // 2]
    qdate = str(d.date())
    row_all = con.execute(
        "SELECT date, code, value FROM signals WHERE arm=? AND variant=?",
        [arm_tag, "mean"],
    ).fetchall()
    by_date: dict[str, dict[str, float]] = {}
    for dte, code, val in row_all:
        by_date.setdefault(str(dte), {})[code] = val
    return qdate, by_date.get(qdate, {}), wide.loc[d].dropna().sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：既有归档为前置"

    con = duckdb.connect(str(DB_PATH))
    existing_arms = {r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()}
    tags = [f"{a}_s{args.seed}" for a in ARM_POOL]
    todo = [t for t in tags if t not in existing_arms]
    if not todo:
        logger.info("全部 H1 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"本次追加 {todo}")

    tables: list[pd.DataFrame] = []
    for tag in todo:
        arm = tag.rsplit("_s", 1)[0]
        for window in WINDOW_BOUNDS:
            wide = pd.read_parquet(
                H1_DATA_DIR / arm / f"daily_signals_{window}_{tag}.parquet")
            tables.append(_long_from_wide(tag, "mean", wide))
    append_df = pd.concat(tables, ignore_index=True)
    logger.info(f"追加长表合计 {len(append_df)} 行（arm×variant 分布见下）")
    print(append_df.groupby(["arm", "variant"]).size().to_string())

    con.executemany(
        "INSERT INTO runs VALUES (?, ?, ?, ?)",
        [_runs_row(t.rsplit("_s", 1)[0], args.seed) for t in todo],
    )
    con.register("append_df", append_df)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM append_df")

    n_total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    logger.info(f"追加完成：signals 总 {n_total} 行，runs 追加 {len(todo)} 行")

    for tag in todo:
        qdate, arc, src = _pick_backtest_day(con, tag)
        assert len(arc) == len(src), f"{tag}/mean/{qdate} 行数不一致：{len(arc)} vs {len(src)}"
        max_diff = max((abs(arc[c] - float(src[c])) for c in src.index), default=0.0)
        assert max_diff < 1e-12, f"{tag}/mean/{qdate} 对拍失败 max|Δ|={max_diff}"
        print(f"对拍 {tag}/mean/backtest/{qdate}: {len(src)} 值, max|Δ|={max_diff:.1e} → 一致")
    con.close()
    logger.info(f"DuckDB 追加归档完成：{DB_PATH}")


if __name__ == "__main__":
    main()
