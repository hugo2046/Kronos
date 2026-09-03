"""DuckDB 归档追加 L1 臂（计划 §4.3，20260903 L1与R1计划）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，逐臂幂等：

- arm=L1L250ZS100/L1L250ZS101/L1L250ZS102：每臂 8 组（4 变体 × 2 窗）；
- arm=L1L500ZS100/L1L250FT100：每臂 4 组（仅 backtest）；
- ``runs`` 元数据按臂追加（lookback/权重/推理口径 JSON）；
- 写入后每臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值对拍（贴输出）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m l1_context.append_duckdb_l1``
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import VARIANTS
from finetune_suite.build_duckdb import _long_from_wide
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START

from l1_context.config import ARMS, WINDOW_DEFS, arm_tag

PKG_DIR = Path(__file__).resolve().parent
L1_DATA_DIR = PKG_DIR / "data"
DB_PATH = PKG_DIR.parent / "finetune_suite" / "data" / "signals.duckdb"

H2_START, H2_END = WINDOW_DEFS["2025h2"]


def _inference_str(lookback: int) -> str:
    return (f"L={lookback}/H=10/N=20/T=1.0/top_p=0.9/seed=42 "
            f"(canonical 除 L 外逐字 paper_replication/config.yaml)")


def _runs_row(tag: str) -> tuple[str, str, str, str]:
    spec = ARMS[tag]
    weights = (
        f"L250-ft 重训 predictor（G1 配方 seed=100、lookback=250、全 A 语料、"
        f"epochs=15、CE 早停；tokenizer 冻结共享 G1 s100）" if spec["kind"] == "ft"
        else f"G1 {spec['seed']} 权重不动（s100=G1、s101/s102=G2 重训族）"
    )
    desc = {
        "weights": weights,
        "tokenizer_path": "G1 s100 tokenizer（冻结只读共享）",
        "predictor_path": spec["model_path"],
        "lookback": spec["lookback"],
        "inference": _inference_str(spec["lookback"]),
        "note": (
            "唯一变量=推理上下文长度 L（ft 臂另含几何对齐重训）；"
            "L90 锚信号既有 parquet 只读，本轮不重跑"
        ),
    }
    windows = f"backtest {BACKTEST_START}~{BACKTEST_END}"
    if "2025h2" in spec["windows"]:
        windows += f"; 2025h2 {H2_START}~{H2_END}"
    return (
        arm_tag(tag), windows, json.dumps(desc, ensure_ascii=False),
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _pick_backtest_day(con, atag: str) -> tuple[str, dict[str, float], pd.Series]:
    """抽检准备：backtest mean 中位日（SQL 参数仅常量域 arm；日期过滤在 pandas 侧）。"""
    wide = pd.read_parquet(L1_DATA_DIR / atag[2:] / f"daily_signals_backtest_{atag}_mean.parquet")
    d = wide.index[len(wide) // 2]
    qdate = str(d.date())
    row_all = con.execute(
        "SELECT date, code, value FROM signals WHERE arm=? AND variant=?",
        [atag, "mean"],
    ).fetchall()
    by_date: dict[str, dict[str, float]] = {}
    for dte, code, val in row_all:
        by_date.setdefault(str(dte), {})[code] = val
    arc = by_date.get(qdate, {})
    src = wide.loc[d].dropna().sort_index()
    return qdate, arc, src


def main() -> None:
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：既有归档为前置"

    con = duckdb.connect(str(DB_PATH))
    existing_arms = {r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()}
    todo = [(t, s) for t, s in ARMS.items() if arm_tag(t) not in existing_arms]
    if not todo:
        logger.info("全部 L1 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"已归档臂 {sorted(existing_arms)}；本次追加 {[arm_tag(t) for t, _ in todo]}")

    tables: list[pd.DataFrame] = []
    for tag, spec in todo:
        atag = arm_tag(tag)
        for window in spec["windows"]:
            for v in VARIANTS:
                wide = pd.read_parquet(
                    L1_DATA_DIR / tag / f"daily_signals_{window}_{atag}_{v}.parquet"
                )
                tables.append(_long_from_wide(atag, v, wide))
    append_df = pd.concat(tables, ignore_index=True)
    logger.info(f"追加长表合计 {len(append_df)} 行（arm×variant 分布见下）")
    print(append_df.groupby(["arm", "variant"]).size().to_string())

    con.executemany(
        "INSERT INTO runs VALUES (?, ?, ?, ?)", [_runs_row(t) for t, _ in todo]
    )
    con.register("append_df", append_df)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM append_df")

    n_total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    logger.info(f"追加完成：signals 总 {n_total} 行，runs 追加 {len(todo)} 行")

    # —— 对拍：每个新臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值比对 ——
    for tag, _ in todo:
        atag = arm_tag(tag)
        qdate, arc, src = _pick_backtest_day(con, atag)
        assert len(arc) == len(src), (
            f"{atag}/mean/{qdate} 行数不一致：归档 {len(arc)} vs 源 {len(src)}"
        )
        max_diff = max(
            (abs(arc[code] - float(src[code])) for code in src.index), default=0.0
        )
        assert max_diff < 1e-12, f"{atag}/mean/{qdate} 对拍失败 max|Δ|={max_diff}"
        print(f"对拍 {atag}/mean/backtest/{qdate}: {len(src)} 值, max|Δ|={max_diff:.1e} → 一致")
    con.close()
    logger.info(f"DuckDB 追加归档完成：{DB_PATH}")


if __name__ == "__main__":
    main()
