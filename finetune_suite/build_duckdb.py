"""阶段 3.3：DuckDB 归档层（计划 §5 修订2 步骤 3.3）。

把本轮全部最终信号 parquet（F1×4 + F0×4 + M，共 9 张宽表）导入单文件
``finetune_suite/data/signals.duckdb``：

- 长表 ``signals(arm TEXT, variant TEXT, date DATE, code TEXT, value DOUBLE)``；
- 元数据 ``runs(arm TEXT, window TEXT, config_json TEXT, created_at TIMESTAMP)``；
- 写入后 SELECT 抽 3 组 (arm, variant, date) 与源 parquet 对拍一致。

**引擎消费仍走宽表 parquet**（不改引擎链路），DuckDB 是查询/管理层；
``.duckdb`` 不入库（.gitignore 排除）。

用法：``python finetune_suite/build_duckdb.py``
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
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START, build_f1_config

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
DB_PATH = DATA_DIR / "signals.duckdb"

CANONICAL_INFERENCE = (
    "L=90/H=10/N=20/T=1.0/top_p=0.9/seed=42 "
    "(paper_replication/config.yaml，canonical)"
)


def _long_from_wide(arm: str, variant: str, wide: pd.DataFrame) -> pd.DataFrame:
    """宽表 → 长表（dropna：引擎只消费有限值）。"""
    long = wide.stack(dropna=True).rename("value").reset_index()
    long.columns = ["date", "code", "value"]
    long.insert(0, "variant", variant)
    long.insert(0, "arm", arm)
    return long


def _runs_rows() -> list[tuple[str, str, str, str]]:
    """runs 元数据：三臂的窗口与配置 JSON。"""
    suite_cfg = build_f1_config()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    window = f"{BACKTEST_START}~{BACKTEST_END}"
    rows = [
        (
            "F1", window,
            json.dumps(
                {
                    "weights": "finetuned tokenizer(f1, best=epoch15) + predictor(f1)",
                    "tokenizer_path": suite_cfg.tokenizer_name,
                    "predictor_path": suite_cfg.model_name,
                    "inference": CANONICAL_INFERENCE,
                    "note": "唯一变量=权重（相对 zero-shot canonical）",
                },
                ensure_ascii=False,
            ),
            created,
        ),
        (
            "F0", window,
            json.dumps(
                {
                    "weights": "NeoQuasar/Kronos-base + Tokenizer-base (zero-shot)",
                    "source": "baseline_suite/data/daily_signals_oos_{variant}.parquet 子集",
                    "inference": CANONICAL_INFERENCE,
                },
                ensure_ascii=False,
            ),
            created,
        ),
        (
            "M", window,
            json.dumps(
                {
                    "definition": "close[t]/close[t-10]-1",
                    "source": "baseline_suite/data/daily_signals_oos_M.parquet 子集",
                },
                ensure_ascii=False,
            ),
            created,
        ),
    ]
    return rows


def main() -> None:
    # —— 收集 9 张宽表 ——
    tables: list[pd.DataFrame] = []
    for arm in ("F1", "F0"):
        for v in VARIANTS:
            wide = pd.read_parquet(DATA_DIR / f"daily_signals_backtest_{arm}_{v}.parquet")
            tables.append(_long_from_wide(arm, v, wide))
    tables.append(_long_from_wide("M", "-", pd.read_parquet(DATA_DIR / "daily_signals_backtest_M.parquet")))
    long_all = pd.concat(tables, ignore_index=True)
    logger.info(f"长表合计 {len(long_all)} 行（arm×variant 分布见下）")
    print(long_all.groupby(["arm", "variant"]).size().to_string())

    if DB_PATH.exists():
        DB_PATH.unlink()  # 重建（幂等）
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        """
        CREATE TABLE signals (
            arm TEXT, variant TEXT, date DATE, code TEXT, value DOUBLE
        )
        """
    )
    # "window" 是 DuckDB 保留字，须加引号（schema 字段名与计划 §5 修订2 一致）
    con.execute(
        'CREATE TABLE runs (arm TEXT, "window" TEXT, config_json TEXT, created_at TEXT)'
    )
    con.executemany("INSERT INTO runs VALUES (?, ?, ?, ?)", _runs_rows())
    con.register("long_df", long_all)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM long_df")

    n = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    logger.info(f"signals.duckdb 写入完成：signals {n} 行，runs 3 行")

    # —— 对拍：抽 3 组 (arm, variant, date) 与源 parquet 逐值比对 ——
    probes = [
        ("F1", "mean", pd.read_parquet(DATA_DIR / "daily_signals_backtest_F1_mean.parquet")),
        ("F0", "min", pd.read_parquet(DATA_DIR / "daily_signals_backtest_F0_min.parquet")),
        ("M", "-", pd.read_parquet(DATA_DIR / "daily_signals_backtest_M.parquet")),
    ]
    for arm, variant, wide in probes:
        d = wide.index[len(wide) // 2]  # 取中间日，确定性
        row = con.execute(
            "SELECT code, value FROM signals WHERE arm=? AND variant=? AND date=? "
            "ORDER BY code",
            [arm, variant, d.date().isoformat()],
        ).fetchall()
        src = wide.loc[d].dropna().sort_index()
        assert len(row) == len(src), f"{arm}/{variant}/{d.date()} 行数不一致"
        max_diff = max(
            abs(value - float(src[code])) for code, value in row
        )
        assert max_diff < 1e-12, f"{arm}/{variant}/{d.date()} 对拍失败 max|Δ|={max_diff}"
        print(
            f"对拍 {arm}/{variant}/{d.date().isoformat()}: {len(row)} 值, "
            f"max|Δ|={max_diff:.1e} → 一致"
        )
    con.close()
    logger.info(f"DuckDB 归档完成：{DB_PATH}")


if __name__ == "__main__":
    main()
