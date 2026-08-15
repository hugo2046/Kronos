"""G2.2b：DuckDB 归档追加 G2 种子臂（计划 §1，20260816 计划）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建：

- ``signals`` 追加 arm='G2S101' / 'G2S102'——每臂 8 组（4 变体 × 2 窗：
  backtest 2026-01-01~2026-07-24 + 2025H2 2025-07-01~2025-12-31，日期天然
  区分两窗）；
- ``runs`` 元数据追加 2 行（种子、窗口、权重、推理口径 JSON）；
- 写入后抽 3 组 (arm, variant, date) 与源 parquet 对拍一致（贴输出）。

用法：``python finetune_suite/append_duckdb_g2.py``
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
from finetune_suite.build_duckdb import CANONICAL_INFERENCE, _long_from_wide
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from finetune_suite.run_g2_signals import WINDOW_DEFS, arm_tag

PKG_DIR = Path(__file__).resolve().parent
G2_DIR = PKG_DIR / "data" / "g2"
DB_PATH = PKG_DIR / "data" / "signals.duckdb"

H2_START, H2_END = WINDOW_DEFS["2025h2"]


def _runs_rows() -> list[tuple[str, str, str, str]]:
    from finetune_suite.train_g2 import G2Config

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    windows = f"backtest {BACKTEST_START}~{BACKTEST_END}; 2025h2 {H2_START}~{H2_END}"
    rows = []
    for seed in (101, 102):
        g2 = G2Config(seed)
        rows.append((
            arm_tag(seed), windows,
            json.dumps(
                {
                    "weights": f"G1 predictor 以训练种子 seed={seed} 重训（唯一变量）",
                    "tokenizer_path": g2.finetuned_tokenizer_path,
                    "predictor_path": g2.finetuned_predictor_path,
                    "corpus": "finetune_suite/data/ashares（与 G1 逐字同 pkl）",
                    "inference": CANONICAL_INFERENCE,
                    "note": "种子诊断臂：协议/超参/epochs/语料逐字复用，推理 seed 恒 42",
                },
                ensure_ascii=False,
            ),
            created,
        ))
    return rows


def main() -> None:
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：既有归档为前置"

    tables: list[pd.DataFrame] = []
    for seed in (101, 102):
        for window in ("backtest", "2025h2"):
            for v in VARIANTS:
                wide = pd.read_parquet(
                    G2_DIR / f"s{seed}" / f"daily_signals_{window}_{arm_tag(seed)}_{v}.parquet"
                )
                tables.append(_long_from_wide(arm_tag(seed), v, wide))
    append_df = pd.concat(tables, ignore_index=True)
    logger.info(f"追加长表合计 {len(append_df)} 行（arm×variant 分布见下）")
    print(append_df.groupby(["arm", "variant"]).size().to_string())

    con = duckdb.connect(str(DB_PATH))
    existing_arms = [r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()]
    clash = set(existing_arms) & {"G2S101", "G2S102"}
    assert not clash, f"G2 臂已存在（{clash}）：幂等保护，拒绝重复追加"
    con.executemany('INSERT INTO runs VALUES (?, ?, ?, ?)', _runs_rows())
    con.register("append_df", append_df)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM append_df")

    n_total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    n_new = con.execute(
        "SELECT COUNT(*) FROM signals WHERE arm IN ('G2S101','G2S102')"
    ).fetchone()[0]
    logger.info(f"追加完成：signals 总 {n_total} 行（本次 +{n_new}），runs 追加 2 行")

    # —— 对拍：抽 3 组 (arm, variant, date) 与源 parquet 逐值比对 ——
    probes = [
        ("G2S101", "mean", "backtest"),
        ("G2S101", "min", "2025h2"),
        ("G2S102", "mean", "2025h2"),
    ]
    for arm, v, window in probes:
        wide = pd.read_parquet(G2_DIR / arm[2:].lower() / f"daily_signals_{window}_{arm}_{v}.parquet")
        d = wide.index[len(wide) // 2]
        row = con.execute(
            "SELECT code, value FROM signals WHERE arm=? AND variant=? AND date=? "
            "ORDER BY code",
            [arm, v, d.date().isoformat()],
        ).fetchall()
        src = wide.loc[d].dropna().sort_index()
        assert len(row) == len(src), f"{arm}/{v}/{d.date()} 行数不一致"
        max_diff = max(abs(val - float(src[code])) for code, val in row)
        assert max_diff < 1e-12, f"{arm}/{v}/{d.date()} 对拍失败 max|Δ|={max_diff}"
        print(
            f"对拍 {arm}/{v}/{window}/{d.date().isoformat()}: {len(row)} 值, "
            f"max|Δ|={max_diff:.1e} → 一致"
        )
    con.close()
    logger.info(f"DuckDB 追加归档完成：{DB_PATH}")


if __name__ == "__main__":
    main()
