"""阶段 2.3b：DuckDB 归档**追加** G0/G1 两臂（计划 §4 步骤 2.3，20260815）。

复用 ``finetune_suite/build_duckdb.py`` 的 ``_long_from_wide``（纪律 §8：
DuckDB append 复用）——区别于第 4 轮的"重建"，本脚本对既有
``finetune_suite/data/signals.duckdb`` **只追加不重建**：

- ``signals`` 长表追加 arm='G0'（F1 权重 @ 2025H2 四变体）与 arm='G1'
  （G1 权重 @ backtest 窗四变体）；
- ``runs`` 元数据追加两行（窗口/权重/推理口径 JSON）；
- 写入后抽 3 组 (arm, variant, date) 与源 parquet 对拍一致（贴输出）。

用法：``python finetune_suite/append_duckdb.py``
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

PKG_DIR = Path(__file__).resolve().parent
G0_DIR = PKG_DIR / "data" / "g0"
G1_DIR = PKG_DIR / "data" / "g1"
DB_PATH = PKG_DIR / "data" / "signals.duckdb"


def _runs_rows() -> list[tuple[str, str, str, str]]:
    """runs 元数据追加行：G0 / G1 两臂的窗口与配置 JSON。"""
    from finetune_suite.config import Config as SuiteConfig
    from finetune_suite.train_g1 import G1Config

    suite, g1 = SuiteConfig(), G1Config()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [
        (
            "G0", f"2025-07-01~2025-12-31",
            json.dumps(
                {
                    "weights": "第 4 轮 F1 权重不动一字（只读补测）",
                    "tokenizer_path": suite.finetuned_tokenizer_path,
                    "predictor_path": suite.finetuned_predictor_path,
                    "inference": CANONICAL_INFERENCE,
                    "note": "唯一变量=评估时段（2025H2 跨时段稳定性）",
                },
                ensure_ascii=False,
            ),
            created,
        ),
        (
            "G1", f"{BACKTEST_START}~{BACKTEST_END}",
            json.dumps(
                {
                    "weights": (
                        "全 A 语料（ashares PIT 并集 5599 股）两阶段微调 "
                        "g1 tokenizer + g1 predictor"
                    ),
                    "corpus": "finetune_suite/data/ashares（build_stats.json）",
                    "tokenizer_path": g1.finetuned_tokenizer_path,
                    "predictor_path": g1.finetuned_predictor_path,
                    "inference": CANONICAL_INFERENCE,
                    "note": "唯一变量=训练语料池（csi300→ashares），回测池仍 csi300",
                },
                ensure_ascii=False,
            ),
            created,
        ),
    ]


def main() -> None:
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：第 4 轮归档为本脚本的前置"

    tables: list[pd.DataFrame] = []
    for v in VARIANTS:
        tables.append(
            _long_from_wide("G0", v, pd.read_parquet(G0_DIR / f"daily_signals_2025h2_G0_{v}.parquet"))
        )
    for v in VARIANTS:
        tables.append(
            _long_from_wide("G1", v, pd.read_parquet(G1_DIR / f"daily_signals_backtest_G1_{v}.parquet"))
        )
    append_df = pd.concat(tables, ignore_index=True)
    logger.info(f"追加长表合计 {len(append_df)} 行（arm×variant 分布见下）")
    print(append_df.groupby(["arm", "variant"]).size().to_string())

    con = duckdb.connect(str(DB_PATH))
    existing_arms = [r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()]
    assert "G0" not in existing_arms and "G1" not in existing_arms, (
        f"G0/G1 已存在（{existing_arms}）：append_duckdb 幂等保护，拒绝重复追加"
    )
    con.executemany('INSERT INTO runs VALUES (?, ?, ?, ?)', _runs_rows())
    con.register("append_df", append_df)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM append_df")

    n_total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    n_new = con.execute(
        "SELECT COUNT(*) FROM signals WHERE arm IN ('G0','G1')"
    ).fetchone()[0]
    logger.info(f"追加完成：signals 总 {n_total} 行（本次 +{n_new}），runs 追加 2 行")

    # —— 对拍：抽 3 组 (arm, variant, date) 与源 parquet 逐值比对 ——
    probes = [
        ("G0", "mean", pd.read_parquet(G0_DIR / "daily_signals_2025h2_G0_mean.parquet")),
        ("G1", "mean", pd.read_parquet(G1_DIR / "daily_signals_backtest_G1_mean.parquet")),
        ("G1", "min", pd.read_parquet(G1_DIR / "daily_signals_backtest_G1_min.parquet")),
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
        max_diff = max(abs(value - float(src[code])) for code, value in row)
        assert max_diff < 1e-12, f"{arm}/{variant}/{d.date()} 对拍失败 max|Δ|={max_diff}"
        print(
            f"对拍 {arm}/{variant}/{d.date().isoformat()}: {len(row)} 值, "
            f"max|Δ|={max_diff:.1e} → 一致"
        )
    con.close()
    logger.info(f"DuckDB 追加归档完成：{DB_PATH}")


if __name__ == "__main__":
    main()
