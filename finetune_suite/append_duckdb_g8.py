"""DuckDB 归档追加 G8 臂（计划 §3.3，20260820 G8+E1 计划）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，逐臂幂等：

- arm='G8S100' / 'G8S101' / 'G8S102'，每臂 4 组（仅 backtest 窗——2025H2 为
  G8 早停窗，准样本内，禁止用作跨窗检验，计划 §1 冻结）；
- ``runs`` 元数据按臂追加（窗口/种子/语料终点/推理口径 JSON）；
- 写入后每臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值对拍（贴输出）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m finetune_suite.append_duckdb_g8``
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
from finetune_suite.run_g8_signals import arm_tag

PKG_DIR = Path(__file__).resolve().parent
G8_DIR = PKG_DIR / "data" / "g8"
DB_PATH = PKG_DIR / "data" / "signals.duckdb"

SEEDS = (100, 101, 102)


def _runs_row(seed: int) -> tuple[str, str, str, str]:
    from finetune_suite.train_g8 import G8Config

    cfg = G8Config(seed=seed)
    desc = {
        "weights": (
            "G8 语料新鲜度臂：train 终点 2024-12-31→2025-06-30（唯一变量），"
            f"predictor 训练种子 seed={seed}"
        ),
        "tokenizer_path": cfg.finetuned_tokenizer_path,
        "predictor_path": cfg.finetuned_predictor_path,
        "corpus": "finetune_suite/data/g8（train 2014-01-02~2025-06-30 / val 2025H2）",
        "inference": CANONICAL_INFERENCE,
        "note": "与 G1 逐字一致仅改两处日期；推理 seed 恒 42；"
                "2025H2 为早停窗，准样本内，不生成该窗信号",
    }
    window = f"backtest {BACKTEST_START}~{BACKTEST_END}"
    return (
        arm_tag(seed), window, json.dumps(desc, ensure_ascii=False),
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def main() -> None:
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：既有归档为前置"

    con = duckdb.connect(str(DB_PATH))
    existing_arms = {r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()}
    todo = [s for s in SEEDS if arm_tag(s) not in existing_arms]
    if not todo:
        logger.info("全部 G8 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"已归档臂 {sorted(existing_arms)}；本次追加 {[arm_tag(s) for s in todo]}")

    tables: list[pd.DataFrame] = []
    for s in todo:
        for v in VARIANTS:
            wide = pd.read_parquet(
                G8_DIR / f"s{s}" / f"daily_signals_backtest_{arm_tag(s)}_{v}.parquet"
            )
            tables.append(_long_from_wide(arm_tag(s), v, wide))
    append_df = pd.concat(tables, ignore_index=True)
    logger.info(f"追加长表合计 {len(append_df)} 行（arm×variant 分布见下）")
    print(append_df.groupby(["arm", "variant"]).size().to_string())

    con.executemany(
        "INSERT INTO runs VALUES (?, ?, ?, ?)", [_runs_row(s) for s in todo]
    )
    con.register("append_df", append_df)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM append_df")

    n_total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    logger.info(f"追加完成：signals 总 {n_total} 行，runs 追加 {len(todo)} 行")

    # —— 对拍：每个新臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值比对 ——
    for s in todo:
        wide = pd.read_parquet(G8_DIR / f"s{s}" / f"daily_signals_backtest_{arm_tag(s)}_mean.parquet")
        d = wide.index[len(wide) // 2]
        row = con.execute(
            "SELECT code, value FROM signals WHERE arm=? AND variant=? AND date=? "
            "ORDER BY code",
            [arm_tag(s), "mean", d.date().isoformat()],
        ).fetchall()
        src = wide.loc[d].dropna().sort_index()
        assert len(row) == len(src), f"{arm_tag(s)}/mean/{d.date()} 行数不一致"
        max_diff = max(abs(val - float(src[code])) for code, val in row)
        assert max_diff < 1e-12, f"{arm_tag(s)}/mean/{d.date()} 对拍失败 max|Δ|={max_diff}"
        print(
            f"对拍 {arm_tag(s)}/mean/backtest/{d.date().isoformat()}: {len(row)} 值, "
            f"max|Δ|={max_diff:.1e} → 一致"
        )
    con.close()
    logger.info(f"DuckDB 追加归档完成：{DB_PATH}")


if __name__ == "__main__":
    main()
