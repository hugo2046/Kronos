"""DuckDB 归档追加 G4 种子臂（G4 计划 §3 1.2）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，**逐臂幂等**（已存在的臂
跳过）：arm='G4S100'/'G4S101'/'G4S102'，每臂 8 组（4 变体 × 2 窗）；
``runs`` 元数据按臂追加；写入后抽 (arm, mean, backtest 中位日) 与源 parquet
对拍一致（贴输出）。

用法：``python g4_features/append_duckdb_g4.py``
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
G4_DIR = PKG_DIR / "data"
DB_PATH = PKG_DIR.parent / "finetune_suite" / "data" / "signals.duckdb"

H2_START, H2_END = "2025-07-01", "2025-12-31"

ARM_SPECS: dict[str, str] = {  # arm -> 子目录（两窗全跑）
    "G4S100": "s100",
    "G4S101": "s101",
    "G4S102": "s102",
}
WINDOWS = ("backtest", "2025h2")


def _runs_row(arm: str) -> tuple[str, str, str, str]:
    from g4_features.config import G1_PREDICTOR_DIRS, G1_TOKENIZER_DIR, G4Config

    seed = int(arm[3:])
    cfg = G4Config(seed=seed)
    desc = {
        "weights": (
            "G4 = 市场上下文特征微调：输入 6→9 列（idx_ret/mkt_vol/ma200_gate），"
            "tokenizer 零初始化手术续训 + predictor 原形装载 G1 族对应种子权重续训"
        ),
        "tokenizer_path": cfg.finetuned_tokenizer_path,
        "predictor_path": cfg.finetuned_predictor_path,
        "warm_start": {
            "tokenizer": str(G1_TOKENIZER_DIR),
            "predictor": str(G1_PREDICTOR_DIRS[seed]),
        },
        "corpus": "g4_features/data（9 列 = G1 ashares pkl 右连接市场三列，前 6 列逐位一致）",
        "inference": CANONICAL_INFERENCE,
        "note": "特征路线臂：AR 解码链路不动，唯一变量 = 输入特征集；推理 seed 恒 42",
    }
    windows = f"backtest {BACKTEST_START}~{BACKTEST_END}; 2025h2 {H2_START}~{H2_END}"
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (arm, windows, json.dumps(desc, ensure_ascii=False), created)


def main() -> None:
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：既有归档为前置"

    con = duckdb.connect(str(DB_PATH))
    existing_arms = {r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()}
    todo = [a for a in ARM_SPECS if a not in existing_arms]
    if not todo:
        logger.info("全部 G4 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"已归档臂 {sorted(existing_arms)}；本次追加 {todo}")

    tables: list[pd.DataFrame] = []
    for arm in todo:
        sub = ARM_SPECS[arm]
        for window in WINDOWS:
            for v in VARIANTS:
                p = G4_DIR / sub / f"daily_signals_{window}_{arm}_{v}.parquet"
                assert p.exists(), f"信号缺失：{p}（先跑 run_g4_signals）"
                wide = pd.read_parquet(p)
                tables.append(_long_from_wide(arm, v, wide))
    append_df = pd.concat(tables, ignore_index=True)
    logger.info(f"追加长表合计 {len(append_df)} 行")
    print(append_df.groupby(["arm", "variant"]).size().to_string())

    con.executemany(
        "INSERT INTO runs VALUES (?, ?, ?, ?)", [_runs_row(a) for a in todo]
    )
    con.register("append_df", append_df)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM append_df")

    n_total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    logger.info(f"追加完成：signals 总 {n_total} 行，runs 追加 {len(todo)} 行")

    # —— 对拍：每个新臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值比对 ——
    for arm in todo:
        wide = pd.read_parquet(
            G4_DIR / ARM_SPECS[arm] / f"daily_signals_backtest_{arm}_mean.parquet"
        )
        d = wide.index[len(wide) // 2]
        row = con.execute(
            "SELECT code, value FROM signals WHERE arm=? AND variant=? AND date=? "
            "ORDER BY code",
            [arm, "mean", d.date().isoformat()],
        ).fetchall()
        src = wide.loc[d].dropna().sort_index()
        assert len(row) == len(src), f"{arm}/mean/{d.date()} 行数不一致"
        max_diff = max(abs(r[1] - s) for r, s in zip(row, src.values))
        assert max_diff < 1e-12, f"{arm} 对拍失败 max|Δ|={max_diff}"
        logger.info(f"对拍 {arm}/mean/{d.date().isoformat()}：{len(src)} 只，"
                    f"max|Δ|={max_diff:.1e} 一致")

    con.close()
    logger.info("G4 臂 DuckDB 追加完成")


if __name__ == "__main__":
    main()
