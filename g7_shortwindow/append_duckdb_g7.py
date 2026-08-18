"""DuckDB 归档追加 G7 W85 臂（G7 计划 §3 3.2）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，**逐臂幂等**（已存在的臂
跳过）：arm='W85S100'/'W85S101'/'W85S102'，每臂 8 组（4 变体 × 2 窗）；
``runs`` 元数据按臂追加；写入后抽 (arm, mean, backtest 中位日) 与源 parquet
对拍一致（贴输出）。只追加原始信号，不含任何评估数字（落盘前置纪律）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m g7_shortwindow.append_duckdb_g7``
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

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
DB_PATH = PKG_DIR.parent / "finetune_suite" / "data" / "signals.duckdb"

H2_START, H2_END = "2025-07-01", "2025-12-31"

ARM_SPECS: dict[str, str] = {  # arm -> 子目录（两窗全跑）
    "W85S100": "s100",
    "W85S101": "s101",
    "W85S102": "s102",
}
WINDOWS = ("backtest", "2025h2")


def _runs_row(arm: str) -> tuple[str, str, str, str]:
    from g7_shortwindow.run_g7_signals import _g1_family_paths

    seed = int(arm[4:])
    model, tokenizer = _g1_family_paths(seed)
    desc = {
        "weights": (
            "G7 W85 = G1 族短窗推理复检：权重只读复用 G1 族种子"
            f"（s{seed}），L/H 为纯推理期参数，换配置零训练"
        ),
        "tokenizer_path": tokenizer,
        "predictor_path": model,
        "inference": (
            "L=8/H=5/N=20/T=1.0/top_p=0.9/seed=42 "
            "（除 L/H 外逐字 paper_replication/config.yaml canonical）"
        ),
        "windows_note": "backtest/2025h2 与在位者同界（H 短导致的多余可结算日不使用）",
        "note": "短窗推理臂：用户经验先验 8 天看 5 天；推理 seed 恒 42",
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
        logger.info("全部 W85 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"已归档臂 {sorted(existing_arms)}；本次追加 {todo}")

    tables: list[pd.DataFrame] = []
    for arm in todo:
        sub = ARM_SPECS[arm]
        for window in WINDOWS:
            for v in VARIANTS:
                p = DATA_DIR / sub / f"daily_signals_{window}_{arm}_{v}.parquet"
                assert p.exists(), f"信号缺失：{p}（先跑 run_g7_signals）"
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
            DATA_DIR / ARM_SPECS[arm] / f"daily_signals_backtest_{arm}_mean.parquet"
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
    logger.info("W85 臂 DuckDB 追加完成")


if __name__ == "__main__":
    main()
