"""DuckDB 归档追加 C4 臂（C4 计划 §3 3.2，20260820）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，**逐臂幂等**（已存在的臂
跳过）：arm='C4S100'/'C4S101'/'C4S102'，每臂 1 组（variant='c4' × merged
260 日连续窗，含预热期前 30 行——评估层负责剔除，归档如实保存变换工件）；
``runs`` 元数据按臂追加；写入后抽 (arm, c4, merged 中位日) 与源 parquet
对拍逐值一致（贴输出）。只追加原始信号，不含任何评估数字（落盘前置纪律）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m c4_temporal.append_duckdb_c4``
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

from c4_temporal.transform import (
    C4_LAMBDA,
    C4_WINDOW,
    HALF_LIFE_DAYS,
    MERGED_WINDOW,
    TRINARIZE_THRESHOLD,
    WARMUP_DAYS,
)
from finetune_suite.build_duckdb import _long_from_wide

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
DB_PATH = PKG_DIR.parent / "finetune_suite" / "data" / "signals.duckdb"

ARM_SPECS: dict[str, str] = {"C4S100": "s100", "C4S101": "s101", "C4S102": "s102"}
VARIANT = "c4"


def _runs_row(arm: str) -> tuple[str, str, str, str]:
    seed = int(arm[3:])
    desc = {
        "weights": (
            "C4 时间维因子 = 既有 G1 族种子逐日 mean 信号 parquet 的确定函数"
            f"（s{seed}，G1 信号只读），零训练零推理"
        ),
        "transform": (
            f"三值化（> +{TRINARIZE_THRESHOLD:.0%} → +1；< −{TRINARIZE_THRESHOLD:.0%} → −1；"
            f"否则 0，恰 ±2% 取 0）→ 半衰期加权 Σ λ^k·e(t−k)，k=0..{C4_WINDOW - 1}，"
            f"λ=0.5^(1/{HALF_LIFE_DAYS})={C4_LAMBDA:.10f}；"
            "NaN 不投票、加权跳过（贡献 0）、全 NaN 窗 → NaN"
        ),
        "hyperparams_frozen": (
            f"阈值 ±2% / 半衰期 {HALF_LIFE_DAYS} / 窗口 {C4_WINDOW} 为提案冻结值，"
            "跑后禁扫描（计划 §5）"
        ),
        "windows_note": (
            f"merged {MERGED_WINDOW[0]}~{MERGED_WINDOW[1]} 连续 260 日"
            f"（2025h2 126 + backtest 134 外连接列并集）；预热前 {WARMUP_DAYS} 日"
            "包含在归档中，评估自首日后第 30 个交易日起"
        ),
        "source_parquets": "G7 封盘同款 G1 族映射（g1/G1、g2/G2S101、g2/G2S102、"
                           "g5_head/G1@2025h2）",
        "note": "时间维处理层首轮：符号化（对症幅度多噪声发现）+ 平滑降换手；判据 T1~T5 跑前冻结",
    }
    windows = f"merged {MERGED_WINDOW[0]}~{MERGED_WINDOW[1]}"
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (arm, windows, json.dumps(desc, ensure_ascii=False), created)


def main() -> None:
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：既有归档为前置"

    con = duckdb.connect(str(DB_PATH))
    existing_arms = {r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()}
    todo = [a for a in ARM_SPECS if a not in existing_arms]
    if not todo:
        logger.info("全部 C4 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"已归档臂 {sorted(existing_arms)}；本次追加 {todo}")

    tables: list[pd.DataFrame] = []
    for arm in todo:
        sub = ARM_SPECS[arm]
        p = DATA_DIR / sub / f"daily_signals_merged_{arm}_{VARIANT}.parquet"
        assert p.exists(), f"信号缺失：{p}（先跑 run_c4_signals）"
        tables.append(_long_from_wide(arm, VARIANT, pd.read_parquet(p)))
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

    # —— 对拍：每个新臂抽 (arm, c4, merged 中位日) 与源 parquet 逐值比对 ——
    for arm in todo:
        wide = pd.read_parquet(
            DATA_DIR / ARM_SPECS[arm] / f"daily_signals_merged_{arm}_{VARIANT}.parquet"
        )
        d = wide.index[len(wide) // 2]
        row = con.execute(
            "SELECT code, value FROM signals WHERE arm=? AND variant=? AND date=? "
            "ORDER BY code",
            [arm, VARIANT, d.date().isoformat()],
        ).fetchall()
        src = wide.loc[d].dropna().sort_index()
        assert len(row) == len(src), f"{arm}/{VARIANT}/{d.date()} 行数不一致"
        max_diff = max(abs(r[1] - s) for r, s in zip(row, src.values))
        assert max_diff < 1e-12, f"{arm} 对拍失败 max|Δ|={max_diff}"
        logger.info(f"对拍 {arm}/{VARIANT}/{d.date().isoformat()}：{len(src)} 只，"
                    f"max|Δ|={max_diff:.1e} 一致")

    con.close()
    logger.info("C4 臂 DuckDB 追加完成")


if __name__ == "__main__":
    main()
