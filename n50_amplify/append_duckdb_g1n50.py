"""DuckDB 归档追加 N50 臂（N50 计划 §3 3.2，20260819）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，**逐臂幂等**（已存在的臂
跳过）：arm='G1N50S100'/'G1N50S101'/'G1N50S102'，每臂 4 组（4 变体 × backtest
单窗，预算裁定 2025h2 不跑）；``runs`` 元数据按臂追加；写入后抽
(arm, mean, backtest 中位日) 与源 parquet 对拍一致（贴输出）。只追加原始信号，
不含任何评估数字（落盘前置纪律）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m n50_amplify.append_duckdb_g1n50``
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

ARM_SPECS: dict[str, str] = {  # arm -> 子目录（仅 backtest，预算裁定）
    "G1N50S100": "s100",
    "G1N50S101": "s101",
    "G1N50S102": "s102",
}
WINDOWS = ("backtest",)


def _runs_row(arm: str) -> tuple[str, str, str, str]:
    from n50_amplify.run_n50_signals import _g1_family_paths

    seed = int(arm[6:])
    model, tokenizer = _g1_family_paths(seed)
    desc = {
        "weights": (
            "N50 采样放大 = G1 族采样路径数复检：权重只读复用 G1 族种子"
            f"（s{seed}），N 为纯推理期参数，换配置零训练"
        ),
        "tokenizer_path": tokenizer,
        "predictor_path": model,
        "inference": (
            "L=90/H=10/N=50/T=1.0/top_p=0.9/seed=42 "
            "（除 sample_count=50 外逐字 paper_replication/config.yaml canonical）"
        ),
        "windows_note": (
            "仅 backtest（预算裁定 N=50 成本 ≈2.5×，2025h2 不跑，跑前声明）"
        ),
        "note": "采样放大臂：论文敏感性曲线延长线（测到 N=20 为止，20→50 检验）；推理 seed 恒 42",
    }
    windows = f"backtest {BACKTEST_START}~{BACKTEST_END}"
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (arm, windows, json.dumps(desc, ensure_ascii=False), created)


def main() -> None:
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：既有归档为前置"

    con = duckdb.connect(str(DB_PATH))
    existing_arms = {r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()}
    todo = [a for a in ARM_SPECS if a not in existing_arms]
    if not todo:
        logger.info("全部 N50 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"已归档臂 {sorted(existing_arms)}；本次追加 {todo}")

    tables: list[pd.DataFrame] = []
    for arm in todo:
        sub = ARM_SPECS[arm]
        for window in WINDOWS:
            for v in VARIANTS:
                p = DATA_DIR / sub / f"daily_signals_{window}_{arm}_{v}.parquet"
                assert p.exists(), f"信号缺失：{p}（先跑 run_n50_signals）"
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
    logger.info("N50 臂 DuckDB 追加完成")


if __name__ == "__main__":
    main()
