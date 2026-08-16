"""G2.2b：DuckDB 归档追加 G2 种子臂（计划 §1 + 跑前增补 37aba7d）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，**逐臂幂等**（已存在的臂
跳过，缺失的臂追加——支持核心臂与增补臂分批落库）：

- 核心（计划 §1）：arm='G2S101' / 'G2S102'，每臂 8 组（4 变体 × 2 窗）；
- 增补（37aba7d）：arm='G2S103' / 'G2S104' / 'DTOK'，每臂 4 组（仅 backtest 窗，
  增补条款明确不做 2025H2）；
- ``runs`` 元数据按臂追加（窗口/种子/权重/推理口径 JSON）；
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

# 臂 → (子目录, 窗口列表)：核心臂两窗；增补臂仅 backtest（增补条款）
ARM_SPECS: dict[str, tuple[str, list[str]]] = {
    "G2S101": ("s101", ["backtest", "2025h2"]),
    "G2S102": ("s102", ["backtest", "2025h2"]),
    "G2S103": ("s103", ["backtest"]),
    "G2S104": ("s104", ["backtest"]),
    "DTOK": ("dtok", ["backtest"]),
}


def _runs_row(arm: str) -> tuple[str, str, str, str]:
    from finetune_suite.train_dtok import DtokConfig
    from finetune_suite.train_g1 import G1Config
    from finetune_suite.train_g2 import G2Config

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    core_windows = f"backtest {BACKTEST_START}~{BACKTEST_END}; 2025h2 {H2_START}~{H2_END}"
    bt_windows = f"backtest {BACKTEST_START}~{BACKTEST_END}（增补条款：仅 backtest）"

    if arm == "DTOK":
        cfg = DtokConfig()
        desc = {
            "weights": "D-tok 增补臂：tokenizer+predictor 全管线 seed=101 重训",
            "tokenizer_path": cfg.finetuned_tokenizer_path,
            "predictor_path": cfg.finetuned_predictor_path,
            "corpus": "finetune_suite/data/ashares（与 G1 逐字同 pkl）",
            "inference": CANONICAL_INFERENCE,
            "note": "补 tokenizer 种子敏感性洞（对照共享 G1 tokenizer 的 G2S101）",
        }
        window = bt_windows
    else:
        seed = int(arm[3:])
        cfg = G2Config(seed)
        desc = {
            "weights": f"G1 predictor 以训练种子 seed={seed} 重训（唯一变量）",
            "tokenizer_path": cfg.finetuned_tokenizer_path,
            "predictor_path": cfg.finetuned_predictor_path,
            "corpus": "finetune_suite/data/ashares（与 G1 逐字同 pkl）",
            "inference": CANONICAL_INFERENCE,
            "note": "种子诊断臂：协议/超参/epochs/语料逐字复用，推理 seed 恒 42",
        }
        window = core_windows if seed in (101, 102) else bt_windows
        if seed in (103, 104):
            desc["note"] += "；增补 D-seed+（5 种子面板）"
    _ = G1Config()  # 显式引用共享 tokenizer 条款来源
    return (arm, window, json.dumps(desc, ensure_ascii=False), created)


def main() -> None:
    assert DB_PATH.exists(), f"{DB_PATH} 缺失：既有归档为前置"

    con = duckdb.connect(str(DB_PATH))
    existing_arms = {r[0] for r in con.execute("SELECT DISTINCT arm FROM signals").fetchall()}
    todo = [(arm, spec) for arm, spec in ARM_SPECS.items() if arm not in existing_arms]
    if not todo:
        logger.info("全部 G2 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"已归档臂 {sorted(existing_arms)}；本次追加 {[a for a, _ in todo]}")

    tables: list[pd.DataFrame] = []
    for arm, (sub, windows) in todo:
        for window in windows:
            for v in VARIANTS:
                wide = pd.read_parquet(
                    G2_DIR / sub / f"daily_signals_{window}_{arm}_{v}.parquet"
                )
                tables.append(_long_from_wide(arm, v, wide))
    append_df = pd.concat(tables, ignore_index=True)
    logger.info(f"追加长表合计 {len(append_df)} 行（arm×variant 分布见下）")
    print(append_df.groupby(["arm", "variant"]).size().to_string())

    con.executemany(
        'INSERT INTO runs VALUES (?, ?, ?, ?)', [_runs_row(arm) for arm, _ in todo]
    )
    con.register("append_df", append_df)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM append_df")

    n_total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    logger.info(f"追加完成：signals 总 {n_total} 行，runs 追加 {len(todo)} 行")

    # —— 对拍：每个新臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值比对 ——
    for arm, _ in todo:
        wide = pd.read_parquet(G2_DIR / ARM_SPECS[arm][0] / f"daily_signals_backtest_{arm}_mean.parquet")
        d = wide.index[len(wide) // 2]
        row = con.execute(
            "SELECT code, value FROM signals WHERE arm=? AND variant=? AND date=? "
            "ORDER BY code",
            [arm, "mean", d.date().isoformat()],
        ).fetchall()
        src = wide.loc[d].dropna().sort_index()
        assert len(row) == len(src), f"{arm}/mean/{d.date()} 行数不一致"
        max_diff = max(abs(val - float(src[code])) for code, val in row)
        assert max_diff < 1e-12, f"{arm}/mean/{d.date()} 对拍失败 max|Δ|={max_diff}"
        print(
            f"对拍 {arm}/mean/backtest/{d.date().isoformat()}: {len(row)} 值, "
            f"max|Δ|={max_diff:.1e} → 一致"
        )
    con.close()
    logger.info(f"DuckDB 追加归档完成：{DB_PATH}")


if __name__ == "__main__":
    main()
