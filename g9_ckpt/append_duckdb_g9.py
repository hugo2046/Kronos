"""DuckDB 归档追加 G9 臂（计划 §4.3，20260821 G9 计划）。

复用 ``build_duckdb._long_from_wide``（纪律：DuckDB append 复用），对既有
``finetune_suite/data/signals.duckdb`` 只追加不重建，逐臂幂等：

- arm='G9E1' / 'G9E15'：每臂 8 组（4 变体 × 2 窗——backtest + 2025H2，判据臂）；
- arm='G9E5' / 'G9E10' / 'G9E0'：每臂 4 组（仅 backtest，描述性曲线臂）；
- ``runs`` 元数据按臂追加（epoch 号/官方底座/tokenizer 只读/推理口径 JSON）；
- 写入后每臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值对拍（贴输出）；
  SQL 参数仅常量域（arm ∈ 冻结字典键），对拍日期在 pandas 侧过滤。

用法：``/home/user/miniconda3/envs/quant/bin/python -m g9_ckpt.append_duckdb_g9``
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
G9_DATA_DIR = PKG_DIR / "data"
DB_PATH = PKG_DIR.parent / "finetune_suite" / "data" / "signals.duckdb"

H2_START, H2_END = "2025-07-01", "2025-12-31"

# 臂 × 窗规格（计划 §1 臂表冻结：E1/E15 双窗；E5/E10/E0 仅 backtest）
ARM_SPECS: dict[str, tuple[str, list[str]]] = {
    "G9E1": ("e1", ["backtest", "2025h2"]),
    "G9E15": ("e15", ["backtest", "2025h2"]),
    "G9E5": ("e5", ["backtest"]),
    "G9E10": ("e10", ["backtest"]),
    "G9E0": ("e0", ["backtest"]),
}


def _runs_row(arm: str) -> tuple[str, str, str, str]:
    from g9_ckpt.run_g9_signals import G9_CKPT_ROOT, OFFICIAL_PREDICTOR, arm_model_path

    epoch = int(arm[3:]) if arm != "G9E0" else 0
    desc = {
        "weights": (
            f"G9 checkpoint 选择臂：epoch {epoch}（"
            + ("官方 Kronos-base 底座，零训练）" if arm == "G9E0"
               else f"{G9_CKPT_ROOT}/epoch_{epoch}，G1 配方 seed=100 重训")
        ),
        "tokenizer_path": "G1 s100 tokenizer（冻结只读共享）",
        "predictor_path": arm_model_path("E0" if arm == "G9E0" else arm[2:]),
        "official_predictor": OFFICIAL_PREDICTOR,
        "inference": CANONICAL_INFERENCE,
        "note": (
            "唯一变量=选用第几个 epoch 的 checkpoint；E15 为唯一预声明对照臂，"
            "E5/E10/E0 描述性不进判据；推理 seed 恒 42"
        ),
    }
    windows = f"backtest {BACKTEST_START}~{BACKTEST_END}"
    if "2025h2" in ARM_SPECS[arm][1]:
        windows += f"; 2025h2 {H2_START}~{H2_END}"
    return (
        arm, windows, json.dumps(desc, ensure_ascii=False),
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _pick_backtest_day(con, arm: str) -> tuple[str, dict[str, float], pd.Series]:
    """抽检准备：backtest mean 中位日 → (qdate, 归档值 dict, 源 Series)。

    SQL 参数仅常量域（arm 为冻结字典键、variant 为字面量）；对拍日期的
    过滤在 pandas 侧完成（数据衍生值不进 SQL）。
    """
    sub = ARM_SPECS[arm][0]
    wide = pd.read_parquet(G9_DATA_DIR / sub / f"daily_signals_backtest_{arm}_mean.parquet")
    d = wide.index[len(wide) // 2]
    qdate = str(d.date())
    row_all = con.execute(
        "SELECT date, code, value FROM signals WHERE arm=? AND variant=?",
        [arm, "mean"],
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
    todo = [(a, spec) for a, spec in ARM_SPECS.items() if a not in existing_arms]
    if not todo:
        logger.info("全部 G9 臂已归档（逐臂幂等）→ 无事可做")
        con.close()
        return
    logger.info(f"已归档臂 {sorted(existing_arms)}；本次追加 {[a for a, _ in todo]}")

    tables: list[pd.DataFrame] = []
    for arm, (sub, windows) in todo:
        for window in windows:
            for v in VARIANTS:
                wide = pd.read_parquet(
                    G9_DATA_DIR / sub / f"daily_signals_{window}_{arm}_{v}.parquet"
                )
                tables.append(_long_from_wide(arm, v, wide))
    append_df = pd.concat(tables, ignore_index=True)
    logger.info(f"追加长表合计 {len(append_df)} 行（arm×variant 分布见下）")
    print(append_df.groupby(["arm", "variant"]).size().to_string())

    con.executemany(
        "INSERT INTO runs VALUES (?, ?, ?, ?)", [_runs_row(a) for a, _ in todo]
    )
    con.register("append_df", append_df)
    con.execute("INSERT INTO signals SELECT arm, variant, date, code, value FROM append_df")

    n_total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    logger.info(f"追加完成：signals 总 {n_total} 行，runs 追加 {len(todo)} 行")

    # —— 对拍：每个新臂抽 (arm, mean, backtest 中位日) 与源 parquet 逐值比对 ——
    for arm, _ in todo:
        qdate, arc, src = _pick_backtest_day(con, arm)
        assert len(arc) == len(src), (
            f"{arm}/mean/{qdate} 行数不一致：归档 {len(arc)} vs 源 {len(src)}"
        )
        max_diff = max(
            (abs(arc[code] - float(src[code])) for code in src.index), default=0.0
        )
        assert max_diff < 1e-12, f"{arm}/mean/{qdate} 对拍失败 max|Δ|={max_diff}"
        print(f"对拍 {arm}/mean/backtest/{qdate}: {len(src)} 值, max|Δ|={max_diff:.1e} → 一致")
    con.close()
    logger.info(f"DuckDB 追加归档完成：{DB_PATH}")


if __name__ == "__main__":
    main()
