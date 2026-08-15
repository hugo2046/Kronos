"""finetune_ashares 阶段 1 契约测试（计划 §3 步骤 1.1/1.2，20260815 计划）。

G0 = 第 4 轮 F1 权重（不动一字）推理 2025-07-01~2025-12-31（oos1 内、F1
从未在此段出过数）。落盘 ``finetune_suite/data/g0/``：

- ``test_g0_signals_alignment``：9 张宽表（G0=F1@2025H2 ×4 + F0 ×4 + M）日期
  索引完全一致、窗界 ∈ [2025-07-01, 2025-12-31]、且为既有 oos parquet 索引
  的子集（F0/M 是切取、G0 推理调仓日应与交易日历一致）；
- ``test_g0_backtest_sealed``：引擎结果 JSON 存在、覆盖 9 组 × 双基准、
  **不含任何判读字段**（纪律 §8：G0 数字落盘后不判读，统一阶段 3 封盘）。

依赖 DDB 推理先完成（约 1.2h，断点续跑），测试本身只读文件。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from baseline_suite.common import DATA_DIR as BL_DATA_DIR, VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[1]
G0_DIR = REPO_ROOT / "finetune_suite" / "data" / "g0"

G0_START, G0_END = "2025-07-01", "2025-12-31"


def _paths() -> dict[str, Path]:
    paths = {f"G0_{v}": G0_DIR / f"daily_signals_2025h2_G0_{v}.parquet" for v in VARIANTS}
    for v in VARIANTS:
        paths[f"F0_{v}"] = G0_DIR / f"daily_signals_2025h2_F0_{v}.parquet"
    paths["M"] = G0_DIR / "daily_signals_2025h2_M.parquet"
    return paths


def test_g0_signals_alignment():
    paths = _paths()
    missing = [p.name for p in paths.values() if not p.exists()]
    assert not missing, f"G0 信号缺失 {missing}：先运行 run_g0_signals.py（断点续跑）"

    arms = {k: pd.read_parquet(p) for k, p in paths.items()}
    ref_idx = arms["G0_mean"].index
    assert len(ref_idx) > 0, "G0_mean 信号为空"
    for name, df in arms.items():
        assert df.index.equals(ref_idx), f"{name} 日期索引与 G0_mean 不一致"
        assert str(ref_idx.min().date()) >= G0_START
        assert str(ref_idx.max().date()) <= G0_END
    # F0/M 切取自 oos parquet → 其索引必须是 oos 索引的子集（同日同名）
    oos_idx = pd.read_parquet(BL_DATA_DIR / "daily_signals_oos_M.parquet").index
    assert ref_idx.isin(oos_idx).all(), "G0 调仓日超出 oos parquet 索引范围"


def test_g0_backtest_sealed():
    path = G0_DIR / "g0_backtest_results.json"
    assert path.exists(), "g0_backtest_results.json 缺失：先运行 run_g0_backtest.py"
    out = json.loads(path.read_text(encoding="utf-8"))

    assert out["window"] == "g0_2025h2"
    assert out["period"] == [G0_START, G0_END]
    groups = out["groups"]
    expected = {"M"} | {f"F0_{v}" for v in VARIANTS} | {f"G0_{v}" for v in VARIANTS}
    assert set(groups) == expected, f"组缺失/多余：{set(groups) ^ expected}"
    for tag, res in groups.items():
        assert set(res) == {"perf_idx", "perf_ew"}, f"{tag} 绩效键漂移：{set(res)}"
        assert {"aer", "ir"} <= set(res["perf_ew"]), f"{tag} AER/IR 缺失"
    # 纪律 §8：阶段 1 只落盘不判读——不得携带任何判定字段
    assert "verdict" not in out and "criterion" not in json.dumps(out).lower()[:200], (
        "G0 结果含判读字段（阶段 3 之前禁止）"
    )
