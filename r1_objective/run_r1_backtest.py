"""R1 引擎全表封存（计划 §4.2，**落盘不判读，一次开封**）。

canonical 引擎（csi300, k=50/n=5/min_hold=5/15bp，双基准 000300.SH + 同池等权）：
两窗 ×（6 R 模型 + 在位者 G1_mean + F0×4 / M 只读参照）。引擎宇宙与
``g5_head.run_g5_eval.UNIVERSE_PARQUETS`` 逐字一致（冻结口径轮次并集——在位者
backtest 与冻结 +14.33% 逐位可比的门禁前提，与 G5 直接可比）。

**纪律 §5**：绩效数字只写封存日志 ``r1_objective/data/r1_backtest_sealed.log``
与结果 JSON（judge 之前不读不贴）；终端只打印结构信息。

产出：``r1_objective/data/r1_backtest_results.json``（RC1~RC4 判读的冻结输入）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m r1_objective.run_r1_backtest``
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

# —— 纪律 §5：移除 console sink，数字只落封存日志 ——
logger.remove()

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
R4 = PKG_DIR.parent / "finetune_suite" / "data"
G5_DATA = PKG_DIR.parent / "g5_head" / "data"

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable
from r1_objective.run_r1_signals import MODEL_NAMES, WINDOW_BOUNDS

# 在位者（G1 s100，L90 锚；与 G5 判据锚同源）
INCUMBENT_AER_EW = 0.1433
INCUMBENT_PARQUET = {
    "backtest": R4 / "g1" / "daily_signals_backtest_G1_mean.parquet",
    "2025h2": G5_DATA / "daily_signals_2025h2_G1_mean.parquet",
}
# 引擎宇宙（冻结口径轮次并集；镜像 g5_head.run_g5_eval，只读 import 逐字）
UNIVERSE_PARQUETS = {
    "backtest": [R4 / "g1" / f"daily_signals_backtest_G1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "daily_signals_backtest_M.parquet"],
    "2025h2": [R4 / "g0" / f"daily_signals_2025h2_G0_{v}.parquet" for v in VARIANTS]
    + [R4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "g0" / "daily_signals_2025h2_M.parquet"],
}


def _load_window_signals(window: str) -> dict[str, pd.DataFrame]:
    """一窗全部宽表：F0×4/M 参照 + 在位者 G1_mean + 6 R 模型（mean 变体）。"""
    signals: dict[str, pd.DataFrame] = {}
    if window == "backtest":
        for v in VARIANTS:
            signals[f"F0_{v}"] = pd.read_parquet(R4 / f"daily_signals_backtest_F0_{v}.parquet")
        signals["M"] = pd.read_parquet(R4 / "daily_signals_backtest_M.parquet")
    else:
        for v in VARIANTS:
            signals[f"F0_{v}"] = pd.read_parquet(R4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet")
        signals["M"] = pd.read_parquet(R4 / "g0" / "daily_signals_2025h2_M.parquet")
    signals["G1_mean"] = pd.read_parquet(INCUMBENT_PARQUET[window])
    for n in MODEL_NAMES:
        p = DATA_DIR / f"daily_signals_{window}_{n}.parquet"
        assert p.exists(), f"R1 信号缺失：{p}（先跑 run_r1_signals）"
        signals[n] = pd.read_parquet(p)
    return signals


def main() -> None:
    sealed_log = DATA_DIR / "r1_backtest_sealed.log"
    logger.add(str(sealed_log), enqueue=False)

    results: dict[str, dict] = {n: {} for n in MODEL_NAMES}
    ctx: dict = {}
    incumbent: dict[str, dict] = {}

    for window, (start, end) in WINDOW_BOUNDS.items():
        cfg = replace(BaselineConfig.load(window="oos"),
                      window=f"r1_{window}", backtest_start=start, backtest_end=end)
        signals = _load_window_signals(window)

        # 索引一致性（在位者 vs 模型打分网格）
        first = signals[MODEL_NAMES[0]]
        assert signals["G1_mean"].index.equals(first.index), (
            f"[{window}] 在位者索引与 R1 打分网格不一致")

        universe_cols = sorted(set().union(*[
            set(pd.read_parquet(p).columns) for p in UNIVERSE_PARQUETS[window]]))
        logger.info(f"[{window}] 引擎宇宙：{len(universe_cols)} 列（冻结口径轮次并集）")

        from kronos_qlib import QlibProvider

        provider = QlibProvider(cfg.pool, start, end)
        rebalances = pd.DatetimeIndex(first.index)
        px, trd = build_px_tradeable(provider, cfg, rebalances, universe_cols)
        bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)
        ctx[window] = {"beta_gap": beta_gap}

        for name, sig in signals.items():
            pi, pe, _, _, _ = run_group(sig, px, trd, bench_idx, bench_ew, cfg=cfg, name=name)
            r = {"perf_idx": pi.to_dict(), "perf_ew": pe.to_dict()}
            if name in results:
                results[name][window] = r
            else:
                incumbent.setdefault(name, {})[window] = r

        # 在位者对拍（backtest 窗与冻结值一致 → 评估链路无误；写入封存日志）
        if window == "backtest":
            rerun = incumbent["G1_mean"]["backtest"]["perf_ew"]["aer"]
            assert abs(rerun - INCUMBENT_AER_EW) < 5e-4, (
                f"在位者引擎复跑 {rerun:+.4%} 与冻结 {INCUMBENT_AER_EW:+.4%} 不一致")
            logger.info(f"对拍在位者 G1_mean：复跑 {rerun:+.4%} vs 冻结 {INCUMBENT_AER_EW:+.4%} 一致")

    out = {
        "experiment": "r1_backtest_sealed",
        "date": "2026-09-03",
        "windows": {w: list(b) for w, b in WINDOW_BOUNDS.items()},
        "engine": {"pool": "csi300", "top_k": 50, "drop_n": 5, "min_hold": 5,
                   "cost_bps": 15.0, "benchmarks": ["000300.SH 指数", "同池等权"],
                   "beta_gap": {w: ctx[w]["beta_gap"] for w in WINDOW_BOUNDS}},
        "results": results,
        "incumbent": incumbent,
        "note": "纪律 §5：数字落盘后不判读不外泄，统一 run_r1_judge 一次开封（RC1~RC4）",
    }
    out_path = DATA_DIR / "r1_backtest_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))

    print(f"R1 引擎出表完成：{len(MODEL_NAMES)} 模型 × {len(WINDOW_BOUNDS)} 窗 × 双基准")
    for window in WINDOW_BOUNDS:
        print(f"窗口 {window}：{len(MODEL_NAMES) + 6} 组（6 R + G1_mean + F0×4 + M）")
    print(f"结果落盘：{out_path}")
    print(f"封存日志：{sealed_log}（judge 之前不读不贴）")


if __name__ == "__main__":
    main()
