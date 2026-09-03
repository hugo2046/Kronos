"""L1 引擎全表封存（计划 §4.3 后半，**落盘不判读，一次开封**）。

canonical 引擎（csi300, k=50/n=5/min_hold=5/15bp）双基准（000300.SH 市值加权 +
同池等权），两窗全组：

- backtest（2026-01-01~2026-07-24）：参照 M + F0×4；L90 锚 ×3（G1_mean(s100) +
  G2S101_mean + G2S102_mean，既有信号只读）；L1 五臂 × 4 变体；
- 2025H2（2025-07-01~2025-12-31）：参照 + L90 锚 ×3（2025h2 parquet）+
  L250-zs 三种子 × 4 变体（计划 §1：可比性窗）。

引擎宇宙与 ``g5_head.run_g5_eval.UNIVERSE_PARQUETS`` 逐字一致（冻结口径轮次并集
——在位者 backtest 复现冻结 +14.33% 的门禁前提；L90 锚 G2S101/102 在该宇宙下
重算，与冻结值（+29.06/+18.67）差 > 1pp 时如实记录于封存 JSON 的
``anchor_recheck``——判读配对一律用**同引擎同宇宙**重算值，冻结值并列披露）。

**纪律 §5**：绩效数字只写封存日志 ``l1_context/data/l1_backtest_sealed.log``
与结果 JSON（judge 之前不读不贴）；终端只打印结构信息。

产出：``l1_context/data/l1_backtest_results.json``（LC1~LC4 判读的冻结输入）。

用法：``/home/user/miniconda3/envs/quant/bin/python -m l1_context.run_l1_backtest``
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
L1_DATA_DIR = PKG_DIR / "data"
R4 = PKG_DIR.parent / "finetune_suite" / "data"

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.pipeline import build_dual_benchmarks, run_group
from baseline_suite.signal import build_px_tradeable

from l1_context.config import ARMS, L90_ANCHOR_PARQUETS, L90_FROZEN_AER_EW, WINDOW_DEFS, arm_tag

# 引擎宇宙（冻结口径轮次并集；只读 import 逐字镜像 g5_head.run_g5_eval）
UNIVERSE_PARQUETS = {
    "backtest": [R4 / "g1" / f"daily_signals_backtest_G1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F1_{v}.parquet" for v in VARIANTS]
    + [R4 / f"daily_signals_backtest_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "daily_signals_backtest_M.parquet"],
    "2025h2": [R4 / "g0" / f"daily_signals_2025h2_G0_{v}.parquet" for v in VARIANTS]
    + [R4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet" for v in VARIANTS]
    + [R4 / "g0" / "daily_signals_2025h2_M.parquet"],
}

ANCHOR_TAG = {"s100": "G1_mean", "s101": "G2S101_mean", "s102": "G2S102_mean"}
INCUMBENT_TOL = 5e-4   # 在位者复现容差（g5 同款）
ANCHOR_TOL = 0.01      # G2 锚重算容差 1pp（K0 同款；超差不阻断、如实披露）


def _window_arms(window: str) -> list[str]:
    return [t for t, spec in ARMS.items() if window in spec["windows"]]


def _load_window_signals(window: str) -> dict[str, pd.DataFrame]:
    """一窗全部宽表：参照 + L90 锚（只读）+ L1 臂 × 4 变体。"""
    signals: dict[str, pd.DataFrame] = {}
    if window == "backtest":
        for v in VARIANTS:
            signals[f"F0_{v}"] = pd.read_parquet(R4 / f"daily_signals_backtest_F0_{v}.parquet")
        signals["M"] = pd.read_parquet(R4 / "daily_signals_backtest_M.parquet")
    else:
        for v in VARIANTS:
            signals[f"F0_{v}"] = pd.read_parquet(R4 / "g0" / f"daily_signals_2025h2_F0_{v}.parquet")
        signals["M"] = pd.read_parquet(R4 / "g0" / "daily_signals_2025h2_M.parquet")
    for seed, p in L90_ANCHOR_PARQUETS[window].items():
        signals[ANCHOR_TAG[seed]] = pd.read_parquet(p)
    for tag in _window_arms(window):
        atag = arm_tag(tag)
        for v in VARIANTS:
            p = L1_DATA_DIR / tag / f"daily_signals_{window}_{atag}_{v}.parquet"
            assert p.exists(), f"L1 信号缺失：{p}（先跑 run_l1_signals）"
            signals[f"{atag}_{v}"] = pd.read_parquet(p)
    return signals


def _order(window: str) -> list[str]:
    head = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")]
    anchors = [ANCHOR_TAG[s] for s in ("s100", "s101", "s102")]
    tail = [f"{arm_tag(t)}_{v}" for t in _window_arms(window)
            for v in ("min", "max", "last", "mean")]
    return head + anchors + tail


def main() -> None:
    sealed_log = L1_DATA_DIR / "l1_backtest_sealed.log"
    logger.add(str(sealed_log), enqueue=False)

    results: dict[str, dict] = {}
    anchors_recheck: dict[str, dict] = {}
    beta_gaps: dict[str, float] = {}
    n_days_total = 0

    for window, (start, end) in WINDOW_DEFS.items():
        cfg = replace(BaselineConfig.load(window="oos"),
                      window=f"l1_{window}", backtest_start=start, backtest_end=end)
        signals = _load_window_signals(window)
        order = _order(window)

        ref_idx = signals["M"].index
        for tag in _window_arms(window):
            atag = arm_tag(tag)
            assert signals[f"{atag}_mean"].index.equals(ref_idx), (
                f"[{window}] {atag} 日期索引与参照网格不一致")

        universe_cols = sorted(set().union(*[
            set(pd.read_parquet(p).columns) for p in UNIVERSE_PARQUETS[window]]))
        logger.info(f"[{window}] 引擎宇宙：{len(universe_cols)} 列（冻结口径轮次并集）")

        from kronos_qlib import QlibProvider

        provider = QlibProvider(cfg.pool, start, end)
        rebalances = pd.DatetimeIndex(ref_idx)
        px, trd = build_px_tradeable(provider, cfg, rebalances, universe_cols)
        bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)
        beta_gaps[window] = beta_gap

        for tag in order:
            pi, pe, _, _, _ = run_group(signals[tag], px, trd, bench_idx, bench_ew,
                                        cfg=cfg, name=tag)
            results[f"{window}:{tag}"] = {"perf_idx": pi.to_dict(), "perf_ew": pe.to_dict()}
        n_days_total += len(rebalances)

        # —— 锚完整性（写封存日志；判读用同引擎重算值，冻结值并列）——
        for seed in ("s100", "s101", "s102"):
            rerun = float(results[f"{window}:{ANCHOR_TAG[seed]}"]["perf_ew"]["aer"])
            frozen = L90_FROZEN_AER_EW[seed] if window == "backtest" else None
            entry = {"rerun_aer_ew": rerun}
            if frozen is not None:
                entry["frozen_aer_ew"] = frozen
                entry["abs_diff"] = abs(rerun - frozen)
                entry["within_tol"] = bool(entry["abs_diff"] <= ANCHOR_TOL)
                if seed == "s100":
                    assert entry["abs_diff"] < INCUMBENT_TOL, (
                        f"在位者引擎复跑 {rerun:+.4%} 与冻结 {frozen:+.4%} 不一致"
                        f"（容差 {INCUMBENT_TOL}）")
                logger.info(
                    f"[{window}] 锚 {ANCHOR_TAG[seed]} 重算 {rerun:+.4%} vs 冻结 "
                    f"{frozen:+.4%} 差 {entry['abs_diff']:.4%}"
                    f"（{'一致' if entry['within_tol'] else '超 1pp——判读用重算值并披露'}）")
            anchors_recheck.setdefault(window, {})[seed] = entry

    out = {
        "experiment": "l1_backtest_sealed",
        "date": "2026-09-03",
        "windows": {w: list(b) for w, b in WINDOW_DEFS.items()},
        "engine": {"pool": "csi300", "top_k": 50, "drop_n": 5, "min_hold": 5,
                   "cost_bps": 15.0, "benchmarks": ["000300.SH 指数", "同池等权"],
                   "beta_gap": beta_gaps},
        "groups": results,
        "anchor_recheck": anchors_recheck,
        "note": "纪律 §5：数字落盘后不判读不外泄，统一 run_l1_judge 一次开封（LC1~LC4）",
    }
    out_path = L1_DATA_DIR / "l1_backtest_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))

    print(f"L1 引擎出表完成：{len(results)} 组 × 双基准，{n_days_total} 日（两窗合计）")
    for window in WINDOW_DEFS:
        print(f"窗口 {window}：{len(_order(window))} 组，组序 {_order(window)}")
    print(f"锚完整性检查：backtest 三锚已写入封存 JSON anchor_recheck（judge 时披露）")
    print(f"结果落盘：{out_path}")
    print(f"封存日志：{sealed_log}（judge 之前不读不贴）")


if __name__ == "__main__":
    main()
