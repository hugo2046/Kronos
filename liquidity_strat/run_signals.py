"""信号生成入口（计划 §2.3）。

用法（解释器 /home/user/miniconda3/envs/quant/bin/python）::

    # 1) Kronos canonical mean（union 宇宙，断点续跑，GPU 重头）
    python -m liquidity_strat.run_signals kronos
    # 2) M/R/P 对照（纯取数 + 随机，快）
    python -m liquidity_strat.run_signals baselines
    # 3) 按 (bucket, st_track) 切片（确定性派生）
    python -m liquidity_strat.run_signals slice

    # 计时校准（只跑 1 日，估算总时长）
    python -m liquidity_strat.run_signals kronos --limit 1

落盘（data/ 不入库）：
    - ``daily_signals_K_union.parquet``（Kronos mean，union 宇宙）
    - ``daily_signals_M_union.parquet`` / ``_R_`` / ``_P_``（union 宇宙）
    - ``signal_<bucket>_<track>_<SIGNAL>.parquet``（切片档内表）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from liquidity_strat.common import (
    DATA_END,
    DATA_DIR,
    NEW_SIGNALS,
    SIGNAL_KRONOS,
    SIGNAL_MOM,
    SIGNAL_PLACEHOLDER,
    SIGNAL_REV,
    LiquidityConfig,
    ensure_dirs,
)
from liquidity_strat.signal_gen import (
    daily_union_universe,
    run_kronos_signals,
    run_mr_signals,
    run_placeholder_signals,
    slice_bucket_signals,
)

STRAT_PATH = DATA_DIR / "strat_membership.parquet"


def build_provider(cfg: LiquidityConfig):
    """QlibProvider（ashares 母池 + 回看缓冲 + 结算余量）。"""
    from kronos_qlib import QlibProvider

    fetch_start = (
        pd.Timestamp(cfg.window_start) - pd.Timedelta(days=cfg.lookback * 2)
    ).strftime("%Y-%m-%d")
    return QlibProvider(cfg.pool, fetch_start, DATA_END)


def load_predictor(cfg: LiquidityConfig):
    """加载 KronosPredictor（与 baseline_suite / paper_replication 同口径）。"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def load_strat() -> pd.DataFrame:
    if not STRAT_PATH.exists():
        raise FileNotFoundError(f"分档成员表缺失：{STRAT_PATH}（先跑 probe_stage0）")
    df = pd.read_parquet(STRAT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def cmd_kronos(cfg: LiquidityConfig, limit: int | None) -> None:
    strat = load_strat()
    provider = build_provider(cfg)
    cal, universe_map = daily_union_universe(strat, provider, cfg.window_start, cfg.window_end)
    if limit:
        cal = cal[:limit]
        universe_map = {d: universe_map[d] for d in cal}
        logger.info(f"计时模式：仅跑前 {limit} 日")
    predictor = load_predictor(cfg)
    t0 = time.time()
    wide = run_kronos_signals(predictor, provider, cfg, cal, universe_map, checkpoint_dir=DATA_DIR)
    elapsed = time.time() - t0
    n_days = wide.shape[0]
    logger.info(f"Kronos 完成：{n_days} 日，耗时 {elapsed:.0f}s（{elapsed/max(n_days,1):.1f}s/日）")
    out = DATA_DIR / "daily_signals_K_union.parquet"
    wide.to_parquet(out)
    logger.info(f"K union 落盘：{out}")


def cmd_baselines(cfg: LiquidityConfig) -> None:
    strat = load_strat()
    provider = build_provider(cfg)
    cal, universe_map = daily_union_universe(strat, provider, cfg.window_start, cfg.window_end)
    mom, rev = run_mr_signals(provider, cal, universe_map, DATA_END)
    mom.to_parquet(DATA_DIR / "daily_signals_M_union.parquet")
    rev.to_parquet(DATA_DIR / "daily_signals_R_union.parquet")
    logger.info("M/R union 落盘")
    p = run_placeholder_signals(cal, universe_map, seed=cfg.seed)
    p.to_parquet(DATA_DIR / "daily_signals_P_union.parquet")
    logger.info("P union 落盘")


def cmd_slice(cfg: LiquidityConfig) -> None:
    strat = load_strat()
    provider = build_provider(cfg)
    union = {
        SIGNAL_KRONOS: DATA_DIR / "daily_signals_K_union.parquet",
        SIGNAL_MOM: DATA_DIR / "daily_signals_M_union.parquet",
        SIGNAL_REV: DATA_DIR / "daily_signals_R_union.parquet",
        SIGNAL_PLACEHOLDER: DATA_DIR / "daily_signals_P_union.parquet",
    }
    union_signals = {}
    for tag, p in union.items():
        if not p.exists():
            raise FileNotFoundError(f"union 信号缺失：{p}（先跑 kronos/baselines）")
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        union_signals[tag] = df
    sliced = slice_bucket_signals(union_signals, strat, provider, cfg.window_start, cfg.window_end)
    for (b, tr, sig), wide in sliced.items():
        out = DATA_DIR / f"signal_{b}_{tr}_{sig}.parquet"
        wide.to_parquet(out)
    logger.info(f"切片落盘：{len(sliced)} 张档内信号表")


def main() -> None:
    parser = argparse.ArgumentParser(description="流动性分层信号生成")
    parser.add_argument("cmd", choices=["kronos", "baselines", "slice"], help="kronos | baselines | slice")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 日（计时校准）")
    args = parser.parse_args()
    ensure_dirs()
    cfg = LiquidityConfig.load()
    if args.cmd == "kronos":
        cmd_kronos(cfg, args.limit)
    elif args.cmd == "baselines":
        cmd_baselines(cfg)
    elif args.cmd == "slice":
        cmd_slice(cfg)


if __name__ == "__main__":
    main()
