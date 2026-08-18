"""G4 种子臂两窗四变体信号生成（G4 计划 §3 1.2）。

镜像 ``finetune_suite/run_g2_signals.py``（种子臂/两窗/断点续跑/落盘命名），
唯一差异 = 推理链路换 9 列：

- 窗口构造：``g4_features.infer.build_inference_windows_9col``（原版
  ``kronos_qlib.build_inference_windows`` 包装 + 市场三列右连接）；
- 预测器：``g4_features.infer.G4Predictor``（9 列版 KronosPredictor，
  z-score/AR 解码/N=20 采样均值聚合全部继承原实现）；
- canonical 推理口径逐字不动：L=90/H=10/N=20/T=1.0/top_p=0.9/推理 seed=42
  （推理种子恒 42，与训练种子无关）。

落盘（g4_features/data/s{seed}/，不入库）：
    daily_signals_backtest_G4S{seed}_{last,mean,max,min}.parquet
    daily_signals_2025h2_G4S{seed}_{last,mean,max,min}.parquet

用法::

    python g4_features/run_g4_signals.py --seed 100 --window backtest
    python g4_features/run_g4_signals.py --seed 100 --window 2025h2
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import VARIANTS, BaselineConfig
from baseline_suite.run_signals import build_provider, build_rebalances
from baseline_suite.signal import compute_variants_from_preds
from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
from paper_replication.signal import predict_batch_chunked

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"

WINDOW_DEFS = {
    "backtest": (BACKTEST_START, BACKTEST_END),
    "2025h2": ("2025-07-01", "2025-12-31"),
}


def arm_tag(seed: int) -> str:
    return f"G4S{seed}"


def build_g4_config(seed: int, window: str) -> BaselineConfig:
    from g4_features.config import G4Config

    g4 = G4Config(seed=seed)
    start, end = WINDOW_DEFS[window]
    return replace(
        BaselineConfig.load(window="oos"),
        window=f"{window}_{arm_tag(seed)}",  # 断点续跑文件名即最终名
        backtest_start=start,
        backtest_end=end,
        model_name=g4.finetuned_predictor_path,
        tokenizer_name=g4.finetuned_tokenizer_path,
    )


def _load_g4_predictor(cfg: BaselineConfig):
    from g4_features.infer import G4Predictor
    from model import Kronos, KronosTokenizer

    logger.info(f"加载 G4 tokenizer（9 列）：{cfg.tokenizer_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"加载 G4 predictor：{cfg.model_name}")
    model = Kronos.from_pretrained(cfg.model_name)
    return G4Predictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def run_one(seed: int, window: str) -> None:
    from g4_features.infer import build_inference_windows_9col, build_market_context

    out_dir = DATA_DIR / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_g4_config(seed, window)
    start, end = WINDOW_DEFS[window]
    logger.info(
        f"G4 臂配置：seed={seed} window={cfg.window} [{start}~{end}] "
        f"pool={cfg.pool} N={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} "
        f"推理seed={cfg.seed}"
    )
    logger.info(f"G4 权重（9 列特征 = 唯一变量）：model={cfg.model_name}")
    logger.info(f"                            tokenizer={cfg.tokenizer_name}")

    market = build_market_context(end)
    logger.info(f"市场上下文三列：{market.index[0].date()}~{market.index[-1].date()} "
                f"（{len(market)} 日）")

    rebalances = build_rebalances(cfg)
    provider = build_provider(cfg)
    predictor = _load_g4_predictor(cfg)

    # —— 断点续跑（checkpoint 名 = 最终落盘名，同 run_variant_signals 惯例）——
    rows: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    done_dates: set[pd.Timestamp] = set()
    ckpt_paths = {
        v: out_dir / f"daily_signals_{window}_{arm_tag(seed)}_{v}.parquet"
        for v in VARIANTS
    }
    if all(p.exists() for p in ckpt_paths.values()):
        existing = {v: pd.read_parquet(ckpt_paths[v]) for v in VARIANTS}
        done_dates = set(pd.to_datetime(existing["mean"].index))
        for v in VARIANTS:
            rows[v] = [existing[v].loc[d].dropna().to_dict() for d in existing[v].index]
        logger.info(f"四变体断点续跑：已有 {len(done_dates)} 日，跳过")

    pending = [d for d in rebalances if d not in done_dates]
    logger.info(f"四变体信号 [{cfg.window}]：{len(rebalances)} 日总计，"
                f"{len(pending)} 日待跑（N={cfg.sample_count}）")

    for i, d in enumerate(pending):
        ds = d.strftime("%Y-%m-%d")
        df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows_9col(
            provider, ds, market=market,
            lookback=cfg.lookback, predict_len=cfg.predict_len, pool=cfg.pool,
        )
        if len(df_list) == 0:
            logger.warning(f"{ds}: 无可用股票（{stats}）")
            for v in VARIANTS:
                rows[v].append({})
            continue

        last_closes = [df["close"].iloc[-1] for df in df_list]
        torch.manual_seed(cfg.seed)
        preds = predict_batch_chunked(
            predictor, df_list, x_ts_list, y_ts_list,
            pred_len=cfg.predict_len, T=cfg.T, top_k=cfg.sample_top_k,
            top_p=cfg.top_p, sample_count=cfg.sample_count,
        )
        day_signals: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
        for j, pred_df in enumerate(preds):
            variants = compute_variants_from_preds(
                pred_df[cfg.signal_field], last_closes[j]
            )
            for v in VARIANTS:
                day_signals[v][codes[j]] = variants[v]
        for v in VARIANTS:
            rows[v].append(day_signals[v])

        if (i + 1) % 10 == 0 or i == 0 or i == len(pending) - 1:
            done = len(done_dates) + i + 1
            logger.info(f"  [{done}/{len(rebalances)}] {ds}: {len(df_list)} 只 "
                        f"（skipped_short={stats['skipped_short']} "
                        f"halt={stats['skipped_halt']}）")
            for v in VARIANTS:
                wide = pd.DataFrame(rows[v], index=[*done_dates, *pending[: i + 1]])
                wide.to_parquet(ckpt_paths[v])

    for v in VARIANTS:
        wide = pd.DataFrame(rows[v], index=rebalances)
        out = out_dir / f"daily_signals_{window}_{arm_tag(seed)}_{v}.parquet"
        wide.to_parquet(out)
        logger.info(f"{arm_tag(seed)} {window} {v} 落盘 {out.name}（{wide.shape[0]} 日）")

    # —— 日期索引对齐断言：与在位者 G1 同窗信号逐日一致 ——
    _assert_aligned(seed, window, rebalances)


def _assert_aligned(seed: int, window: str, rebalances: pd.DatetimeIndex) -> None:
    r4 = PKG_DIR.parent / "finetune_suite" / "data"
    incumbent = {
        "backtest": {s: r4 / f"daily_signals_backtest_{a}_{v}.parquet"
                     for s, a in ((100, "G1"), (101, "G2S101"), (102, "G2S102"))
                     for v in ("mean",)},
        "2025h2": {s: r4 / "g2" / f"s{s}" / f"daily_signals_2025h2_G2S{s}_mean.parquet"
                   for s in (101, 102)},
    }
    refs = incumbent[window]
    for s, p in refs.items():
        if not p.exists():
            logger.warning(f"对齐参照缺失（跳过）：{p}")
            continue
        ref_idx = pd.read_parquet(p).index
        assert ref_idx.equals(rebalances), (
            f"在位者 s{s} {window} 日期索引与 G4 不一致："
            f"{ref_idx.min()}~{ref_idx.max()} vs {rebalances.min()}~{rebalances.max()}"
        )
    logger.info(f"对齐断言通过：{len(refs)} 个在位者参照日期索引一致"
                f"（{len(rebalances)} 日）")


def main() -> None:
    parser = argparse.ArgumentParser(description="G4 种子臂两窗四变体推理（9 列）")
    parser.add_argument("--seed", type=int, choices=[100, 101, 102], required=True)
    parser.add_argument("--window", choices=list(WINDOW_DEFS), required=True)
    args = parser.parse_args()
    run_one(args.seed, args.window)


if __name__ == "__main__":
    main()
