"""长窗四变体信号生成（L1 专用：lookback 自适应分块，其余逐字 canonical）。

与 ``baseline_suite.signal.run_variant_signals`` 同一推理链路（同
``predict_batch_chunked``、同 seed、同 N、同四变体聚合与断点续跑协议），唯一差别 =
``chunk_size`` 随 lookback 缩放：分块是纯显存簿记（不改变任何数学），L=90 冻结
口径用 32；L=250/500 序列 token 数 ~2.8×/5.6×，按 token 等价缩到 10/5
（RTX 5090 32GB 实测安全档，N=20 采样叠加）。

用法：见 :func:`run_lb_variant_signals`（由 ``run_l1_signals`` 按 臂表 调用）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import VARIANTS, BaselineConfig

# L=90 冻结口径的实测安全块（paper_replication 既有事实）
_BASE_CHUNK, _BASE_LOOKBACK = 32, 90


def lb_chunk_size(lookback: int) -> int:
    """token 等价分块：chunk ∝ 1/lookback，下限 4（L500 → 5）。"""
    return max(4, int(_BASE_CHUNK * _BASE_LOOKBACK / lookback))


def run_lb_variant_signals(
    predictor,
    provider,
    cfg: BaselineConfig,
    rebalances: pd.DatetimeIndex,
    *,
    chunk_size: int,
    checkpoint_dir=None,
    progress_every: int = 10,
) -> dict[str, pd.DataFrame]:
    """逐交易日四变体信号宽表（镜像 run_variant_signals + chunk_size 透传）。"""
    import torch

    from baseline_suite.signal import compute_variants_from_preds
    from kronos_qlib import build_inference_windows
    from paper_replication.signal import predict_batch_chunked

    rows: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    done_dates: set[pd.Timestamp] = set()

    if checkpoint_dir is not None:
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)
        ckpt_paths = {
            v: checkpoint_dir / f"daily_signals_{cfg.window}_{v}.parquet" for v in VARIANTS
        }
        if all(p.exists() for p in ckpt_paths.values()):
            existing = {v: pd.read_parquet(ckpt_paths[v]) for v in VARIANTS}
            done_dates = set(pd.to_datetime(existing["mean"].index))
            for v in VARIANTS:
                rows[v] = [existing[v].loc[d].dropna().to_dict() for d in existing[v].index]
            logger.info(f"四变体断点续跑：已有 {len(done_dates)} 日，跳过")

    pending = [d for d in rebalances if d not in done_dates]
    logger.info(
        f"四变体信号 [{cfg.window}]：{len(rebalances)} 日总计，{len(pending)} 日待跑"
        f"（L={cfg.lookback} N={cfg.sample_count} chunk={chunk_size}）"
    )

    for i, d in enumerate(pending):
        ds = d.strftime("%Y-%m-%d")
        df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows(
            provider, ds, lookback=cfg.lookback, predict_len=cfg.predict_len, pool=cfg.pool,
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
            top_p=cfg.top_p, sample_count=cfg.sample_count, chunk_size=chunk_size,
        )
        day_signals: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
        for j, pred_df in enumerate(preds):
            variants = compute_variants_from_preds(pred_df[cfg.signal_field], last_closes[j])
            for v in VARIANTS:
                day_signals[v][codes[j]] = variants[v]
        for v in VARIANTS:
            rows[v].append(day_signals[v])

        if (i + 1) % progress_every == 0 or i == 0:
            logger.info(
                f"四变体信号 [{i + 1}/{len(pending)}] {ds}: {len(codes)} 只"
                f"（kept={stats.get('n_kept')} short={stats.get('skipped_short')} "
                f"halt={stats.get('skipped_halt')}），"
                f"mean std={np.std(list(day_signals['mean'].values())):.4f}"
            )
            if checkpoint_dir is not None:
                _dump_partial(rows, rebalances, pending[: i + 1], done_dates, ckpt_paths)

    wide = {}
    for v in VARIANTS:
        wide[v] = pd.DataFrame(rows[v], index=rebalances)
        logger.info(
            f"{v} 信号宽表 [{cfg.window}]：{wide[v].shape[0]} 日 × "
            f"平均 {wide[v].notna().sum(axis=1).mean():.0f} 只/日"
        )
    return wide


def _dump_partial(rows, rebalances, pending, done_dates, ckpt_paths) -> None:
    """已完成日合并落盘（与 baseline_suite.signal._dump_partial 同构）。"""
    done_sorted = sorted(done_dates)
    n_done = len(done_sorted)
    for v in VARIANTS:
        done_rows = rows[v][:n_done] if n_done > 0 else []
        new_rows = rows[v][n_done:]
        all_idx = done_sorted + list(pending[: len(new_rows)])
        partial = pd.DataFrame(done_rows + new_rows, index=all_idx)
        partial.to_parquet(ckpt_paths[v])


__all__ = ["run_lb_variant_signals", "lb_chunk_size"]
