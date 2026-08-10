"""四变体信号聚合（计划 §1）。

四变体**全部除以现价** ``close_t``（修正上游 demo ``qlib_test.py`` 的价格尺度偏差）：

    - ``last = pred_close[t+H] / close_t − 1``
    - ``mean = mean(pred_close[t+1..t+H]) / close_t − 1``  ← canonical 主线
    - ``max  = max(pred_close[t+1..t+H]) / close_t − 1``
    - ``min  = min(pred_close[t+1..t+H]) / close_t − 1``

复用 ``paper_replication.signal.predict_batch_chunked`` + ``kronos_qlib.build_inference_windows``
（同一推理链路），区别仅在：每次推理把**四个聚合一次性算全**，不重复推理。

宽表约定：``index=date, columns=code, values=signal``（引擎消费）。

对拍门禁（§2.2）：本模块的 mean 聚合必须与既有
``paper_replication/data/daily_signals_K.parquet`` 逐位一致——同 seed 同 N 同推理路径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import VARIANTS, BaselineConfig
from paper_replication.signal import predict_batch_chunked


def compute_variants_from_preds(
    pred_close_path: pd.Series, last_close: float
) -> dict[str, float]:
    """从一条预测 close 路径算四变体信号（全部除以现价）。

    :param pred_close_path: ``predict_batch`` 返回的 pred_df 的 ``close`` 列
        （已对 sample_count 次采样取均值，长度 = H）。
    :param last_close: 决策日 t 的后复权 close（窗口最后一行）。
    :returns: ``{"last":..., "mean":..., "max":..., "min":...}``。

    **mean 必须与 ``paper_replication.signal.compute_signal_from_preds`` 逐字一致**
    （对拍门禁）：那边是 ``np.mean(values)/last_close - 1``，本函数 mean 分支同样用
    ``np.mean``（非 nanmean——pred 路径无 NaN，predict_batch 上游已断言）。
    """
    vals = pred_close_path.values
    mean_val = float(np.mean(vals) / last_close - 1.0)
    return {
        "last": float(vals[-1] / last_close - 1.0),
        "mean": mean_val,
        "max": float(np.max(vals) / last_close - 1.0),
        "min": float(np.min(vals) / last_close - 1.0),
    }


def run_variant_signals(
    predictor,
    provider,
    cfg: BaselineConfig,
    rebalances: pd.DatetimeIndex,
    *,
    progress_every: int = 10,
    checkpoint_dir=None,
) -> dict[str, pd.DataFrame]:
    """逐交易日生成四变体信号宽表（last/mean/max/min）。

    一次推理 → 四个聚合同时落盘。推理链路与
    ``paper_replication.signal.run_kronos_signals`` 逐字一致（同 chunked predict_batch、
    同 seed、同 N），保证 mean 列可与既有 K 组逐位对拍。

    :param checkpoint_dir: 可选，每 ``progress_every`` 日把已完成日写盘到
        ``<checkpoint_dir>/daily_signals_paper_<variant>.parquet``（断点续跑）。
    :returns: ``{variant: wide_df}``，四张宽表同形状同 index 同 columns。
    """
    import torch

    from kronos_qlib import build_inference_windows

    # 四变体逐日行 dict
    rows: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    done_dates: set[pd.Timestamp] = set()

    # 断点续跑：读已有 partial
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
        f"四变体信号 [{cfg.window}]：{len(rebalances)} 日总计，{len(pending)} 日待跑（N={cfg.sample_count}）"
    )

    for i, d in enumerate(pending):
        ds = d.strftime("%Y-%m-%d")
        df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows(
            provider,
            ds,
            lookback=cfg.lookback,
            predict_len=cfg.predict_len,
            pool=cfg.pool,
        )
        if len(df_list) == 0:
            logger.warning(f"{ds}: 无可用股票（{stats}）")
            for v in VARIANTS:
                rows[v].append({})
            continue

        last_closes = [df["close"].iloc[-1] for df in df_list]
        torch.manual_seed(cfg.seed)
        preds = predict_batch_chunked(
            predictor,
            df_list,
            x_ts_list,
            y_ts_list,
            pred_len=cfg.predict_len,
            T=cfg.T,
            top_k=cfg.sample_top_k,
            top_p=cfg.top_p,
            sample_count=cfg.sample_count,
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
                f"四变体信号 [{i + 1}/{len(pending)}] {ds}: "
                f"{len(codes)} 只，mean std={np.std(list(day_signals['mean'].values())):.4f}"
            )
            # 增量落盘（断点续跑）
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
    """把已完成日（done + 当前）合并落盘（断点续跑用）。"""
    done_sorted = sorted(done_dates)
    n_done = len(done_sorted)
    for v in VARIANTS:
        done_rows = rows[v][:n_done] if n_done > 0 else []
        new_rows = rows[v][n_done:]
        all_idx = done_sorted + list(pending[: len(new_rows)])
        partial = pd.DataFrame(done_rows + new_rows, index=all_idx)
        partial.to_parquet(ckpt_paths[v])
    logger.debug(f"四变体 checkpoint 落盘（{len(done_sorted) + len(pending[: len(rows['mean']) - n_done])} 日）")


def run_momentum_reversal(
    provider, cfg: BaselineConfig, rebalances: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐交易日算 10 日动量 / 反转信号（M / R 组）。

    与 ``paper_replication.signal.run_momentum_reversal`` 逐字一致——仅窗口不同。
    """
    from paper_replication.signal import run_momentum_reversal as _run_mr
    from paper_replication.common import ReplicationConfig

    # 借用 paper_replication 的实现，但用本 cfg 的窗口
    rc = _to_replication_config(cfg)
    return _run_mr(provider, rc, rebalances)


def run_placeholder(
    cfg: BaselineConfig,
    rebalances: pd.DatetimeIndex,
    columns: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """随机占位信号（P 组）。与 ``paper_replication.signal.run_placeholder`` 一致。"""
    from paper_replication.signal import run_placeholder as _run_p

    return _run_p(cfg, rebalances, columns, seed=seed)


def build_px_tradeable(
    provider,
    cfg: BaselineConfig,
    rebalances: pd.DatetimeIndex,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """取全区间后复权 close + 可交易掩码宽表。

    与 ``paper_replication.signal.build_px_tradeable`` 一致——仅窗口用本 cfg。
    """
    from paper_replication.signal import build_px_tradeable as _build
    from paper_replication.common import ReplicationConfig

    rc = _to_replication_config(cfg)
    return _build(provider, rc, rebalances, columns)


def _to_replication_config(cfg: BaselineConfig):
    """把 BaselineConfig 桥接成 ReplicationConfig（复用 paper_replication 实现）。

    paper_replication 的取数 / 基线信号函数只读 cfg 的数据层 + 推理字段，
    本桥接保证字段对齐、口径不漂移。
    """
    from paper_replication.common import ReplicationConfig

    return ReplicationConfig(
        pool=cfg.pool,
        lookback=cfg.lookback,
        predict_len=cfg.predict_len,
        backtest_start=cfg.backtest_start,
        backtest_end=cfg.backtest_end,
        data_end=cfg.data_end,
        filter_pipe=cfg.filter_pipe,
        model_name=cfg.model_name,
        tokenizer_name=cfg.tokenizer_name,
        T=cfg.T,
        top_p=cfg.top_p,
        sample_top_k=cfg.sample_top_k,
        sample_count=cfg.sample_count,
        seed=cfg.seed,
        device=cfg.device,
        max_context=cfg.max_context,
        signal_field=cfg.signal_field,
        top_k=cfg.top_k,
        drop_n=cfg.drop_n,
        min_hold=cfg.min_hold,
        cost_bps=cfg.cost_bps,
        baselines=cfg.baselines,
    )
