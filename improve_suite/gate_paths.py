"""实数据对拍门禁（计划 §2.0.4 阶段 0）。

论文窗口取前 ``n_days`` 个交易日，用 :func:`predict_batch_paths` 重推理，路径均值算出
的 mean 信号与既有 ``baseline_suite/data/daily_signals_paper_mean.parquet`` 对应日
逐位对拍（``max|Δ| == 0`` 期望）。

既有 parquet 由 ``baseline_suite.signal.run_variant_signals``（同 seed / 同 N / 同
``predict_batch_chunked``）产出；路径版推理函数 RNG 序逐字不动 → 均值逐位一致。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m improve_suite.gate_paths
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from baseline_suite.common import DATA_DIR as BL_DATA_DIR
from improve_suite.common import ImproveConfig


def _load_predictor(cfg: ImproveConfig):
    """加载 KronosPredictor（与 baseline_suite.run_signals 同口径）。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def _build_provider(cfg: ImproveConfig):
    from kronos_qlib import QlibProvider

    fetch_start = (
        pd.Timestamp(cfg.backtest_start) - pd.Timedelta(days=cfg.lookback * 2)
    ).strftime("%Y-%m-%d")
    return QlibProvider(cfg.pool, fetch_start, cfg.backtest_end)


def run_paths_gate(n_days: int = 3, window: str = "paper") -> dict:
    """对拍门禁主体。

    :param n_days: 取前几个交易日对拍（默认 3）。
    :param window: ``paper`` 或 ``oos``（默认 paper，与既有 parquet 同窗）。
    :returns: ``{"max_abs_diff": float, "n_cells": int, "passed": bool}``。
    """
    from improve_suite.path_inference import predict_batch_paths
    from kronos_qlib import build_inference_windows, QlibProvider

    cfg = ImproveConfig.load(window=window)
    ref_path = BL_DATA_DIR / f"daily_signals_{window}_mean.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(f"对拍基准缺失：{ref_path}（先跑 baseline_suite variants）")
    ref = pd.read_parquet(ref_path)

    # 取前 n_days 个交易日
    cal = QlibProvider(cfg.pool, cfg.backtest_start, cfg.backtest_end)
    rebalances = cal.trading_days(cfg.backtest_start, cfg.backtest_end)
    sample_days = rebalances[:n_days]
    logger.info(f"对拍门禁 [{window}]：取前 {n_days} 日 {sample_days[0].date()}~{sample_days[-1].date()}")

    provider = _build_provider(cfg)
    predictor = _load_predictor(cfg)

    import torch

    new_rows = []
    for d in sample_days:
        ds = d.strftime("%Y-%m-%d")
        df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows(
            provider, ds, lookback=cfg.lookback, predict_len=cfg.predict_len, pool=cfg.pool,
        )
        if len(df_list) == 0:
            logger.warning(f"{ds}: 无可用股票")
            new_rows.append({})
            continue
        last_closes = [df["close"].iloc[-1] for df in df_list]
        torch.manual_seed(cfg.seed)  # 与 run_variant_signals 同：每日重置 seed
        preds, paths = predict_batch_paths(
            predictor, df_list, x_ts_list, y_ts_list,
            pred_len=cfg.predict_len, T=cfg.T, top_k=cfg.sample_top_k,
            top_p=cfg.top_p, sample_count=cfg.sample_count,
        )
        day_sig = {}
        for j, pred_df in enumerate(preds):
            # mean 信号 = mean(pred_close)/last_close - 1（与 compute_variants_from_preds mean 分支同口径）
            day_sig[codes[j]] = float(np.mean(pred_df[cfg.signal_field].values) / last_closes[j] - 1.0)
        new_rows.append(day_sig)
        logger.info(f"{ds}: {len(codes)} 只，逐路径矩阵 {paths[0].shape}")

    new = pd.DataFrame(new_rows, index=sample_days)
    # 对拍：取公共日 + 公共列
    common_dates = new.index.intersection(ref.index)
    common_cols = new.columns.intersection(ref.columns)
    a = new.loc[common_dates, common_cols]
    b = ref.loc[common_dates, common_cols]
    both_valid = a.notna() & b.notna()
    diff = (a - b).where(both_valid)
    max_abs_diff = float(np.nanmax(np.abs(diff.values))) if both_valid.any().any() else 0.0
    n_cells = int(both_valid.sum().sum())
    na_mismatch = int((a.isna() != b.isna()).sum().sum())

    passed = (max_abs_diff < 1e-8) and (na_mismatch == 0)
    logger.info(
        f"对拍结果 [{window} 前 {n_days} 日]：有效单元 {n_cells}，"
        f"max|Δ|={max_abs_diff:.3e}，NaN 不一致 {na_mismatch}"
    )
    if passed:
        logger.info(f"✓ 对拍门禁通过：路径均值与既有 mean 逐位一致（max|Δ|={max_abs_diff:.3e}）")
    else:
        logger.error(f"✗ 对拍门禁未通过：max|Δ|={max_abs_diff:.3e}——查 RNG / 预处理差异")
    return {"max_abs_diff": max_abs_diff, "n_cells": n_cells, "na_mismatch": na_mismatch, "passed": passed}


def main() -> None:
    res = run_paths_gate(n_days=3, window="paper")
    if not res["passed"]:
        raise SystemExit(f"对拍门禁未通过：max|Δ|={res['max_abs_diff']:.3e}")


if __name__ == "__main__":
    main()
