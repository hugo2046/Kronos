"""9 列推理封装（G4 计划 §2 0.2：包装，不改 ``kronos_qlib/`` 与 ``model/``）。

- :func:`build_inference_windows_9col`：先调原版 ``kronos_qlib.build_inference_windows``
  （6 列、池/停牌/行数语义逐字不动），再对每个窗口右连接市场三列；
- :class:`G4Predictor`：``KronosPredictor`` 子类，仅把窗口张量的列选择从 6 列
  换成 9 列——z-score/clip、AR 解码（``auto_regressive_inference``）、N 次采样
  均值聚合全部继承原实现，一字不动。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kronos_qlib.windows import REQUIRED_COLS, build_inference_windows
from model.kronos import KronosPredictor, calc_time_stamps

from g4_features.market_context import MARKET_COLS

# 推理侧 9 列（KronosPredictor 的 volume/amount 命名约定 + 市场三列）
FEATURE_COLS_9 = REQUIRED_COLS + MARKET_COLS


def build_inference_windows_9col(
    provider,
    rebalance_date: str,
    *,
    market: pd.DataFrame,
    lookback: int = 90,
    predict_len: int = 10,
    pool: str = "csi300",
    filter_pipe: list | None = None,
):
    """原版 6 列窗口 + 市场三列右连接（列序 = REQUIRED_COLS + MARKET_COLS）。

    :param market: :func:`g4_features.market_context.compute_market_context` 产物，
        须覆盖窗口日期（2014 起 DDB 段 + 2013 预热即全覆盖）。
    :returns: 与原版同构的五元组（df_list 为 9 列版）。
    """
    df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows(
        provider,
        rebalance_date,
        lookback=lookback,
        predict_len=predict_len,
        pool=pool,
        filter_pipe=filter_pipe,
    )
    mkt = market[MARKET_COLS].sort_index()
    out: list[pd.DataFrame] = []
    for df in df_list:
        joined = df.join(mkt, how="left")
        if joined[MARKET_COLS].isna().any().any():
            missing = joined.index[joined[MARKET_COLS].isna().any(axis=1)]
            raise ValueError(
                f"{rebalance_date} 窗口市场列未覆盖（{len(missing)} 行，"
                f"首 {missing[0].date()}）——预热/对齐破坏，禁止静默继续"
            )
        out.append(joined[FEATURE_COLS_9])
    return out, x_ts_list, y_ts_list, codes, stats


class G4Predictor(KronosPredictor):
    """9 列版 KronosPredictor：仅列选择不同，归一化/AR 解码/聚合全继承。"""

    FEATURE_COLS_9 = FEATURE_COLS_9

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len,
                      T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        """父类 ``predict_batch`` 的 9 列版（逻辑逐字同构，列集换成 FEATURE_COLS_9）。"""
        if not isinstance(df_list, (list, tuple)) or not isinstance(
            x_timestamp_list, (list, tuple)
        ) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)
        cols = self.FEATURE_COLS_9

        x_list, x_stamp_list, y_stamp_list = [], [], []
        means, stds, seq_lens, y_lens = [], [], [], []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"DataFrame at index {i} is missing columns {missing}.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[cols].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}.")

            if np.isnan(x).any():
                raise ValueError(f"DataFrame at index {i} contains NaN values.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)
            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len,
                              T, top_k, top_p, sample_count, verbose)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_dfs.append(pd.DataFrame(preds_i, columns=cols, index=y_timestamp_list[i]))
        return pred_dfs


def build_market_context(end_date: str) -> pd.DataFrame:
    """市场三列全表（预热 + DDB 到 end_date），供信号生成逐日复用。"""
    from g4_features.market_context import build_index_series, compute_market_context

    return compute_market_context(build_index_series(None, end_date))
