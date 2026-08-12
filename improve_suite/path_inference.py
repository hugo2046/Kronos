"""逐路径推理仪器（计划 §2 阶段 0 核心）。

复制 ``model/kronos.py:auto_regressive_inference`` 为
:func:`auto_regressive_inference_paths`——**唯一改动**是在原 L465 reshape 出
``(bs, sample_count, total_seq, d)`` 后，同时返回逐路径张量与原均值（RNG 调用序
逐字不动）。外层 :func:`predict_batch_paths` 镜像 ``predict_batch_chunked`` 签名，
额外返回每股 ``pred_close`` 的逐路径矩阵 ``(N, H)``。

**不改 ``model/``**：本模块独立复制推理函数，差异仅限出口（对拍门禁保证逐位一致）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from model.kronos import sample_from_logits


def auto_regressive_inference_paths(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=5,
    verbose=False,
):
    """与 ``model.kronos.auto_regressive_inference`` 逐字同构，出口多返回逐路径张量。

    :returns: ``(mean, paths)``：
        ``mean`` ``(bs, total_seq, d)`` = 原函数返回值（路径沿 sample_count 维均值）；
        ``paths`` ``(bs, sample_count, total_seq, d)`` = 逐路径解码张量。
    """
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        if verbose:
            ran = trange
        else:
            ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [
                    pre_buffer[:, :window_len],
                    post_buffer[:, :window_len]
                ]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous()
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        # —— 出口差异（唯一改动）：保留逐路径张量，均值与原函数同公式 ——
        z_np = z.cpu().numpy()
        mean = np.mean(z_np, axis=1)

        return mean, z_np


# ------------------------------------------------------------------
# predict_batch_paths：镜像 predict_batch_chunked，额外返回逐路径 pred_close
# ------------------------------------------------------------------


def _generate_paths(
    predictor,
    x,
    x_stamp,
    y_stamp,
    pred_len,
    T,
    top_k,
    top_p,
    sample_count,
    verbose,
):
    """镜像 ``KronosPredictor.generate``，但调用路径版推理，返回 (mean, paths)。"""
    x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(predictor.device)
    x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(predictor.device)
    y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(predictor.device)

    mean, paths = auto_regressive_inference_paths(
        predictor.tokenizer,
        predictor.model,
        x_tensor,
        x_stamp_tensor,
        y_stamp_tensor,
        predictor.max_context,
        pred_len,
        predictor.clip,
        T,
        top_k,
        top_p,
        sample_count,
        verbose,
    )
    # 与原 generate 一致：只取最后 pred_len 步
    mean = mean[:, -pred_len:, :]
    paths = paths[:, :, -pred_len:, :]
    return mean, paths


def _predict_batch_paths_one(
    predictor,
    df_list,
    x_timestamp_list,
    y_timestamp_list,
    *,
    pred_len,
    T,
    top_k,
    top_p,
    sample_count,
    verbose=False,
):
    """镜像 ``KronosPredictor.predict_batch`` 预处理 + 反归一化，额外产出逐路径 close。

    预处理（z-score + clip + stack）逐字照搬 ``predict_batch``，保证同 seed 同输入时
    均值与原链路逐位一致。逐路径张量按相同反归一化还原到价格尺度。

    :returns: ``(pred_dfs, paths_close)``：
        ``pred_dfs`` 与 ``predict_batch`` 同构（均值预测 DataFrame 列表）；
        ``paths_close`` 每股一个 ``(sample_count, pred_len)`` 数组（逐路径 close 价）。
    """
    num_series = len(df_list)
    price_cols = predictor.price_cols  # ['open','high','low','close']
    vol_col = predictor.vol_col
    amt_vol = predictor.amt_vol
    feat_cols = price_cols + [vol_col, amt_vol]

    x_list, x_stamp_list, y_stamp_list = [], [], []
    means, stds = [], []

    for i in range(num_series):
        df = df_list[i]
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
        if not all(col in df.columns for col in price_cols):
            raise ValueError(f"DataFrame at index {i} is missing price columns {price_cols}.")

        df = df.copy()
        if vol_col not in df.columns:
            df[vol_col] = 0.0
            df[amt_vol] = 0.0
        if amt_vol not in df.columns and vol_col in df.columns:
            df[amt_vol] = df[vol_col] * df[price_cols].mean(axis=1)

        if df[price_cols + [vol_col, amt_vol]].isnull().values.any():
            raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

        from model.kronos import calc_time_stamps

        x_timestamp = x_timestamp_list[i]
        y_timestamp = y_timestamp_list[i]

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[feat_cols].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -predictor.clip, predictor.clip)

        x_list.append(x_norm)
        x_stamp_list.append(x_stamp)
        y_stamp_list.append(y_stamp)
        means.append(x_mean)
        stds.append(x_std)

    x_batch = np.stack(x_list, axis=0).astype(np.float32)
    x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
    y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)

    mean, paths = _generate_paths(
        predictor, x_batch, x_stamp_batch, y_stamp_batch,
        pred_len, T, top_k, top_p, sample_count, verbose,
    )
    # mean: (B, pred_len, feat)，paths: (B, sample_count, pred_len, feat)
    close_idx = feat_cols.index("close")

    pred_dfs = []
    paths_close = []
    for i in range(num_series):
        mean_i = mean[i] * (stds[i] + 1e-5) + means[i]
        pred_df = pd.DataFrame(mean_i, columns=feat_cols, index=y_timestamp_list[i])
        pred_dfs.append(pred_df)
        # 逐路径反归一化 close
        paths_i = paths[i] * (stds[i] + 1e-5) + means[i]  # (sample_count, pred_len, feat)
        paths_close.append(paths_i[:, :, close_idx].copy())
    return pred_dfs, paths_close


def predict_batch_paths(
    predictor,
    df_list,
    x_timestamp_list,
    y_timestamp_list,
    *,
    pred_len,
    T,
    top_k,
    top_p,
    sample_count,
    chunk_size=32,
    verbose=False,
):
    """显存友好的分块逐路径推理（镜像 ``predict_batch_chunked`` 签名）。

    :returns: ``(pred_dfs, paths_close)``——
        ``pred_dfs`` 与 ``predict_batch_chunked`` 同构（顺序一致）；
        ``paths_close`` 每股一个 ``(sample_count, pred_len)`` 逐路径 close 矩阵。

    同 seed 同输入时，``pred_dfs`` 的 close 列与原 ``predict_batch_chunked`` 逐位一致
    （推理函数 RNG 序逐字不动，对拍门禁保证）。
    """
    n = len(df_list)
    pred_dfs: list = []
    paths_close: list = []
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        dfs, pcs = _predict_batch_paths_one(
            predictor,
            df_list[s:e],
            x_timestamp_list[s:e],
            y_timestamp_list[s:e],
            pred_len=pred_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        pred_dfs.extend(dfs)
        paths_close.extend(pcs)
        torch.cuda.empty_cache()
    return pred_dfs, paths_close
