"""兼容现有 ``predict_batch`` 的 TAS 推理包装器（计划 §7）。

与 ``KronosPredictor.predict_batch``（model/kronos.py:562）同接口、同预处理
（列校验 → 窗口 z-score + clip → 时间特征）、同返回（``pred_df`` 列表），
内部改走 :class:`ConditionalKronos` 两遍条件生成；聚合固定 mean
（复用 ``baseline_suite.signal.compute_variants_from_preds`` 语义）。

SHUFFLE（计划 §4）：同日股票状态按代码排序循环错配一位——仅推理破坏
实验；当天股票数 < 2 时报错，不跨日期抽取状态。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from model.kronos import calc_time_stamps
from tas_state.config import TASConfig
from tas_state.model import ConditionalKronos


class TASPredictor:
    """TAS1 推理包装器（PRE / POST / SHUFFLE / baseline 共用一类）。

    :param ck: 已加载权重的 :class:`ConditionalKronos`。
    :param cfg: 冻结实验配置。
    :param layout: ``"pre"`` / ``"post"`` / ``"baseline"``。
    :param shuffle_states: True 时同日状态按代码排序循环错配一位
        （T-SHUFFLE 臂；与 ``layout`` 独立叠加）。
    """

    price_cols = ["open", "high", "low", "close"]
    vol_col = "volume"
    amt_vol = "amount"

    def __init__(
        self,
        ck: ConditionalKronos,
        cfg: TASConfig,
        *,
        layout: str = "pre",
        shuffle_states: bool = False,
    ) -> None:
        if layout not in ("pre", "post", "baseline"):
            raise ValueError(f"未知布局 {layout!r}")
        self.ck = ck
        self.cfg = cfg
        self.layout = layout
        self.shuffle_states = shuffle_states
        self.device = cfg.device

    # —— 预处理（与 KronosPredictor.predict_batch 逐字同构）——
    def _prepare(self, df_list, x_ts_list, y_ts_list, pred_len):
        if not (len(df_list) == len(x_ts_list) == len(y_ts_list)):
            raise ValueError("df_list / x_ts_list / y_ts_list 长度不一致")
        x_list, xs_list, ys_list, means, stds = [], [], [], [], []
        for i, df in enumerate(df_list):
            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)
            cols = self.price_cols + [self.vol_col, self.amt_vol]
            if df[cols].isnull().values.any():
                raise ValueError(f"索引 {i} 输入含 NaN")
            x = df[cols].values.astype(np.float32)
            x_stamp = calc_time_stamps(x_ts_list[i]).values.astype(np.float32)
            y_stamp = calc_time_stamps(y_ts_list[i]).values.astype(np.float32)
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"索引 {i} y 长度 {y_stamp.shape[0]} ≠ {pred_len}")
            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.cfg.clip, self.cfg.clip)
            x_list.append(x_norm)
            xs_list.append(x_stamp)
            ys_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)
        return (
            np.stack(x_list).astype(np.float32),
            np.stack(xs_list).astype(np.float32),
            np.stack(ys_list).astype(np.float32),
            np.stack(means).astype(np.float32),
            np.stack(stds).astype(np.float32),
        )

    def _maybe_shuffle(self, S: torch.Tensor, codes: list[str]) -> torch.Tensor:
        """同日状态循环错配：按代码排序后移一位（计划 §4 T-SHUFFLE）。"""
        if not self.shuffle_states:
            return S
        if len(codes) < 2:
            raise ValueError(
                f"T-SHUFFLE 要求当天股票数 ≥2，收到 {len(codes)}（不允许跨日期抽取）"
            )
        order = np.argsort(np.array(codes))
        shifted = np.roll(order, 1)  # sorted[k] 的状态 ← sorted[k-1]
        out = S.clone()
        out[order] = S[shifted]
        return out

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
        codes: list[str] | None = None,
    ):
        """与 ``KronosPredictor.predict_batch`` 同构的批量推理。

        :param codes: 股票代码（SHUFFLE 排序用；调用方传入与 df_list 对齐）。
        :returns: ``pred_df`` 列表（close 等 6 列，index = y_timestamp），
            对 ``sample_count`` 条采样路径的 close 已取均值——mean 聚合
            语义与原版一致。
        """
        if pred_len != self.cfg.predict_len:
            raise ValueError(f"pred_len={pred_len} ≠ 冻结配置 {self.cfg.predict_len}")
        x_norm, x_stamp, y_stamp, means, stds = self._prepare(
            df_list, x_timestamp_list, y_timestamp_list, pred_len
        )
        t = lambda a: torch.from_numpy(a).to(self.device)
        x_t, xs_t, ys_t = t(x_norm), t(x_stamp), t(y_stamp)

        S, hist_tokens = self.ck.encode_state(x_t, xs_t)
        if codes is not None:
            S = self._maybe_shuffle(S, codes)
        gen_s1, gen_s2 = self.ck.generate(
            x_t, xs_t, ys_t,
            S=S, layout=self.layout,
            sample_count=sample_count,
            temperature=T, top_k=top_k, top_p=top_p,
        )
        # decode：[B, N, H, 6]（真实量纲），对 N 条路径的 close 取均值
        z = self.ck.decode_tokens(
            hist_tokens, gen_s1, gen_s2,
            t(means), t(stds),
        )
        preds = z.mean(dim=1)  # [B, H, 6]

        cols = self.price_cols + [self.vol_col, self.amt_vol]
        return [
            pd.DataFrame(preds[i].cpu().numpy(), columns=cols, index=y_timestamp_list[i])
            for i in range(len(df_list))
        ]
