"""TAS1 训练 / 验证数据契约（计划 §3.1、§5.1、§7）。

从既有全 A 语料 pickle（G1 同源 ``finetune_suite/data/ashares/{train,val}_data.pkl``，
只读）构造合法 ``(code, decision_date)`` 窗口池：

- 窗口 = ``lookback`` 行历史 + ``predict_len`` 行未来（共 100 行）；
- 归一化 ``mean/std`` 只用历史 90 行（未来不参与，防泄漏），全序列 ``clip``；
- 未来标签仅作为监督目标与 teacher-forcing 前缀（计划 §3.4）；
- purge：训练目标末日 ≤ ``train_target_end``，验证目标末日 ≤ ``val_target_end``。

数据契约（计划 §7）：``X=[B,90,6]``、``stamp=[B,90,5]``、``Y=[B,10,6]``
（归一化标签）、``y_stamp=[B,10,5]``；所有结果必须携带 ``(date, code)``。
"""
from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from tas_state.config import REPO_ROOT, TASConfig

# 语料列顺序 = 官方 feature_list（finetune/dataset.py 同源；build_dataset.py 落盘）
FEATURES: tuple[str, ...] = ("open", "high", "low", "close", "vol", "amt")
TIME_FEATURES: tuple[str, ...] = ("minute", "hour", "weekday", "day", "month")


@dataclass(frozen=True)
class TasWindow:
    """单个合法训练窗口（已归一化，携带 (date, code) 身份）。

    ``x_norm`` 长度 100：前 90 行 = 模型输入 X，后 10 行 = 未来标签 Y；
    二者共用同一窗口归一化参数（mean/std 只来自前 90 行）。
    """

    code: str
    decision_date: pd.Timestamp  # 决策日 t（历史最后一行）
    target_end: pd.Timestamp  # 最后标签日（purge 边界锚点）
    x_norm: np.ndarray  # [100, 6] float32，z-score + clip
    x_stamp: np.ndarray  # [90, 5] float32 历史时间特征
    y_stamp: np.ndarray  # [10, 5] float32 未来时间特征

    @property
    def history(self) -> np.ndarray:
        """模型输入 X = [90, 6]。"""
        return self.x_norm[:90]

    @property
    def future(self) -> np.ndarray:
        """未来标签 Y = [10, 6]（监督目标 / teacher forcing 专用）。"""
        return self.x_norm[90:]


class TASCorpus:
    """全 A 语料窗口池（只读 pickle + 索引预计算）。

    :param pkl_path: ``{symbol: DataFrame}`` 语料（DatetimeIndex + FEATURES 列）。
    :param cfg: 冻结实验配置。
    :param target_end: purge 边界——窗口最后标签日必须 ≤ 此日。
    """

    def __init__(
        self,
        pkl_path: str | Path,
        cfg: TASConfig,
        *,
        target_end: str,
        split_name: str = "train",
    ) -> None:
        self.cfg = cfg
        self.target_end = pd.Timestamp(target_end)
        self.split_name = split_name
        self.path = Path(pkl_path)
        if not self.path.is_absolute():
            self.path = REPO_ROOT / self.path

        with open(self.path, "rb") as f:
            raw: dict[str, pd.DataFrame] = pickle.load(f)

        win = cfg.lookback + cfg.predict_len  # 100 行
        self._data: dict[str, pd.DataFrame] = {}
        self._keys: list[tuple[str, pd.Timestamp]] = []  # (code, decision_date)
        self._index: dict[tuple[str, pd.Timestamp], int] = {}
        n_short = 0
        for code in sorted(raw):
            df = raw[code]
            if len(df) < win:
                n_short += 1
                continue
            df = df.sort_index()
            # 窗口 [i, i+win)，决策日 = 第 lookback-1 行，标签末日 = 第 win-1 行
            dates = df.index
            starts = np.arange(len(df) - win + 1)
            if len(starts) == 0:
                continue
            # purge：标签末日 = dates[start + win - 1] ≤ target_end
            end_dates = dates[starts + win - 1]
            ok = end_dates <= self.target_end
            if not ok.any():
                continue
            self._data[code] = df
            base = len(self._keys)
            for j in np.nonzero(ok)[0]:
                key = (code, pd.Timestamp(dates[starts[j] + cfg.lookback - 1]))
                self._keys.append(key)
                self._index[key] = base + j
        logger.info(
            f"TASCorpus[{split_name}] {self.path.name}: {len(self._data)} symbols, "
            f"{len(self._keys)} 合法窗口（purge≤ {target_end}, 短窗剔除 {n_short}）"
        )

    # —— 键与身份 ——
    def keys(self) -> list[tuple[str, pd.Timestamp]]:
        """全部合法 ``(code, decision_date)``（构造序，code 字典序）。"""
        return list(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    # —— 窗口构造 ——
    def window(self, code: str, decision_date: pd.Timestamp) -> TasWindow:
        """按键取窗（批处理顺序无关：同键同内容）。"""
        df = self._data[code]
        loc = df.index.get_loc(decision_date)
        if loc < self.cfg.lookback - 1:
            raise KeyError(f"{code}@{decision_date}: 历史不足 {self.cfg.lookback} 行")
        w = df.iloc[loc - self.cfg.lookback + 1 : loc + self.cfg.predict_len + 1]
        return _build_window(code, w, self.cfg)

    def batch(self, keys: list[tuple[str, pd.Timestamp]]) -> dict[str, np.ndarray]:
        """按键列表组 batch（顺序敏感：输出与 keys 一一对应）。

        :returns: ``{"X":[B,90,6], "stamp":[B,90,5], "Y":[B,10,6],
            "y_stamp":[B,10,5], "decision_dates":[B], "codes":[B]}``。
        """
        wins = [self.window(c, d) for c, d in keys]
        return {
            "X": np.stack([w.history for w in wins]).astype(np.float32),
            "stamp": np.stack([w.x_stamp for w in wins]).astype(np.float32),
            "Y": np.stack([w.future for w in wins]).astype(np.float32),
            "y_stamp": np.stack([w.y_stamp for w in wins]).astype(np.float32),
            "decision_dates": [w.decision_date for w in wins],
            "codes": [w.code for w in wins],
        }

    # —— 采样 ——
    def sampler(self, seed: int) -> "CorpusSampler":
        """均匀窗口采样器（独立 RNG，不干扰模型初始化随机性）。"""
        return CorpusSampler(self, seed)

    # —— STATIC 固定平均 Z 的样本清单（计划 §4）——
    def static_sample_keys(self, n: int) -> list[tuple[str, pd.Timestamp]]:
        """合法键按 SHA256 字典序取前 ``n``（不足则全量）。

        排序键 = ``sha256(f"{code}|{date:%Y-%m-%d}")``——与键插入顺序、
        code 原始顺序都无关，三个种子共用同一清单。
        """
        ranked = sorted(
            self._keys,
            key=lambda k: hashlib.sha256(
                f"{k[0]}|{k[1]:%Y-%m-%d}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < n:
            logger.warning(
                f"STATIC 样本不足：合法键 {len(ranked)} < {n}，用全量并记录"
            )
        return ranked[:n]


class CorpusSampler:
    """均匀随机窗口采样器（官方 QlibDataset 同模式：random.Random(seed)）。"""

    def __init__(self, corpus: TASCorpus, seed: int) -> None:
        self._corpus = corpus
        self._rng = random.Random(seed)

    def draw(self, n: int) -> list[tuple[str, pd.Timestamp]]:
        """有放回均匀抽 n 个键（有效 batch 由调用方按步数切分）。"""
        keys = self._corpus.keys()
        return [self._rng.choice(keys) for _ in range(n)]


def _build_window(code: str, w: pd.DataFrame, cfg: TASConfig) -> TasWindow:
    """100 行窗口 → 归一化 TasWindow（mean/std 只用历史 90 行）。"""
    feats = w[list(FEATURES)]
    if feats.isnull().values.any():
        raise ValueError(f"{code}: 窗口含 NaN（语料清洗不应发生）")
    x = feats.values.astype(np.float32)
    hist = x[: cfg.lookback]
    mean = np.mean(hist, axis=0)
    std = np.std(hist, axis=0)
    x_norm = np.clip((x - mean) / (std + 1e-5), -cfg.clip, cfg.clip)

    stamp = _time_features(w.index[: cfg.lookback])
    y_stamp = _time_features(w.index[cfg.lookback :])
    return TasWindow(
        code=code,
        decision_date=pd.Timestamp(w.index[cfg.lookback - 1]),
        target_end=pd.Timestamp(w.index[-1]),
        x_norm=x_norm.astype(np.float32),
        x_stamp=stamp,
        y_stamp=y_stamp,
    )


def _time_features(idx: pd.DatetimeIndex) -> np.ndarray:
    """五列时间特征（与 model.kronos.calc_time_stamps / 官方 dataset 一致）。"""
    df = pd.DataFrame(index=idx)
    df["minute"] = idx.minute
    df["hour"] = idx.hour
    df["weekday"] = idx.weekday
    df["day"] = idx.day
    df["month"] = idx.month
    return df[list(TIME_FEATURES)].values.astype(np.float32)


def load_corpora(cfg: TASConfig) -> tuple[TASCorpus, TASCorpus]:
    """训练 / 验证语料（计划 §5.1 分割；验证目标日全在 2025H1 内）。"""
    train = TASCorpus(
        cfg.train_corpus_path, cfg, target_end=cfg.train_target_end, split_name="train"
    )
    val = TASCorpus(
        cfg.val_corpus_path,
        cfg,
        target_end=cfg.val_target_end,
        split_name="val",
    )
    return train, val
