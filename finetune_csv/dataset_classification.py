"""方向分类数据集。

实现 :class:`KlineClassificationDataset`，按时间窗构造 (特征, 时间戳, 涨跌平标签) 样本。
相比 ``finetune_csv/finetune_base_model.py`` 的 :class:`CustomKlineDataset`，本数据集
**独立实现**（不继承后打补丁），并修复两处问题：

1. **未来函数防护（硬性验收项）**：z-score 只在 lookback 段上计算，原实现用
   全窗口（lookback + predict）的均值方差，会把未来 K 线泄露进特征归一化；
2. 按收益率 ``r = close[t+H]/close[t] - 1`` 与阈值分 3 类，阈值在训练集估计后冻结
   并透传给 val/test，避免分位数泄露。

设计参考 ``docs/Kronos方向分类头改造方案_20260809.md`` 第 3.2 节。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class KlineClassificationDataset(Dataset):
    """K 线方向分类数据集。

    特征窗为 ``[t-L+1, t]``（L=lookback_window，t 为窗口末尾），标签为未来 H 步
    收益率 ``r = close[t+H] / close[t] - 1`` 按阈值分 3 类（0=跌，1=平，2=涨）。
    一个样本共占用 L+H 个连续时间点，特征归一化**严格只用前 L 个**。

    :param data_path: CSV 文件路径，需含 timestamps/open/high/low/close/volume/amount 列。
    :param data_type: 数据集划分，'train' / 'val' / 'test'。
    :param lookback_window: 特征回看窗口 L，默认 90。
    :param predict_window: 未来预测步数 H，默认 10。
    :param clip: z-score 后的截断阈值。
    :param train_ratio: 训练集时间比例。
    :param val_ratio: 验证集时间比例。
    :param test_ratio: 测试集时间比例。
    :param threshold_mode: 阈值模式，'quantile' 或 'fixed'。
    :param fixed_threshold: fixed 模式下收益率绝对值阈值（如 0.005 表示 ±0.5%）。
    :param quantile_low: quantile 模式下训练集 r 的下分位数（默认 0.30）。
    :param quantile_high: quantile 模式下训练集 r 的上分位数（默认 0.70）。
    :param low_threshold: 透传的下阈值（val/test 由 train 估计后传入）。
    :param high_threshold: 透传的上阈值（val/test 由 train 估计后传入）。
    """

    FEATURE_LIST = ["open", "high", "low", "close", "volume", "amount"]
    TIME_FEATURE_LIST = ["minute", "hour", "weekday", "day", "month"]

    def __init__(
        self,
        data_path: str | Path,
        data_type: str = "train",
        lookback_window: int = 90,
        predict_window: int = 10,
        clip: float = 5.0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        threshold_mode: str = "quantile",
        fixed_threshold: float = 0.005,
        quantile_low: float = 0.30,
        quantile_high: float = 0.70,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ):
        self.data_path = Path(data_path)
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        # 一个样本占用 L(特征) + H(标签用未来步) 个时间点
        self.window = lookback_window + predict_window
        self.clip = clip
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        # 阈值冻结后保存于此（train 自行估计，val/test 由外部透传）
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self._load_and_preprocess()
        self._split_by_time()

        # 在已划分的数据上估计或复用阈值
        self._resolve_thresholds()

        self.n_samples = max(len(self.data) - self.window + 1, 0)
        print(f"[{data_type.upper()}] 划分后长度 {len(self.data)}，可用样本 {self.n_samples}")

    def _load_and_preprocess(self) -> None:
        """读取 CSV、解析时间特征、前向填充缺失值。"""
        df = pd.read_csv(self.data_path)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)

        self.timestamps = df["timestamps"].copy()

        # 衍生时间特征列（与 CustomKlineDataset 一致）
        df["minute"] = df["timestamps"].dt.minute
        df["hour"] = df["timestamps"].dt.hour
        df["weekday"] = df["timestamps"].dt.weekday
        df["day"] = df["timestamps"].dt.day
        df["month"] = df["timestamps"].dt.month

        cols = self.FEATURE_LIST + self.TIME_FEATURE_LIST
        if df[cols].isnull().any().any():
            print("警告：检测到缺失值，执行前向填充")
            df[cols] = df[cols].ffill()

        # 保留原始量纲的 close 列用于标签计算
        self.close_series = df["close"].astype(np.float32).to_numpy()
        self.data = df[cols].astype(np.float32).to_numpy()

        print(
            f"原始数据时间范围：{self.timestamps.min()} ~ {self.timestamps.max()}，"
            f"共 {len(df)} 条"
        )

    def _split_by_time(self) -> None:
        """按时间顺序划分 train/val/test，同步切片 close 与 timestamps。"""
        n = len(self.data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        if self.data_type == "train":
            lo, hi = 0, train_end
        elif self.data_type == "val":
            lo, hi = train_end, val_end
        elif self.data_type == "test":
            lo, hi = val_end, n
        else:
            raise ValueError(f"未知 data_type: {self.data_type}")

        self.data = self.data[lo:hi].copy()
        self.close_series = self.close_series[lo:hi].copy()
        self.timestamps = self.timestamps.iloc[lo:hi].copy()
        print(f"[{self.data_type.upper()}] 时间范围：{self.timestamps.min()} ~ {self.timestamps.max()}")

    def _resolve_thresholds(self) -> None:
        """估计或校验阈值。

        train 集在 quantile 模式下自行估计 30/70 分位数并冻结；fixed 模式直接用
        ``±fixed_threshold``。val/test 必须由外部透传已冻结阈值，否则用 fixed 模式兜底。
        """
        if self.threshold_mode == "fixed":
            self.low_threshold = -self.fixed_threshold
            self.high_threshold = self.fixed_threshold
            return

        # quantile 模式
        if self.data_type == "train":
            rates = self._collect_train_rates()
            self.low_threshold = float(np.quantile(rates, self.quantile_low))
            self.high_threshold = float(np.quantile(rates, self.quantile_high))
            print(
                f"[TRAIN] 阈值分位数估计：low={self.low_threshold:.6f}, "
                f"high={self.high_threshold:.6f}"
            )
        else:
            # val/test 必须透传，未透传时报错以防泄露/误用
            if self.low_threshold is None or self.high_threshold is None:
                raise ValueError(
                    "quantile 模式下 val/test 必须透传 train 估计的 low_threshold/high_threshold"
                )

    def _collect_train_rates(self) -> np.ndarray:
        """收集训练集所有可用样本的未来 H 步收益率，用于分位数估计。

        :return: 收益率一维数组。
        """
        l, h = self.lookback_window, self.predict_window
        close = self.close_series
        n_samples = max(len(close) - (l + h) + 1, 0)
        if n_samples <= 0:
            return np.array([], dtype=np.float32)
        # t = idx + l - 1；r = close[t+h]/close[t] - 1
        t_idx = np.arange(n_samples) + l - 1
        rates = close[t_idx + h] / close[t_idx] - 1.0
        return rates.astype(np.float32)

    def get_frozen_thresholds(self) -> tuple[float, float]:
        """返回已冻结的 (low_threshold, high_threshold)，供 val/test 透传。

        :return: (下阈值, 上阈值)。
        """
        if self.low_threshold is None or self.high_threshold is None:
            raise RuntimeError("阈值尚未冻结，请先在 train 集上初始化")
        return self.low_threshold, self.high_threshold

    def compute_class_weights(self) -> np.ndarray:
        """按训练集类别频率倒数计算 class_weights。

        :return: 长度为 3 的权重数组，类别频率越低权重越高。
        """
        labels = self._collect_labels()
        counts = np.bincount(labels, minlength=3).astype(np.float64)
        counts = np.where(counts == 0, 1.0, counts)
        # 各类频率倒数，再归一化到均值为 1
        weights = 1.0 / counts
        weights = weights / weights.mean()
        return weights.astype(np.float32)

    def _collect_labels(self) -> np.ndarray:
        """计算训练集所有样本的标签，用于 class_weights 估计。

        :return: 标签一维数组（值域 {0,1,2}）。
        """
        l, h = self.lookback_window, self.predict_window
        rates = self._collect_train_rates()
        if rates.size == 0:
            return np.array([], dtype=np.int64)
        return self._rate_to_label(rates)

    def _rate_to_label(self, rates: np.ndarray) -> np.ndarray:
        """按冻结阈值把收益率分箱为标签。

        :param rates: 收益率数组。
        :return: 标签数组，r < low → 0(跌)，r > high → 2(涨)，其余 → 1(平)。
        """
        labels = np.ones_like(rates, dtype=np.int64)
        labels[rates < self.low_threshold] = 0
        labels[rates > self.high_threshold] = 2
        return labels

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """取第 idx 个样本。

        特征窗 ``data[idx : idx+L]``（仅用 lookback 段归一化），标签由
        ``close[idx+L-1+H] / close[idx+L-1] - 1`` 分箱得到。篡改 ``data[idx+L:]``
        不影响特征、只影响标签——这是无未来函数的构造性保证。

        :param idx: 样本下标。
        :return: (features[L, 6], stamp[L, 5], label[])。
        """
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"下标 {idx} 超出范围 [0, {self.n_samples})")

        l = self.lookback_window
        h = self.predict_window

        # 特征窗严格只用前 L 个时间点
        feat_block = self.data[idx : idx + l]
        x = feat_block[..., : len(self.FEATURE_LIST)].copy()
        x_stamp = feat_block[..., len(self.FEATURE_LIST) :].copy()

        # 未来函数防护：归一化统计量只在 lookback 段计算
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        # 标签：t = idx + l - 1，r = close[t+h]/close[t] - 1
        close = self.close_series
        t = idx + l - 1
        rate = close[t + h] / close[t] - 1.0
        label = self._rate_to_label(np.array([rate], dtype=np.float32))[0]

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(x_stamp.astype(np.float32)),
            torch.tensor(int(label), dtype=torch.long),
        )


def build_classification_datasets(
    config,
) -> tuple[KlineClassificationDataset, KlineClassificationDataset, KlineClassificationDataset]:
    """构建 train/val/test 三个数据集，阈值在 train 估计后透传给 val/test。

    :param config: 提供 data_path/lookback_window/predict_window/clip/划分比例/
        阈值相关字段的对象（如 :class:`config_loader.ClassifierConfig`）。
    :return: (train_dataset, val_dataset, test_dataset)。
    """
    common = dict(
        data_path=config.data_path,
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        threshold_mode=config.threshold_mode,
        fixed_threshold=config.fixed_threshold,
        quantile_low=config.quantile_low,
        quantile_high=config.quantile_high,
    )

    train_ds = KlineClassificationDataset(data_type="train", **common)

    # 阈值在 train 集冻结后透传，防止 val/test 重复估计造成泄露
    low, high = train_ds.get_frozen_thresholds()
    val_ds = KlineClassificationDataset(
        data_type="val", low_threshold=low, high_threshold=high, **common
    )
    test_ds = KlineClassificationDataset(
        data_type="test", low_threshold=low, high_threshold=high, **common
    )
    return train_ds, val_ds, test_ds
