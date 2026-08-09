"""方向分类头测试。

覆盖方案 3.5 节的 5 项验收测试：

1. 前向 shape 测试（方案 A）；
2. 方案 B probe 的冻结断言与前向 shape；
3. 无未来函数构造性测试（篡改 t+1 之后，特征逐位不变）；
4. 标签正确性（quantile / fixed 两模式）；
5. 训练冒烟测试（随机数据 2 epoch，loss 下降且无 NaN）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# 让 tests/ 能导入仓库根目录的包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import Kronos, KronosClassifier, KronosProbeClassifier, KronosTokenizer
from finetune_csv.dataset_classification import KlineClassificationDataset


def set_seed(seed: int = 42) -> None:
    """固定随机源。

    :param seed: 种子。
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1. 方案 A 前向 shape
# ---------------------------------------------------------------------------
def _rand_stamp(B: int, L: int) -> torch.Tensor:
    """生成合法的 5 维时间特征张量，各维度不越 TemporalEmbedding 的嵌入表边界。

    各列上限：minute<60, hour<24, weekday<7, day<32, month<13。

    :param B: batch size。
    :param L: 序列长度。
    :return: ``[B, L, 5]`` float 张量。
    """
    minute = torch.randint(0, 60, (B, L, 1))
    hour = torch.randint(0, 24, (B, L, 1))
    weekday = torch.randint(0, 7, (B, L, 1))
    day = torch.randint(1, 29, (B, L, 1))
    month = torch.randint(1, 13, (B, L, 1))
    return torch.cat([minute, hour, weekday, day, month], dim=-1).float()


def test_classifier_forward_shape():
    """方案 A：``(B, L, 6) + (B, L, 5) stamp → (B, 3)`` logits。"""
    set_seed()
    model = KronosClassifier(d_in=6, d_model=64, n_heads=4, ff_dim=128, n_layers=2, n_classes=3)
    model.eval()

    B, L = 4, 12
    features = torch.randn(B, L, 6)
    stamp = _rand_stamp(B, L)

    with torch.no_grad():
        logits = model(features, stamp)

    assert logits.shape == (B, 3), f"期望 (4, 3)，得到 {logits.shape}"
    assert torch.isfinite(logits).all(), "logits 含 NaN/Inf"


# ---------------------------------------------------------------------------
# 2. 方案 B probe 冻结断言 + 前向 shape
# ---------------------------------------------------------------------------
def test_probe_forward_shape_and_frozen():
    """方案 B：随机初始化 Kronos+Tokenizer 包成 probe，仅 probe 可训练，前向 shape 正确。"""
    set_seed()
    tokenizer = KronosTokenizer(
        d_in=6, d_model=64, n_heads=4, ff_dim=128, n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=10, s2_bits=10, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
    )
    kronos = Kronos(
        s1_bits=10, s2_bits=10, n_layers=2, d_model=64, n_heads=4, ff_dim=128,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    )
    model = KronosProbeClassifier(tokenizer, kronos, n_classes=3)

    # 断言：仅 probe 参数可训练
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(n.startswith("probe") for n in trainable_names), f"存在非 probe 的可训练参数: {trainable_names}"
    assert len(trainable_names) > 0, "probe 应有可训练参数"

    # train() 调用后主干仍为 eval
    model.train()
    assert not model.tokenizer.training, "tokenizer 应恒为 eval"
    assert not model.kronos.training, "kronos 应恒为 eval"

    B, L = 4, 12
    features = torch.randn(B, L, 6)
    stamp = _rand_stamp(B, L)
    with torch.no_grad():
        logits = model(features, stamp)
    assert logits.shape == (B, 3), f"期望 (4, 3)，得到 {logits.shape}"
    assert torch.isfinite(logits).all(), "logits 含 NaN/Inf"


# ---------------------------------------------------------------------------
# 3. 无未来函数构造性测试
# ---------------------------------------------------------------------------
def _make_tmp_csv(tmp_path: Path, n_rows: int = 400, seed: int = 7) -> Path:
    """生成含 timestamps/OHLCV/amount 的临时 CSV。

    :param tmp_path: pytest 临时目录。
    :param n_rows: 行数。
    :param seed: 随机种子。
    :return: CSV 路径。
    """
    set_seed(seed)
    start = pd.Timestamp("2023-01-01 09:30")
    ts = pd.date_range(start, periods=n_rows, freq="5min")
    base = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.1)
    df = pd.DataFrame({
        "timestamps": ts,
        "open": base,
        "high": base + np.abs(np.random.randn(n_rows)) * 0.05,
        "low": base - np.abs(np.random.randn(n_rows)) * 0.05,
        "close": base,
        "volume": np.random.randint(1000, 100000, n_rows).astype(np.float32),
        "amount": np.random.randint(1000, 100000, n_rows).astype(np.float32),
    })
    p = tmp_path / "cls_data.csv"
    df.to_csv(p, index=False)
    return p


def test_no_lookahead_leakage(tmp_path):
    """篡改 ``data[idx+L:]`` 与 ``close_series[idx+L:]`` 源头，特征逐位不变且标签必须改变。

    样本 idx=0 的特征窗为 ``data[0:L]``，t=L-1，标签依赖 ``close[L-1+H]``。
    构造平稳 close（r≈0 → label 1），再整段放大 t 之后的所有行（含 close[t+H]）使 r 远超
    阈值 → label 2。特征窗 ``data[0:L]`` 完全不含被篡改位置，故特征逐位不变；标签则必然改变。
    """
    # 构造平稳 close：标签依赖的 close[t+H] 原本使 r 落在 |r|<threshold 的“平”区间
    set_seed(5)
    n = 200
    L, H = 30, 5
    close = np.full(n, 100.0, dtype=np.float32)  # 全平稳 → 各样本 r≈0
    ts = pd.date_range("2023-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "timestamps": ts,
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.ones(n, dtype=np.float32) * 1000,
        "amount": np.ones(n, dtype=np.float32) * 1000,
    })
    csv = tmp_path / "leak.csv"
    df.to_csv(csv, index=False)

    ds = KlineClassificationDataset(
        data_path=csv, data_type="train",
        lookback_window=L, predict_window=H,
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
        threshold_mode="fixed", fixed_threshold=0.01,
    )
    assert ds.n_samples > 0
    idx = 0
    feat0, stamp0, label0 = ds[idx]
    assert int(label0) == 1, f"平稳序列首样本应为平(1)，得到 {int(label0)}"

    # 从源头整段篡改：放大 t 之后所有行的特征列（前 6 列 OHLCV+amount）与 close_series，
    # 堵住任何中间行泄露变体（而非仅放大 close[t+H] 单格）
    ds.close_series = ds.close_series.copy()
    ds.data = ds.data.copy()
    t_pos = idx + L - 1
    ds.close_series[t_pos + 1:] *= 10.0
    ds.data[t_pos + 1:, :6] *= 10.0

    feat1, stamp1, label1 = ds[idx]

    # 特征窗 data[0:L] 不含被篡改位置 → 特征逐位不变（核心防泄露断言）
    assert torch.equal(feat0, feat1), "篡改 t+1 之后的数据改变了特征 → 存在未来函数泄露"
    assert torch.equal(stamp0, stamp1), "时间特征被篡改（异常）"
    # 标签由 close[t+H] 决定，被放大后 r≫threshold → label 必为涨(2)
    assert int(label1) == 2, f"篡改 close[t+H] 后标签应为涨(2)，得到 {int(label1)}"


# ---------------------------------------------------------------------------
# 4. 标签正确性
# ---------------------------------------------------------------------------
def test_label_correctness_fixed():
    """fixed 模式：手工构造已知收益率序列，验证分箱。

    样本 i 的 t = i + L - 1，标签由 close[t+H]/close[t]-1 与阈值决定。
    令 close 呈三段：前段下跌（label 0）、中段足够宽的平台（label 1）、后段上涨（label 2）。
    """
    set_seed(1)
    L, H = 10, 3
    # 三段：下跌 / 宽平台 / 上涨；平台足够宽以保证中段样本的 t 与 t+H 都落在平台内
    close = np.concatenate([
        100.0 - np.arange(40) * 0.2,                         # [0,40) 单调下跌
        np.full(80, 92.0, dtype=np.float32),                 # [40,120) 平台
        92.0 + np.arange(60) * 0.2,                          # [120,180) 单调上涨
    ]).astype(np.float32)
    n = len(close)

    ts = pd.date_range("2023-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "timestamps": ts,
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.ones(n, dtype=np.float32) * 1000,
        "amount": np.ones(n, dtype=np.float32) * 1000,
    })
    csv = tmp_path_for_test(df)

    ds = KlineClassificationDataset(
        data_path=csv, data_type="train",
        lookback_window=L, predict_window=H,
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
        threshold_mode="fixed", fixed_threshold=0.005,
    )
    assert ds.n_samples > 0
    # 前段样本（下跌段）：label 应为 0
    head_label = int(ds[0][2])
    assert head_label == 0, f"前段样本应为跌(0)，得到 {head_label}"
    # 后段样本（上涨段）：label 应为 2
    tail_label = int(ds[ds.n_samples - 1][2])
    assert tail_label == 2, f"后段样本应为涨(2)，得到 {tail_label}"
    # 平台中段：取平台中部的样本，t 与 t+H 都在平台内 → r=0 → label 必为 1
    # 平台 [40,120)，t=i+L-1，令 t 落在 [70,100) 且 t+H<120 → i 落在 [61,91)
    for i in [65, 70, 75]:
        mid_label = int(ds[i][2])
        assert mid_label == 1, f"平台中段样本 i={i} 应为平(1)，得到 {mid_label}"


def test_label_correctness_quantile(tmp_path):
    """quantile 模式：阈值在 train 估计后冻结，val/test 透传且一致。"""
    csv = _make_tmp_csv(tmp_path, n_rows=600, seed=3)
    L, H = 30, 5
    common = dict(
        data_path=csv, lookback_window=L, predict_window=H,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        threshold_mode="quantile", quantile_low=0.3, quantile_high=0.7,
    )
    train_ds = KlineClassificationDataset(data_type="train", **common)
    low, high = train_ds.get_frozen_thresholds()
    val_ds = KlineClassificationDataset(data_type="val", low_threshold=low, high_threshold=high, **common)

    # val 的阈值必须与 train 冻结值完全一致
    assert val_ds.low_threshold == low
    assert val_ds.high_threshold == high
    # 阈值合理：low < 0 < high
    assert low < 0 < high, f"阈值不合理: low={low}, high={high}"


def tmp_path_for_test(df: pd.DataFrame) -> Path:
    """把 DataFrame 写入临时文件并返回路径（供标签测试复用）。

    :param df: 待写入数据。
    :return: 临时 CSV 路径。
    """
    import tempfile

    p = Path(tempfile.mkdtemp()) / "data.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 5. 训练冒烟测试
# ---------------------------------------------------------------------------
def test_smoke_training(tmp_path):
    """随机数据跑 2 epoch，loss 下降且无 NaN。"""
    csv = _make_tmp_csv(tmp_path, n_rows=500, seed=11)
    L, H = 30, 5
    ds = KlineClassificationDataset(
        data_path=csv, data_type="train",
        lookback_window=L, predict_window=H,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        threshold_mode="fixed", fixed_threshold=0.005,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)

    set_seed()
    model = KronosClassifier(d_in=6, d_model=32, n_heads=4, ff_dim=64, n_layers=2, n_classes=3)
    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_losses = []
    for _ in range(2):
        model.train()
        running = 0.0
        nb = 0
        for features, stamp, labels in loader:
            features = features.to(device)
            stamp = stamp.to(device)
            labels = labels.to(device)
            logits = model(features, stamp)
            loss = criterion(logits, labels)
            assert torch.isfinite(loss), "训练 loss 出现 NaN/Inf"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            nb += 1
        epoch_losses.append(running / max(nb, 1))

    # 第 2 个 epoch 平均 loss 应低于第 1 个（允许小幅波动，这里要求严格下降）
    assert epoch_losses[1] < epoch_losses[0], f"loss 未下降: {epoch_losses}"
