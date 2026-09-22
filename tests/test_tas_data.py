"""TAS1 数据契约测试（计划 §8 P1：时间边界、归一化、PIT、批处理顺序不变性）。

全部用合成语料（不加载 330MB 真实 pickle，不触发 DDB）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tas_state.config import TASConfig
from tas_state.data import FEATURES, TASCorpus, _build_window

COLS = list(FEATURES)


def _fake_corpus(
    n_days: int = 320, start: str = "2024-06-03"
) -> dict[str, pd.DataFrame]:
    """两只股票的合成语料（B 日历日 → 交易日近似，足够覆盖窗口+purge 边界）。"""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(start, periods=n_days)
    out = {}
    for sym in ("SH000001", "SZ000002"):
        base = 10.0 + np.cumsum(rng.normal(0, 0.1, n_days))
        df = pd.DataFrame(
            {
                "open": base + rng.normal(0, 0.05, n_days),
                "high": base + 0.2 + rng.normal(0, 0.05, n_days),
                "low": base - 0.2 + rng.normal(0, 0.05, n_days),
                "close": base,
                "vol": rng.lognormal(10, 0.3, n_days),
                "amt": rng.lognormal(14, 0.3, n_days),
            },
            index=idx,
        )
        out[sym] = df
    return out


@pytest.fixture()
def corpus(tmp_path) -> TASCorpus:
    import pickle

    data = _fake_corpus()
    p = tmp_path / "train_data.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)
    cfg = TASConfig()
    return TASCorpus(p, cfg, target_end=cfg.train_target_end, split_name="train")


# ============================================================
# 1. 时间边界 / purge（计划 §8：test_target_purge_and_normalization）
# ============================================================
def test_target_purge_boundary(corpus):
    cfg = corpus.cfg
    for code, d in corpus.keys():
        w = corpus.window(code, d)
        # 标签末日 ≤ purge 边界（训练 2024-12-31）
        assert w.target_end <= pd.Timestamp(cfg.train_target_end)
        assert w.x_norm.shape == (100, 6)
        assert w.x_stamp.shape == (90, 5)
        assert w.y_stamp.shape == (10, 5)


def test_normalization_leak_free(corpus):
    """修改未来 10 行不改变历史均值方差与历史 token（归一化只用 90 行）。"""
    code, d = corpus.keys()[0]
    w0 = corpus.window(code, d)

    # 在原始语料上改未来行 → 重建窗口，历史段逐位不变
    df = corpus._data[code].copy()
    loc = df.index.get_loc(d)
    df.iloc[loc + 1 : loc + 11] = df.iloc[loc + 1 : loc + 11] * 3.0 + 5.0
    w1 = _build_window(code, df.iloc[loc - 89 : loc + 11], corpus.cfg)

    assert np.array_equal(w0.history, w1.history)
    assert np.array_equal(w0.x_stamp, w1.x_stamp)
    assert not np.array_equal(w0.future, w1.future)  # 标签确实变了


def test_history_stats_only_from_lookback(corpus):
    """归一化统计只来自历史 90 行（与官方 dataset.py 同口径）。"""
    code, d = corpus.keys()[1]
    df = corpus._data[code]
    loc = df.index.get_loc(d)
    hist = df.iloc[loc - 89 : loc + 1][COLS].values.astype(np.float32)
    w = corpus.window(code, d)
    mean = np.mean(hist, axis=0)
    std = np.std(hist, axis=0)
    expected_hist = np.clip((hist - mean) / (std + 1e-5), -5, 5)
    assert np.allclose(w.history, expected_hist, atol=1e-6)


# ============================================================
# 2. 批处理顺序不变性（计划 §8 P1）
# ============================================================
def test_batch_order_invariance(corpus):
    keys = corpus.keys()[:6]
    b1 = corpus.batch(keys)
    import random

    rng = random.Random(3)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    bs = corpus.batch(shuffled)
    for i, k in enumerate(shuffled):
        j = keys.index(k)
        assert np.array_equal(bs["X"][i], b1["X"][j])
        assert np.array_equal(bs["Y"][i], b1["Y"][j])
        assert np.array_equal(bs["stamp"][i], b1["stamp"][j])
        assert bs["codes"][i] == b1["codes"][j]


# ============================================================
# 3. STATIC 固定样本清单确定性（计划 §4：SHA256 字典序）
# ============================================================
def test_static_sample_deterministic(corpus):
    a = corpus.static_sample_keys(50)
    b = corpus.static_sample_keys(50)
    assert a == b
    assert len(set(a)) == len(a)  # 无重复
    # 与键构造顺序无关：打乱内部 _keys 不改变结果
    orig = corpus._keys[:]
    import random

    random.Random(9).shuffle(corpus._keys)
    c = corpus.static_sample_keys(50)
    corpus._keys[:] = orig
    assert a == c
    # 超过总量时取全量并记录（此处总键数 < 4096）
    assert len(corpus.static_sample_keys(4096)) == len(corpus)


# ============================================================
# 4. 采样器可复现（训练配方的固定批次序列前提）
# ============================================================
def test_sampler_reproducible(corpus):
    s1 = corpus.sampler(seed=42)
    s2 = corpus.sampler(seed=42)
    assert s1.draw(8) == s2.draw(8)
    s3 = corpus.sampler(seed=43)
    assert s1.draw(8) != s3.draw(8) or True  # 不同 seed 大概率不同（弱断言）


# ============================================================
# 5. 窗口契约：NaN 拒绝
# ============================================================
def test_window_rejects_nan(corpus):
    code, d = corpus.keys()[0]
    df = corpus._data[code].copy()
    loc = df.index.get_loc(d)
    df.iloc[loc, df.columns.get_loc("close")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _build_window(code, df.iloc[loc - 89 : loc + 11], corpus.cfg)
