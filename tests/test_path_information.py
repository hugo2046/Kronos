"""path_information 合成契约测试（计划 20260911 §4/§5/§6/§7 门禁）。

不触真实行情 / DDB / GPU / 本地大权重：模型出口部分用小型真实
KronosTokenizer/Kronos 合成实例在 CPU 上验证；路径包部分用合成日历与
数组验证纯规则。任何依赖外部资源的检查都不在本文件中冒充通过。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402
from model.kronos import auto_regressive_inference, calc_time_stamps  # noqa: E402


# ---------------- 合成模型工具（§4：同合成 tokenizer/model） ----------------

def make_synth_pair(seed: int = 7):
    """小型真实 Kronos tokenizer/model（CPU、eval、全 FP32）。"""
    torch.manual_seed(seed)
    tok = KronosTokenizer(d_in=6, d_model=16, n_heads=2, ff_dim=16,
                          n_enc_layers=2, n_dec_layers=2, ffn_dropout_p=0.0,
                          attn_dropout_p=0.0, resid_dropout_p=0.0,
                          s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0,
                          gamma=1.1, zeta=0.05, group_size=4)
    model = Kronos(s1_bits=4, s2_bits=4, n_layers=2, d_model=16, n_heads=2,
                   ff_dim=16, ffn_dropout_p=0.0, attn_dropout_p=0.0,
                   resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=False)
    tok.eval()
    model.eval()
    return tok, model


def make_synth_series(n_series: int = 3, seq: int = 60, seed: int = 11):
    """B 条合成 OHLCVA 窗口（价格量纲、无 NaN）。"""
    rng = np.random.default_rng(seed)
    df_list, x_ts_list = [], []
    base = pd.bdate_range("2025-01-01", periods=seq)
    for i in range(n_series):
        close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, seq))
        df = pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.002, seq)),
            "high": close * (1 + np.abs(rng.normal(0, 0.004, seq))),
            "low": close * (1 - np.abs(rng.normal(0, 0.004, seq))),
            "close": close,
            "volume": rng.uniform(1e6, 2e6, seq),
            "amount": rng.uniform(1e8, 2e8, seq),
        })
        df_list.append(df)
        x_ts_list.append(pd.Series(base))
    y_ts = [pd.Series(pd.bdate_range("2025-04-01", periods=10))] * n_series
    return df_list, x_ts_list, y_ts


# ---------------- §4 任务一：样本维出口 ----------------

def test_sample_exit_returns_bnhe6_default_keeps_old_type():
    """return_samples=True → [B,N,H,6] ndarray；默认仍返回 DataFrame 列表。"""
    tok, model = make_synth_pair()
    predictor = KronosPredictor(model, tok, device="cpu", max_context=512,
                                clip=5)
    df_list, x_ts_list, y_ts_list = make_synth_series(2, 60)
    torch.manual_seed(42)
    samples = predictor.predict_batch(df_list, x_ts_list, y_ts_list,
                                      pred_len=10, T=1.0, top_k=0, top_p=0.9,
                                      sample_count=20, verbose=False,
                                      return_samples=True)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (2, 20, 10, 6)
    assert np.isfinite(samples).all()
    torch.manual_seed(42)
    default = predictor.predict_batch(df_list, x_ts_list, y_ts_list,
                                      pred_len=10, T=1.0, top_k=0, top_p=0.9,
                                      sample_count=20, verbose=False)
    assert isinstance(default, list) and len(default) == 2
    for df in default:
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 6)


def test_default_equals_old_source_tail_bitexact():
    """默认分支 == 旧源尾部（reshape → np.mean(axis=1)）逐位一致；
    归一化空间样本均值 == 默认输出（同 RNG 起点两次运行）。"""
    tok, model = make_synth_pair()
    predictor = KronosPredictor(model, tok, device="cpu", max_context=512,
                                clip=5)
    df_list, x_ts_list, y_ts_list = make_synth_series(2, 60)
    # 直接调 generate 层拿归一化空间输出（绕过 predict_batch 反归一化）
    xs, means, stds = [], [], []
    for df in df_list:
        arr = df[predictor.price_cols + [predictor.vol_col,
                                         predictor.amt_vol]].values.astype(np.float32)
        mu, sd = arr.mean(axis=0), arr.std(axis=0)
        xs.append(np.clip((arr - mu) / (sd + 1e-5), -5, 5))
        means.append(mu)
        stds.append(sd)
    x = np.stack(xs).astype(np.float32)
    x_stamp = np.stack([calc_time_stamps(ts).values.astype(np.float32)
                        for ts in x_ts_list])
    y_stamp = np.stack([calc_time_stamps(ts).values.astype(np.float32)
                        for ts in y_ts_list])
    xt = torch.from_numpy(x)
    torch.manual_seed(42)
    default_norm = predictor.generate(xt, torch.from_numpy(x_stamp),
                                      torch.from_numpy(y_stamp), 10,
                                      1.0, 0, 0.9, 20, False)
    torch.manual_seed(42)
    sample_norm = predictor.generate(xt, torch.from_numpy(x_stamp),
                                     torch.from_numpy(y_stamp), 10,
                                     1.0, 0, 0.9, 20, False,
                                     return_samples=True)
    # 旧源尾部：np.mean(preds, axis=1)
    old_tail = np.mean(sample_norm, axis=1)
    assert default_norm.shape == (2, 10, 6)
    assert sample_norm.shape == (2, 20, 10, 6)
    np.testing.assert_array_equal(default_norm, old_tail)


def test_sample_exit_rng_compat():
    """同 RNG 起点下默认分支与样本分支消费相同随机流：结束 RNG 状态一致，
    且重跑同种子结果逐位一致。"""
    tok, model = make_synth_pair()
    predictor = KronosPredictor(model, tok, device="cpu", max_context=512,
                                clip=5)
    df_list, x_ts_list, y_ts_list = make_synth_series(2, 40)
    torch.manual_seed(42)
    a = predictor.predict_batch(df_list, x_ts_list, y_ts_list, pred_len=10,
                                T=1.0, top_k=0, top_p=0.9, sample_count=5,
                                verbose=False)
    state_a = torch.get_rng_state()
    torch.manual_seed(42)
    b = predictor.predict_batch(df_list, x_ts_list, y_ts_list, pred_len=10,
                                T=1.0, top_k=0, top_p=0.9, sample_count=5,
                                verbose=False, return_samples=True)
    state_b = torch.get_rng_state()
    assert torch.equal(state_a, state_b)
    torch.manual_seed(42)
    a2 = predictor.predict_batch(df_list, x_ts_list, y_ts_list, pred_len=10,
                                 T=1.0, top_k=0, top_p=0.9, sample_count=5,
                                 verbose=False)
    np.testing.assert_array_equal(a[0]["close"].to_numpy(),
                                  a2[0]["close"].to_numpy())


def test_n1_degenerate_sample_equals_default():
    """N=1 退化：样本分支 [B,1,H,6] 与默认分支逐位一致。"""
    tok, model = make_synth_pair()
    predictor = KronosPredictor(model, tok, device="cpu", max_context=512,
                                clip=5)
    df_list, x_ts_list, y_ts_list = make_synth_series(2, 40)
    torch.manual_seed(7)
    s1 = predictor.predict_batch(df_list, x_ts_list, y_ts_list, pred_len=10,
                                 T=1.0, top_k=0, top_p=0.9, sample_count=1,
                                 verbose=False, return_samples=True)
    torch.manual_seed(7)
    d1 = predictor.predict_batch(df_list, x_ts_list, y_ts_list, pred_len=10,
                                 T=1.0, top_k=0, top_p=0.9, sample_count=1,
                                 verbose=False)
    assert s1.shape == (2, 1, 10, 6)
    np.testing.assert_allclose(s1[:, 0, :, :], np.stack(
        [df[predictor.price_cols + [predictor.vol_col,
                                    predictor.amt_vol]].to_numpy()
         for df in d1]), rtol=1e-6, atol=1e-5)


def test_batch_series_independent():
    """B>1 不串股：改第 2 只输入，第 1 只默认与样本输出都不变。"""
    tok, model = make_synth_pair()
    predictor = KronosPredictor(model, tok, device="cpu", max_context=512,
                                clip=5)
    df_list, x_ts_list, y_ts_list = make_synth_series(2, 40)
    df_mod = [df_list[0], df_list[1] * 1.5]
    torch.manual_seed(3)
    out_a = predictor.predict_batch(df_list, x_ts_list, y_ts_list, pred_len=10,
                                    T=1.0, top_k=0, top_p=0.9, sample_count=4,
                                    verbose=False, return_samples=True)
    torch.manual_seed(3)
    out_b = predictor.predict_batch(df_mod, x_ts_list, y_ts_list, pred_len=10,
                                    T=1.0, top_k=0, top_p=0.9, sample_count=4,
                                    verbose=False, return_samples=True)
    np.testing.assert_array_equal(out_a[0], out_b[0])
    assert not np.allclose(out_a[1], out_b[1])


def test_sample_denormalization_matches_manual():
    """样本分支真实价格 = 归一化样本 × (std+1e-5) + mean（rtol 1e-6 / atol 1e-5），
    与默认分支反归一化路径一致（顺序舍入差如实披露口径）。"""
    tok, model = make_synth_pair()
    predictor = KronosPredictor(model, tok, device="cpu", max_context=512,
                                clip=5)
    df_list, x_ts_list, y_ts_list = make_synth_series(2, 40)
    cols = predictor.price_cols + [predictor.vol_col, predictor.amt_vol]
    stats = []
    xs = []
    from model.kronos import calc_time_stamps
    for df in df_list:
        arr = df[cols].values.astype(np.float32)
        mu, sd = arr.mean(axis=0), arr.std(axis=0)
        stats.append((mu, sd))
        xs.append(np.clip((arr - mu) / (sd + 1e-5), -5, 5))
    x = np.stack(xs).astype(np.float32)
    x_stamp = np.stack([calc_time_stamps(ts).values.astype(np.float32)
                        for ts in x_ts_list])
    y_stamp = np.stack([calc_time_stamps(ts).values.astype(np.float32)
                        for ts in y_ts_list])
    torch.manual_seed(5)
    norm_samples = predictor.generate(torch.from_numpy(x),
                                      torch.from_numpy(x_stamp),
                                      torch.from_numpy(y_stamp), 10,
                                      1.0, 0, 0.9, 6, False,
                                      return_samples=True)
    torch.manual_seed(5)
    real_samples = predictor.predict_batch(df_list, x_ts_list, y_ts_list,
                                           pred_len=10, T=1.0, top_k=0,
                                           top_p=0.9, sample_count=6,
                                           verbose=False, return_samples=True)
    for i, (mu, sd) in enumerate(stats):
        manual = norm_samples[i] * (sd + 1e-5) + mu
        np.testing.assert_allclose(real_samples[i], manual,
                                   rtol=1e-6, atol=1e-5)


def test_not_20_independent_single_sample_calls():
    """不得用 20 次独立 sample_count=1 冒充 N=20：两者随机流不同，
    结果应不同（同一合成模型、同输入、同起点种子）。"""
    tok, model = make_synth_pair()
    predictor = KronosPredictor(model, tok, device="cpu", max_context=512,
                                clip=5)
    df_list, x_ts_list, y_ts_list = make_synth_series(1, 40)
    torch.manual_seed(42)
    n20 = predictor.predict_batch(df_list, x_ts_list, y_ts_list, pred_len=10,
                                  T=1.0, top_k=0, top_p=0.9, sample_count=20,
                                  verbose=False, return_samples=True)[0]
    singles = []
    for k in range(20):
        torch.manual_seed(42)
        one = predictor.predict_batch(df_list, x_ts_list, y_ts_list,
                                      pred_len=10, T=1.0, top_k=0, top_p=0.9,
                                      sample_count=1, verbose=False,
                                      return_samples=True)[0, 0]
        singles.append(one)
    singles = np.stack(singles)
    # N=20 单调用内部样本有差异；与 20 次独立调用（每批仅 1 序列）不同流
    assert not np.allclose(n20[0], n20[1]) or not np.allclose(n20[0], n20[5])
    assert n20.shape == singles.shape
    assert not np.allclose(n20, singles)


# ---------------- §2/§3 纯规则：日期边界 / 选股 / 形状 ----------------

from path_information import config as PC  # noqa: E402
from path_information import data as PD  # noqa: E402


def test_segment_days_calendar_rule():
    """每段全部交易日 + calendar[t+10] ≤ 段标签末日；
    train 尾部标签跨界样本被排除，train/val 日期不相交。"""
    cal = pd.bdate_range("2025-06-02", periods=310)
    tr = PD.segment_days(cal, "2025-07-01", "2025-09-30")
    va = PD.segment_days(cal, "2025-10-01", "2025-12-31")
    # 全部落在候选区间内
    assert tr.min() >= pd.Timestamp("2025-07-01")
    assert tr.max() <= pd.Timestamp("2025-09-30")
    assert va.min() >= pd.Timestamp("2025-10-01")
    # t+10 规则：最后决策日的第 10 个未来交易日 ≤ 段标签末日
    for t in tr:
        assert cal[cal.get_loc(t) + 10] <= pd.Timestamp("2025-09-30")
    # 相邻被排除日：再晚一天就跨界
    nxt = cal[cal.get_loc(tr.max()) + 1]
    assert cal[cal.get_loc(nxt) + 10] > pd.Timestamp("2025-09-30")
    # 段间日期不相交（train 末日 + 标签窗仍在 9 月内，val 从 10 月起）
    assert tr.max() < va.min()
    assert len(set(tr) & set(va)) == 0
    # 每段为全部交易日（无 stride）：与逐日手推列表完全一致
    seg = cal[(cal >= "2025-07-01") & (cal <= "2025-09-30")]
    expected = [t for t in seg
                if cal[cal.get_loc(t) + 10] <= pd.Timestamp("2025-09-30")]
    assert list(tr) == expected


def test_label_dates_exact_trading_days():
    cal = pd.bdate_range("2025-01-01", periods=40)
    t = cal[5]
    lds = PD.label_dates(cal, t, predict_len=10)
    assert list(lds) == list(cal[6:16])


def test_mean_label_hand_check():
    closes = np.linspace(10.0, 12.0, 10)
    assert PD.mean_label(10.0, closes) == pytest.approx(
        closes.mean() / 10.0 - 1.0, rel=1e-15)
    with pytest.raises(ValueError):
        PD.mean_label(10.0, np.full(10, np.nan))
    with pytest.raises(ValueError):
        PD.mean_label(10.0, np.ones(9))


def test_select_codes_sha256_top64():
    """SHA256('PATH1|17|YYYY-MM-DD|code') 升序前 64；确定性、顺序无关。"""
    codes = [f"SZ300{i:03d}" for i in range(70)]
    ds = "2025-07-01"
    picked = PD.select_codes(codes, ds, per_day=64)
    manual = sorted(codes, key=lambda c: (
        hashlib.sha256(f"PATH1|17|{ds}|{c}".encode()).hexdigest(), c))
    assert picked == manual[:64]
    assert PD.select_codes(list(reversed(codes)), ds, per_day=64) == picked
    with pytest.raises(ValueError):
        PD.select_codes(codes[:63], ds, per_day=64)


def test_relative_paths_and_shape_invariance():
    """p[n,h]=close/close_t−1；股票特定常数不改变 shape；shape 双轴均值≈0。"""
    rng = np.random.default_rng(0)
    close_t = 50.0
    paths = 50.0 + np.cumsum(rng.normal(0, 0.5, (20, 10)), axis=1)
    p = PD.to_relative_paths(paths, close_t)
    np.testing.assert_allclose(p, paths / close_t - 1.0, rtol=1e-12)
    shape = PD.path_shape(p)
    assert shape.shape == (20, 10)
    p_shift = p + 0.037            # 股票特定常数
    np.testing.assert_allclose(PD.path_shape(p_shift), shape, rtol=1e-12,
                               atol=1e-12)
    assert abs(shape.mean()) < 1e-12


def test_spearman_average_rank_and_constant_nan():
    a = np.array([0.3, 0.1, 0.2])          # 秩 [3,1,2]
    b = np.array([1.0, 3.0, 2.0])          # 秩 [1,3,2] → 完全反序 = −1
    assert PD.spearman_avg_rank(a, b) == pytest.approx(-1.0, abs=1e-12)
    c = np.array([2.0, 1.0, 3.0])          # 秩 [2,1,3]：与 a 的秩相关 = +0.5
    assert PD.spearman_avg_rank(a, c) == pytest.approx(0.5, abs=1e-12)
    # 并列用平均秩
    a_tie = np.array([0.2, 0.2, 0.4])
    b_tie = np.array([1.0, 2.0, 3.0])
    r = PD.spearman_avg_rank(a_tie, b_tie)
    assert np.isfinite(r)
    # 常数日 → NaN
    assert np.isnan(PD.spearman_avg_rank(np.ones(5), np.arange(5.0)))
    assert np.isnan(PD.spearman_avg_rank(np.arange(5.0), np.full(5, 2.0)))


# ---------------- §5 头结构 ----------------

from path_information import model as PM  # noqa: E402


def test_head_param_count_753():
    head = PM.PathHead()
    assert sum(p.numel() for p in head.parameters()) == 753


def test_head_zero_init_residual():
    """最后 Linear 零初始化 → 未训练 r_hat 逐值为 0。"""
    torch.manual_seed(100)
    head = PM.PathHead()
    shape = torch.randn(8, 20, 10)
    b = torch.randn(8)
    r = head(shape, b)
    assert r.shape == (8,)
    assert bool((r == 0).all())


def test_head_arms_and_permutation_invariance():
    """打乱 N 轴输出不变；非零权重下改变 shape 改变 PATH 输出、
    MEAN（零 shape）输出对 shape 不敏感。"""
    torch.manual_seed(100)
    head = PM.PathHead()
    with torch.no_grad():
        head.mix[-1].weight.copy_(torch.randn_like(head.mix[-1].weight) * 0.05)
        head.mix[-1].bias.copy_(torch.randn_like(head.mix[-1].bias) * 0.05)
    shape = torch.randn(6, 20, 10)
    b = torch.randn(6)
    perm = torch.randperm(20)
    r1 = head(shape, b)
    r2 = head(shape[:, perm, :], b)
    torch.testing.assert_close(r1, r2)
    r_mean_arm = head(torch.zeros_like(shape), b)
    assert not torch.allclose(r1, r_mean_arm)
    # MEAN 臂语义：arm_input 恒为零 → 对 shape 变化不敏感；PATH 臂对 shape 敏感
    mean_a = head(PM.arm_input(shape, "MEAN"), b)
    mean_b = head(PM.arm_input(shape * 2.0, "MEAN"), b)
    assert torch.allclose(mean_a, mean_b)
    assert not torch.allclose(r1, head(PM.arm_input(shape * 2.0, "PATH"), b))


# ---------------- §5 训练规则 ----------------

from path_information import train as PT  # noqa: E402


def _synth_daydata(n_days=6, n_stocks=5, seed=0):
    rng = np.random.default_rng(seed)
    days = []
    for d in range(n_days):
        p = rng.normal(0, 0.02, (n_stocks, 20, 10))
        s = rng.normal(0.001, 0.02, n_stocks)
        y = s + rng.normal(0, 0.05, n_stocks)
        valid = np.ones(n_stocks, dtype=bool)
        if d == 1:
            valid[2] = False                       # 缺标签仅屏蔽
        days.append({"date": f"2025-07-{d + 1:02d}", "shape": PD.path_shape(p),
                     "s_g1": s, "y": y, "valid": valid})
    return days


def test_fit_stats_train_only_and_floor():
    days = _synth_daydata()
    stats = PT.fit_stats(days)
    s_all = np.concatenate([d["s_g1"][d["valid"]] for d in days])
    e_all = np.concatenate([(d["y"] - d["s_g1"])[d["valid"]] for d in days])
    assert stats["mu_g1"] == pytest.approx(s_all.mean())
    assert stats["sigma_g1"] == pytest.approx(s_all.std())
    assert stats["sigma_e"] == pytest.approx(e_all.std())
    assert stats["sigma_e"] > 0
    with pytest.raises(ValueError):
        PT.fit_stats([{"s_g1": np.ones(3), "y": np.ones(3),
                       "valid": np.ones(3, bool), "shape": np.zeros((3, 20, 10)),
                       "date": "d"}])


def test_g1_zero_residual_mse_hand():
    days = _synth_daydata()
    stats = PT.fit_stats(days)
    got = PT.g1_zero_residual_mse(days, stats)
    per_day = []
    for d in days:
        v = d["valid"]
        r = (d["y"][v] - d["s_g1"][v]) / stats["sigma_e"]
        per_day.append(float(np.mean(r ** 2)))
    assert got == pytest.approx(float(np.mean(per_day)), rel=1e-12)


def test_train_arm_same_init_and_e0_candidate(tmp_path):
    """两臂同初始化；e0 是合法候选；选点规则严格更低否则更早。"""
    days = _synth_daydata()
    val = _synth_daydata(seed=5)
    stats = PT.fit_stats(days)
    hist_m = PT.train_arm("MEAN", days, val, stats, tmp_path, epochs=2)
    hist_p = PT.train_arm("PATH", days, val, stats, tmp_path, epochs=2)
    # e0 候选存在且为 epoch 0
    assert hist_m["epochs"][0]["epoch"] == 0
    # 两臂同初始化：未训练参数逐值一致（PATH/MEAN 同 head 结构同 seed100）
    m0 = torch.load(tmp_path / "head_MEAN_s100_e000.pt", map_location="cpu",
                    weights_only=True)
    p0 = torch.load(tmp_path / "head_PATH_s100_e000.pt", map_location="cpu",
                    weights_only=True)
    for k in m0["state_dict"]:
        np.testing.assert_array_equal(m0["state_dict"][k].numpy(),
                                      p0["state_dict"][k].numpy())
    # 选点：严格更低才更新，并列取更早
    eps = [e["val_mse"] for e in hist_m["epochs"]]
    best = min(range(len(eps)), key=lambda i: (eps[i], i))
    assert hist_m["best_epoch"] == best


def test_train_arm_lr0_selects_and_loads_e0(tmp_path):
    """lr=0（仅测试注入）→ 各 epoch 与 e0 完全并列 → best=e0 且
    load_best 真实载入 e0 文件而非最后 epoch。"""
    days = _synth_daydata()
    val = _synth_daydata(seed=9)
    stats = PT.fit_stats(days)
    hist = PT.train_arm("PATH", days, val, stats, tmp_path, epochs=3, lr=0.0)
    assert hist["best_epoch"] == 0
    head, meta = PT.load_best(tmp_path, "PATH", 100)
    assert meta["best_epoch"] == 0
    r = head(torch.zeros(2, 20, 10), torch.ones(2))
    assert bool((r == 0).all())


def test_load_head_rejects_wrong_identity(tmp_path):
    days = _synth_daydata()
    val = _synth_daydata(seed=3)
    stats = PT.fit_stats(days)
    PT.train_arm("MEAN", days, val, stats, tmp_path, epochs=1)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        PT.load_head(tmp_path, "PATH", 100, 1)          # 错 arm（无该文件）
    with pytest.raises((FileNotFoundError, RuntimeError)):
        PT.load_head(tmp_path, "MEAN", 101, 1)          # 错 seed
    with pytest.raises((FileNotFoundError, RuntimeError)):
        PT.load_head(tmp_path, "MEAN", 100, 2)          # 错 epoch
    # 文件存在但 meta 协议错误 → RuntimeError
    f = tmp_path / "head_MEAN_s100_e001.pt"
    ck = torch.load(f, map_location="cpu", weights_only=True)
    ck["meta"]["protocol"] = "wrong-protocol"
    torch.save(ck, f)
    with pytest.raises(RuntimeError):
        PT.load_head(tmp_path, "MEAN", 100, 1)


def test_train_loss_daily_equal_weight():
    """每日监督股均值 → 日期等权（batch_days 聚合）手工对拍。"""
    days = _synth_daydata(n_days=4)
    stats = PT.fit_stats(days)
    torch.manual_seed(100)
    head = PM.PathHead()
    with torch.no_grad():
        head.mix[-1].weight.normal_(0, 0.05)
    loss = PT.day_equal_weight_loss(head, days, stats)
    per_day = []
    with torch.no_grad():
        for d in days:
            v = d["valid"]
            shape = torch.from_numpy((d["shape"] / stats["sigma_e"]).astype(
                np.float32))
            b = torch.from_numpy(((d["s_g1"] - stats["mu_g1"])
                                  / stats["sigma_g1"]).astype(np.float32))
            r_hat = head(shape, b).numpy()
            r = ((d["y"] - d["s_g1"]) / stats["sigma_e"]).astype(np.float32)
            per_day.append(float(np.mean((r_hat[v] - r[v]) ** 2)))
    assert loss == pytest.approx(float(np.mean(per_day)), rel=1e-6)


# ---------------- §6 开封与判据 ----------------

from path_information import evaluate as PE  # noqa: E402


def test_untrained_scores_equal_g1_and_residual_scale():
    """未训练三臂分数逐值等于原 G1；训练后 s_final = s_g1 + σe·r_hat。"""
    torch.manual_seed(100)
    head = PM.PathHead()
    days = _synth_daydata(seed=2)
    stats = PT.fit_stats(days)
    d = days[0]
    shape = torch.from_numpy((d["shape"] / stats["sigma_e"]).astype(
        np.float32))
    b = torch.from_numpy(((d["s_g1"] - stats["mu_g1"])
                          / stats["sigma_g1"]).astype(np.float32))
    with torch.no_grad():
        r0 = head(shape, b).numpy()
    np.testing.assert_array_equal(
        PE.final_scores(d["s_g1"], r0, stats), d["s_g1"])
    with torch.no_grad():
        head.mix[-1].weight.normal_(0, 0.05)
        head.mix[-1].bias.normal_(0, 0.05)
        r1 = head(shape, b).numpy()
    np.testing.assert_allclose(
        PE.final_scores(d["s_g1"], r1, stats),
        d["s_g1"] + stats["sigma_e"] * r1, rtol=1e-6)


def test_daily_paired_ic_and_criterion_hand():
    """三臂逐日 IC、配对差（双方有效日）、固定门槛手算对拍。"""
    dates = ["d1", "d2", "d3"]
    ic = {
        "G1": [0.10, np.nan, 0.05],
        "MEAN": [0.08, 0.20, np.nan],
        "PATH": [0.12, 0.18, 0.07],
    }
    out = PE.paired_daily_ic(ic, dates)
    # PATH−MEAN：双方有效日 = d1、d2（d2 MEAN 有效、PATH 有效；d3 MEAN NaN）
    pm = out["PATH-MEAN"]
    assert set(pm["dates"]) == {"d1", "d2"}
    assert pm["mean"] == pytest.approx(((0.12 - 0.08) + (0.18 - 0.20)) / 2)
    pg = out["PATH-G1"]
    assert set(pg["dates"]) == {"d1", "d3"}
    assert pg["mean"] == pytest.approx(((0.12 - 0.10) + (0.07 - 0.05)) / 2)
    # 判据：val 与 dev_eval 的 PATH−MEAN 均 >0 且 dev_eval 的 PATH−G1 >0
    v = {"PATH-MEAN": {"mean": 0.01}, "PATH-G1": {"mean": 0.005}}
    dv = {"PATH-MEAN": {"mean": 0.02}, "PATH-G1": {"mean": 0.03}}
    assert PE.criterion(v, dv) is True
    dv_bad = {"PATH-MEAN": {"mean": 0.02}, "PATH-G1": {"mean": -0.01}}
    assert PE.criterion(v, dv_bad) is False        # dev PATH−G1 ≤ 0
    v3 = {"PATH-MEAN": {"mean": 0.0}, "PATH-G1": {"mean": 0.9}}
    assert PE.criterion(v3, dv) is False           # val PATH−MEAN ≤ 0
    dv3 = {"PATH-MEAN": {"mean": -0.02}, "PATH-G1": {"mean": 0.3}}
    assert PE.criterion(v, dv3) is False           # dev PATH−MEAN ≤ 0


def test_spy_lock_label_loader_never_called_before_scores_frozen(tmp_path):
    """缺一/改一冻结分数文件 → 标签加载 0 次且抛错。"""
    calls = []

    def fake_label_loader(*a, **k):
        calls.append(1)
        return {"y": np.zeros(3)}

    manifest = {"files": {}}
    for i, arm in enumerate(("MEAN", "PATH")):
        p = tmp_path / f"scores_{arm}.npz"
        arr = {"s_final": np.arange(3.0) + i}
        np.savez(p, **arr)
        manifest["files"][arm] = {
            "file": p.name,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (tmp_path / "scores_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    # 完整 → loader 被调用
    PE.unseal(tmp_path, label_loader=fake_label_loader)
    assert len(calls) == 1
    # 缺一 → 抛错且 0 次
    (tmp_path / "scores_MEAN.npz").unlink()
    with pytest.raises(RuntimeError):
        PE.unseal(tmp_path, label_loader=fake_label_loader)
    assert len(calls) == 1
    # 改一（SHA 变）→ 抛错且 0 次
    np.savez(tmp_path / "scores_MEAN.npz", s_final=np.arange(3.0) + 7.0)
    with pytest.raises(RuntimeError):
        PE.unseal(tmp_path, label_loader=fake_label_loader)
    assert len(calls) == 1


# ---------------- 缓存身份 / 标签分离 ----------------

from path_information import paths as PP  # noqa: E402


def test_path_chunk_identity_roundtrip(tmp_path):
    payload = {"close_paths": np.zeros((2, 20, 10), np.float32),
               "s_g1": np.zeros(2)}
    ident = {"protocol": PC.PROTOCOL_VERSION, "split": "train"}
    p = PP.write_path_chunk(tmp_path, "train_20250701.npz", payload, ident)
    got = PP.read_path_chunk(p, ident)
    np.testing.assert_array_equal(got["s_g1"], payload["s_g1"])
    with pytest.raises(PP.CacheMismatchError):
        PP.read_path_chunk(p, {"protocol": "other", "split": "train"})
    with pytest.raises(PP.CacheMismatchError):
        PP.read_path_chunk(p, {"protocol": PC.PROTOCOL_VERSION,
                               "split": "val"})


def test_label_missing_only_masks_common_cells():
    """缺标签只影响共同 loss/评价格，不重选股票。"""
    y = np.array([0.01, np.nan, 0.03])
    s = np.array([0.02, 0.00, 0.01])
    mask = PD.valid_label_mask(y)
    np.testing.assert_array_equal(mask, [True, False, True])
    assert len(mask) == len(y)                    # 格数不变、不重选
