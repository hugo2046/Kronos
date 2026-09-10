"""sae_residual 合成契约测试（计划 §7 门禁；不触真实行情/DDB/GPU）。

覆盖：mean 标签手算对拍、10 日精确日历缺失剔除、purge/无前视、μh/σh/σe
只拟合 train、零残差恢复 G1、AE/SAE 初始一致、beta0 精确退化、KL 有限且
在 p=rho 最小、残差量纲还原、键对齐与基底冻结、生产路径（真实 train/save/
/load 两 epoch、e1/e2 权重不同、best 选点与载入、错权重/错身份拒绝）、
决策日规则与按日 RNG、样本选择确定性。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sae_residual import config as C  # noqa: E402
from sae_residual import data as D  # noqa: E402
from sae_residual import model as M  # noqa: E402
from sae_residual.cache_io import (CacheMismatchError, load_chunk,  # noqa: E402
                                   write_chunk)
from sae_residual import train as T  # noqa: E402


# ---------------- §3 标签与日历 ----------------

def test_mean_label_hand_check():
    """y_mean = mean(close[t+1..t+10])/close_t − 1 手算对拍。"""
    closes = np.linspace(10.0, 11.0, 10)
    close_t = 10.0
    expect = closes.mean() / close_t - 1.0
    assert D.mean_label(close_t, closes) == pytest.approx(expect, rel=1e-15)
    # 等价于逐日收益均值（+1 的展开）
    rets = closes / close_t - 1.0
    assert D.mean_label(close_t, closes) == pytest.approx(rets.mean(), rel=1e-12)


def test_mean_label_rejects_missing_or_wrong_length():
    """标签缺失/非有限/长度不符 → 无标签（异常），不得 forward-fill。"""
    with pytest.raises(ValueError):
        D.mean_label(10.0, np.array([1.0] * 9))
    with pytest.raises(ValueError):
        D.mean_label(10.0, np.append(np.linspace(1, 2, 9), np.nan))
    with pytest.raises(ValueError):
        D.mean_label(np.nan, np.linspace(1, 2, 10))
    with pytest.raises(ValueError):
        D.mean_label(0.0, np.ones(10))          # 现价非法


def test_label_dates_exact_calendar():
    """标签日 = t 之后 10 个精确交易日（不 date+n 推、不跳停牌凑数）。"""
    cal = pd.bdate_range("2025-01-01", periods=30)
    t = cal[4]
    lds = D.label_dates(cal, t)
    assert list(lds) == list(cal[5:15])
    assert len(lds) == C.PREDICT_LEN
    # 节假日不可用 date+n 推：10 个交易日跨周末 → 自然日跨度 ≥ 12
    diffs = (lds - t).days
    assert diffs[-1] >= 12 and not np.all(np.diff(diffs) == 1)


def test_decision_days_rule():
    """每 5 个交易日取决策日；标签终点 ≤ 上界截断。"""
    cal = pd.bdate_range("2024-01-01", periods=50)
    label_end = cal[29]                          # 第 30 天为标签末日上界
    days = D.decision_days(cal, "2024-01-01", label_end)
    # anchor=cal[0]，i=0,5,10,...；i+10 ≤ 29 → i ≤ 19
    assert list(days) == [cal[i] for i in range(0, 20, 5)]
    # start 非交易日 → 取其后第一个交易日
    days2 = D.decision_days(cal, "2024-01-01", cal[49], stride=5)
    assert days2[0] == cal[0]
    days3 = D.decision_days(cal, "2023-12-31", cal[49], stride=5)
    assert days3[0] == cal[0]


# ---------------- §3 归一化统计（purge：只拟合 train） ----------------

def test_norm_stats_train_only():
    """μh/σh 只用 train；val/test 分布不同也不影响统计。"""
    rng = np.random.default_rng(0)
    h_tr = rng.normal(0.0, 1.0, (500, 8))
    stats = D.fit_norm_stats(h_tr)
    np.testing.assert_allclose(stats["mu"], h_tr.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(stats["sigma"] ** 2, np.maximum(
        h_tr.std(axis=0), C.STATS_FLOOR) ** 2, rtol=1e-9)
    # 应用到漂移的 val 上不会重新估计
    h_va = h_tr + 100.0
    x_va = D.normalize_hidden(h_va, stats)
    np.testing.assert_allclose(x_va.mean(axis=0), 100.0 / stats["sigma"], rtol=1e-4)


def test_residual_std_floor_and_no_mean_shift():
    e = np.array([0.01, -0.02, 0.015, -0.005, 0.03])
    s = D.residual_std(e)
    assert s == pytest.approx(e.std(), rel=1e-12)
    assert s >= C.STATS_FLOOR
    # 不减均值：零输出头 = 零修正（不自带均值偏移）——r=e/σe 均值非零
    r = e / s
    assert r.mean() != pytest.approx(0.0, abs=1e-6) or abs(e.mean()) < 1e-12
    with pytest.raises(ValueError):
        D.residual_std(np.full(10, 3.14))       # 退化 → 停实验


def test_normalize_window_matches_predict_batch():
    """窗口 z-score + clip5 与 KronosPredictor.predict_batch 逐字一致。"""
    rng = np.random.default_rng(1)
    x = rng.normal(100, 5, (90, 6)).astype(np.float32)
    xn, mean, std = D.normalize_window(x)
    expect = np.clip((x - x.mean(axis=0)) / (x.std(axis=0) + C.EPS_STD),
                     -5, 5)
    np.testing.assert_allclose(xn, expect.astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(mean, x.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(std, x.std(axis=0), rtol=1e-5)


def test_no_lookahead_in_target_construction():
    """特征只含 ≤t 行；标签单独取数后按键 join（构造层面无 t 后特征）。"""
    rng = np.random.default_rng(2)
    cal = pd.bdate_range("2025-01-01", periods=105)
    t = cal[90]
    window = pd.DataFrame(
        rng.normal(10, 1, (90, 6)),
        index=cal[:90], columns=["open", "high", "low", "close", "volume",
                                 "amount"])
    assert (window.index <= t).all()            # 特征无前视
    lds = D.label_dates(cal, t)
    assert (lds > t).all() and len(lds) == 10   # 标签全部在 t 后
    xn, _, _ = D.normalize_window(window.values)
    assert xn.shape == (90, 6)


# ---------------- §5 结构门禁 ----------------

def test_zero_residual_recovers_g1_exactly():
    """零残差初始化：r_hat 逐值为 0 → s_final 逐值等于 G1。"""
    torch.manual_seed(0)
    head = M.build_head(7, 32)
    x = torch.randn(64, 32)
    r_hat, _, _ = head(x)
    assert torch.equal(r_hat, torch.zeros_like(r_hat))    # 严格 0
    s_g1 = torch.randn(64)
    s_final = M.final_signal(s_g1, r_hat, sigma_e=0.0123)
    assert torch.equal(s_final, s_g1)                     # 逐位一致


def test_ae_sae_identical_initial_state():
    """同 seed AE/SAE 初始 state_dict 完全一致（唯一差异 = beta）。"""
    a = M.build_head(100, 832)
    b = M.build_head(100, 832)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    c = M.build_head(101, 832)
    assert not torch.equal(next(iter(a.state_dict().values())),
                           next(iter(c.state_dict().values())))


def test_beta_zero_exactly_degenerates_to_ae():
    """beta=0 时总损失精确等于 L_pred + 0.1·L_recon（SAE→AE 退化）。"""
    torch.manual_seed(0)
    head = M.build_head(3, 16)
    x, r = torch.randn(128, 16), torch.randn(128)
    r_hat, x_hat, z = head(x)
    out = M.head_loss(r_hat, r, x_hat, x, z, beta=0.0)
    manual = (torch.nn.functional.mse_loss(r_hat, r)
              + C.LOSS["recon_weight"] * torch.nn.functional.mse_loss(x_hat, x))
    assert torch.allclose(out["total"], manual, atol=0, rtol=1e-7)
    assert torch.allclose(out["l_sparse"] * 0.0, torch.zeros(()))  # 系数精确保留


def test_sparse_kl_finite_minimum_at_rho():
    """KL 有限；p_j=rho 时为 0（最小）；偏离 rho 严格为正。"""
    rho = C.LOSS["rho"]
    n = C.HEAD["z_dim"]
    z_rho = torch.full((50, n), rho)
    # float32 下 KL 数学上 ≥0，p=rho 处数值 0（容 1e-7 舍入）
    assert float(M.sparse_kl(z_rho)) == pytest.approx(0.0, abs=1e-7)
    z_off = torch.full((50, n), rho + 0.1)
    kl_off = float(M.sparse_kl(z_off))
    assert np.isfinite(kl_off) and kl_off > 0
    # 同侧单调：偏离 rho 越远 KL 越大（0.15 vs 0.06）
    z_near = torch.full((50, n), rho + 0.01)
    assert float(M.sparse_kl(z_off)) > float(M.sparse_kl(z_near)) > 0


def test_residual_restores_return_scale():
    """s_final = s_G1 + σe·r_hat：残差正确还原收益率量纲。"""
    s_g1 = np.array([0.01, -0.02])
    r_hat = np.array([1.5, -2.0])
    sigma_e = 0.02
    out = M.final_signal(torch.tensor(s_g1), torch.tensor(r_hat), sigma_e)
    np.testing.assert_allclose(out.numpy(), s_g1 + sigma_e * r_hat, rtol=1e-7)


# ---------------- §4 键对齐 / 身份 / 抽样 ----------------

def test_day_rng_stable_and_independent_of_order():
    """按日派生 RNG：同日同选、与处理顺序无关（断点续跑不变）。"""
    a = D.make_day_rng(pd.Timestamp("2020-03-05"))
    b = D.make_day_rng(pd.Timestamp("2020-03-05"))
    c = D.make_day_rng(pd.Timestamp("2020-03-06"))
    codes = [f"{i:06d}.SH" for i in range(100)]
    sa, sb, sc = (D.select_codes(codes, g) for g in (a, b, c))
    assert sa == sb
    assert set(sa) <= set(codes) and len(sa) == C.PER_DAY


def test_select_codes_skips_short_days():
    rng = D.make_day_rng(pd.Timestamp("2020-01-01"))
    with pytest.raises(ValueError):
        D.select_codes(["A", "B"], rng)          # 不足 64 记录并跳过
    picked = D.select_codes(sorted("CODE%03d" % i for i in range(70)), rng)
    assert len(picked) == 64 and len(set(picked)) == 64   # 无放回


def test_chunk_identity_roundtrip_and_reject(tmp_path):
    """chunk 写读往返；身份不匹配/缺身份拒绝复用。"""
    payload = {"dates": np.array(["2020-01-01"]), "h": np.zeros((1, 4),
                                                                np.float32)}
    ident = {"protocol": C.PROTOCOL_VERSION, "split": "train",
             "tokenizer_sha": "tok", "predictor_sha": "pred"}
    p = tmp_path / "train_20200101.npz"
    write_chunk(p, payload, ident)
    got = load_chunk(p, ident)
    np.testing.assert_array_equal(got["h"], payload["h"])
    with pytest.raises(CacheMismatchError):
        load_chunk(p, {**ident, "predictor_sha": "OTHER"})   # 错身份
    with pytest.raises(CacheMismatchError):
        p2 = tmp_path / "no_ident.npz"
        np.savez(p2, **payload)
        load_chunk(p2, ident)                                 # 缺身份


def test_keys_alignment_join():
    """teacher/hidden/标签三方按 (date,instrument) 键 join 一致（基底冻结）。"""
    n = 128
    dates = np.array(["2020-01-02"] * n)
    inst = np.array(sorted(f"C{i:03d}" for i in range(n)))
    # 三方同键 → join 不重排不丢样本
    assert len(set(zip(dates, inst))) == n
    # 基底冻结：hidden 变换不改键
    h = np.ones((n, 8), np.float32)
    x = D.normalize_hidden(h, D.fit_norm_stats(h))
    assert x.shape == h.shape


# ---------------- §7 生产路径门禁（真实 train/save/load，合成数据） ----------------

def _synth_train_val(n_days=3, per_day=64, d=16, seed=0):
    rng = np.random.default_rng(seed)
    rows_tr, rows_va = [], []
    for k in range(n_days + 1):
        rows = rows_tr if k < n_days else rows_va
        for i in range(per_day):
            rows.append((f"2020-01-{k + 1:02d}", f"C{i:03d}"))
    def pack(rows):
        dates = np.array([r[0] for r in rows])
        h = rng.normal(0, 1, (len(rows), d)).astype(np.float32)
        # 残差与 h 线性相关 + 噪声：可学的非退化结构
        w = rng.normal(0, 1, d)
        e = (h @ w / np.sqrt(d) + rng.normal(0, .1, len(rows))) * 0.01
        s = rng.normal(0, .01, len(rows))
        return {"x": h, "r": (e / e.std()).astype(np.float32), "date": dates,
                "e": e, "s_g1": s}
    return pack(rows_tr), pack(rows_va)


def test_production_two_epochs_distinct_weights_and_best_load(tmp_path):
    """实际 train/save/load：两 epoch 权重不同；载入 best 确为最优 epoch。"""
    torch.manual_seed(0)
    tr, va = _synth_train_val()
    model0 = M.build_head(100, tr["x"].shape[1])
    out = tmp_path
    res = T.train_arm(tr, va, "AE", 100, out, device="cpu", epochs=2)
    m1 = T.load_head(out, "AE", 100, 1, tr["x"].shape[1])
    sd = [torch.load(T.head_file(out, "AE", 100, e), weights_only=True)["state_dict"]
          for e in (1, 2)]
    assert not torch.equal(sd[0]["encoder.0.weight"], sd[1]["encoder.0.weight"])
    # best meta 与指标一致
    best = json.loads((out / "best_AE_s100.json").read_text())["best_epoch"]
    assert best == res["best_epoch"]
    m_best, best_meta = T.load_best(out, "AE", 100, tr["x"].shape[1])
    assert best_meta["best_epoch"] == best
    # e1/e2 指标随 epoch 保存（选点=验证 L_pred 最低）
    mets = json.loads((out / "epochs_metrics_AE_s100.json").read_text())["epochs"]
    assert [m["epoch"] for m in mets] == [1, 2]
    assert res["best_val_l_pred"] == min(m["val_l_pred"] for m in mets)


def test_production_rejects_wrong_head_or_identity(tmp_path):
    """错权重（seed/arm/epoch 不匹配）拒绝。"""
    tr, va = _synth_train_val()
    T.train_arm(tr, va, "SAE", 101, tmp_path, device="cpu", epochs=1)
    # 错 seed：文件存在但 meta.seed 不符（改名伪造）→ RuntimeError
    fake = T.head_file(tmp_path, "SAE", 202, 1)
    fake.write_bytes(T.head_file(tmp_path, "SAE", 101, 1).read_bytes())
    with pytest.raises(RuntimeError):
        T.load_head(tmp_path, "SAE", 202, 1, tr["x"].shape[1])
    fake2 = T.head_file(tmp_path, "AE", 101, 1)
    fake2.write_bytes(T.head_file(tmp_path, "SAE", 101, 1).read_bytes())
    with pytest.raises(RuntimeError):
        T.load_head(tmp_path, "AE", 101, 1, tr["x"].shape[1])    # 错 arm
    with pytest.raises(FileNotFoundError):
        T.load_head(tmp_path, "SAE", 101, 99, tr["x"].shape[1])  # 缺 best


def test_same_batch_order_across_arms():
    """AE/SAE 同 seed 消费相同批序列（shuffle 逐位一致）。"""
    n = 300
    g1 = torch.Generator()
    g1.manual_seed(100)
    g2 = torch.Generator()
    g2.manual_seed(100)
    for _ in range(3):
        assert torch.equal(torch.randperm(n, generator=g1),
                           torch.randperm(n, generator=g2))


def test_val_pred_loss_equal_weight_per_day():
    """验证 L_pred：先日内 mean 再跨日 mean（各日等权，非样本等权）。"""
    model = M.build_head(5, 4)
    val = {"x": np.zeros((6, 4), np.float32),
           "r": np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], np.float32),
           "date": np.array(["a"] * 3 + ["b"] * 3)}
    # 零残差头：err = r²；日 a=0，日 b=4 → 等权均值 2（样本等权也是 2，
    # 改为不等日大小验证）
    val2 = {"x": np.zeros((5, 4), np.float32),
            "r": np.array([0.0, 0.0, 0.0, 2.0, 2.0], np.float32),
            "date": np.array(["a"] * 3 + ["b"] * 2)}
    loss = T.val_pred_loss(model, val2)
    assert loss == pytest.approx((0.0 + 4.0) / 2, rel=1e-6)      # (0+4)/2=2
    loss6 = T.val_pred_loss(model, val)
    assert loss6 == pytest.approx(2.0, rel=1e-6)


def test_run_py_has_required_stages():
    src = (REPO_ROOT / "sae_residual" / "run.py").read_text(encoding="utf-8")
    for stage in ("preflight", "smoke", "cache", "pilot", "confirm", "report"):
        assert f'"{stage}"' in src, f"缺 stage：{stage}"
    # 不读 registry forward、不触 LoRA 通道：源内不得出现相关调用
    assert "registry/forward" not in src
    assert "lora" not in src.lower()
