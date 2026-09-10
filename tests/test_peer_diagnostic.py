"""peer_residual 诊断与执行纠偏契约测试（计划 20260910 §4~§6）。

先失败后实现：任务一（冻结 MSE 公式/跨日等权/监督隔离/排名诊断/CLI 拒绝
覆盖/legacy 头精确 SHA 只读路径）、任务二（读取边界 spy 复现与适配层）、
任务三（checkpoint 身份门禁/缓存清单校验/复用门禁/信号先冻结后回测顺序）。
全部合成数据，不触真实行情/DDB/GPU/大缓存。
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

from peer_residual import config as C  # noqa: E402
from peer_residual import data as PD  # noqa: E402
from peer_residual import model as M  # noqa: E402
from peer_residual import train as T  # noqa: E402
from sae_residual.cache_io import write_chunk  # noqa: E402

D = 16
STATS = {"mu": np.zeros(D, np.float32), "sigma": np.ones(D, np.float32),
         "sigma_e": 0.1, "d_in": D}


# ---------------- §4 任务一：冻结公式 ----------------

def test_day_errors_hand_check():
    """手算：r=[1,3]，G1 r_hat=[0,0] → MSE 5 / 收益量纲 0.05；头 [1,1] → 2 / 0.02。"""
    from peer_residual.diagnose import day_errors

    r = np.array([1.0, 3.0])
    g1 = day_errors(r, np.zeros(2), 0.1)
    assert g1["mse_norm"] == pytest.approx(5.0, rel=1e-12)
    assert g1["mse_return"] == pytest.approx(0.05, rel=1e-12)
    assert g1["bias_return"] == pytest.approx(0.2, rel=1e-12)   # mean(1,3)*0.1
    head = day_errors(r, np.array([1.0, 1.0]), 0.1)
    assert head["mse_norm"] == pytest.approx(2.0, rel=1e-12)
    assert head["mse_return"] == pytest.approx(0.02, rel=1e-12)


def test_cross_day_equal_weight_unequal_sizes():
    """两日期样本数不同：跨日等权（非合并全部格平均）。"""
    from peer_residual.diagnose import day_errors

    d1 = day_errors(np.array([1.0, 3.0]), np.zeros(2), 1.0)    # 日1：MSE 5
    d2 = day_errors(np.array([2.0]), np.zeros(1), 1.0)         # 日2：MSE 4
    eq_weight = (d1["mse_norm"] + d2["mse_norm"]) / 2          # = 4.5
    pooled = (5 * 2 + 4 * 1) / 3                               # = 4.667（错误口径）
    assert eq_weight == pytest.approx(4.5)
    assert eq_weight != pytest.approx(pooled)


def test_nonsupervised_cells_excluded_from_mse():
    """非监督格加入极大目标不影响 MSE；改变 loss mask 不改变 forward 输入。"""
    from peer_residual.diagnose import day_errors

    r_full = np.array([0.5, -0.5, 1000.0])     # 第 3 格非监督（loss 外）
    r_hat_full = np.array([0.2, -0.2, 0.0])
    loss = np.array([True, True, False])
    e1 = day_errors(r_full[loss], r_hat_full[loss], 1.0)      # 已截取 LossSet
    r_full2 = r_full.copy()
    r_full2[2] = 7.0                           # 非监督格任意扰动
    e2 = day_errors(r_full2[loss], r_hat_full[loss], 1.0)
    assert e1["mse_norm"] == pytest.approx(0.09)
    assert e2["mse_norm"] == pytest.approx(0.09)              # 监督格结果一致
    # forward 输入与 loss mask 无关：DayData 装配后 x 不含标签通道
    day = PD.DayData(date="t", codes=["a", "b"],
                     x=np.ones((2, D), np.float32),
                     loss_mask=np.array([True, False]),
                     r=np.array([1.0, 999.0]), s_g1=np.zeros(2))
    x, valid, loss, r_t = PD.collate_batch([day])
    assert x.shape == (1, 2, D) and loss[0].tolist() == [True, False]
    assert not np.array_equal(x.numpy()[0, 1], np.zeros(D))    # 第 2 格仍是真实输入


def test_spearman_average_rank_ties_and_nan():
    """平均秩并列处理；常数向量 → NaN（不擅自记零）。"""
    from peer_residual.diagnose import daily_spearman

    a = np.array([1.0, 2.0, 2.0, 3.0])
    b = np.array([1.0, 2.5, 1.5, 3.0])
    rho = daily_spearman(a, b)
    assert 0 < rho < 1 and np.isfinite(rho)
    # 与 scipy 平均秩 spearman 对拍
    from scipy.stats import spearmanr
    assert rho == pytest.approx(spearmanr(a, b).statistic, rel=1e-12)
    assert np.isnan(daily_spearman(np.ones(4), b))             # 常数 → NaN
    assert np.isnan(daily_spearman(a, np.full(4, 2.0)))


# ---------------- §4 legacy 头只读路径 ----------------

def _save_legacy_head(tmp_path, name="head_OFF_s100_e018.pt"):
    model = M.build_head(100, D)
    p = tmp_path / name
    torch.save({"state_dict": model.state_dict(),
                "meta": {"arm": "OFF", "seed": 100, "epoch": 18,
                         "protocol": C.PROTOCOL_VERSION, "d_in": D}}, p)
    return p


def test_legacy_head_exact_sha_only(tmp_path):
    """legacy 路径只接收交接清单精确 SHA；错 SHA 拒绝；标记 identity 不完整。"""
    from peer_residual.identity import load_legacy_head
    from sae_residual.cache_io import sha256_file

    p = _save_legacy_head(tmp_path)
    sha = sha256_file(p)
    model, note = load_legacy_head(p, sha, "OFF", 100, 18, D)
    assert note["legacy_identity_incomplete"] is True
    with pytest.raises(RuntimeError):
        load_legacy_head(p, "0" * 64, "OFF", 100, 18, D)       # 错 SHA
    with pytest.raises(RuntimeError):
        load_legacy_head(p, sha, "ON", 100, 18, D)             # 错 arm


def test_legacy_head_forbidden_for_new_training(tmp_path):
    """legacy 头不得用于新训练续跑/新实验验收。"""
    from peer_residual.identity import (assert_training_eligible,
                                        load_legacy_head)
    from sae_residual.cache_io import sha256_file

    p = _save_legacy_head(tmp_path)
    _, note = load_legacy_head(p, sha256_file(p), "OFF", 100, 18, D)
    with pytest.raises(RuntimeError):
        assert_training_eligible(note)                         # 缺新身份字段


# ---------------- §6.1 checkpoint 身份门禁 ----------------

FULL_IDENTITY = {"run_id": C.RUN_ID, "protocol_digest": "pd" * 32,
                 "tokenizer_sha": "t" * 64, "predictor_sha": "p" * 64,
                 "sae_manifest_sha": "s" * 64, "peer_manifest_sha": "q" * 64,
                 "norm_stats_sha": "n" * 64, "arm": "ON", "seed": 101,
                 "epoch": 7}


def test_checkpoint_identity_roundtrip_and_rejects(tmp_path):
    """新 schema：字段齐全往返；错 run/base/cache、缺字段、改字节拒绝。"""
    from peer_residual.identity import (load_head_checkpoint,
                                        save_head_checkpoint)
    from sae_residual.cache_io import sha256_file

    model = M.build_head(101, D)
    p = tmp_path / "head.pt"
    save_head_checkpoint(model, p, dict(FULL_IDENTITY))
    got = load_head_checkpoint(p, FULL_IDENTITY, D)
    assert torch.equal(got.state_dict()["proj.weight"],
                       model.state_dict()["proj.weight"])
    # 错 run_id / base / cache → 拒绝
    for key, bad in (("run_id", "other-run"), ("tokenizer_sha", "x" * 64),
                     ("peer_manifest_sha", "y" * 64)):
        bad_expect = dict(FULL_IDENTITY, **{key: bad})
        with pytest.raises(RuntimeError):
            load_head_checkpoint(p, bad_expect, D)
    # 缺身份字段的 ckpt → 拒绝（不自动补写/升级）
    ck = torch.load(p, weights_only=True)
    del ck["identity"]["norm_stats_sha"]
    torch.save(ck, tmp_path / "missing.pt")
    with pytest.raises(RuntimeError):
        load_head_checkpoint(tmp_path / "missing.pt", FULL_IDENTITY, D)
    # 字节被改 → 文件 SHA 门禁拒绝
    sha = sha256_file(p)
    load_head_checkpoint(p, FULL_IDENTITY, D, expected_file_sha=sha)
    ck2 = torch.load(p, weights_only=True)
    with torch.no_grad():
        ck2["state_dict"]["proj.weight"].add_(0.5)
    torch.save(ck2, p)
    with pytest.raises(RuntimeError):
        load_head_checkpoint(p, FULL_IDENTITY, D, expected_file_sha=sha)


def _mini_manifest(split: str, chunks: list[tuple[str, str, int]]) -> dict:
    return {"splits": {split: {"n_chunks": len(chunks), "n_days": len(chunks),
                               "chunks": [{"file": f, "sha256": s, "n": n}
                                          for f, s, n in chunks]}}}


def test_cache_verify_complete_keys_and_sha(tmp_path):
    """缓存校验按冻结清单：缺 chunk / 改字节 / 元数据对但字节变 → 拒绝。"""
    from peer_residual.identity import verify_cache_dir
    from sae_residual.cache_io import sha256_file

    d = tmp_path / "cache" / "train"
    d.mkdir(parents=True)
    files = []
    for i, day in enumerate(("20250102", "20250107")):
        p = d / f"peer_train_{day}.npz"
        write_chunk(p, {"h": np.zeros((1, D), np.float32),
                        "peer_codes": np.array(["a"])},
                    {"protocol": C.PROTOCOL_VERSION, "split": "train",
                     "date": f"2025-01-0{i + 2}", "n_peers": 1, "n_extra": 1})
        files.append((p.name, sha256_file(p), 1))
    mf = _mini_manifest("train", files)
    verify_cache_dir(d, mf, "train")                           # 完整 → 通过
    # 缺 chunk（清单有、目录无）
    mf_missing = _mini_manifest("train", files + [("ghost.npz", "z" * 64, 1)])
    with pytest.raises(RuntimeError):
        verify_cache_dir(d, mf_missing, "train")
    # 元数据相符但数据字节被改 → SHA 拒绝
    tampered = dict(mf)
    tampered["splits"] = {"train": dict(mf["splits"]["train"])}
    bad = list(files)
    p2 = d / files[0][0]
    write_chunk(p2, {"h": np.ones((1, D), np.float32),
                     "peer_codes": np.array(["a"])},
                {"protocol": C.PROTOCOL_VERSION, "split": "train",
                 "date": "2025-01-02", "n_peers": 1, "n_extra": 1})
    with pytest.raises(RuntimeError):
        verify_cache_dir(d, mf, "train")                       # 原 SHA 不再匹配


def test_reuse_gate_rejects_tampered_head(tmp_path):
    """旧"best.json 存在就跳过"不能绕过身份：篡改头文件 → 复用拒绝。"""
    from peer_residual.identity import verify_reusable_head
    from sae_residual.cache_io import sha256_file

    model = M.build_head(100, D)
    p = tmp_path / "head_OFF_s100_e001.pt"
    torch.save({"state_dict": model.state_dict(),
                "meta": {"arm": "OFF", "seed": 100, "epoch": 1,
                         "protocol": C.PROTOCOL_VERSION, "d_in": D}}, p)
    sha = sha256_file(p)
    (tmp_path / "best_OFF_s100.json").write_text(
        json.dumps({"best_epoch": 1}), encoding="utf-8")
    verify_reusable_head(tmp_path, "OFF", 100, 1, {"head_OFF_s100_e001.pt": sha})
    ck = torch.load(p, weights_only=True)
    with torch.no_grad():
        ck["state_dict"]["proj.weight"].add_(1.0)
    torch.save(ck, p)
    with pytest.raises(RuntimeError):
        verify_reusable_head(tmp_path, "OFF", 100, 1, {"head_OFF_s100_e001.pt": sha})
    with pytest.raises(RuntimeError):
        verify_reusable_head(tmp_path, "OFF", 100, 1, {})     # 清单缺文件


# ---------------- §5 任务二：读取边界适配层 ----------------

def _synthetic_panel():
    """合成 csi300 面板：3 只股 × 2026-04-01..2026-08-31 工作日（跨 cutoff）。"""
    days = pd.bdate_range("2026-04-01", "2026-08-31")
    codes = ["000001.SZ", "000002.SZ", "600000.SH"]
    idx = pd.MultiIndex.from_product(
        [days, codes], names=["datetime", "instrument"])
    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "$open": rng.normal(10, 0.1, len(idx)),
        "$high": rng.normal(11, 0.1, len(idx)),
        "$low": rng.normal(9, 0.1, len(idx)),
        "$close": rng.normal(10, 0.1, len(idx)),
        "$volume": rng.normal(1e6, 1e4, len(idx)),
        "$amount": rng.normal(1e7, 1e5, len(idx)),
        "$preclose": rng.normal(10, 0.1, len(idx)),
        "$tradestatuscode": -1.0,
    }, index=idx)
    return df, codes


class SpyProvider:
    """记录全部底层 fetch 入参的合成 provider（不触真实行情）。"""

    def __init__(self, data, members):
        self._data, self.instruments_ = data, members
        self._start_date = self._end_date = None
        self.fetch_calls: list[dict] = []

    def fetch(self, fields, *, filter_pipe=None, freq="day"):
        self.fetch_calls.append({"start": self._start_date,
                                 "end": self._end_date,
                                 "instruments": self.instruments_})
        df = self._data[self._data.index.get_level_values("instrument").isin(
            self.instruments_)].copy()
        if self._start_date is not None:
            df = df[df.index.get_level_values("datetime")
                    >= pd.Timestamp(self._start_date)]
        if self._end_date is not None:
            df = df[df.index.get_level_values("datetime")
                    <= pd.Timestamp(self._end_date)]
        df = df[list(fields)]
        df.columns = df.columns.str.replace("$", "", regex=False)
        return df

    def trading_days(self, start=None, end=None):
        cal = pd.DatetimeIndex(sorted(self._data.index.get_level_values(
            "datetime").unique()))
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal <= pd.Timestamp(end)]
        return cal

    def list_pool_at(self, pool, t):
        return list(self.instruments_)


def test_old_gap_reproduced_without_adapter():
    """复现缺口：无适配层时 fetch_end 越过决策日 t（2026-07-24 → 2026-08-07）。"""
    from kronos_qlib import build_inference_windows

    data, codes = _synthetic_panel()
    spy = SpyProvider(data, codes)
    build_inference_windows(spy, "2026-07-24", lookback=20, predict_len=10,
                            pool="csi300")
    assert spy.fetch_calls, "应发生 fetch"
    ends = [pd.Timestamp(c["end"]) for c in spy.fetch_calls]
    assert max(ends) == pd.Timestamp("2026-08-07")            # 越界请求被复现
    assert max(ends) > pd.Timestamp("2026-07-24")


def test_adapter_clamps_fetch_end_to_decision_day():
    """适配层：所有传给底层 fetch 的 end ≤ t；PeerSet 与窗口不变。"""
    from kronos_qlib import build_inference_windows
    from peer_residual.history_provider import PeerHistoryProvider

    data, codes = _synthetic_panel()
    spy = SpyProvider(data, codes)
    hist = PeerHistoryProvider(spy)
    hist.set_decision_bound("2026-07-24")
    df_list, x_ts_list, _, kept, _ = build_inference_windows(
        hist, "2026-07-24", lookback=20, predict_len=10, pool="csi300")
    assert spy.fetch_calls
    assert all(pd.Timestamp(c["end"]) <= pd.Timestamp("2026-07-24")
               for c in spy.fetch_calls)
    assert sorted(kept) == sorted(codes)                       # PeerSet 不缩
    assert all(pd.DatetimeIndex(s).max() <= pd.Timestamp("2026-07-24")
               for s in x_ts_list)                             # 窗口上界 = t


def test_adapter_rejects_beyond_cutoff_zero_calls():
    """t > 2026-07-24：下层 fetch 前拒绝且底层调用计数为 0。"""
    from peer_residual.history_provider import PeerHistoryProvider

    data, codes = _synthetic_panel()
    spy = SpyProvider(data, codes)
    hist = PeerHistoryProvider(spy)
    with pytest.raises(RuntimeError):
        hist.set_decision_bound("2026-07-27")                  # cutoff 后一交易日
    assert spy.fetch_calls == []                               # 底层 0 次
    hist.set_decision_bound("2026-07-24")                      # 截止日当天有效
    assert spy.fetch_calls == []


def test_adapter_future_calendar_still_usable():
    """未来日历仍可用：trading_days 委派不受钳制（不随日历拉未来价格）。"""
    from peer_residual.history_provider import PeerHistoryProvider

    data, codes = _synthetic_panel()
    spy = SpyProvider(data, codes)
    hist = PeerHistoryProvider(spy)
    hist.set_decision_bound("2026-07-24")
    cal = hist.trading_days()
    assert cal.max() == pd.Timestamp("2026-08-31")             # 日历含未来
    assert spy.fetch_calls == []


def test_future_perturbation_does_not_change_windows():
    """合成性质：t 后价格/停牌状态扰动不影响 PeerSet、历史输入（仅合成验证）。"""
    from kronos_qlib import build_inference_windows
    from peer_residual.history_provider import PeerHistoryProvider

    data, codes = _synthetic_panel()
    hist = PeerHistoryProvider(SpyProvider(data, codes))
    hist.set_decision_bound("2026-07-24")
    out1 = build_inference_windows(hist, "2026-07-24", lookback=20,
                                   predict_len=10, pool="csi300")
    data2 = data.copy()
    future = data2.index.get_level_values("datetime") > pd.Timestamp("2026-07-24")
    data2.loc[future, "$close"] = 999.0                        # 未来价格扰动
    data2.loc[future, "$tradestatuscode"] = 4.0                # 未来停牌态扰动
    hist2 = PeerHistoryProvider(SpyProvider(data2, codes))
    hist2.set_decision_bound("2026-07-24")
    out2 = build_inference_windows(hist2, "2026-07-24", lookback=20,
                                   predict_len=10, pool="csi300")
    assert sorted(out1[3]) == sorted(out2[3])                  # PeerSet 同
    for a, b in zip(out1[0], out2[0]):                         # 历史窗口逐值同
        pd.testing.assert_frame_equal(a, b)


def test_run_cache_uses_history_provider():
    """生产接线：peer cache 构建路径包 PeerHistoryProvider（源码级守卫）。"""
    src = (REPO_ROOT / "peer_residual" / "cache.py").read_text(encoding="utf-8")
    assert "PeerHistoryProvider" in src
    assert "set_decision_bound" in src


# ---------------- §6.2 信号先冻结、再回测 ----------------

def _fake_events():
    return []


def test_freeze_then_backtest_order_and_no_infer_in_backtest(tmp_path):
    """编排顺序：全部 infer/write/freeze/verify 严格先于第一笔 backtest；
    回测只读文件，不再调用 infer。"""
    from peer_residual import run as RUN

    events: list[str] = []

    def fake_infer(arm, seed, w, stats, heads_dir, weight_shas):
        events.append(f"infer:{arm}:{seed}:{w}")
        wide = pd.DataFrame({c: [0.1, 0.2] for c in ("a", "b")},
                            index=pd.bdate_range("2026-01-05", periods=2))
        return wide, {"best_epoch": 1}

    def fake_write(art_dir, arm, seed, w, wide, best):
        events.append(f"write:{arm}:{seed}:{w}")
        p = art_dir / f"signal_{w}_{arm}_s{seed}.parquet"
        wide.to_parquet(p)
        from sae_residual.cache_io import sha256_file
        return {"arm": arm, "seed": seed, "window": w, "file": p.name,
                "sha256": sha256_file(p), "n_cells": 4, "n_days": 2}

    manifest = RUN.freeze_signals_stage(
        tmp_path, {"d_in": D, "sigma_e": 0.1}, [100], {"t": "s"},
        tmp_path, infer_signal_fn=fake_infer, write_signal_fn=fake_write)
    assert (tmp_path / "frozen_signals_manifest.json").is_file()

    def fake_runner(art_dir, seeds, read_wide_fn, backtest_fn):
        events.append("backtest")
        # 回测阶段不得调用 infer（fake_infer 只在 freeze 阶段出现）
        assert not [e for e in events[:-1] if e.startswith("infer") and
                    events.index("backtest") < events.index(e)]
        wide = read_wide_fn("PEER_OFF_s100", "W3")             # 只读冻结文件
        assert wide is not None and wide.shape == (2, 2)
        return {"per_seed": {}, "created_at": "t"}

    RUN.backtest_frozen_stage(tmp_path, [100], runner_fn=fake_runner)
    first_bt = events.index("backtest")
    assert all(events.index(e) < first_bt for e in events
               if e.startswith(("infer:", "write:")))
    assert "signals" in manifest and len(manifest["signals"]) == 4
    assert events.count("backtest") == 1


def test_missing_frozen_signal_zero_backtests(tmp_path):
    """缺一份冻结信号 → 回测调用次数为 0（校验先于任何回测）。"""
    from peer_residual import run as RUN

    def fake_infer(arm, seed, w, stats, heads_dir, weight_shas):
        wide = pd.DataFrame({c: [0.1] for c in ("a",)},
                            index=pd.bdate_range("2026-01-05", periods=1))
        return wide, {"best_epoch": 1}

    def fake_write(art_dir, arm, seed, w, wide, best):
        p = art_dir / f"signal_{w}_{arm}_s{seed}.parquet"
        wide.to_parquet(p)
        from sae_residual.cache_io import sha256_file
        return {"arm": arm, "seed": seed, "window": w, "file": p.name,
                "sha256": sha256_file(p), "n_cells": 1, "n_days": 1}

    RUN.freeze_signals_stage(tmp_path, {"d_in": D, "sigma_e": 0.1}, [100],
                             {"t": "s"}, tmp_path, infer_signal_fn=fake_infer,
                             write_signal_fn=fake_write)
    # 删掉一份 → 回测拒绝且 runner 0 次
    victim = tmp_path / "signal_W4_ON_s100.parquet"
    victim.unlink()
    calls = []

    def fake_runner(*a, **k):
        calls.append(1)
        return {}

    with pytest.raises(RuntimeError):
        RUN.backtest_frozen_stage(tmp_path, [100], runner_fn=fake_runner)
    assert calls == []


def test_tampered_frozen_signal_rejected_no_regen(tmp_path):
    """冻结文件被篡改 → 拒绝，不自动重生（infer 未被再次调用）。"""
    from peer_residual import run as RUN

    infer_calls = []

    def fake_infer(arm, seed, w, stats, heads_dir, weight_shas):
        infer_calls.append((arm, seed, w))
        wide = pd.DataFrame({c: [0.1] for c in ("a",)},
                            index=pd.bdate_range("2026-01-05", periods=1))
        return wide, {"best_epoch": 1}

    def fake_write(art_dir, arm, seed, w, wide, best):
        p = art_dir / f"signal_{w}_{arm}_s{seed}.parquet"
        wide.to_parquet(p)
        from sae_residual.cache_io import sha256_file
        return {"arm": arm, "seed": seed, "window": w, "file": p.name,
                "sha256": sha256_file(p), "n_cells": 1, "n_days": 1}

    RUN.freeze_signals_stage(tmp_path, {"d_in": D, "sigma_e": 0.1}, [100],
                             {"t": "s"}, tmp_path, infer_signal_fn=fake_infer,
                             write_signal_fn=fake_write)
    n_infer_before = len(infer_calls)
    victim = tmp_path / "signal_W3_OFF_s100.parquet"
    pd.DataFrame({c: [9.9] for c in ("a",)},
                 index=pd.bdate_range("2026-01-05", periods=1)).to_parquet(victim)
    with pytest.raises(RuntimeError):
        RUN.backtest_frozen_stage(tmp_path, [100], runner_fn=lambda *a, **k: {})
    assert len(infer_calls) == n_infer_before                 # 未自动重生


# ---------------- §4 CLI 契约 ----------------

def test_diagnose_cli_refuses_overwrite(tmp_path):
    """输出目录已有完成 manifest 时拒绝覆盖。"""
    from peer_residual import diagnose as DG

    out = tmp_path / "out"
    out.mkdir()
    (out / "diagnostic_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        DG.main(["--sae-cache", str(tmp_path), "--peer-cache", str(tmp_path),
                 "--output", str(out)])
