"""任务3（方案 §7）：教师 keyed RNG、3 replicas、断点恢复测试（假教师，无 HF/GPU）。

假教师依赖 torch 随机数——所有一致性断言真正检验 keyed RNG 协议，而非
比较两次调用同一确定性函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from dhead_distill.teacher import (
    TeacherRunner,
    teacher_seed,
)


class FakePredictor:
    """依赖 torch 随机数的假教师：predict 签名对齐 KronosPredictor.predict。

    输出 close 路径 = close_t × exp(累积噪声)，噪声由当前全局 RNG 状态决定
    （fork_rng 的生效点）。
    """

    def __init__(self, close_t_by_key: dict[tuple[str, str], float]):
        self.calls = 0
        self._close_t = close_t_by_key

    def predict(self, df, x_timestamp, y_timestamp, pred_len,
                T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False):
        self.calls += 1
        closes = df["close"].values.astype(np.float64)
        rets = torch.randn(pred_len).numpy() * 0.01  # 全局 RNG（fork_rng 生效点）
        path = closes[-1] * np.exp(np.cumsum(rets))
        out = np.stack([path for _ in range(6)], axis=1)  # 6 列 OHLCVA
        return pd.DataFrame(
            out, columns=["open", "high", "low", "close", "volume", "amount"],
            index=pd.DatetimeIndex(y_timestamp),
        )


def _tiny_manifest():
    """2 日 × 3 股的迷你清单（绕开 DDB，直接构造 DayManifest）。"""
    from dhead_distill.data import DayManifest, Sample, _calc_stamps

    m = DayManifest(split="train", pool="ashares", profile="pilot",
                    protocol="proto-hash-x", list_seed=20260905)
    dates = [pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-04")]
    codes = ["SZ0001", "SZ0002", "SZ0003"]
    for d in dates:
        d_iso = d.strftime("%Y-%m-%d")
        x_cal = pd.bdate_range(d - pd.Timedelta(days=130), d)[-90:]
        y_cal = pd.bdate_range(d + pd.Timedelta(days=1), periods=14)[:10]
        m.x_stamp[d_iso] = _calc_stamps(x_cal)
        m.y_stamp[d_iso] = _calc_stamps(y_cal)
        m.x_cal[d_iso] = x_cal.values.astype("datetime64[D]")
        m.y_cal[d_iso] = y_cal.values.astype("datetime64[D]")
        for c in codes:
            key = (d_iso, c)
            m.x_raw[key] = np.full((90, 6), 10.0, dtype=np.float32)
            m.x_raw[key][:, 3] = 100.0
            m.close_t[key] = 100.0
            m.samples.append(Sample(date=d, code=c,
                                    y_real=np.full(10, 0.01, dtype=np.float32),
                                    label_ok=True))
    from dhead_distill.data import content_hash
    m.content_hash = content_hash(m)
    return m


def _runner(tmp_path, monkeypatch, m, weight_hash="w" * 64, n_paths=4, replicas=3):
    monkeypatch.setenv("DHEAD_BASE_REPO", str(tmp_path / "base"))
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art"))
    (tmp_path / "base").mkdir(exist_ok=True)
    close_t = {
        (s.date.strftime("%Y-%m-%d"), s.code): 100.0 for s in m.samples
    }
    return TeacherRunner(
        manifest=m,
        predict_fn=FakePredictor(close_t).predict,
        weight_hash=weight_hash,
        n_paths=n_paths,
        replicas=replicas,
        teacher_T=1.0, teacher_top_p=0.9, teacher_top_k=0,
        predict_len=10,
        model_eval_verified=True,  # 假教师无 dropout，断言语义由测试方担保
    )


def test_teacher_seed_protocol() -> None:
    """keyed seed：同参一致；replica/样本/协议不同 → seed 不同。"""
    a = teacher_seed("p" * 8, "2024-06-03", "SZ0001", 0)
    b = teacher_seed("p" * 8, "2024-06-03", "SZ0001", 0)
    assert a == b
    assert a != teacher_seed("p" * 8, "2024-06-03", "SZ0001", 1)
    assert a != teacher_seed("p" * 8, "2024-06-03", "SZ0002", 0)
    assert a != teacher_seed("q" * 8, "2024-06-03", "SZ0001", 0)


def test_replica_order_and_batch_independence(tmp_path, monkeypatch) -> None:
    """顺序反转、逐日重跑后逐样本逐 replica target 一致（keyed RNG 协议）。"""
    m = _tiny_manifest()
    r1 = _runner(tmp_path, monkeypatch, m)
    full = r1.run()

    # 顺序反转：逆日期重跑（同 run 目录语义由 hash 隔离，这里换新产物根）
    from dhead_distill.data import DayManifest

    m_rev = DayManifest(split=m.split, pool=m.pool, profile=m.profile,
                        protocol=m.protocol, list_seed=m.list_seed)
    for d in sorted({s.date for s in m.samples}, reverse=True):
        d_iso = d.strftime("%Y-%m-%d")
        m_rev.x_stamp[d_iso] = m.x_stamp[d_iso]
        m_rev.y_stamp[d_iso] = m.y_stamp[d_iso]
        m_rev.x_cal[d_iso] = m.x_cal[d_iso]
        m_rev.y_cal[d_iso] = m.y_cal[d_iso]
        for s in [s for s in m.samples if s.date == d]:
            k = (d_iso, s.code)
            m_rev.x_raw[k] = m.x_raw[k]
            m_rev.close_t[k] = m.close_t[k]
            m_rev.samples.append(s)
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art-rev"))
    base = _runner(tmp_path, monkeypatch, m)
    r2 = TeacherRunner(
        manifest=m_rev, predict_fn=base._predict_fn, weight_hash=base.weight_hash,
        n_paths=base.n_paths, replicas=base.replicas,
        teacher_T=base.teacher_T, teacher_top_p=base.teacher_top_p,
        teacher_top_k=base.teacher_top_k, predict_len=base.predict_len,
        model_eval_verified=True,
    )
    rev = r2.run()

    for s in m.samples:
        k = (s.date.strftime("%Y-%m-%d"), s.code)
        np.testing.assert_array_equal(full[k], rev[k])  # [R,H] 逐位一致


def test_resume_after_partial_completion(tmp_path, monkeypatch) -> None:
    """断点恢复：首跑中途抛错 → 恢复补齐 → 与独立完整跑逐位一致。"""
    m = _tiny_manifest()

    class Boom(Exception):
        pass

    base = _runner(tmp_path, monkeypatch, m)

    # 首跑即崩溃：第 1 天 9 次调用完成后，第 2 天第 1 次调用抛错（分片未落盘）
    calls = {"n": 0}

    def flaky(df, x_timestamp, y_timestamp, pred_len, **kw):
        calls["n"] += 1
        if calls["n"] > 9:
            raise Boom("mid-run crash")
        return base._predict_fn(df, x_timestamp, y_timestamp, pred_len, **kw)

    r_crash = TeacherRunner(
        manifest=m, predict_fn=flaky, weight_hash=base.weight_hash,
        n_paths=base.n_paths, replicas=base.replicas,
        teacher_T=base.teacher_T, teacher_top_p=base.teacher_top_p,
        teacher_top_k=base.teacher_top_k, predict_len=base.predict_len,
        model_eval_verified=True,
    )
    with pytest.raises(Boom):
        r_crash.run()
    assert calls["n"] == 10  # 第 1 天完整落盘，第 2 天第 1 次即崩

    # 恢复：同 run 目录继续，第 1 天走分片缓存、第 2 天重新生成
    r_resume = TeacherRunner(
        manifest=m, predict_fn=base._predict_fn, weight_hash=base.weight_hash,
        n_paths=base.n_paths, replicas=base.replicas,
        teacher_T=base.teacher_T, teacher_top_p=base.teacher_top_p,
        teacher_top_k=base.teacher_top_k, predict_len=base.predict_len,
        model_eval_verified=True,
    )
    resumed = r_resume.run()

    # 独立完整跑（新产物根）作对照：keyed RNG 保证逐位一致
    monkeypatch.setenv("DHEAD_ARTIFACT_ROOT", str(tmp_path / "art-full"))
    r_full = _runner(tmp_path, monkeypatch, m)
    full = r_full.run()

    for s in m.samples:
        k = (s.date.strftime("%Y-%m-%d"), s.code)
        np.testing.assert_array_equal(full[k], resumed[k])


def test_weight_hash_change_rejects_old_cache(tmp_path, monkeypatch) -> None:
    """教师权重 hash 改变 → 拒绝复用旧缓存（防混配）。"""
    m = _tiny_manifest()
    r1 = _runner(tmp_path, monkeypatch, m, weight_hash="a" * 64)
    r1.run()
    # 身份核验在构造期完成：不同权重 hash 的 runner 不得复用同一 run 目录
    with pytest.raises(RuntimeError, match="身份不一致"):
        _runner(tmp_path, monkeypatch, m, weight_hash="b" * 64)


def test_three_replicas_differ(tmp_path, monkeypatch) -> None:
    """三个 replica 种子不同 → 同一样本 3 组 target 不同（教师波动真实）。"""
    m = _tiny_manifest()
    r = _runner(tmp_path, monkeypatch, m)
    out = r.run()
    k = (m.samples[0].date.strftime("%Y-%m-%d"), m.samples[0].code)
    y = out[k]  # [R,H]
    assert y.shape == (3, 10)
    assert not np.allclose(y[0], y[1])
    assert not np.allclose(y[1], y[2])


def test_shard_atomic_and_done_key(tmp_path, monkeypatch) -> None:
    """分片落盘原子 + 完成键核验：run 目录内每个日期分片带 done 标记。"""
    m = _tiny_manifest()
    r = _runner(tmp_path, monkeypatch, m)
    r.run()
    for shard in sorted(r.run_dir.glob("day-*.npz")):
        z = np.load(shard)
        assert int(z["done"]) == 1, f"{shard.name} 缺完成键"
    assert not list(r.run_dir.glob("*.tmp"))  # 无 .tmp 残留（原子替换）
