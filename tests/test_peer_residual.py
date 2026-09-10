"""peer_residual 机制契约测试（计划 §6；不触真实行情/DDB/GPU）。

覆盖：跨股作用（ON 改变邻居输出、OFF 不变）、排列等变、跨日隔离、
padding 隔离与全 padding 拒绝、标签隔离（标签只进损失不进 forward）、
零残差锚与 σe 还原、掩码语义（OFF 对角/ON 有效全可见）、底座冻结与
OFF/ON 配对（同 seed 初始哈希/同日键/同 shuffle）、生产保存/加载与
错误身份拒绝、完整交易网格（尾部无标签日仍有预测、覆盖全部输出格）。
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
         "sigma_e": 0.0490884, "d_in": D}


def synth_day(date: str, codes: list[str], seed: int, loss_codes=None,
              r_val: float = 0.3) -> PD.DayData:
    """合成 DayData：PeerSet=codes，LossSet=loss_codes（默认前 3 个）。"""
    rng = np.random.default_rng(seed)
    n = len(codes)
    x = rng.normal(size=(n, D)).astype(np.float32)
    loss_set = set(loss_codes if loss_codes is not None else codes[:3])
    loss_mask = np.array([c in loss_set for c in codes], dtype=bool)
    r = np.zeros(n, np.float32)
    r[loss_mask] = r_val
    s = rng.normal(size=n)
    return PD.DayData(date=date, codes=list(codes), x=x, loss_mask=loss_mask,
                      r=r, s_g1=s, n_extra=n - len(loss_set))


def synth_head(seed: int = 100, nonzero_out: bool = False) -> M.PeerResidualHead:
    model = M.build_head(seed, D)
    if nonzero_out:  # 机制测试须用非零且非均匀的读出权重（LN 后特征零均值，
        # 常数权重会退化成与输入无关的常数输出）
        with torch.no_grad():
            model.out.weight.normal_(0.0, 0.3)
            model.out.bias.fill_(0.1)
    return model


# ---------------- §5 掩码语义 ----------------

def test_mask_semantics_off_diagonal_on_all_valid():
    """OFF：有效 query 只看自己；ON：有效 key 全可见；padding query 连首个有效 key。"""
    valid = torch.tensor([[True, True, False]])
    off = M.build_blocked_mask(valid, "OFF")[0]        # [N,N] True=禁止
    on = M.build_blocked_mask(valid, "ON")[0]
    assert not off[0, 0] and off[0, 1] and off[0, 2]   # 0 只看自己
    assert off[1, 0] and not off[1, 1] and off[1, 2]   # 1 只看自己
    assert not on[0, 0] and not on[0, 1] and on[0, 2]  # 0 看 0/1，pad 键被遮
    assert not on[1, 0] and not on[1, 1] and on[1, 2]
    # padding query（两种模式）只连该日首个有效 key=0，其余全遮
    assert not off[2, 0] and off[2, 1] and off[2, 2]
    assert not on[2, 0] and on[2, 1] and on[2, 2]
    with pytest.raises(ValueError):
        M.build_blocked_mask(valid, "WRONG")


def test_param_table_matches_module():
    """参数表精确数目 = 模块实际参数（计划 §5：构造前列参数表）。"""
    table = M.param_table(D)
    model = M.PeerResidualHead(D)
    total = sum(p.numel() for p in model.parameters())
    assert table["total"] == total
    t832 = M.param_table(832)["total"]                   # 真实 D=832 规模
    assert t832 == sum(p.numel() for p in M.PeerResidualHead(832).parameters())
    assert 80_000 < t832 < 95_000                        # 预期约 8.7 万


# ---------------- §6 机制测试 ----------------

def test_cross_stock_effect_on_changes_neighbor_off_not():
    """跨股作用：固定第 1 只输入、改变第 2 只——ON 第 1 只输出改变，OFF 不变。"""
    torch.manual_seed(0)
    x = torch.randn(1, 3, D)
    x2 = x.clone()
    x2[0, 1] += torch.randn(D) * 3.0
    valid = torch.ones(1, 3, dtype=torch.bool)
    for arm, expect_change in (("ON", True), ("OFF", False)):
        model = synth_head(nonzero_out=True)
        with torch.no_grad():
            r_a = model(x, valid, arm)[0, 0]
            r_b = model(x2, valid, arm)[0, 0]
        changed = not torch.allclose(r_a, r_b, atol=1e-6)
        assert changed == expect_change, f"{arm} 跨股作用与预期不符"


def test_permutation_equivariance_both_arms():
    """排列等变：任意排列股票，输出逆排列后相同（1e-5，OFF/ON 都要）。"""
    torch.manual_seed(1)
    x = torch.randn(1, 6, D)
    valid = torch.tensor([[True, True, True, False, True, True]])
    perm = [4, 0, 3, 5, 2, 1]
    inv = torch.argsort(torch.tensor(perm))
    for arm in C.ARMS:
        model = synth_head(nonzero_out=True)
        with torch.no_grad():
            r = model(x, valid, arm)
            r_p = model(x[:, perm], valid[:, perm], arm)[:, inv]
        assert torch.allclose(r, r_p, atol=1e-5), f"{arm} 排列不等变"


def test_cross_day_isolation():
    """跨日隔离：改变 batch 中另一日期任意 peer，当前日输出不变。"""
    torch.manual_seed(2)
    x = torch.randn(2, 5, D)
    valid = torch.ones(2, 5, dtype=torch.bool)
    x2 = x.clone()
    x2[1] += torch.randn(5, D) * 5.0
    for arm in C.ARMS:
        model = synth_head(nonzero_out=True)
        with torch.no_grad():
            r = model(x, valid, arm)[0]
            r2 = model(x2, valid, arm)[0]
        assert torch.allclose(r, r2, atol=1e-6), f"{arm} 跨日泄漏"


def test_padding_isolation_and_full_pad_rejected():
    """padding 隔离：追加/改动 padding 股，真实输出不变且无 NaN；全 padding 日期拒绝。"""
    torch.manual_seed(3)
    days = [synth_day("2025-01-02", [f"c{i}" for i in range(3)], 0),
            synth_day("2025-01-03", [f"d{i}" for i in range(5)], 1)]
    x, valid, loss, r = PD.collate_batch(days)
    assert x.shape == (2, 5, D) and not valid[0, 3:].any()   # day0 补 2 个 padding
    model = synth_head(nonzero_out=True)
    with torch.no_grad():
        out1 = model(x, valid, "ON")
    assert torch.isfinite(out1).all()
    x2 = x.clone()
    x2[0, 3:] = torch.randn(2, D) * 100.0                    # 改 padding 内容
    with torch.no_grad():
        out2 = model(x2, valid, "ON")
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6)
    assert (out1[0, 3:] == 0).all()                          # padding 输出置零
    empty = PD.DayData("2025-01-04", [], np.zeros((0, D), np.float32),
                       np.zeros(0, bool), np.zeros(0, np.float32),
                       np.zeros(0))
    with pytest.raises(ValueError):
        PD.collate_batch([empty])


def test_label_isolation_labels_never_enter_forward():
    """标签隔离：标签有/无/变值只改 r 与损失，不改 PeerSet 与 forward。"""
    sae = {"instruments": np.array(["a", "b", "c"]),
           "h": np.random.default_rng(0).normal(size=(3, D)).astype(np.float32),
           "s_g1": np.array([0.01, -0.02, 0.03]),
           "y_mean": np.array([0.02, -0.01, 0.04]),
           "x_end": np.array(["2025-01-02"] * 3),
           "label_start": np.array(["2025-01-03"] * 3)}
    peer = {"instruments": np.array(["d"]),
            "h": np.random.default_rng(1).normal(size=(1, D)).astype(np.float32),
            "peer_codes": np.array(["a", "b", "c", "d"])}
    d1 = PD.assemble_day(sae, peer, STATS, "2025-01-02", "train")
    sae2 = {**sae, "y_mean": np.array([0.9, -0.9, 0.5])}
    d2 = PD.assemble_day(sae2, peer, STATS, "2025-01-02", "train")
    assert d1.codes == d2.codes and d1.n_extra == d2.n_extra
    assert np.array_equal(d1.x, d2.x)                       # forward 输入不变
    assert np.array_equal(d1.loss_mask, d2.loss_mask)
    assert not np.array_equal(d1.r, d2.r)                   # 只损失目标变
    sae3 = {k: v for k, v in sae.items() if k != "y_mean"}  # 无标签 → 无监督
    d3 = PD.assemble_day(sae3, peer, STATS, "2025-01-02", "W3")
    assert (d3.r == 0).all()
    # 未来标签窗必须晚于历史窗（无前视断言在装配路径内）
    with pytest.raises(AssertionError):
        bad = {**sae, "label_start": np.array(["2025-01-01"] * 3)}
        PD.assemble_day(bad, peer, STATS, "2025-01-02", "train")


def test_zero_residual_anchor_and_sigma_e_restore():
    """零残差锚：未训练头 s_final 逐值等于 G1；σe 还原公式手算对拍。"""
    model = synth_head()                                    # 初始出口全 0
    x, valid, _, _ = PD.collate_batch([synth_day("t", [f"c{i}" for i in range(4)], 0)])
    with torch.no_grad():
        r_hat = model(x, valid, "ON")
    assert bool((r_hat == 0).all())
    s_g1 = torch.tensor([0.01, -0.02, 0.03, 0.005])
    s_final = M.final_signal(s_g1, r_hat[0], STATS["sigma_e"])
    assert torch.equal(s_final, s_g1)                       # 逐值相等
    r_half = torch.full((4,), 0.5)
    hand = s_g1 + 0.5 * STATS["sigma_e"]
    assert torch.allclose(M.final_signal(s_g1, r_half, STATS["sigma_e"]), hand)


def test_frozen_stats_only_from_file_no_refit():
    """μh/σh/σe 只引用冻结训练文件：两次读取逐位一致、形状/量纲锁定。"""
    stats = PD.load_frozen_stats()
    stats2 = PD.load_frozen_stats()
    assert stats["sigma_e"] == stats2["sigma_e"]
    assert np.array_equal(stats["mu"], stats2["mu"])
    assert stats["mu"].shape == (832,) and stats["sigma"].shape == (832,)
    assert stats["d_in"] == 832
    assert abs(stats["sigma_e"] - 0.0490884) < 1e-6         # f6c0726 冻结值
    x = PD.normalize_hidden(np.ones((2, 832), np.float32), stats)
    assert x.shape == (2, 832) and np.isfinite(x).all()


def test_backbone_frozen_one_update_only_head_changes():
    """底座冻结：一次更新后仅头参数变化；输入 h 无梯度；冻结底座权重逐位不变。"""
    bb = torch.nn.Linear(D, D, bias=False).requires_grad_(False)
    bb_w0 = bb.weight.detach().clone()
    raw = torch.randn(1, 3, D)
    with torch.no_grad():
        x = bb(raw)                                         # 生产路径：h 预计算
    model = synth_head()
    w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    days = [synth_day("t", [f"c{i}" for i in range(3)], 0)]
    _, valid, loss, r = PD.collate_batch(days)
    T.train_step(model, opt, x.detach(), valid, loss, r, "ON")
    assert x.grad is None                                   # h 无梯度
    assert torch.equal(bb.weight, bb_w0)                    # 底座不变
    changed = [k for k in w0 if not torch.equal(w0[k], model.state_dict()[k])]
    assert changed, "一次更新后头参数应变化"
    assert all(k.startswith(("proj.", "ln", "attn.", "ffn", "out.")) for k in changed)


def test_off_on_pairing_init_hash_and_shuffle():
    """同 seed OFF/ON：初始哈希相同、日键摘要相同、首 epoch shuffle 相同。"""
    h_off = T.state_dict_hash(M.build_head(100, D))
    h_on = T.state_dict_hash(M.build_head(100, D))
    assert h_off == h_on
    days = [synth_day(f"2025-01-{i:02d}", [f"c{j}" for j in range(5)], i)
            for i in range(2, 12)]
    assert PD.day_keys_digest(days) == PD.day_keys_digest(days)
    g1 = torch.Generator(); g1.manual_seed(100)
    g2 = torch.Generator(); g2.manual_seed(100)
    for _ in range(3):
        assert torch.equal(torch.randperm(len(days), generator=g1),
                           torch.randperm(len(days), generator=g2))


def test_production_two_epochs_best_load_and_rejects(tmp_path):
    """生产保存/加载：真实两 epoch、预定 best=e1（训练目标与验证目标反向）、
    e1/e2 权重不同、best 实际载入 e1；缺 best/错 epoch/错身份拒绝。"""
    tr = [synth_day(f"2025-01-{i:02d}", [f"c{j}" for j in range(6)], i,
                    r_val=1.0) for i in range(2, 6)]
    va = [synth_day(f"2025-02-{i:02d}", [f"v{j}" for j in range(6)], 100 + i,
                    r_val=-1.0) for i in range(2, 5)]
    res_off = T.train_arm(tr, va, "OFF", 100, tmp_path, device="cpu", epochs=2)
    res_on = T.train_arm(tr, va, "ON", 100, tmp_path, device="cpu", epochs=2)
    assert res_off["best_epoch"] == res_on["best_epoch"] == 1  # 反向目标 → e1 最优
    sd = [torch.load(T.head_file(tmp_path, "ON", 100, e), weights_only=True)["state_dict"]
          for e in (1, 2)]
    assert not torch.equal(sd[0]["proj.weight"], sd[1]["proj.weight"])
    m_best, meta = T.load_best(tmp_path, "ON", 100, D)
    assert meta["best_epoch"] == 1
    assert torch.equal(m_best.state_dict()["proj.weight"], sd[0]["proj.weight"])
    assert res_off["day_keys_digest"] == res_on["day_keys_digest"]
    assert res_off["first_epoch_perm"] == res_on["first_epoch_perm"]
    # 拒绝：缺 epoch / 错 seed（伪造改名）/ 错 arm / 错协议
    with pytest.raises(FileNotFoundError):
        T.load_head(tmp_path, "ON", 100, 99, D)
    fake = T.head_file(tmp_path, "ON", 202, 1)
    fake.write_bytes(T.head_file(tmp_path, "ON", 100, 1).read_bytes())
    with pytest.raises(RuntimeError):
        T.load_head(tmp_path, "ON", 202, 1, D)
    fake2 = T.head_file(tmp_path, "OFF", 100, 1)
    fake2.write_bytes(T.head_file(tmp_path, "ON", 100, 1).read_bytes())
    with pytest.raises(RuntimeError):
        T.load_head(tmp_path, "OFF", 100, 1, D)
    ck = torch.load(T.head_file(tmp_path, "ON", 100, 1), weights_only=True)
    ck["meta"]["protocol"] = "wrong"
    torch.save(ck, fake)
    with pytest.raises(RuntimeError):
        T.load_head(tmp_path, "ON", 202, 1, D)


def test_full_trading_grid_tail_dates_have_predictions(tmp_path, monkeypatch):
    """完整交易网格：尾部日期只缺未来标签仍有预测；输出覆盖全部输出格。"""
    from peer_residual.evaluate import infer_arm_signals

    codes = [f"s{i:03d}" for i in range(6)]
    rng = np.random.default_rng(7)
    sae = {"instruments": np.array(codes),
           "h": rng.normal(size=(6, D)).astype(np.float32),
           "s_g1": rng.normal(size=6) * 0.02,
           "x_end": np.array(["2026-07-24"] * 6)}
    peer = {"instruments": np.array(["s999"]),
            "h": rng.normal(size=(1, D)).astype(np.float32),
            "peer_codes": np.array(sorted(codes + ["s999"]))}
    # 通过生产装配路径构造两日（后一日 = 只缺未来标签的尾部日）
    d1 = PD.assemble_day({**sae, "x_end": np.array(["2026-07-23"] * 6)}, peer,
                         STATS, "2026-07-23", "W4")
    d2 = PD.assemble_day(sae, peer, STATS, "2026-07-24", "W4")
    model = synth_head(nonzero_out=True)
    wide = infer_arm_signals([d1, d2], model, "ON", STATS["sigma_e"])
    assert list(wide.index) == [pd.Timestamp("2026-07-23"),
                                pd.Timestamp("2026-07-24")]
    assert set(wide.columns) == set(codes)                  # 额外 peer 不是候选
    assert wide.notna().all().all()                         # 全部输出格有值
    assert "s999" not in wide.columns


def test_chunk_identity_roundtrip_and_reject(tmp_path, monkeypatch):
    """chunk 身份：协议/权重 SHA 不匹配拒绝复用；源 SAE chunk SHA 绑定。"""
    from sae_residual.cache_io import CacheMismatchError, load_chunk

    ident = {"protocol": C.PROTOCOL_VERSION, "split": "train",
             "tokenizer_sha": "t0", "predictor_sha": "p0",
             "sae_chunk_sha256": "x" * 64}
    p = tmp_path / "peer_train_20250102.npz"
    write_chunk(p, {"h": np.zeros((1, D), np.float32),
                    "peer_codes": np.array(["a"])}, ident)
    got = load_chunk(p, {"protocol": C.PROTOCOL_VERSION, "split": "train"})
    assert got["peer_codes"].tolist() == ["a"]
    with pytest.raises(CacheMismatchError):
        load_chunk(p, {"protocol": "other-v9", "split": "train"})
    with pytest.raises(CacheMismatchError):
        load_chunk(p, {"predictor_sha": "p1"})
    # 生产装配路径的 peer chunk 身份拒绝（伪造权重 SHA）
    monkeypatch.setattr(C, "SAE_CACHE_DIR", tmp_path / "sae")
    monkeypatch.setattr(C, "PEER_CACHE_DIR", tmp_path / "peer")
    (tmp_path / "sae" / "train").mkdir(parents=True)
    (tmp_path / "peer" / "train").mkdir(parents=True)
    sae_payload = {"dates": np.array(["2025-01-02"]),
                   "instruments": np.array(["a"]),
                   "h": np.zeros((1, D), np.float32),
                   "s_g1": np.array([0.01]), "x_end": np.array(["2025-01-02"])}
    sae_p = tmp_path / "sae" / "train" / "train_20250102.npz"
    write_chunk(sae_p, sae_payload,
                {"protocol": C.SAE_PROTOCOL, "split": "train",
                 "tokenizer_sha": "t0", "predictor_sha": "p0"})
    pp = tmp_path / "peer" / "train" / "peer_train_20250102.npz"
    write_chunk(pp, {"instruments": np.array(["b"]),
                     "h": np.zeros((1, D), np.float32),
                     "peer_codes": np.array(["a", "b"])},
                {**ident, "sae_chunk_sha256": "deadbeef",
                 "date": "2025-01-02", "n_peers": 2, "n_extra": 1})
    with pytest.raises(RuntimeError):
        PD.load_day("train", "2025-01-02", STATS,
                    weight_shas={"tokenizer_sha": "t0", "predictor_sha": "p0"})


def test_lossset_must_be_subset_of_peerset():
    """LossSet/基线格 ⊄ PeerSet → 停止排查（不静默替换、不缩池）。"""
    sae = {"instruments": np.array(["a", "z"]),
           "h": np.zeros((2, D), np.float32), "s_g1": np.zeros(2)}
    peer = {"instruments": np.array(["b"]),
            "h": np.zeros((1, D), np.float32),
            "peer_codes": np.array(["a", "b"])}             # 缺 z
    with pytest.raises(AssertionError):
        PD.assemble_day(sae, peer, STATS, "2025-01-02", "train")


def test_val_loss_equal_weight_per_day():
    """验证 MSE：先日内 mean 再跨日等权 mean（非样本等权）。"""
    model = synth_head()                                    # r_hat ≡ 0
    d1 = synth_day("a", [f"c{i}" for i in range(4)], 0, r_val=0.0)
    d2 = synth_day("b", [f"d{i}" for i in range(4)], 1,
                   loss_codes=["d0"], r_val=2.0)            # 日 b：1 股 r=2
    loss = T.val_loss_days(model, [d1, d2], "ON", "cpu")
    assert loss == pytest.approx((0.0 + 4.0) / 2, rel=1e-6)


def test_run_py_has_required_stages_and_guards():
    src = (REPO_ROOT / "peer_residual" / "run.py").read_text(encoding="utf-8")
    for stage in ("preflight", "smoke", "cache", "pilot", "confirm", "report"):
        assert f'"{stage}"' in src, f"缺 stage：{stage}"
    # confirm 必须是条件入口（读取 pilot 门禁后才可执行）
    assert "pilot_verdict.json" in src
    assert "forward" not in src.lower() or "forward_cutoff" in src.lower()
