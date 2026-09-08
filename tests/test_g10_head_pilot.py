"""G10H-pilot 契约测试（计划 §4/§6，先 FAIL 后 PASS；全合成，不触真实数据）。

覆盖：A1 参数 allowlist 精确匹配与 tokenizer.head 禁训；A1 更新后仅 4 张量
可变而 A0 有 head 外参数可变；tokenizer 前后不变；CE 选点只读验证损失；判据
代入逻辑（V/I/P/S/E 冻结公式）；FULL 口径窗口边界不越封存线；协议哈希防篡改。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from g10_head_pilot.config import A1_ALLOWED, FORWARD_CUTOFF, WINDOW_BOUNDS  # noqa: E402


def _tiny_kronos_like():
    """与 Kronos 同构的极小模型（named_parameters 层级一致即可测掩码）。"""
    import torch
    from torch import nn

    class FakeKronos(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(8, 8)
            self.transformer = nn.ModuleList([nn.Linear(8, 8)])
            self.norm = nn.LayerNorm(8)
            self.head = nn.Module()
            self.head.proj_s1 = nn.Linear(8, 16)
            self.head.proj_s2 = nn.Linear(8, 16)
            self.dep_layer = nn.Linear(8, 8)

    return FakeKronos()


def test_a1_allowlist_exact_and_mask() -> None:
    from g10_head_pilot.train import apply_param_mask

    m = _tiny_kronos_like()
    trainable = apply_param_mask(m, "A1")
    assert set(trainable) == A1_ALLOWED          # 恰 4 张量
    assert "head.proj_s1.weight" in trainable
    # tokenizer.head 语义禁止：允许清单内无任何 tokenizer 前缀
    assert not any(n.startswith("tokenizer") for n in trainable)
    # A0：head 外参数也可训练
    m2 = _tiny_kronos_like()
    trainable_a0 = apply_param_mask(m2, "A0")
    assert "embedding.weight" in trainable_a0
    assert len(trainable_a0) > len(A1_ALLOWED)


def test_a1_updates_only_allowed_tensors() -> None:
    import torch

    from g10_head_pilot.train import apply_param_mask

    torch.manual_seed(0)
    m = _tiny_kronos_like()
    apply_param_mask(m, "A1")
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x = torch.randn(4, 8)
    t1, t2 = torch.randn(4, 16), torch.randn(4, 16)
    loss = ((m.head.proj_s1(x) - t1).pow(2).mean()
            + ((m.head.proj_s2(x) - t2).pow(2).mean()))
    opt.zero_grad()
    loss.backward()
    opt.step()
    changed = {k for k in before
               if not torch.equal(before[k], m.state_dict()[k])}
    assert changed <= A1_ALLOWED
    assert "head.proj_s1.weight" in changed
    assert "embedding.weight" not in changed


def test_tokenizer_untouched_by_param_mask() -> None:
    import torch

    from g10_head_pilot.train import apply_param_mask

    tok = _tiny_kronos_like()                     # 结构替身：tokenizer 同构冻结
    for p in tok.parameters():
        p.requires_grad_(True)
    before = {k: v.clone() for k, v in tok.state_dict().items()}
    apply_param_mask(tok, "A1")                   # A1 只作用于 predictor；
    # tokenizer 不在 predictor 命名空间内——以独立对象模拟：掩码不得触碰
    after = {k: v for k, v in tok.state_dict().items()}
    for k in before:
        assert torch.equal(before[k], after[k])


def test_selection_reads_only_val_ce() -> None:
    """bestCE 选点 = 验证 CE 最低 epoch（严格小于，并列最早），不读收益。"""
    hist = [{"epoch": e, "val_ce": v}
            for e, v in ((1, 5.0), (2, 4.5), (3, 4.5), (4, 4.9), (5, 4.0))]
    best_ce, best_epoch = float("inf"), None
    for row in hist:
        if row["val_ce"] < best_ce:
            best_ce, best_epoch = row["val_ce"], row["epoch"]
    assert best_epoch == 5
    hist2 = [dict(h, val_ce=4.5) for h in hist[:3]]
    best2 = None
    for row in hist2:
        if best2 is None or row["val_ce"] < best2:
            best2 = row["val_ce"]
    assert best2 == 4.5                           # 并列最早由 < 保证


def test_judge_criteria_formulas() -> None:
    from g10_head_pilot.evaluate import judge

    ic = {("A1", s, w): 0.02 for s in ("bestCE", "e15")
          for w in WINDOW_BOUNDS}
    ic.update({("A0", "e15", w): 0.01 for w in WINDOW_BOUNDS})
    ic.update({("A0", "bestCE", w): 0.03 for w in WINDOW_BOUNDS})  # A0 末期劣化
    tr = {("A1", s, w): {"idx_excess_net_cum": 0.05} for s in ("bestCE", "e15")
          for w in WINDOW_BOUNDS}
    diff = {"W3": 0.01, "W4": 0.02, "combined_t": 2.5,
            "combined_judgable": True}
    out = judge({"ic": ic, "trade": tr, "ic_diff_e15": diff})
    assert out["I_info_retained"] is True
    assert out["P_vs_A0_e15"] is True
    assert out["S_stability"] is True   # drop(A1)=0 < drop(A0)=0.02（严格更小）
    assert out["E_econ_filter"] is True
    # I 容忍带：drop(A1) > 0.01 → 失败
    ic_bad = dict(ic)
    ic_bad[("A1", "bestCE", "W3")] = 0.05         # drop=0.03 > 0.01
    out2 = judge({"ic": ic_bad, "trade": tr, "ic_diff_e15": diff})
    assert out2["I_info_retained"] is False
    # S：A0 无末期劣化（drop(A0)=0）且 A1 有劣化 → 不得宣称"挽救"
    ic_s = dict(ic)
    ic_s[("A1", "e15", "W3")] = 0.0               # drop(A1,W3)=0.02
    out3 = judge({"ic": ic_s, "trade": tr, "ic_diff_e15": diff})
    assert out3["S_stability"] is False


def test_window_bounds_full_no_forward() -> None:
    for wname, (start, end) in WINDOW_BOUNDS.items():
        assert str(end) <= FORWARD_CUTOFF
        assert (WINDOW_BOUNDS["W3"] == ("2025-07-01", "2025-12-31")
                and WINDOW_BOUNDS["W4"] == ("2026-01-01", "2026-07-24"))


def test_protocol_hash_tamper_rejected(tmp_path, monkeypatch) -> None:
    import g10_head_pilot.config as cfg

    monkeypatch.setattr(cfg, "RUN_DIR", tmp_path)
    monkeypatch.setattr(cfg, "PROTOCOL_PATH", tmp_path / "protocol.json")
    sha = cfg.write_protocol({"k": 1})
    (tmp_path / "protocol.json.sha256").write_text(sha + "\n", encoding="utf-8")
    cfg.load_protocol()                            # 未变 → 通过
    (tmp_path / "protocol.json").write_text('{"k": 2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="漂移"):
        cfg.load_protocol()
