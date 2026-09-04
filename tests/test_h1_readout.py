"""H1 读出头全语料版 契约测试（计划 §3.1，20260905 H1 计划）。

四个计划指定测试（先 FAIL 后 PASS）：

- ``test_label_purge``：训练标签末日 ≤ 2024-12-17（决策日语义：t+10 交易日
  落在 train pkl 末 2024-12-31 内）、早停段标签 ∈ 2025H1；
- ``test_daily_batch_same_date``：批内 128 股同一交易日截面（stamp 日期字段
  批内唯一）、批大小 ≤ 128、y 与 x 行对齐；
- ``test_protocol_frozen``：lr/wd/batch/patience/损失与 R1 逐字段一致（零漂移
  import 对拍），H1 侧冻结（steps 2000 / epochs ≤ 15 / 数据窗 / 臂表 / 早停窗）；
- ``test_backbone_frozen``：G1 s100 底座全参数 requires_grad=False、train()
  重写后恒 eval、H1 头可训练参数恰为 833 / 1,209,937（底座不在可训练集）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# test_label_purge（csi300 真语料，构建 + 断言）
# ============================================================


def test_label_purge() -> None:
    from h1_readout.corpus import (
        ES_END, ES_START, TRAIN_LABEL_END, TRAIN_START, build_corpus,
    )

    assert (TRAIN_START, TRAIN_LABEL_END) == ("2014-01-02", "2024-12-17")
    assert (ES_START, ES_END) == ("2025-01-01", "2025-06-30")

    corpus = build_corpus("csi300")
    cal = corpus.calendar
    cal_pos = {d: i for i, d in enumerate(cal)}

    # 训练决策日 ≤ 2024-12-17 且标签日（t+10 交易日）≤ 2024-12-31（train 段内）
    train_days = corpus.train_days
    assert len(train_days) > 2000, f"csi300 11 年训练日应 ~2600，实测 {len(train_days)}"
    for d in train_days[:5] + train_days[-5:] + train_days[len(train_days) // 2:len(train_days) // 2 + 5]:
        assert str(d.date.date()) <= TRAIN_LABEL_END
        label_date = cal[cal_pos[d.date] + 10]
        assert str(label_date.date()) <= "2024-12-31", (
            f"训练标签 {label_date.date()} 越过 train pkl 末（泄露早停段）")
    max_train_day = max(d.date for d in train_days)
    assert str(max_train_day.date()) == TRAIN_LABEL_END, "末日决策日应恰为 2024-12-17"

    # 早停段：决策日与标签日均 ∈ 2025H1（决策日上界 = 06-30 回退 10 交易日，
    # +10 标签恰为 06-30——不溢出到 2025H2 评估窗）
    es_days = corpus.es_days
    assert len(es_days) > 100, f"2025H1 早停日应 ~110+，实测 {len(es_days)}"
    n_cal = len(cal)
    for d in es_days:
        assert ES_START <= str(d.date.date()) <= ES_END
        c = cal_pos[d.date]
        assert c + 10 < n_cal, f"早停日 {d.date.date()} 距语料日历末不足 10 日"
        assert str(cal[c + 10].date()) <= ES_END, (
            f"早停标签 {cal[c + 10].date()} 越过 2025-06-30（溢出 2025H2 评估窗）")

    # 训练段与早停段之间 ≥ 10 交易日 purge（2024-12-18~12-31）
    gap = cal_pos[es_days[0].date] - cal_pos[max_train_day] - 1
    assert gap >= 10, f"purge 间隔仅 {gap} 交易日 < 10"


# ============================================================
# test_daily_batch_same_date
# ============================================================


def test_daily_batch_same_date() -> None:
    from h1_readout.corpus import build_corpus
    from h1_readout.sampler import DailyBatchSampler

    corpus = build_corpus("csi300")
    sampler = DailyBatchSampler(corpus, seed=42)
    for _ in range(5):
        x, stamp, y, day = sampler.sample()
        assert x.shape == (min(128, len(y)), 90, 6), x.shape
        assert stamp.shape == (x.shape[0], 90, 5)
        assert y.shape == (x.shape[0],)
        # 批内同一交易日：stamp 的日/月/星期字段批内唯一
        key = {(int(s[3]), int(s[4]), int(s[2])) for s in stamp[:, -1, :].tolist()}
        assert len(key) == 1, f"批内出现 {len(key)} 个不同日期：{key}"
        # 该日期即采样器声称的决策日
        (dd, mm, wd) = key.pop()
        assert (dd, mm) == (day.date.day, day.date.month)
        # 数值口径：窗口 z-score + clip ±5
        assert float(x.abs().max()) <= 5.0 + 1e-4
        assert abs(float(x.mean())) < 0.5  # 逐窗去均值后的批均值应近零


# ============================================================
# test_protocol_frozen
# ============================================================


def test_protocol_frozen() -> None:
    from r1_objective import run_r1_train
    from r1_objective.ic_loss import ic_loss

    from h1_readout import train_h1

    # 与 R1 逐字段一致（唯一继承变量集；零漂移 import 对拍）
    assert train_h1.H1_LR == run_r1_train.R1_LR == 3e-4
    assert train_h1.H1_WD == run_r1_train.R1_WD == 0.01
    assert train_h1.H1_BATCH == run_r1_train.R1_BATCH == 128
    assert train_h1.H1_PATIENCE == run_r1_train.R1_PATIENCE == 5
    assert train_h1.LOSS_NAME == run_r1_train.LOSS_NAME
    assert train_h1._make_loss_fn() is ic_loss

    # H1 侧冻结（计划 §1）
    assert train_h1.H1_STEPS_PER_EPOCH == 2000
    assert train_h1.H1_EPOCHS == 15
    assert train_h1.H1_ARMS == ("H1a-lin", "H1a-kda", "H1b-lin", "H1b-kda")
    assert train_h1.ARM_POOL == {"H1a-lin": "csi300", "H1a-kda": "csi300",
                                 "H1b-lin": "ashares", "H1b-kda": "ashares"}
    assert train_h1.H1_SEEDS == (42,)          # 两阶段：seed 42 先行
    assert train_h1.ES_POOL == "csi300"          # 早停段恒 csi300（R1 口径）
    # 数据窗冻结
    from h1_readout.corpus import ES_END, ES_START, TRAIN_LABEL_END, TRAIN_START

    assert (TRAIN_START, TRAIN_LABEL_END, ES_START, ES_END) == (
        "2014-01-02", "2024-12-17", "2025-01-01", "2025-06-30")


# ============================================================
# test_backbone_frozen
# ============================================================


def test_backbone_frozen() -> None:
    from h1_readout.train_h1 import _make_head

    backbone = _load_backbone_cpu()
    n_bb_params = 0
    for p in backbone.parameters():
        n_bb_params += 1
        assert p.requires_grad is False, "G1 底座参数必须冻结（requires_grad=False）"

    # train() 重写：恒 eval（token_drop/dropout 不扰动表征）
    backbone.train(True)
    assert backbone.tokenizer.training is False
    assert backbone.kronos.training is False

    # 头可训练参数恰为头自身（底座不在可训练集），数值核算 833 / 1,209,937
    lin = _make_head("H1a-lin", backbone)
    kda = _make_head("H1b-kda", backbone)
    n_lin = sum(p.numel() for p in lin.parameters() if p.requires_grad)
    n_kda = sum(p.numel() for p in kda.parameters() if p.requires_grad)
    assert n_lin == 833, f"H1a-lin 可训练 {n_lin} ≠ 833"
    assert n_kda == 1_209_937, f"H1b-kda 可训练 {n_kda} ≠ 1,209,937"

    # 前向不污染底座梯度：extract 后底座参数 grad 恒 None
    x = torch.randn(4, 90, 6)
    stamp = torch.zeros(4, 90, 5)
    _ = backbone.extract(x, stamp)
    assert all(p.grad is None for p in backbone.parameters())


def _load_backbone_cpu():
    """装载真实 G1 s100 底座（CPU；~1 分钟内）。"""
    from g5_head.backbone_g1 import load_g1_backbone

    return load_g1_backbone("cpu")
