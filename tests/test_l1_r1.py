"""L1 长上下文 + R1 读出头换目标 契约测试（计划 §4.1，20260903 L1与R1计划）。

三个计划指定测试 + 两个产物门禁：

- ``test_l1_config``：L1 各臂 BaselineConfig 除 lookback 与权重路径（及窗口
  字段）外与 canonical（paper_replication/config.yaml 加载）逐字段一致；
  L250-ft 训练配方相对 G1 配方仅 lookback_window/数据与落盘目录/实验标签可异；
  在位者 L90 冻结锚逐位一致；
- ``test_r1_loss_ic``：玩具批 IC 损失手算对拍（−Pearson）、对 MSE 的尺度平移
  不变性（两分支互斥的数学证据）、训练路径挂 IC 而非 MSE；
- ``test_r1_protocol_frozen``：R1 协议常量与 G5 镜像零漂移、臂/种子/窗冻结、
  批结构=每日截面、两头可训练参数量（833 / 1,209,937）；
- ``test_l250ft_dataset``：4.3 产物门禁（先 FAIL 后 PASS）——lookback=250 数据
  窗重建 pkl 结构/最短行数/时间范围；
- ``test_r1_day_groups_cache``：日边界重建与冻结 g5 缓存 y 逐位对拍（r1 训练
  数据完整性；构建后 PASS）。
"""
from __future__ import annotations

import sys
from dataclasses import asdict, fields as dc_fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baseline_suite.common import BaselineConfig


# ============================================================
# test_l1_config
# ============================================================

# 与 canonical 逐字段对比时允许差异的字段（计划 §1：唯一变量 = lookback + 权重路径；
# window/backtest_start/backtest_end 是窗口簿记，随臂窗枚举确定）
L1_ALLOWED_DIFF = {
    "lookback", "model_name", "tokenizer_name",
    "window", "backtest_start", "backtest_end",
}


def test_l1_config() -> None:
    from l1_context.config import ARMS, L250FT_PREDICTOR_PATH, WINDOW_DEFS

    canonical = asdict(BaselineConfig.load(window="oos"))
    for tag, spec in ARMS.items():
        from l1_context.config import build_arm_config

        for window in spec["windows"]:
            cfg = asdict(build_arm_config(tag, window))
            diffs = {
                k for k in canonical
                if canonical[k] != cfg[k]
            }
            assert diffs <= L1_ALLOWED_DIFF, (
                f"{tag}@{window} 出现计划外差异字段：{diffs - L1_ALLOWED_DIFF}"
            )
            assert cfg["lookback"] == spec["lookback"], tag
            assert cfg["model_name"] == spec["model_path"], tag
            assert cfg["tokenizer_name"] == spec["tokenizer_path"], tag
            # canonical 冻结值逐字（H=10/N=20/T=1.0/top_p=0.9/seed=42/k50/n5/c15bp）
            assert (cfg["predict_len"], cfg["sample_count"], cfg["T"]) == (10, 20, 1.0)
            assert (cfg["top_p"], cfg["seed"], cfg["top_k"]) == (0.9, 42, 50)
            assert (cfg["drop_n"], cfg["min_hold"], cfg["cost_bps"]) == (5, 5, 15.0)
            assert cfg["pool"] == "csi300" and cfg["max_context"] == 512

    # 臂表冻结（计划 §1）
    assert set(ARMS) == {
        "L250ZS100", "L250ZS101", "L250ZS102", "L500ZS100", "L250FT100"
    }, "L1 臂名集合与计划 §1 冻结表不一致"
    assert ARMS["L250FT100"]["model_path"] == str(L250FT_PREDICTOR_PATH)
    assert ARMS["L250FT100"]["lookback"] == 250
    # 双窗臂仅 L250-zs 三种子（计划 §1 评估窗声明）
    for tag in ("L250ZS100", "L250ZS101", "L250ZS102"):
        assert ARMS[tag]["windows"] == ("backtest", "2025h2")
    for tag in ("L500ZS100", "L250FT100"):
        assert ARMS[tag]["windows"] == ("backtest",)
    assert WINDOW_DEFS == {
        "backtest": ("2026-01-01", "2026-07-24"),
        "2025h2": ("2025-07-01", "2025-12-31"),
    }

    # —— L250-ft 配方：相对 G1 配方只改 lookback_window/数据/落盘/标签 ——
    from finetune_suite.train_g1 import G1Config
    from l1_context.config import L250FTConfig

    g1, ft = G1Config().__dict__, L250FTConfig().__dict__
    allowed = {
        "lookback_window", "val_time_range", "dataset_path", "save_path",
        "predictor_save_folder_name", "comet_tag", "comet_name",
        "finetuned_predictor_path",
    }
    diffs = {k for k in g1 if g1[k] != ft[k]}
    assert diffs <= allowed, f"L250-ft 配方出现计划外差异：{diffs - allowed}"
    assert ft["lookback_window"] == 250
    # 最小声明修正（计划内不可行性：官方 6 个月 val < window 261，CE 早停不可运转）
    assert ft["val_time_range"] == ["2024-01-02", "2025-06-30"]
    assert ft["seed"] == 100 and ft["epochs"] == 15
    assert ft["pretrained_predictor_path"] == g1["pretrained_predictor_path"]
    # tokenizer 冻结共享 G1（阶段 2.1 产物只读）
    assert ft["finetuned_tokenizer_path"] == g1["finetuned_tokenizer_path"]

    # —— 在位者 L90 冻结锚（计划 §0 表，逐位）——
    from l1_context.config import L90_FROZEN_AER_EW, L90_ANCHOR_PARQUETS

    assert L90_FROZEN_AER_EW == {
        "s100": pytest.approx(0.14333, abs=1e-4),
        "s101": pytest.approx(0.29059, abs=1e-4),
        "s102": pytest.approx(0.18671, abs=1e-4),
    }
    for window in ("backtest", "2025h2"):
        assert set(L90_ANCHOR_PARQUETS[window]) == {"s100", "s101", "s102"}
        for p in L90_ANCHOR_PARQUETS[window].values():
            assert p.exists(), f"L90 锚信号缺失：{p}"


# ============================================================
# test_r1_loss_ic
# ============================================================


def test_r1_loss_ic() -> None:
    from r1_objective.ic_loss import ic_loss

    # —— 手算对拍：p=[1,2,3,4], y=[2,1,4,3] → Pearson r = +0.6，损失 = −0.6 ——
    p = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([2.0, 1.0, 4.0, 3.0])
    loss = ic_loss(p, y)
    assert float(loss) == pytest.approx(-0.6, abs=1e-6)

    # 与 scipy 逐位对拍（非平凡玩具批）
    rng = np.random.default_rng(0)
    p2 = torch.tensor(rng.normal(size=137).astype(np.float32))
    y2 = torch.tensor(rng.normal(size=137).astype(np.float32))
    from scipy import stats

    r = stats.pearsonr(p2.numpy(), y2.numpy()).statistic
    assert float(ic_loss(p2, y2)) == pytest.approx(-r, abs=1e-5)

    # 完全同向 / 完全反向
    assert float(ic_loss(y, y)) == pytest.approx(-1.0, abs=1e-5)
    assert float(ic_loss(y, -y)) == pytest.approx(+1.0, abs=1e-5)

    # —— 与 MSE 分支互斥的数学证据：IC 对 pred 的仿射变换不变，MSE 不变才怪 ——
    mse = torch.nn.functional.mse_loss
    base_ic, shift_ic = float(ic_loss(p, y)), float(ic_loss(2.0 * p + 1.0, y))
    assert shift_ic == pytest.approx(base_ic, abs=1e-6)
    assert float(mse(p, y)) != pytest.approx(float(mse(2.0 * p + 1.0, y)))

    # —— 训练路径挂 IC 而非 MSE ——
    from r1_objective import run_r1_train

    assert run_r1_train.LOSS_NAME == "IC(-Pearson, 每日截面批内)"
    loss_fn = run_r1_train._make_loss_fn()
    assert not isinstance(loss_fn, torch.nn.MSELoss)
    assert float(loss_fn(p, y)) == pytest.approx(-0.6, abs=1e-6)


# ============================================================
# test_r1_protocol_frozen
# ============================================================


class _StubBackbone(torch.nn.Module):
    """头构造桩：只需 d_model 属性（G5 头不触权重内容）。"""

    d_model = 832


def test_r1_protocol_frozen() -> None:
    from g5_head.run_g5_head import (
        G5_BATCH, G5_EPOCHS, G5_ES_END, G5_ES_START, G5_LR,
        G5_PATIENCE, G5_PURGE, G5_TRAIN_END, G5_TRAIN_START, G5_WD,
    )
    from r1_objective import run_r1_train

    # 协议常量逐字镜像 G5（零漂移）
    assert run_r1_train.R1_LR == G5_LR == 3e-4
    assert run_r1_train.R1_WD == G5_WD == 0.01
    assert run_r1_train.R1_BATCH == G5_BATCH == 128  # 批粒度上限沿用；实际批=每日截面
    assert run_r1_train.R1_EPOCHS == G5_EPOCHS == 50
    assert run_r1_train.R1_PATIENCE == G5_PATIENCE == 5
    assert (run_r1_train.R1_TRAIN_START, run_r1_train.R1_TRAIN_END) == (G5_TRAIN_START, G5_TRAIN_END)
    assert (run_r1_train.R1_ES_START, run_r1_train.R1_ES_END) == (G5_ES_START, G5_ES_END)
    assert run_r1_train.R1_PURGE == G5_PURGE

    # 臂 / 种子 / 评估窗冻结（计划 §2 表）
    assert run_r1_train.R1_ARMS == ("R-lin", "R-kda")
    assert run_r1_train.R1_SEEDS == (42, 43, 44)
    assert run_r1_train.R1_WINDOWS == ("backtest", "2025h2")
    # 唯一变量 = 损失函数：批结构随损失按 qlib 惯例改为每日截面
    assert run_r1_train.R1_BATCHING == "per-day-cross-section"
    assert run_r1_train.LOSS_NAME == "IC(-Pearson, 每日截面批内)"

    # 头族与 G5 一一对应（同构构造 + 可训练参数量数值核算）
    from cross_section_kda.models import B2LinearProbe
    from g5_head.heads import G5KdaHead

    stub = _StubBackbone()
    lin = run_r1_train._make_head("R-lin", stub)
    kda = run_r1_train._make_head("R-kda", stub)
    assert isinstance(lin, B2LinearProbe) and isinstance(kda, G5KdaHead)
    n_lin = sum(t.numel() for t in lin.parameters() if t.requires_grad)
    n_kda = sum(t.numel() for t in kda.parameters() if t.requires_grad)
    assert n_lin == 833, f"R-lin 可训练参数 {n_lin} ≠ 833（832→1 线性探测）"
    assert n_kda == 1_209_937, f"R-kda 可训练参数 {n_kda} ≠ 1,209,937（G5 H-kda 同构）"


# ============================================================
# 4.3 产物门禁：test_l250ft_dataset（先 FAIL 后 PASS）
# ============================================================


def test_l250ft_dataset() -> None:
    """lookback=250 数据窗重建产物：结构/最短行数/时间范围（4.3 格式测试）。

    装载走官方 ``finetune.dataset.QlibDataset``（注入 L250FTConfig，与训练期同一路径）
    ——同时验证 window=261 与可采样索引非空；时间范围/行数走 ``build_stats.json``
    （json 安全加载）。不在测试内新增 pickle 反序列化面（Mimosa 门禁合规）。
    """
    import json
    import sys

    from l1_context.config import L250FTConfig

    cfg = L250FTConfig()
    ds_dir = Path(cfg.dataset_path)
    stats_path = ds_dir / "build_stats.json"
    if not stats_path.exists():
        pytest.skip("l250ft 数据窗未构建（4.3 产物门禁：构建后强制 PASS，缺失不豁免断言）")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["n_train_symbols"] > 5000, "全 A 语料训练符号数异常"
    assert stats["train_range"] == ["2011-01-04", "2024-12-31"]
    # val 已按最小声明修正扩展（2024-01-02~2025-06-30，容纳 window=261）
    assert stats["val_range"] == ["2024-01-02", "2025-06-30"], stats["val_range"]
    assert stats["n_rows_train"] > 10_000_000, "训练行数异常（全 A 语料千万级）"

    sys.path.insert(0, str(REPO_ROOT / "finetune"))
    import dataset as official_dataset

    assert official_dataset.Config is not L250FTConfig
    official_dataset.Config = L250FTConfig  # 注入（与 train_l250ft 同款机制）
    train_ds = official_dataset.QlibDataset("train")
    val_ds = official_dataset.QlibDataset("val")

    # 官方窗口语义：window = lookback(250) + predict(10) + 1 = 261
    assert train_ds.window == 261, f"window={train_ds.window} ≠ 261"
    assert val_ds.window == 261
    assert len(train_ds.indices) > 0, "train 无可采样窗口"
    assert len(val_ds.indices) > 0, "val 无可采样窗口"
    assert train_ds.n_samples == min(train_ds.n_samples, len(train_ds.indices))

    # 样本级格式：官方 __getitem__ 走一遍（特征 [261,6] / 时间特征 [261,5]）
    x, x_stamp = train_ds[0]
    assert x.shape == (261, 6) and x_stamp.shape == (261, 5)
    assert float(x.abs().max()) <= cfg.clip
    feature_cols = set(cfg.feature_list)
    for df in list(train_ds.data.values())[:50]:
        assert feature_cols <= set(df.columns)


# ============================================================
# R1 日边界重建完整性（构建后 PASS）
# ============================================================


def test_r1_day_groups_cache() -> None:
    """日边界重建 JSON 存在且与冻结缓存行数自洽（深度对拍在构建时做+落盘记录）。"""
    import json

    from r1_objective.day_groups import DAY_GROUPS_JSON, HIDDEN_CACHE_PATH

    assert HIDDEN_CACHE_PATH.exists(), "g5 隐状态缓存缺失（只读前置）"
    if not DAY_GROUPS_JSON.exists():
        pytest.skip("day_groups 尚未构建（run_r1_train 首次运行时构建+深度对拍）")
    doc = json.loads(DAY_GROUPS_JSON.read_text(encoding="utf-8"))
    assert doc["verify"]["y_allclose"] is True
    assert doc["verify"]["max_abs_diff"] < 1e-6
    groups = doc["groups"]
    assert len(groups) > 400, "train 日数应 ~480（2022-01~2023-12-15）"
    total = sum(g["length"] for g in groups)
    assert total == doc["n_train_samples"]
    assert all(g["length"] >= 5 for g in groups), "出现 <5 只的截面日（IC 损失退化为跳过）"
