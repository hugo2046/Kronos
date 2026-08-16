"""finetune_ashares G2 种子诊断契约测试（计划 §1，20260816 计划）。

- ``test_g2_config_single_variable``：G2Config(seed) 相对 G1Config 的差异**仅限**
  seed / predictor 输出目录 / 实验名——语料、tokenizer（共享条款复用 g1）、
  一切训练超参逐字一致（唯一变量 = predictor 训练种子）；
- 判据 S1~S4 逻辑（合成数字，跑前可测）：
  - S1 保级：三种子 backtest AER(等权) 的**中位种子**双基准 > 0；
  - S2 稳健强度：三种子 AER(等权) 全部 > 0；
  - S3 降级：中位种子任一基准 ≤ 0（与 S1 互补）；
  - S4 跨窗一致：新种子 2025H2 较 F0_mean 改善 ≥ +5pp 的复现比例（如实记录）。
"""
from __future__ import annotations

from finetune_suite.run_g2_judge import judge_g2

# 协议字段（纪律 §3：G2 必须逐字复用 G1/第 5 轮，唯一变量 = 种子）
_PROTOCOL_FIELDS = [
    "dataset_path", "lookback_window", "predict_window", "max_context",
    "feature_list", "time_feature_list", "train_time_range", "val_time_range",
    "dataset_begin_time", "dataset_end_time",
    "clip", "epochs", "log_interval", "batch_size",
    "n_train_iter", "n_val_iter",
    "tokenizer_learning_rate", "predictor_learning_rate", "accumulation_steps",
    "adam_beta1", "adam_beta2", "adam_weight_decay",
    "pretrained_tokenizer_path", "pretrained_predictor_path",
    "finetuned_tokenizer_path",  # 共享条款：G2 复用 G1 tokenizer（不重训）
    "tokenizer_save_folder_name",
]


def test_g2_config_single_variable():
    from finetune_suite.train_g1 import G1Config
    from finetune_suite.train_g2 import G2Config

    g1 = G1Config()
    # 101/102 = 计划 §1 核心；103/104 = 跑前增补 D-seed+（均共享 G1 tokenizer）
    for seed in (101, 102, 103, 104):
        g2 = G2Config(seed=seed)
        drift = [f for f in _PROTOCOL_FIELDS if getattr(g1, f) != getattr(g2, f)]
        assert not drift, f"seed={seed} 协议字段漂移：{drift}"
        assert g2.seed == seed and g1.seed == 100  # 唯一变量
        assert g2.predictor_save_folder_name == f"finetune_predictor_g2_s{seed}"
        assert g2.finetuned_predictor_path.endswith(
            f"finetune_predictor_g2_s{seed}/checkpoints/best_model"
        )
        # tokenizer 路径 = G1 共享工件（逐字复用，不重训）
        assert g2.finetuned_tokenizer_path == g1.finetuned_tokenizer_path
        # G1 seed=100 权重目录不被 G2 触碰
        assert g2.predictor_save_folder_name != g1.predictor_save_folder_name


def test_dtok_config_full_pipeline_seed():
    """增补臂 D-tok：tokenizer+predictor 全管线 seed=101（补 tokenizer 种子洞）。"""
    from finetune_suite.train_dtok import DtokConfig
    from finetune_suite.train_g1 import G1Config

    g1 = G1Config()
    dt = DtokConfig()
    drift = [
        f for f in _PROTOCOL_FIELDS
        if f not in ("finetuned_tokenizer_path", "tokenizer_save_folder_name")
        and getattr(g1, f) != getattr(dt, f)
    ]
    assert not drift, f"D-tok 协议字段漂移：{drift}"
    assert dt.seed == 101
    assert dt.tokenizer_save_folder_name == "finetune_tokenizer_dtok"
    assert dt.predictor_save_folder_name == "finetune_predictor_dtok"
    # D-tok 用自己的 tokenizer（与共享 G1 tokenizer 的 s101 形成对照）
    assert dt.finetuned_tokenizer_path != g1.finetuned_tokenizer_path
    assert dt.finetuned_tokenizer_path.endswith("finetune_tokenizer_dtok/checkpoints/best_model")


def _case(s100_ew, s101_ew, s102_ew, idx=None, h2=None, f0_h2=-0.10):
    """构造 judge_g2 输入：三种子 backtest (ew, idx) + 新种子 2025H2 ew + F0_mean@2025H2。"""
    idx = idx or {}
    h2 = h2 if h2 is not None else {}
    return {
        "backtest": {
            "s100": {"aer_ew": s100_ew, "aer_idx": idx.get("s100", s100_ew)},
            "s101": {"aer_ew": s101_ew, "aer_idx": idx.get("s101", s101_ew)},
            "s102": {"aer_ew": s102_ew, "aer_idx": idx.get("s102", s102_ew)},
        },
        "h2_new_seeds": {
            "s101": {"aer_ew": h2.get("s101", h2.get("s", 0.0))},
            "s102": {"aer_ew": h2.get("s102", h2.get("s", 0.0))},
        },
        "f0_mean_h2_aer_ew": f0_h2,
    }


def test_s1_median_seed_dual_positive():
    # 中位（按 AER 等权排序）= s101，双基准为正 → S1 通过
    v = judge_g2(_case(0.20, 0.10, -0.05, idx={"s101": 0.02}))
    assert v["S1_keep"]["passed"]
    assert not v["S3_downgrade"]["triggered"]
    assert v["S1_keep"]["median_seed"] == "s101"


def test_s1_fails_when_median_idx_le0():
    # 中位种子 AER(等权)>0 但 AER(指数)≤0 → S1 失败、S3 触发
    v = judge_g2(_case(0.20, 0.10, -0.05, idx={"s101": 0.0}))
    assert not v["S1_keep"]["passed"]
    assert v["S3_downgrade"]["triggered"]
    assert v["S3_downgrade"]["median_seed"] == "s101"


def test_s2_all_positive():
    v = judge_g2(_case(0.14, 0.09, 0.03))
    assert v["S2_all_positive"]["passed"]
    v2 = judge_g2(_case(0.14, 0.09, -0.03))
    assert not v2["S2_all_positive"]["passed"]


def test_s4_replication_ratio():
    # F0_mean@2025H2=-10%：新种子 -4%（+6pp≥5pp 复现）、-6%（+4pp<5pp 未复现）→ 1/2
    v = judge_g2(_case(0.14, 0.10, 0.05, h2={"s101": -0.04, "s102": -0.06}, f0_h2=-0.10))
    assert v["S4_cross_window"]["ratio"] == 0.5
    # 恰好 +5pp → 复现（≥ 含等于）
    v2 = judge_g2(_case(0.14, 0.10, 0.05, h2={"s101": -0.05, "s102": -0.05}, f0_h2=-0.10))
    assert v2["S4_cross_window"]["ratio"] == 1.0


def test_median_by_aer_ew():
    # 中位按 AER(等权) 排序取中，不按种子号
    v = judge_g2(_case(-0.02, 0.30, 0.05))
    assert v["S1_keep"]["median_seed"] == "s102"


def test_g2_signals_alignment():
    """G2 两窗信号落盘与既有对照索引逐日一致（同窗可比）。"""
    import pandas as pd

    from finetune_suite.run_g2_signals import WINDOW_DEFS, arm_tag

    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    refs = {
        "backtest": repo / "finetune_suite" / "data" / "daily_signals_backtest_M.parquet",
        "2025h2": repo / "finetune_suite" / "data" / "g0" / "daily_signals_2025h2_M.parquet",
    }
    for seed in (101, 102):
        for window in WINDOW_DEFS:
            paths = [
                repo / "finetune_suite" / "data" / "g2" / f"s{seed}"
                / f"daily_signals_{window}_{arm_tag(seed)}_{v}.parquet"
                for v in ("last", "mean", "max", "min")
            ]
            missing = [p.name for p in paths if not p.exists()]
            assert not missing, f"s{seed}/{window} 信号缺失 {missing}：先跑 run_g2_signals.py"
            ref_idx = pd.read_parquet(refs[window]).index
            for p in paths:
                assert pd.read_parquet(p).index.equals(ref_idx), (
                    f"{p.name} 索引与 {window} 窗对照不一致"
                )


def test_g2_supp_signals_alignment():
    """增补臂（D-seed+ 103/104 与 D-tok）backtest 窗信号对齐（只跑 backtest，增补条款）。"""
    import pandas as pd

    from finetune_suite.run_g2_signals import arm_tag

    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    ref_idx = pd.read_parquet(
        repo / "finetune_suite" / "data" / "daily_signals_backtest_M.parquet"
    ).index
    for seed, sub in ((103, "s103"), (104, "s104"), ("dtok", "dtok")):
        paths = [
            repo / "finetune_suite" / "data" / "g2" / sub
            / f"daily_signals_backtest_{arm_tag(seed)}_{v}.parquet"
            for v in ("last", "mean", "max", "min")
        ]
        missing = [p.name for p in paths if not p.exists()]
        assert not missing, f"{sub} 信号缺失 {missing}：先跑增补臂推理"
        for p in paths:
            assert pd.read_parquet(p).index.equals(ref_idx), f"{p.name} 索引不一致"
