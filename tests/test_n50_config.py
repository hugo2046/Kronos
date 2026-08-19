"""n50_amplify 单测（N50 计划 §3 3.1，20260819）。

``test_n50_config``：N50 配置**除 sample_count=50 外与 canonical 逐字一致**的
门禁——先于实现写成（模块未建时 import 失败 → FAIL；实现后 → PASS）。

逐项断言（N50 计划 §0/§1/§4 冻结内容）：
    - N=50（唯一自由度）；L=90/H=10 与 canonical 逐字相等（本实验**不碰**
      窗口参数——G7 已终审关闭 L/H 议题，本轮只动采样路径数）；
    - 推理采样（T=1.0/top_p=0.9/top_k=0/推理seed=42/device/max_context/
      signal_field/pool/data_end/filter_pipe）与 ``BaselineConfig.load("oos")``
      （= paper_replication/config.yaml canonical）逐字段相等；
    - 引擎参数（k=50/n=5/min_hold=5/15bp/baselines）逐字段相等（引擎零改动纪律）；
    - 窗口边界逐字 import 自既有常量（backtest=第 4 轮 BACKTEST_START/END），
      不自定义；**2025h2 不提供**（预算裁定跑前声明，runner 不暴露该选项）；
    - G1 族三种子权重映射（s100=G1，s101/102=G2S101/102，tokenizer 共享 G1）
      与训练模块派生路径一致且在盘（G1 权重只读纪律）。
"""
from __future__ import annotations

from dataclasses import fields

SEEDS = (100, 101, 102)

# N50 相对 canonical 的**全部**合法差异字段：sample_count + 权重 + 窗口标签/边界。
# 此集合之外的任何字段漂移 = 协议漂移 → 测试失败。
# （lookback/predict_len **不在**其中——L/H 不是本轮变量，必须逐字等于 canonical。）
ALLOWED_DIFF = {
    "sample_count",
    "model_name",
    "tokenizer_name",
    "window",
    "backtest_start",
    "backtest_end",
}


def _canonical():
    from baseline_suite.common import BaselineConfig

    return BaselineConfig.load(window="oos")


class TestN50Config:
    def test_sample_count_only_knob(self) -> None:
        """唯一自由度 N=50；L=90/H=10 与 canonical 逐字相等（不碰窗口参数）。"""
        from n50_amplify.run_n50_signals import build_n50_config

        base = _canonical()
        cfg = build_n50_config(100)
        assert cfg.sample_count == 50, f"N={cfg.sample_count}，期望 50"
        assert base.sample_count == 20, f"canonical N={base.sample_count}，期望 20"
        assert cfg.lookback == base.lookback == 90, (
            f"L={cfg.lookback} 漂移（canonical {base.lookback}）"
        )
        assert cfg.predict_len == base.predict_len == 10, (
            f"H={cfg.predict_len} 漂移（canonical {base.predict_len}）"
        )

    def test_verbatim_canonical_except_n(self) -> None:
        """除 ALLOWED_DIFF 外逐字段与 canonical 相等（三种子全覆盖）。"""
        from n50_amplify.run_n50_signals import build_n50_config

        base = _canonical()
        for seed in SEEDS:
            cfg = build_n50_config(seed)
            for f in fields(cfg):
                if f.name in ALLOWED_DIFF:
                    continue
                got, want = getattr(cfg, f.name), getattr(base, f.name)
                assert got == want, (
                    f"s{seed} 字段 {f.name} 漂移：{got!r} != {want!r}"
                )

    def test_engine_params_frozen(self) -> None:
        """引擎参数与 yaml 冻结值逐字相等（k=50/n=5/min_hold=5/15bp）。"""
        from n50_amplify.run_n50_signals import build_n50_config

        cfg = build_n50_config(101)
        assert (cfg.top_k, cfg.drop_n, cfg.min_hold, cfg.cost_bps) == (50, 5, 5, 15)
        assert list(cfg.baselines) == ["momentum_10d", "reversal_10d"]

    def test_windows_verbatim_backtest_only(self) -> None:
        """窗口常量逐字 import 自第 4 轮 backtest 常量；无 2025h2（预算裁定）。"""
        from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START

        from n50_amplify.run_n50_signals import WINDOW_DEFS, build_n50_config

        assert WINDOW_DEFS == {"backtest": (BACKTEST_START, BACKTEST_END)}
        for seed in SEEDS:
            cfg = build_n50_config(seed)
            assert (cfg.backtest_start, cfg.backtest_end) == (
                BACKTEST_START, BACKTEST_END,
            )

    def test_g1_family_weight_mapping(self) -> None:
        """s100=G1 / s101=G2S101 / s102=G2S102，tokenizer 共享 G1，路径在盘。"""
        from pathlib import Path

        from finetune_suite.train_g1 import G1Config
        from finetune_suite.train_g2 import G2Config

        from n50_amplify.run_n50_signals import build_n50_config

        g1 = G1Config()
        derived = {
            100: (G1Config().finetuned_predictor_path, g1.finetuned_tokenizer_path),
            101: (
                G2Config(seed=101).finetuned_predictor_path,
                g1.finetuned_tokenizer_path,
            ),
            102: (
                G2Config(seed=102).finetuned_predictor_path,
                g1.finetuned_tokenizer_path,
            ),
        }
        for seed in SEEDS:
            cfg = build_n50_config(seed)
            assert (cfg.model_name, cfg.tokenizer_name) == derived[seed], (
                f"s{seed} 权重映射漂移"
            )
            assert Path(cfg.model_name).exists(), f"s{seed} predictor 不在盘：{cfg.model_name}"
            assert Path(cfg.tokenizer_name).exists(), (
                f"s{seed} tokenizer 不在盘：{cfg.tokenizer_name}"
            )
