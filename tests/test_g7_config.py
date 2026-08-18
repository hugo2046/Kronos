"""g7_shortwindow 单测（G7 计划 §3 3.1，20260818）。

``test_g7_config``：W85 配置**除 L/H 外与 canonical 逐字一致**的门禁——
先于实现写成（模块未建时 import 失败 → FAIL；实现后 → PASS）。

逐项断言（G7 计划 §0/§4 冻结内容）：
    - L=8/H=5（W85 唯一自由度）且 H=5 与 min_hold=5 对齐（无持有期错配）；
    - 推理采样（N=20/T=1.0/top_p=0.9/top_k=0/seed=42/device/max_context/
      signal_field/pool/data_end/filter_pipe）与 ``BaselineConfig.load("oos")``
      （= paper_replication/config.yaml canonical）逐字段相等；
    - 引擎参数（k=50/n=5/min_hold=5/15bp/baselines）逐字段相等（引擎零改动纪律）；
    - 窗口边界逐字 import 自既有常量（backtest=第 4 轮 BACKTEST_START/END，
      2025h2=run_g2_signals.WINDOW_DEFS），不自定义；
    - G1 族三种子权重映射（s100=G1，s101/102=G2S101/102，tokenizer 共享 G1）
      与训练模块派生路径一致且在盘（G1 权重只读纪律）。
"""
from __future__ import annotations

from dataclasses import fields

SEEDS = (100, 101, 102)

# W85 相对 canonical 的**全部**合法差异字段：L/H + 权重 + 窗口标签/边界。
# 此集合之外的任何字段漂移 = 协议漂移 → 测试失败。
ALLOWED_DIFF = {
    "lookback",
    "predict_len",
    "model_name",
    "tokenizer_name",
    "window",
    "backtest_start",
    "backtest_end",
}


def _canonical():
    from baseline_suite.common import BaselineConfig

    return BaselineConfig.load(window="oos")


class TestW85Config:
    def test_l_h(self) -> None:
        """W85 = L=8/H=5，且 H 与 min_hold 对齐（计划 §0 错配披露仅涉 H=3）。"""
        from g7_shortwindow.run_g7_signals import build_w85_config

        cfg = build_w85_config(100, "backtest")
        assert cfg.lookback == 8, f"L={cfg.lookback}，期望 8"
        assert cfg.predict_len == 5, f"H={cfg.predict_len}，期望 5"
        assert cfg.predict_len == cfg.min_hold, "H=5 应与 min_hold=5 对齐"

    def test_verbatim_canonical_except_lh(self) -> None:
        """除 ALLOWED_DIFF 外逐字段与 canonical 相等（三种子 × 两窗全覆盖）。"""
        from g7_shortwindow.run_g7_signals import build_w85_config

        base = _canonical()
        for seed in SEEDS:
            for window in ("backtest", "2025h2"):
                cfg = build_w85_config(seed, window)
                for f in fields(cfg):
                    if f.name in ALLOWED_DIFF:
                        continue
                    got, want = getattr(cfg, f.name), getattr(base, f.name)
                    assert got == want, (
                        f"s{seed}/{window} 字段 {f.name} 漂移：{got!r} != {want!r}"
                    )

    def test_engine_params_frozen(self) -> None:
        """引擎参数与 yaml 冻结值逐字相等（k=50/n=5/min_hold=5/15bp）。"""
        from g7_shortwindow.run_g7_signals import build_w85_config

        cfg = build_w85_config(101, "backtest")
        assert (cfg.top_k, cfg.drop_n, cfg.min_hold, cfg.cost_bps) == (50, 5, 5, 15)
        assert list(cfg.baselines) == ["momentum_10d", "reversal_10d"]

    def test_windows_verbatim(self) -> None:
        """窗口常量逐字 import 自既有模块（与在位者同界可比）。"""
        from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
        from finetune_suite.run_g2_signals import WINDOW_DEFS as G2_WINDOWS

        from g7_shortwindow.run_g7_signals import WINDOW_DEFS

        assert WINDOW_DEFS["backtest"] == (BACKTEST_START, BACKTEST_END)
        assert WINDOW_DEFS["2025h2"] == G2_WINDOWS["2025h2"]
        assert WINDOW_DEFS["2025h2"] == ("2025-07-01", "2025-12-31")
        from g7_shortwindow.run_g7_signals import build_w85_config

        for window, (s, e) in WINDOW_DEFS.items():
            for seed in SEEDS:
                cfg = build_w85_config(seed, window)
                assert (cfg.backtest_start, cfg.backtest_end) == (s, e)

    def test_g1_family_weight_mapping(self) -> None:
        """s100=G1 / s101=G2S101 / s102=G2S102，tokenizer 共享 G1，路径在盘。"""
        from pathlib import Path

        from finetune_suite.train_g1 import G1Config
        from finetune_suite.train_g2 import G2Config

        from g7_shortwindow.run_g7_signals import build_w85_config

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
            cfg = build_w85_config(seed, "backtest")
            assert (cfg.model_name, cfg.tokenizer_name) == derived[seed], (
                f"s{seed} 权重映射漂移"
            )
            assert Path(cfg.model_name).exists(), f"s{seed} predictor 不在盘：{cfg.model_name}"
            assert Path(cfg.tokenizer_name).exists(), (
                f"s{seed} tokenizer 不在盘：{cfg.tokenizer_name}"
            )
