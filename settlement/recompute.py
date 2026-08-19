"""复算级臂信号生成器（forward结算计划_20260818.md §1/§4.2）。

复算级 = 权重/协议在结算前已冻结，结算时确定性重放生成 forward 信号
（事后计算，先后无密码学证明——结论一律附"复算级"标签）。四臂冻结清单：

- F0 zero-shot：官方 Kronos-base + Tokenizer-base，canonical 协议 + seed42；
- F1：第 4 轮微调权重（finetune_predictor_f1 / finetune_tokenizer_f1）；
- B3：KDA 冻结 checkpoint（cross_section_kda/data/B3_best.pt，经
  cross_section_kda.models 装载）；
- R1：零推理组装（gate True→M / False→F0_mean，规则见 settlement.rules）。

信号源注入（``RecomputeSource`` 协议）：真实结算用推理链路（canonical 逐字）；
合成演习用 ``SyntheticRecomputeSource``（确定性合成，绝不触碰真实数据/权重）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# 协议串逐字 canonical（与 paper_replication/config.yaml 一致；结算计划 §1）
CANONICAL_PROTOCOL = "L=90/H=10/N=20/T=1.0/top_p=0.9/seed=42 canonical"

RECOMPUTE_ARM_SPECS: dict[str, dict] = {
    "F0": {
        "kind": "inference",
        "weights": "NeoQuasar/Kronos-base",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "protocol": CANONICAL_PROTOCOL,
        "level": "复算级",
        "note": "链条零点参照（zero-shot）",
    },
    "F1": {
        "kind": "inference",
        "weights": str(REPO_ROOT / "finetune_suite/outputs/models/finetune_predictor_f1/checkpoints/best_model"),
        "tokenizer": str(REPO_ROOT / "finetune_suite/outputs/models/finetune_tokenizer_f1/checkpoints/best_model"),
        "protocol": CANONICAL_PROTOCOL,
        "level": "复算级",
        "note": "第 4 轮权重，'改善但未存活'，观察 forward 方向",
    },
    "B3": {
        "kind": "inference",
        "weights": str(REPO_ROOT / "cross_section_kda/data/B3_best.pt"),
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "loader": "cross_section_kda.models",
        "protocol": CANONICAL_PROTOCOL,
        "level": "复算级",
        "note": "冻结 checkpoint，已降级'疑似种子运气'，复核用",
    },
    "R1": {
        "kind": "assembly",
        "rule": "gate True→M / False→F0_mean（settlement.rules.r1_assemble）",
        "protocol": CANONICAL_PROTOCOL,
        "level": "复算级",
        "note": "零推理组装，M=登记列，F0=本表 F0 臂重放",
    },
}


def verify_specs_on_disk() -> None:
    """冻结权重路径在盘断言（仅存在性检查，不加载任何权重）。"""
    for arm in ("F1", "B3"):
        p = Path(RECOMPUTE_ARM_SPECS[arm]["weights"])
        assert p.exists(), f"{arm} 冻结权重缺失：{p}"
    f1_tok = Path(RECOMPUTE_ARM_SPECS["F1"]["tokenizer"])
    assert f1_tok.exists(), f"F1 tokenizer 缺失：{f1_tok}"


def generate_arm_signals(
    arm: str,
    dates: pd.DatetimeIndex,
    source: "RecomputeSource",
    *,
    variant: str = "mean",
) -> pd.DataFrame:
    """逐日向信号源取 ``arm`` 的 ``variant`` 列，拼 date×code 宽表（确定性重放）。"""
    if arm not in RECOMPUTE_ARM_SPECS:
        raise KeyError(f"复算臂 {arm!r} 不在冻结清单 {sorted(RECOMPUTE_ARM_SPECS)}")
    rows = {}
    for d in dates:
        day = source.day_frame(arm, d)
        if variant not in day.columns:
            raise KeyError(f"{arm} 信号源缺变体列 {variant!r}（现有 {list(day.columns)}）")
        rows[d] = day[variant].dropna()
    return pd.DataFrame(rows).T.reindex(dates)


class RecomputeSource:  # pragma: no cover - 协议文档
    """信号源协议：真实结算 = canonical 推理链路；演习 = 合成确定性源。"""

    def day_frame(self, arm: str, date: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError

    def variants(self, arm: str) -> tuple[str, ...]:
        raise NotImplementedError


class SyntheticRecomputeSource(RecomputeSource):
    """演习用合成源：从 SyntheticWorld 取 F0/F1/B3 四变体日信号（零真实读取）。"""

    def __init__(self, world) -> None:
        self._world = world

    def day_frame(self, arm: str, date: pd.Timestamp) -> pd.DataFrame:
        if arm == "R1":
            raise KeyError("R1 为组装臂：由 executor 按 gate 逐日组装，不直取")
        return self._world.recompute[arm][date]

    def variants(self, arm: str) -> tuple[str, ...]:
        if arm == "R1":
            return ("mean",)
        return ("last", "mean", "max", "min")


class RealInferenceSource(RecomputeSource):
    """真实结算推理源（2026-11 开封时执行；canonical 逐字，断点续跑由调用方管理）。

    演习与测试**绝不实例化**本类：F0/F1 走 Kronos.from_pretrained +
    predict_batch_chunked（与 finetune_suite/run_registry.compute_day_signals
    同链路），B3 经 cross_section_kda.models + state_dict 装载，R1 由
    executor 组装（本源不提供 R1）。
    """

    def __init__(self, *, device: str = "cuda:0") -> None:
        from baseline_suite.common import BaselineConfig  # noqa: F401（协议冻结锚）

        self._cfg = BaselineConfig.load(window="oos")
        self._device = device
        self._cache: dict[str, object] = {}

    def day_frame(self, arm: str, date: pd.Timestamp) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError(
            "真实推理源在 2026-11 结算分支实现启用（本包为预注册演习基建）"
        )

    def variants(self, arm: str) -> tuple[str, ...]:
        return ("mean",) if arm == "R1" else ("last", "mean", "max", "min")
