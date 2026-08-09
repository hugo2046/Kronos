"""阶段 5：预注册判读（计划 §6，先否决项）。

严格按预注册口径判读，不做任何参数搜索（111 期样本上调参必然过拟合）。

1. **信号存在性**：|RankIC 均值| < 0.02 或 |ICIR| < 0.3 →
   "zero-shot 无可用截面信号"，停止；IC 为负也算有信号（反转性），如实记录方向；
2. **相对基线**：Kronos 的 |RankIC| 与多空净值**未同时超过**动量、反转两条基线中较强者 →
   "信号存在但不优于免费价量因子，zero-shot 路线无增量价值"；
3. 两关都过 → "zero-shot 截面信号成立"，下一步才谈 A 股微调增强（另立计划）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from cross_section.common import DATA_DIR, ExperimentConfig

EVAL_JSON_PATH = DATA_DIR / "eval_metrics.json"

# 预注册阈值（计划 §6，不可漂移）
GATE_RANKIC_ABS = 0.02
GATE_ICIR_ABS = 0.3


def judge(metrics: dict) -> dict:
    """执行 §6 两关判读，返回结构化结论。

    :param metrics: ``stage4_evaluate.run_evaluation`` 返回的 metrics 字典。
    :returns: ``{gate1, gate2, conclusion, detail}``。
    """
    m = {k: v for k, v in metrics.items()}
    kronos = m["kronos"]
    mom = m["momentum"]
    rev = m["reversal"]

    # ===== 关 1：信号存在性 =====
    rankic_abs = abs(kronos["rankic_mean"])
    icir_abs = abs(kronos["icir"])
    direction = "正（追涨）" if kronos["rankic_mean"] > 0 else "负（反转性）"
    gate1_pass = (rankic_abs >= GATE_RANKIC_ABS) and (icir_abs >= GATE_ICIR_ABS)
    gate1 = {
        "pass": gate1_pass,
        "rankic_mean": kronos["rankic_mean"],
        "rankic_abs": rankic_abs,
        "icir": kronos["icir"],
        "icir_abs": icir_abs,
        "direction": direction,
        "threshold_rankic": GATE_RANKIC_ABS,
        "threshold_icir": GATE_ICIR_ABS,
    }
    logger.info(
        f"关1 信号存在性：|RankIC均值|={rankic_abs:.4f}（阈 {GATE_RANKIC_ABS}），"
        f"|ICIR|={icir_abs:.3f}（阈 {GATE_ICIR_ABS}）方向={direction} → "
        f"{'✅通过' if gate1_pass else '❌未通过（无可用截面信号）'}"
    )

    if not gate1_pass:
        return {
            "gate1": gate1,
            "gate2": None,
            "conclusion": "zero-shot 无可用截面信号",
            "conclusion_detail": (
                f"Kronos-base 零样本信号在 A 股 csi300 截面上 |RankIC均值|={rankic_abs:.4f} "
                f"或 |ICIR|={icir_abs:.3f} 未达预注册门槛，判定为无可用截面信号。"
                "按预注册纪律停止，不做参数搜索。"
            ),
        }

    # ===== 关 2：相对基线（必须同时超过两条基线中较强者）=====
    # "较强者"用 |RankIC| 与多空年化净值分别取两条基线 max
    baselines = {"momentum": mom, "reversal": rev}
    best_rankic = max(abs(mom["rankic_mean"]), abs(rev["rankic_mean"]))
    best_ann = max(mom["long_short_annualized_net"], rev["long_short_annualized_net"])
    best_label_rankic = "momentum" if abs(mom["rankic_mean"]) >= abs(rev["rankic_mean"]) else "reversal"
    best_label_ann = "momentum" if mom["long_short_annualized_net"] >= rev["long_short_annualized_net"] else "reversal"

    kronos_rankic_better = rankic_abs > best_rankic
    kronos_ann_better = kronos["long_short_annualized_net"] > best_ann
    gate2_pass = kronos_rankic_better and kronos_ann_better
    gate2 = {
        "pass": gate2_pass,
        "kronos_rankic_abs": rankic_abs,
        "best_baseline_rankic_abs": best_rankic,
        "best_baseline_rankic_label": best_label_rankic,
        "kronos_rankic_better": kronos_rankic_better,
        "kronos_ann": kronos["long_short_annualized_net"],
        "best_baseline_ann": best_ann,
        "best_baseline_ann_label": best_label_ann,
        "kronos_ann_better": kronos_ann_better,
    }
    logger.info(
        f"关2 相对基线：Kronos |RankIC|={rankic_abs:.4f} vs 基线最强 "
        f"{best_label_rankic}={best_rankic:.4f} → "
        f"{'✅超过' if kronos_rankic_better else '❌未超过'}；"
        f"Kronos 多空年化={kronos['long_short_annualized_net']:+.2%} vs 基线最强 "
        f"{best_label_ann}={best_ann:+.2%} → "
        f"{'✅超过' if kronos_ann_better else '❌未超过'}"
    )

    if gate2_pass:
        conclusion = "zero-shot 截面信号成立"
        detail = (
            "Kronos-base 零样本信号在 RankIC 与多空净值上**同时**超过动量、反转两条基线中较强者，"
            "判定 zero-shot 截面信号成立。下一步可谈 A 股微调增强（另立计划）。"
        )
    else:
        conclusion = "信号存在但不优于免费价量因子，zero-shot 路线无增量价值"
        detail = (
            "Kronos-base 零样本信号虽达到信号存在性门槛，但在 RankIC 或多空净值上"
            "未能**同时**超过动量、反转两条基线中较强者。判定 zero-shot 路线无增量价值。"
        )

    return {"gate1": gate1, "gate2": gate2, "conclusion": conclusion, "conclusion_detail": detail}


def main() -> int:
    cfg = ExperimentConfig.load()
    with open(EVAL_JSON_PATH, encoding="utf-8") as f:
        eval_data = json.load(f)
    result = judge(eval_data["metrics"])
    logger.info(f"最终结论：{result['conclusion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
