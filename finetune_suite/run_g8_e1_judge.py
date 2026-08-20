"""G8+E1 统一封盘判读（计划 §3.4，20260820）——G8-1~4 + E1-1~4 一次开封。

**预注册判据（跑前冻结，§1/§2 表，只定义在 mean / 中位种子）**：

G8（backtest 窗中位种子，2025H2 为早停窗准样本内、禁止跨窗检验）：
    G8-1 存活/注册：中位 AER(等权) > 0 且 AER(指数) > 0 → 注册 G1v2025H1 臂，
        forward 结算与 G1 同场对照（新鲜度终审在干净前向数据上）；
    G8-2 显著性：中位对中位差（G8−G1）> +26pp（噪声底）→ 滚动再训协议改半年度；
    G8-3 描述项：同种子配对差（G8−G1）三对必列，带内一律"不可判"；
    G8-4 否定：中位任一基准 ≤ 0 → 不注册，不做终点日期搜索。

E1（backtest 窗三种子；2025H2 为附加记录窗，E1 无准样本内问题）：
    E1-1 机械判据：日均换手较 canonical 下降 ≥ 30%（确定性量，精确可读）；
    E1-2 无害判据：E1 净 AER(等权) 中位 − canonical 中位 > −26pp；
    E1-3 注册：E1-1 且 E1-2 → 缓冲带候选成立（规则文档注册，同 C 系）；
    E1-4 否定：E1-1 未达 → "滞回带压不动机械换手底，成本侧该形态关闭"，
        不做阈值搜索（30/100 之外不加点）。

判读输入 = 3.3/3.1 已落盘 JSON（只读）；本脚本追加 verdict 到
``finetune_suite/data/g8/g8_e1_verdict.json`` 并打印判读（贴入结果文档）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PKG_DIR = Path(__file__).resolve().parent
G8_DIR = PKG_DIR / "data" / "g8"
E1_RESULTS = PKG_DIR.parent / "e1_buffer" / "data" / "e1_replay_results.json"

SEEDS = ("s100", "s101", "s102")
NOISE_BAND = 0.26  # 窗口 AER 标准误 ±26pp（134 日）
E1_TURNOVER_DROP_REQ = 0.30


def judge_g8(med_ew: float, med_idx: float, med_to_med_diff: float,
             paired: dict[str, float]) -> dict:
    """G8-1~4（纯函数，合成数字单元覆盖）。"""
    g8_1 = med_ew > 0 and med_idx > 0
    g8_2 = med_to_med_diff > NOISE_BAND
    g8_4 = not g8_1
    paired_notes = {
        s: ("带内（|Δ|≤26pp）→ 不可判" if abs(d) <= NOISE_BAND else
            ("高于噪声底 → 新鲜度显著（该种子）" if d > 0 else
             "低于噪声底 → 新鲜度显著为负（该种子）"))
        for s, d in paired.items()
    }
    return {
        "G8_1_survival_register": {
            "median_aer_ew": med_ew, "median_aer_idx": med_idx,
            "passed": bool(g8_1),
            "note": (
                f"中位种子 AER(等权)={med_ew:+.2%} 且 AER(指数)={med_idx:+.2%} "
                "双双 > 0 → 通过 G1 同款准入，注册 G1v2025H1 臂，forward 结算"
                "与 G1 同场对照（新鲜度终审在干净前向数据上，不在本轮）"
                if g8_1 else
                f"中位种子 AER(等权)={med_ew:+.2%} / AER(指数)={med_idx:+.2%} "
                "未双双为正 → 未过准入"
            ),
        },
        "G8_2_significance": {
            "median_to_median_ew_diff": med_to_med_diff, "threshold": NOISE_BAND,
            "passed": bool(g8_2),
            "note": (
                f"中位对中位差 {med_to_med_diff:+.2%} > +26pp → 新鲜度显著，"
                "滚动再训协议改半年度" if g8_2 else
                f"中位对中位差 {med_to_med_diff:+.2%} ≤ +26pp（噪声底）→ "
                "未达显著线（预期难达，如实设线）"
            ),
        },
        "G8_3_paired_diffs": {
            "paired_ew_diff": paired, "notes": paired_notes,
            "note": "同种子配对差（G8−G1，AER 等权）三对全列；带内一律不可判",
        },
        "G8_4_negative": {
            "triggered": bool(g8_4),
            "note": (
                "中位任一基准 ≤ 0 → +6 个月新鲜度未过准入，语料新鲜度臂不注册，"
                "不做终点日期搜索" if g8_4 else "未触发（G8-1 通过）"
            ),
        },
    }


def judge_e1(bt: dict) -> dict:
    """E1-1~4（纯函数）。``bt`` = replay JSON 的 backtest 窗块（table+medians）。"""
    drops = {s: bt["table"][s]["delta"]["turnover_drop_pct"] for s in SEEDS}
    med_drop = bt["medians"]["turnover_drop_pct_median"]
    min_drop = bt["medians"]["turnover_drop_pct_min"]
    net_delta = bt["medians"]["net_aer_ew_median_delta"]
    e1_1 = med_drop >= E1_TURNOVER_DROP_REQ
    e1_2 = net_delta > -NOISE_BAND
    e1_4 = not e1_1
    return {
        "E1_1_mechanical": {
            "turnover_drop_by_seed": drops,
            "median_drop": med_drop, "min_drop": min_drop,
            "threshold": E1_TURNOVER_DROP_REQ,
            "passed": bool(e1_1),
            "note": (
                f"三种子换手降幅中位 {med_drop:+.2%}（最小 {min_drop:+.2%}）"
                f"≥ 30% → 机械换手底被压动" if e1_1 else
                f"三种子换手降幅中位 {med_drop:+.2%}（最小 {min_drop:+.2%}，"
                f"逐种子 {', '.join(f'{s}:{d:+.1%}' for s, d in drops.items())}）"
                "均远未达 30% 门槛（降幅为负=换手不降反微升）→ 滞回带压不动机械换手底"
            ),
        },
        "E1_2_harmless": {
            "net_aer_ew_median_delta": net_delta, "threshold": -NOISE_BAND,
            "passed": bool(e1_2),
            "note": (
                f"E1−canonical 净 AER(等权) 中位差 {net_delta:+.2%} > −26pp → "
                "滞回迟钝未显著吃掉成本节约" if e1_2 else
                f"E1−canonical 净 AER(等权) 中位差 {net_delta:+.2%} ≤ −26pp → "
                "信号迟钝损失超噪声底"
            ),
        },
        "E1_3_register": {
            "triggered": bool(e1_1 and e1_2),
            "note": (
                "E1-1 且 E1-2 → 缓冲带候选成立，以规则文档注册（同 C 系），"
                "forward 结算同场对照" if (e1_1 and e1_2) else
                "未触发（E1-1 未达）→ 不注册"
            ),
        },
        "E1_4_negative": {
            "triggered": bool(e1_4),
            "note": (
                "E1-1 未达 → 滞回带压不动机械换手底，成本侧该形态关闭；"
                "不做阈值搜索（30/100 之外不加点）" if e1_4 else "未触发"
            ),
        },
    }


def main() -> None:
    g8 = json.loads((G8_DIR / "g8_backtest_results.json").read_text(encoding="utf-8"))
    e1 = json.loads(E1_RESULTS.read_text(encoding="utf-8"))

    med = g8["medians"]
    paired = {
        s: g8["groups"][f"{s}_mean"]["perf_ew"]["aer"]
        - g8["g1_frozen_refs"][s]["mean"]["aer_ew"]
        for s in SEEDS
    }
    verdict = {
        "experiment": "g8_e1_unified_verdict",
        "date": "2026-08-20",
        "G8": judge_g8(
            med["g8_mean_aer_ew_median"], med["g8_mean_aer_idx_median"],
            med["median_to_median_ew_diff"], paired,
        ),
        "E1": judge_e1({"table": e1["table"]["backtest"],
                        "medians": e1["medians"]["backtest"]}),
    }
    out_path = G8_DIR / "g8_e1_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")

    print("=== 统一封盘判读（G8-1~4 + E1-1~4 一次开封）===")
    for arm in ("G8", "E1"):
        for k, v in verdict[arm].items():
            head = v.get("passed", v.get("triggered"))
            print(f"[{k}] {'通过/触发' if head else '未通过/未触发'}：{v['note']}")
    print(f"判读落盘：{out_path}")


if __name__ == "__main__":
    main()
