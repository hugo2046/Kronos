"""R1 一次开封判读（计划 §2 判据 RC1~RC4，§4.5）。

六训信号全部落盘 + 引擎全表封存后，本脚本**一次性**开封：

- 全表：两窗 ×（6 R 模型 + 在位者 G1_mean + F0/M 参照）× 双基准 AER/IR；
- RC1 存活：某头中位种子（backtest AER 等权排序取中位，g5 同口径）backtest
  双基准 AER > 0；
- RC2 目标效应：同头 IC vs MSE 配对差（种子对齐；R-kda 3 对真实种子配对；
  R-lin 仅 s42 有 G5 MSE 对照 → s43/s44 以 G5 H-lin_s42 单锚配对并如实披露），
  三差取中位；> +26pp 才许"目标函数是死因"，带内只做"不可判"；
- RC3 对在位者：中位种子与 G1 s100（G1_mean 行）配对差，同 ±26pp 噪声框架；
- RC4 终审关闭：R-lin、R-kda 中位均 ≤ 0 → "读出议题两目标（回归/排序）× 三头族
  × 两底座终审关闭"，不做损失函数搜索；
- 注册：RC1 通过者需 2025H2 AER(等权) > 0 方可注册（复算级新臂）。

诚实先验（计划 §2）：15~25%——头仍在 2022-23 标签上学，"早停段强、样本外崩"
大概率重演；通过判据上限"待前向确认"。

产出：``r1_objective/data/r1_verdict.json`` + ``r1_objective/figures/r1_aer_bar.png``。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"
G5_RESULTS = PKG_DIR.parent / "g5_head" / "data" / "g5_stage2_results.json"

# —— 冻结常量（计划 §2 跑前冻结）——
NOISE_BAND = 0.26          # ±26pp 噪声框架（134 日窗）
INCUMBENT = "G1_mean"      # 在位者锚（G1 s100，L90）
ARM_OF = {"R-lin": "H-lin", "R-kda": "H-kda"}   # R1 → G5 对照头
SEEDS = (42, 43, 44)


def _aer(doc: dict, name: str, window: str, bench: str) -> float:
    src = doc["results"] if name in doc["results"] else doc["incumbent"]
    return float(src[name][window][bench]["aer"])


def _ir(doc: dict, name: str, window: str, bench: str) -> float:
    src = doc["results"] if name in doc["results"] else doc["incumbent"]
    return float(src[name][window][bench]["ir"])


def _median_seed(doc: dict, arm: str) -> str:
    ranked = sorted(SEEDS, key=lambda s: _aer(doc, f"{arm}_s{s}", "backtest", "perf_ew"))
    return f"{arm}_s{ranked[1]}"


def main() -> None:
    doc = json.loads((DATA_DIR / "r1_backtest_results.json").read_text(encoding="utf-8"))
    g5 = json.loads(G5_RESULTS.read_text(encoding="utf-8"))["results"]

    # —— 全表 ——
    print("=" * 96)
    print("R1 一次开封：引擎全表（AER：年化超额；idx=000300.SH 市值加权，ew=同池等权）")
    print("=" * 96)
    order = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")] + list(doc["results"]) + [INCUMBENT]
    for window in doc["windows"]:
        print(f"\n—— 窗口 {window} {doc['windows'][window]} ——")
        print(f"{'模型':<14}{'AER(idx)':>10}{'IR(idx)':>9}{'AER(ew)':>10}{'IR(ew)':>9}")
        for name in order:
            src = doc["results"] if name in doc["results"] else doc["incumbent"]
            if window not in src.get(name, {}):
                continue
            print(f"{name:<14}"
                  f"{_aer(doc, name, window, 'perf_idx'):+10.2%}"
                  f"{_ir(doc, name, window, 'perf_idx'):+9.3f}"
                  f"{_aer(doc, name, window, 'perf_ew'):+10.2%}"
                  f"{_ir(doc, name, window, 'perf_ew'):+9.3f}")

    # —— 判据 ——
    inc_bt_ew = _aer(doc, INCUMBENT, "backtest", "perf_ew")
    per_head = {}
    for arm, g5_arm in ARM_OF.items():
        med = _median_seed(doc, arm)
        bt_ew = _aer(doc, med, "backtest", "perf_ew")
        bt_idx = _aer(doc, med, "backtest", "perf_idx")
        h2_ew = _aer(doc, med, "2025h2", "perf_ew")
        rc1 = bt_ew > 0 and bt_idx > 0

        # RC2：逐种子配对差（IC − MSE），中位
        diffs, pairs_note = [], []
        for s in SEEDS:
            r_aer = _aer(doc, f"{arm}_s{s}", "backtest", "perf_ew")
            m_key = f"{g5_arm}_s{s}"
            if m_key not in g5:
                m_key = f"{g5_arm}_s42"  # R-lin s43/s44：G5 单锚配对（如实披露）
                pairs_note.append(f"{arm}_s{s}↔H-lin_s42(单锚)")
            m_aer = float(g5[m_key]["backtest"]["perf_ew"]["aer"])
            diffs.append(r_aer - m_aer)
        diffs_sorted = sorted(diffs)
        rc2_median = diffs_sorted[1]
        rc2_verdict = ("target_objective" if rc2_median > NOISE_BAND
                       else "不可判(带内)" if rc2_median > 0 else "带内负/零(不可判)" if abs(rc2_median) <= NOISE_BAND
                       else "IC 更差(超带)")

        rc3_diff = bt_ew - inc_bt_ew
        per_head[arm] = {
            "median_model": med,
            "backtest_aer_ew": bt_ew, "backtest_aer_idx": bt_idx,
            "2025h2_aer_ew": h2_ew,
            "RC1_survive": bool(rc1),
            "RC2_paired_diffs": diffs, "RC2_median": rc2_median,
            "RC2_verdict": rc2_verdict, "RC2_pairs": pairs_note,
            "RC3_vs_incumbent": rc3_diff,
            "RC3_significant": bool(abs(rc3_diff) > NOISE_BAND),
            "2025h2_positive": bool(h2_ew > 0),
        }

    rc4 = all(not per_head[a]["RC1_survive"] for a in ARM_OF)
    verdict = {
        "experiment": "r1_judge", "date": "2026-09-03",
        "noise_band": NOISE_BAND,
        "incumbent_backtest_aer_ew": inc_bt_ew,
        "per_head": per_head,
        "RC4_final_close": bool(rc4),
        "RC4_statement": (
            "读出议题两目标（回归/排序）× 三头族 × 两底座终审关闭，不做损失函数搜索"
            if rc4 else None),
        "registration": {
            a: bool(per_head[a]["RC1_survive"] and per_head[a]["2025h2_positive"])
            for a in ARM_OF
        },
        "prior_on_record": "诚实先验 15~25%（计划 §2）；通过判据上限「待前向确认」",
    }

    out = DATA_DIR / "r1_verdict.json"
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== R1 预注册判据 RC1~RC4 一次开封 ====")
    for arm, h in per_head.items():
        print(f"[{arm}] 中位种子={h['median_model']} | backtest AER(等权)={h['backtest_aer_ew']:+.2%} "
              f"AER(指数)={h['backtest_aer_idx']:+.2%} | 2025H2 AER(等权)={h['2025h2_aer_ew']:+.2%}")
        print(f"        RC1 存活={h['RC1_survive']} | RC2 配对差中位={h['RC2_median']:+.2%}"
              f"（{h['RC2_verdict']}）| RC3 对在位者={h['RC3_vs_incumbent']:+.2%}")
        if h["RC2_pairs"]:
            print(f"        RC2 配对披露：{'; '.join(h['RC2_pairs'])}")
    print(f"[RC4 终审关闭] {rc4} — {verdict['RC4_statement'] or '未触发（存在存活头）'}")
    print(f"[注册触发] {[a for a, ok in verdict['registration'].items() if ok] or '无'}"
          f"（RC1 通过且 2025H2>0）")
    print(f"判读落盘 → {out}")

    # —— 图：backtest AER(等权) 条形 ——
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        names, aers = [], []
        for arm in ARM_OF:
            for s in SEEDS:
                names.append(f"{arm}_s{s}")
                aers.append(_aer(doc, f"{arm}_s{s}", "backtest", "perf_ew"))
        for a in ARM_OF:
            names.append(f"{a}·中位")
            aers.append(per_head[a]["backtest_aer_ew"])
        names.append("G1_mean(在位)")
        aers.append(inc_bt_ew)
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = ["#2171b5" if "lin" in n else "#006d2c" for n in names]
        colors = ["#08306b" if "在位" in n else c for n, c in zip(names, colors)]
        ax.bar(names, aers, color=colors)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
        for i, a in enumerate(aers):
            ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}", ha="center", fontsize=8)
        ax.set_ylabel("AER (等权基准, with cost)")
        ax.set_title("R1 IC 损失六训 — backtest AER(等权) 2026-01-01~2026-07-24")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG_DIR / "r1_aer_bar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"图落盘 → {FIG_DIR / 'r1_aer_bar.png'}")
    except Exception as e:  # 图失败不阻断判读
        print(f"[warn] 图绘制失败（不阻断判读）：{e}")


if __name__ == "__main__":
    main()
