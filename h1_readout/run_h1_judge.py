"""H1 一次开封判读（计划 §2 判据 HC1~HC4，§3.4）。

四臂打分全部落盘 + 引擎全表封存后，本脚本**一次性**开封：

- 全表：两窗 ×（四 H1 臂 + R1 锚 s42 + 在位者 G1_mean + F0/M 参照）× 双基准；
- HC1 存活：某臂 backtest 双基准 AER > 0（seed 42 首轮）；
- HC2 数据效应：同头配对——H1a − R1（年数效应，R1 基线取冻结中位种子）、
  H1b − H1a（广度效应）；> +26pp 才许"显著"，带内方向性描述；
- HC3 对在位者：与 G1 s100（+14.33%/+10.66%）配对，同 ±26pp 噪声框架；
- HC4 终审关闭：四臂 backtest 均 ≤ 0 → "读出出头在 70× 数据下仍无 oos 存活
  信息——读出议题终审关闭（架构/目标/数据三维度均已覆盖）"，不做任何搜索；
- 结局地图（计划 §2）：按 HC1/HC2 组合给出广度/年数归因方向；
- 注册：HC1 ∧ 2025H2 > 0 ∧ 三种子中位双正 → 进入 3.5 种子确认后注册。

R1 锚复跑一致性（§3.3）：R-lin_s42/R-kda_s42 引擎复跑值与
r1_objective/data/r1_backtest_results.json 冻结值对拍（容差 5e-4）。

产出：``h1_readout/data/h1_verdict.json`` + ``h1_readout/figures/h1_aer_bar.png``。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"
R1_RESULTS = PKG_DIR.parent / "r1_objective" / "data" / "r1_backtest_results.json"

# —— 冻结常量（计划 §0/§2 跑前冻结）——
NOISE_BAND = 0.26
INCUMBENT_AER_EW = 0.1433
INCUMBENT_AER_IDX = 0.1066
# R1 冻结基线（中位种子，docs/L1与R1实验结果_20260903.md §1）
R1_BASELINE_EW = {"lin": -0.1232, "kda": -0.0118}
SEED = 42


def _aer(doc: dict, name: str, window: str, bench: str) -> float:
    src = doc["results"] if name in doc["results"] else doc["incumbent"]
    return float(src[name][window][bench]["aer"])


def _ir(doc: dict, name: str, window: str, bench: str) -> float:
    src = doc["results"] if name in doc["results"] else doc["incumbent"]
    return float(src[name][window][bench]["ir"])


def main() -> None:
    doc = json.loads((DATA_DIR / "h1_backtest_results.json").read_text(encoding="utf-8"))
    r1_frozen = json.loads(R1_RESULTS.read_text(encoding="utf-8"))["results"]

    arms = [f"{a}_s{SEED}" for a in ("H1a-lin", "H1a-kda", "H1b-lin", "H1b-kda")]

    # —— 全表 ——
    print("=" * 96)
    print("H1 一次开封：引擎全表（AER：年化超额；idx=000300.SH 市值加权，ew=同池等权）")
    print("=" * 96)
    order = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")] + [
        "R-lin_s42", "R-kda_s42"] + arms + ["G1_mean"]
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

    # —— R1 锚复跑一致性（§3.3 贴实数）——
    print("\n—— R1 锚复跑一致性（同引擎同宇宙 vs r1_backtest_results.json）——")
    anchor_ok = True
    for n in ("R-lin_s42", "R-kda_s42"):
        rerun = _aer(doc, n, "backtest", "perf_ew")
        frozen = float(r1_frozen[n]["backtest"]["perf_ew"]["aer"])
        ok = abs(rerun - frozen) < 5e-4
        anchor_ok &= ok
        print(f"  {n}: 复跑 {rerun:+.4%} vs 冻结 {frozen:+.4%} "
              f"差 {abs(rerun - frozen):.4%} {'一致' if ok else '不一致！'}")
    assert anchor_ok, "R1 锚复跑不一致——评估链路异常，判读中止"

    # —— 判据 ——
    per_arm = {}
    for name in arms:
        bt_ew = _aer(doc, name, "backtest", "perf_ew")
        bt_idx = _aer(doc, name, "backtest", "perf_idx")
        h2_ew = _aer(doc, name, "2025h2", "perf_ew")
        per_arm[name] = {
            "backtest_aer_ew": bt_ew, "backtest_aer_idx": bt_idx,
            "2025h2_aer_ew": h2_ew,
            "HC1_survive": bool(bt_ew > 0 and bt_idx > 0),
            "HC3_vs_incumbent": bt_ew - INCUMBENT_AER_EW,
            "2025h2_positive": bool(h2_ew > 0),
        }

    # HC2：同头配对（lin/kda 各两组）
    hc2 = {}
    for head in ("lin", "kda"):
        d_years = _aer(doc, f"H1a-{head}_s{SEED}", "backtest", "perf_ew") - R1_BASELINE_EW[head]
        d_breadth = _aer(doc, f"H1b-{head}_s{SEED}", "backtest", "perf_ew") - \
            _aer(doc, f"H1a-{head}_s{SEED}", "backtest", "perf_ew")
        hc2[head] = {
            "d_years_H1a_minus_R1": d_years,
            "d_breadth_H1b_minus_H1a": d_breadth,
            "years_significant": bool(d_years > NOISE_BAND),
            "breadth_significant": bool(d_breadth > NOISE_BAND),
        }

    hc4 = all(not per_arm[n]["HC1_survive"] for n in arms)

    # 结局地图（计划 §2）
    def _alive(tag):
        return per_arm[f"{tag}_s{SEED}"]["HC1_survive"]

    a_alive, b_alive = _alive("H1a-lin") or _alive("H1a-kda"), _alive("H1b-lin") or _alive("H1b-kda")
    if a_alive and b_alive and (hc2["lin"]["d_breadth_H1b_minus_H1a"] > 0 or
                                hc2["kda"]["d_breadth_H1b_minus_H1a"] > 0):
        outcome = "广度是主因（与 G1 定律一致）"
    elif not a_alive and b_alive:
        outcome = "广度是必要条件"
    elif a_alive and not (b_alive and (hc2["lin"]["d_breadth_H1b_minus_H1a"] > 0 or
                                       hc2["kda"]["d_breadth_H1b_minus_H1a"] > 0)):
        outcome = "年数是主因（H1a 活而 H1b 不更强）"
    elif hc4:
        outcome = "饥饿不是死因，读出终审关闭"
    else:
        outcome = "混合/带内（见配对差明细）"

    verdict = {
        "experiment": "h1_judge", "date": "2026-09-05", "seed": SEED,
        "noise_band": NOISE_BAND,
        "per_arm": per_arm, "HC2": hc2,
        "HC4_final_close": bool(hc4),
        "HC4_statement": (
            "读出头在 70× 数据下仍无 oos 存活信息——读出议题终审关闭"
            "（架构/目标/数据三维度均已覆盖），不做任何搜索" if hc4 else None),
        "outcome_map": outcome,
        "HC1_passers": [n for n in arms if per_arm[n]["HC1_survive"]],
        "registration_gate": (
            "HC1 ∧ 2025H2>0 ∧ 三种子中位双正 → 3.5 种子确认（seed 43/44）后注册"),
        "prior_on_record": "半污染窗声明沿用；通过判据上限「待前向确认」",
    }
    out = DATA_DIR / "h1_verdict.json"
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== H1 预注册判据 HC1~HC4 一次开封（backtest，AER）====")
    for name in arms:
        h = per_arm[name]
        print(f"[{name}] AER(等权)={h['backtest_aer_ew']:+.2%} AER(指数)="
              f"{h['backtest_aer_idx']:+.2%} | 2025H2 AER(等权)={h['2025h2_aer_ew']:+.2%} "
              f"| HC1={h['HC1_survive']} | HC3 对在位者={h['HC3_vs_incumbent']:+.2%}")
    for head, h in hc2.items():
        print(f"[HC2-{head}] 年数效应 H1a−R1={h['d_years_H1a_minus_R1']:+.2%}"
              f"（{'显著' if h['years_significant'] else '带内方向'}）| "
              f"广度效应 H1b−H1a={h['d_breadth_H1b_minus_H1a']:+.2%}"
              f"（{'显著' if h['breadth_significant'] else '带内方向'}）")
    print(f"[HC4 终审关闭] {hc4} — {verdict['HC4_statement'] or '未触发'}")
    print(f"[结局地图] {outcome}")
    print(f"[HC1 通过者] {verdict['HC1_passers'] or '无'}（进入 3.5 种子确认）")
    print(f"判读落盘 → {out}")

    # —— 图 ——
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        names = ["R-lin_s42", "R-kda_s42"] + arms + ["G1_mean"]
        aers = [_aer(doc, n, "backtest", "perf_ew") for n in names]
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = ["#4d4d4d", "#7a6c5d", "#2171b5", "#006d2c", "#6baed6", "#74c476", "#08306b"]
        ax.bar(names, aers, color=colors)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
        for i, a in enumerate(aers):
            ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}", ha="center", fontsize=8)
        ax.set_ylabel("AER (等权基准, with cost)")
        ax.set_title("H1 全语料读出头 — backtest AER(等权) 2026-01-01~2026-07-24")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG_DIR / "h1_aer_bar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"图落盘 → {FIG_DIR / 'h1_aer_bar.png'}")
    except Exception as e:
        print(f"[warn] 图绘制失败（不阻断判读）：{e}")


if __name__ == "__main__":
    main()
