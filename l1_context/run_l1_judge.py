"""L1 一次开封判读（计划 §1 判据 LC1~LC4，§4.5）。

各臂信号全部落盘 + 引擎全表封存后，本脚本**一次性**开封：

- 全表：两窗 ×（参照 + L90 锚 ×3 + L1 臂）× 双基准 AER/IR；
- LC1 靶存在：d_ft = (L250-ft − L90 s100) > 0 **且** d_zs_med = median{(L250-zs_s −
  L90_s)}（s=100/101/102 逐种子配对差取中位）> 0——两条同向 → "长上下文有增量"；
- LC2 换骨架：LC1 通过 且 d_500 = (L500-zs s100 − L250-zs s100) > 0 → "逼近 512
  仍在涨，S1 立项"；LC1 通过但 d_500 ≤ 0 → "增量在 Transformer 上限内即可获得
  ——采用 G1@L250 路线，S1 撤案"；
- LC3 显著：任一配对差 > +26pp 才许"显著"；带内只做方向描述；
- LC4 关闭：LC1 任一条 ≤ 0 → "上下文于 90 附近饱和/过峰，长上下文议题关闭，
  S1 撤案"，不做 L 网格（250/500 之外不加点）；
- 注册：LC1 通过的 L250 臂（ft 优先）以规则文档注册进结算对照（复算级新臂）。

锚口径：判读配对一律用**同引擎同宇宙**重算锚值（与 L1 臂同表）；冻结值
（+14.33/+29.06/+18.67）与重算差 > 1pp 时如实披露（G9 K0 同款）。

产出：``l1_context/data/l1_verdict.json`` + ``l1_context/figures/l1_aer_bar.png``。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PKG_DIR = Path(__file__).resolve().parent
L1_DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"

# —— 冻结常量（计划 §1 跑前冻结）——
NOISE_BAND = 0.26          # ±26pp 噪声框架（134 日窗）
SEEDS = ("s100", "s101", "s102")
ZS_ARM = {"s100": "L1L250ZS100", "s101": "L1L250ZS101", "s102": "L1L250ZS102"}
ANCHOR_TAG = {"s100": "G1_mean", "s101": "G2S101_mean", "s102": "G2S102_mean"}
FT_ARM, L500_ARM = "L1L250FT100", "L1L500ZS100"
L90_FROZEN = {"s100": 0.14333, "s101": 0.29059, "s102": 0.18671}


def _aer(doc: dict, window: str, tag: str) -> float:
    return float(doc["groups"][f"{window}:{tag}"]["perf_ew"]["aer"])


def _ir(doc: dict, window: str, tag: str) -> float:
    return float(doc["groups"][f"{window}:{tag}"]["perf_ew"]["ir"])


def main() -> None:
    doc = json.loads((L1_DATA_DIR / "l1_backtest_results.json").read_text(encoding="utf-8"))

    # —— 全表 ——
    print("=" * 96)
    print("L1 一次开封：引擎全表（AER：年化超额；idx=000300.SH 市值加权，ew=同池等权）")
    print("=" * 96)
    order_backtest = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")] + [
        ANCHOR_TAG[s] for s in SEEDS] + [
        f"{a}_{v}" for a in ("L1L250ZS100", "L1L250ZS101", "L1L250ZS102",
                             "L1L500ZS100", "L1L250FT100")
        for v in ("min", "max", "last", "mean")]
    order_h2 = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")] + [
        ANCHOR_TAG[s] for s in SEEDS] + [
        f"{a}_{v}" for a in ("L1L250ZS100", "L1L250ZS101", "L1L250ZS102")
        for v in ("min", "max", "last", "mean")]
    for window, order in (("backtest", order_backtest), ("2025h2", order_h2)):
        print(f"\n—— 窗口 {window} {doc['windows'][window]} ——")
        print(f"{'组':<16}{'AER(ew)':>10}{'IR(ew)':>9}")
        for tag in order:
            key = f"{window}:{tag}"
            if key not in doc["groups"]:
                continue
            print(f"{tag:<16}{_aer(doc, window, tag):+10.2%}{_ir(doc, window, tag):+9.3f}")

    # —— 锚重算披露（判读一律用重算值；冻结值并列）——
    print("\n—— L90 锚重算 vs 冻结（同引擎同宇宙）——")
    for seed in SEEDS:
        rc = doc["anchor_recheck"]["backtest"][seed]
        flag = "" if rc.get("within_tol", True) else " ← 超 1pp，判读用重算值"
        print(f"  {ANCHOR_TAG[seed]}: 重算 {rc['rerun_aer_ew']:+.4%} vs 冻结 "
              f"{L90_FROZEN[seed]:+.4%}{flag}")

    # —— 配对差（backtest，mean 变体，AER 等权）——
    d_ft = _aer(doc, "backtest", f"{FT_ARM}_mean") - _aer(doc, "backtest", f"{ANCHOR_TAG['s100']}")
    d_zs = {
        s: _aer(doc, "backtest", f"{ZS_ARM[s]}_mean") - _aer(doc, "backtest", f"{ANCHOR_TAG[s]}")
        for s in SEEDS
    }
    d_zs_med = sorted(d_zs.values())[1]
    d_500 = _aer(doc, "backtest", f"{L500_ARM}_mean") - _aer(doc, "backtest", f"{ZS_ARM['s100']}_mean")

    lc1 = d_ft > 0 and d_zs_med > 0
    lc3_significant = max(d_ft, *d_zs.values(), d_500) > NOISE_BAND
    if lc1:
        lc2_pass = d_500 > 0
        lc4 = False
    else:
        lc2_pass, lc4 = None, True

    # 2025H2 描述性（可比性窗，不进判据）
    d_zs_h2 = {
        s: _aer(doc, "2025h2", f"{ZS_ARM[s]}_mean") - _aer(doc, "2025h2", f"{ANCHOR_TAG[s]}")
        for s in SEEDS
    }

    verdict = {
        "experiment": "l1_judge", "date": "2026-09-03",
        "noise_band": NOISE_BAND,
        "paired_diffs": {"d_ft_vs_s100": d_ft, "d_zs_by_seed": d_zs,
                         "d_zs_median": d_zs_med, "d_L500_minus_L250_s100": d_500,
                         "d_zs_2025h2_by_seed_descriptive": d_zs_h2},
        "LC1_target_exists": bool(lc1),
        "LC2_s1_project": (None if lc2_pass is None else bool(lc2_pass)),
        "LC2_statement": (
            (f"S1（突破 512 上限）立项" if lc2_pass
             else "增量在 Transformer 上限内即可获得——采用 G1@L250 路线，S1 撤案")
            if lc1 else None),
        "LC3_significant": bool(lc3_significant),
        "LC4_topic_closed": bool(lc4),
        "LC4_statement": (
            "上下文于 90 附近饱和/过峰，长上下文议题关闭，S1 撤案（不做 L 网格）"
            if lc4 else None),
        "registration": (
            {"arm": FT_ARM, "note": "LC1 通过：ft 优先，规则文档注册进结算对照（复算级新臂，待前向确认）"}
            if lc1 else "无（LC1 未通过）"),
        "anchor_rerun_values": {
            s: doc["anchor_recheck"]["backtest"][s]["rerun_aer_ew"] for s in SEEDS},
    }

    out = L1_DATA_DIR / "l1_verdict.json"
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== L1 预注册判据 LC1~LC4 一次开封（配对差 = AER 等权，backtest）====")
    print(f"d_ft（L250-ft − L90 s100）      = {d_ft:+.2%}")
    for s in SEEDS:
        print(f"d_zs[{s}]（L250-zs − L90 {s}）    = {d_zs[s]:+.2%}")
    print(f"d_zs 中位                        = {d_zs_med:+.2%}")
    print(f"d_500（L500-zs − L250-zs, s100） = {d_500:+.2%}")
    print(f"[LC1 靶存在] {lc1}（两腿同向 > 0）")
    print(f"[LC2 换骨架] {verdict['LC2_s1_project']} — {verdict['LC2_statement']}")
    print(f"[LC3 显著]   {lc3_significant}（阈值 +26pp；带内只做方向描述）")
    print(f"[LC4 关闭]   {lc4} — {verdict['LC4_statement'] or '未触发'}")
    print(f"[2025H2 描述性] " + " ".join(f"{s}:{d_zs_h2[s]:+.2%}" for s in SEEDS))
    print(f"[注册] {verdict['registration']}")
    print(f"判读落盘 → {out}")

    # —— 图：backtest AER(等权) 条形 ——
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        names = [ANCHOR_TAG[s] for s in SEEDS] + [
            f"{a}_mean" for a in ("L1L250ZS100", "L1L250ZS101", "L1L250ZS102",
                                  "L1L500ZS100", "L1L250FT100")]
        aers = [_aer(doc, "backtest", n) for n in names]
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = ["#08306b"] * 3 + ["#2171b5", "#4292c6", "#6baed6", "#f28e2b", "#e15759"]
        ax.bar(names, aers, color=colors)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
        for i, a in enumerate(aers):
            ax.text(i, a + (0.004 if a >= 0 else -0.012), f"{a:+.1%}", ha="center", fontsize=8)
        ax.set_ylabel("AER (等权基准, with cost)")
        ax.set_title("L1 长上下文先导 — backtest AER(等权) 2026-01-01~2026-07-24（mean 变体）")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG_DIR / "l1_aer_bar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"图落盘 → {FIG_DIR / 'l1_aer_bar.png'}")
    except Exception as e:
        print(f"[warn] 图绘制失败（不阻断判读）：{e}")


if __name__ == "__main__":
    main()
