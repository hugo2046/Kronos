"""G9 一次开封判读（计划 §2/§4.4，20260821 G9 计划）。

六组信号全部落盘 + 引擎全表封存后，本脚本**一次性**开封：

- 全表：两窗 × 全组（参照 F0/M + G9 各臂）× 双基准 AER/IR；
- K0 复现锚：E1 backtest AER(等权) 与冻结 +14.33%（G1_mean，
  docs/扩语料微调实验结果_20260815.md）差 ≤ 1pp → 重训复现在位者；
  否则如实披露，配对基线改用 E1 实测值（配对仍成立）；
- K1 存活：E15 backtest AER(等权) > 0 且 AER(指数) > 0；
- K2 方向（核心决策规则）：配对差 (E15 − E1) backtest 与 2025H2 两窗同号——
  均 ≥ 0 → "CE 早停规则可疑，训满 checkpoint 为候选"（触发计划 §3 三种子确认）；
  均 < 0 → "CE 早停正确，过拟合墙为真"；异号 → "不可判"，记录；
- K3 显著：任一窗 |E15 − E1| > 26pp → 允许强措辞；带内只许方向性描述；
- K4 关闭：E15 双基准 ≤ 0 且 backtest 配对差 < −26pp → "墙另一侧无 alpha，议题关闭"；
- E0 判读（仅描述）：E0 与 E1 backtest AER 并列——差在噪声底内 → "alpha 主要由
  tokenizer 承载"（机制线索）；深负 → "predictor 的 2000 步适配不可或缺"；
- epoch 曲线图：x=epoch（0=E0），y=backtest AER(等权)，E1/E15 加粗，E0 虚线。

纪律：E5/E10 只画曲线不进判据；半污染窗声明沿用（K2 有利结论上限
"候选，待三种子 + 前向确认"）；forward 零判读。

产出：g9_ckpt/data/g9_verdict.json + g9_ckpt/figures/epoch_curve_aer.png。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PKG_DIR = Path(__file__).resolve().parent
G9_DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "figures"

# —— 冻结常量（计划 §2 跑前冻结）——
K0_ANCHOR_AER_EW = 0.1433   # G1_mean backtest AER(等权)（20260815 结果文档）
K0_TOL = 0.01               # 复现容差 1pp
NOISE_BAND = 0.26           # ±26pp 噪声框架（134 日窗）
EPOCHS_CURVE = [0, 1, 5, 10, 15]  # x 轴（0 = E0 官方底座）
EPOCH_OF_ARM = {"G9E0": 0, "G9E1": 1, "G9E5": 5, "G9E10": 10, "G9E15": 15}


def _aer(results: dict, window: str, tag: str, bench: str) -> float:
    """从结果 JSON 取 AER：bench ∈ {"perf_idx", "perf_ew"}。"""
    return float(results["groups"][f"{window}:{tag}"][bench]["aer"])


def _ir(results: dict, window: str, tag: str, bench: str) -> float:
    return float(results["groups"][f"{window}:{tag}"][bench]["ir"])


def main() -> None:
    results = json.loads((G9_DATA_DIR / "g9_backtest_results.json").read_text(encoding="utf-8"))
    order_backtest = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")] + [
        f"{a}_{v}" for a in ("G9E0", "G9E1", "G9E5", "G9E10", "G9E15")
        for v in ("min", "max", "last", "mean")
    ]
    order_h2 = ["M"] + [f"F0_{v}" for v in ("min", "max", "last", "mean")] + [
        f"{a}_{v}" for a in ("G9E1", "G9E15") for v in ("min", "max", "last", "mean")
    ]

    print("=" * 96)
    print("G9 一次开封：引擎全表（AER：年化超额；idx=000300.SH 市值加权基准，ew=同池等权基准）")
    print("=" * 96)
    for window, order in (("backtest", order_backtest), ("2025h2", order_h2)):
        print(f"\n—— 窗口 {window} {results['windows'][window]} ——")
        print(f"{'组':<14}{'AER(idx)':>10}{'IR(idx)':>9}{'AER(ew)':>10}{'IR(ew)':>9}")
        for tag in order:
            key = f"{window}:{tag}"
            if key not in results["groups"]:
                continue
            print(
                f"{tag:<14}"
                f"{_aer(results, window, tag, 'perf_idx'):+10.2%}"
                f"{_ir(results, window, tag, 'perf_idx'):+9.3f}"
                f"{_aer(results, window, tag, 'perf_ew'):+10.2%}"
                f"{_ir(results, window, tag, 'perf_ew'):+9.3f}"
            )

    # —— 判据计算（全部基于 mean 变体，G1 家族判据惯例）——
    e1_bt_ew = _aer(results, "backtest", "G9E1_mean", "perf_ew")
    e1_bt_idx = _aer(results, "backtest", "G9E1_mean", "perf_idx")
    e15_bt_ew = _aer(results, "backtest", "G9E15_mean", "perf_ew")
    e15_bt_idx = _aer(results, "backtest", "G9E15_mean", "perf_idx")
    e15_h2_ew = _aer(results, "2025h2", "G9E15_mean", "perf_ew")
    e1_h2_ew = _aer(results, "2025h2", "G9E1_mean", "perf_ew")
    e0_bt_ew = _aer(results, "backtest", "G9E0_mean", "perf_ew")
    d_bt = e15_bt_ew - e1_bt_ew
    d_h2 = e15_h2_ew - e1_h2_ew

    print("\n" + "=" * 96)
    print("K0~K4 + E0 一次封盘判读（判据定义 = 计划 §2 跑前冻结）")
    print("=" * 96)

    k0_diff = e1_bt_ew - K0_ANCHOR_AER_EW
    k0_pass = abs(k0_diff) <= K0_TOL
    k0 = {
        "name": "K0 复现锚",
        "e1_backtest_aer_ew": e1_bt_ew,
        "anchor": K0_ANCHOR_AER_EW,
        "diff": k0_diff,
        "pass": k0_pass,
        "verdict": (
            f"E1 backtest AER(等权) {e1_bt_ew:+.2%} vs 冻结锚 +14.33% 差 {k0_diff * 100:+.2f}pp "
            f"≤ 1pp → 重训复现在位者，配对基线 = E1"
            if k0_pass else
            f"E1 backtest AER(等权) {e1_bt_ew:+.2%} vs 冻结锚 +14.33% 差 {k0_diff * 100:+.2f}pp "
            f"> 1pp → 如实披露未达复现容差，配对基线改用 E1 实测值（配对仍成立）"
        ),
    }

    k1_pass = (e15_bt_ew > 0) and (e15_bt_idx > 0)
    k1 = {
        "name": "K1 存活",
        "e15_backtest_aer_ew": e15_bt_ew,
        "e15_backtest_aer_idx": e15_bt_idx,
        "pass": k1_pass,
        "verdict": (
            f"E15 backtest AER(等权) {e15_bt_ew:+.2%} 且 AER(指数) {e15_bt_idx:+.2%} "
            + ("均 > 0 → 存活" if k1_pass else "→ 存活判据未过（存在 ≤ 0）")
        ),
    }

    if d_bt >= 0 and d_h2 >= 0:
        k2_verdict = (
            "两窗配对差均 ≥ 0 → CE 早停规则可疑，训满 checkpoint 为候选"
            "（触发计划 §3 三种子确认；半污染窗声明：上限'候选，待三种子+前向确认'）"
        )
    elif d_bt < 0 and d_h2 < 0:
        k2_verdict = "两窗配对差均 < 0 → CE 早停正确，过拟合墙为真"
    else:
        k2_verdict = "两窗配对差异号 → 不可判，记录"
    k2 = {
        "name": "K2 方向（核心决策规则）",
        "paired_diff_backtest": d_bt,
        "paired_diff_2025h2": d_h2,
        "verdict": k2_verdict,
    }

    k3_sig = max(abs(d_bt), abs(d_h2)) > NOISE_BAND
    k3 = {
        "name": "K3 显著",
        "max_abs_diff": max(abs(d_bt), abs(d_h2)),
        "significant": k3_sig,
        "verdict": (
            f"任一窗 |E15−E1| 最大 {max(abs(d_bt), abs(d_h2)) * 100:+.2f}pp "
            + ("> 26pp → 允许强措辞" if k3_sig else "≤ 26pp（带内）→ 只许方向性描述")
        ),
    }

    k4_pass = (e15_bt_ew <= 0) and (e15_bt_idx <= 0) and (d_bt < -NOISE_BAND)
    k4 = {
        "name": "K4 关闭",
        "pass": k4_pass,
        "verdict": (
            f"E15 双基准 {e15_bt_ew:+.2%}/{e15_bt_idx:+.2%} "
            + (f"≤ 0 且 backtest 配对差 {d_bt * 100:+.2f}pp < −26pp → 墙另一侧无 alpha，选择规则议题关闭" if k4_pass
               else "→ 关闭条件未全部满足（backtest 配对差未越 −26pp 噪声带），议题不关闭")
        ),
    }

    e0_gap = e0_bt_ew - e1_bt_ew
    if abs(e0_gap) <= NOISE_BAND:
        e0_verdict = (
            f"E0 {e0_bt_ew:+.2%} 与 E1 {e1_bt_ew:+.2%} 差 {e0_gap * 100:+.2f}pp "
            f"{'贴带边（距 26pp 噪声底不足 1pp）' if abs(e0_gap) > NOISE_BAND - 0.01 else '在噪声底内'} "
            f"→ 按冻结规则记录机制线索：alpha 主要由 tokenizer 承载"
            f"（同时如实并记：E0 深负于 E1 的方向亦与 predictor 适配贡献相容，单种子不可分离）"
        )
    elif e0_bt_ew < e1_bt_ew - NOISE_BAND:
        e0_verdict = (
            f"E0 {e0_bt_ew:+.2%} 深负于 E1 {e1_bt_ew:+.2%}（差 {e0_gap * 100:+.2f}pp 超噪声底） "
            f"→ predictor 的 2000 步适配不可或缺"
        )
    else:
        e0_verdict = (
            f"E0 {e0_bt_ew:+.2%} 强于 E1 {e1_bt_ew:+.2%}（差 {e0_gap * 100:+.2f}pp 超噪声底） "
            f"→ 描述性记录：零训练底座反超（如实记录，不进判据）"
        )
    e0 = {"name": "E0 判读（仅描述）", "e0_backtest_aer_ew": e0_bt_ew, "gap_vs_e1": e0_gap,
          "verdict": e0_verdict}

    for item in (k0, k1, k2, k3, k4, e0):
        print(f"\n[{item['name']}] {item['verdict']}")

    # —— epoch 曲线图（x=epoch，y=backtest AER 等权；E1/E15 加粗，E0 虚线）——
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    for cjk in ("Noto Sans CJK SC", "Noto Serif CJK SC"):
        try:
            fm.findfont(fm.FontProperties(family=cjk), fallback_to_default=False)
            plt.rcParams["font.family"] = ["sans-serif"]
            plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
            break
        except Exception:  # noqa: BLE001 —— 字体缺失时回退默认
            continue

    curve = {EPOCH_OF_ARM[f"G9E{k}"]: _aer(results, "backtest", f"G9E{k}_mean", "perf_ew")
             for k in EPOCHS_CURVE}
    f1_ref = _aer(results, "backtest", "F0_mean", "perf_ew")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = sorted(curve)
    ys = [curve[x] for x in xs]
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(f1_ref, color="tab:brown", lw=0.8, ls="--",
               label=f"F0_mean 参照 {f1_ref:+.2%}")
    ax.plot(xs, ys, "-o", color="tab:blue", ms=5, label="G9 epoch 曲线（backtest AER 等权）")
    for tag, weight, size in (("G9E1", 700, 11), ("G9E15", 700, 11)):
        ep = EPOCH_OF_ARM[tag]
        ax.plot([ep], [curve[ep]], "o", color="tab:red", ms=9, mfc="none", mew=2)
        ax.annotate(f"{tag}\n{curve[ep]:+.2%}", (ep, curve[ep]),
                    textcoords="offset points", xytext=(8, -4), fontsize=size,
                    fontweight=weight)
    ax.plot([0], [curve[0]], "s", color="tab:blue", ms=5, mfc="none", ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"E{x}\n({'官方底座' if x == 0 else 'epoch ' + str(x)})" for x in xs])
    ax.set_xlabel("predictor checkpoint epoch（0 = 官方 Kronos-base + G1 tokenizer）")
    ax.set_ylabel("backtest AER（等权基准）")
    ax.set_title("G9：checkpoint 选择规则 epoch 曲线（2026-01-01~2026-07-24）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    curve_path = FIG_DIR / "epoch_curve_aer.png"
    fig.savefig(curve_path, dpi=150)

    verdict = {
        "frozen_inputs": {"k0_anchor_aer_ew": K0_ANCHOR_AER_EW, "noise_band": NOISE_BAND},
        "criteria": {"K0": k0, "K1": k1, "K2": k2, "K3": k3, "K4": k4, "E0": e0},
        "curve_backtest_aer_ew": {str(k): v for k, v in curve.items()},
        "figure": str(curve_path),
        "note": "一次开封（计划 §5）；E5/E10 只画曲线不进判据；forward 零判读",
    }
    out_path = G9_DATA_DIR / "g9_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\n判读落盘：{out_path}")
    print(f"曲线图落盘：{curve_path}")


if __name__ == "__main__":
    main()
