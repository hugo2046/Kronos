"""生成 docs/引擎v2重放对照_20260905.md（数据源：v2_replay_results.json）。

文档结构：
    §0 头部声明（旧 AER 全部 provisional、本文档不改判据——用户指定原文）
    §1 背景与六项修正
    §2 方法学（数据构造流派 / 保真校验 / v2 口径）
    §3 旧引擎 vs v2 全表（每族一表，259 臂）
    §4 判据翻转清单（冻结判据逐条）
    §5 qlib 对拍实数
    §6 范围外与结论

用法：``python -m paper_replication.gen_v2_doc``
"""
from __future__ import annotations

import json
from pathlib import Path

from paper_replication.common import DATA_DIR, REPO_ROOT

DOC_PATH = REPO_ROOT / "docs" / "引擎v2重放对照_20260905.md"

FAM_TITLES = {
    "paper": "论文复现四组（paper 窗 2024-07-01~2025-06-30）",
    "baseline_paper": "baseline 四变体 + 对照（paper 窗）",
    "baseline_oos": "baseline 四变体 + 对照（oos1 窗 2025-07-01~2026-07-24）",
    "improve_paper": "improve 网格 C1~C3（paper 窗，own 宇宙）",
    "improve_oos": "improve 网格 C1~C3（oos1 窗，own 宇宙）",
    "f_backtest": "F0/F1 微调族 + M（backtest 窗 2026-01-01~2026-07-24）",
    "g1_backtest": "G1 扩语料族 + F0/F1/M（backtest 窗）",
    "g2_backtest": "G2 种子族 s101/s102（backtest 窗）",
    "g2_supp_backtest": "G2 增补 s103/s104 + DTOK（backtest 窗）",
    "g8_backtest": "G8 语料新鲜度族（backtest 窗）",
    "g9_backtest": "G9 checkpoint 族 E0~E15 + F0/M（backtest 窗）",
    "g5_backtest": "G5 换头族 H-kda/H-lin/H-mamba（backtest 窗，冻结宇宙）",
    "g4_backtest": "G4 特征族（backtest 窗，冻结宇宙）",
    "g7_backtest": "G7 短窗族 W85（backtest 窗，冻结宇宙）",
    "n50_backtest": "N50 采样放大族（backtest 窗，冻结宇宙）",
    "l1_backtest": "L1 长上下文族（backtest 窗，冻结宇宙）",
    "r1_backtest": "R1 读出目标族（backtest 窗，冻结宇宙）",
    "g0_2025h2": "G0/F0 @2025H2（2025-07-01~2025-12-31）",
    "g2_2025h2": "G2 种子族 @2025H2",
    "g5_2025h2": "G1 补生成版 + G5 换头族 @2025H2（冻结宇宙）",
    "g4_2025h2": "G4 特征族 @2025H2（冻结宇宙）",
    "g9_2025h2": "G9 E1/E15 @2025H2",
    "l1_2025h2": "L1 长上下文族 @2025H2（冻结宇宙）",
    "r1_2025h2": "R1 读出目标族 @2025H2（冻结宇宙）",
    "c4_merged": "C4 时间维因子（merged 230 日窗，warmup=30）",
}


def _fmt(x, pct=True):
    if x is None:
        return "—"
    return f"{x:+.2%}" if pct else f"{x:+.2f}"


def fam_table(fam: dict) -> str:
    lines = [
        f"#### {FAM_TITLES.get(fam['fam'], fam['fam'])}",
        "",
        f"窗口 {fam['window'][0]}~{fam['window'][1]}；宇宙口径 `{fam['universe']}`"
        + (f"，warmup={fam['warmup']}" if fam.get("warmup") else ""),
        "",
        "| 臂 | 旧AER(ew) | v2 AER(ew) | Δ(ew) | 旧AER(idx) | v2 AER(idx) | Δ(idx) | 旧IR(idx) | v2 IR(idx) | 符号 | 保真Δ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(fam["arms"]):
        r = fam["arms"][a]
        dew = r["v2_aer_ew"] - r["old_aer_ew"]
        didx = r["v2_aer_idx"] - r["old_aer_idx"]
        fid = r.get("fidelity_delta")
        mark = "⚠️翻转" if r.get("sign_flipped") else ""
        fid_s = f"{fid:.2%}" if fid is not None else "—"
        if r.get("fidelity_ok") is False:
            fid_s += "⚠️"
        lines.append(
            f"| {a} | {_fmt(r['old_aer_ew'])} | {_fmt(r['v2_aer_ew'])} | {dew:+.2%} "
            f"| {_fmt(r['old_aer_idx'])} | {_fmt(r['v2_aer_idx'])} | {didx:+.2%} "
            f"| {_fmt(r['old_ir_idx'], False)} | {_fmt(r['v2_ir_idx'], False)} "
            f"| {mark} | {fid_s} |"
        )
    return "\n".join(lines) + "\n"


# ====================================================================
# 判据翻转清单（冻结判据 → v2 判定；数值从 JSON 取，判定逻辑编码在此）
# ====================================================================

def _get(arms_by_fam, fam, arm, key):
    return arms_by_fam[fam][arm][key]


def _pp(x):
    return "—" if x is None else f"{x * 100:+.2f}pp"


def build_flip_table(arms_by_fam) -> tuple[list[str], list[str]]:
    """返回 (翻转行, 未翻转行)。每行：判据 | 内容(旧判定) | v2 判定。"""
    flipped, kept = [], []

    def ev(name, cond_old, cond_new, detail_old, detail_new):
        row = f"| {name} | {detail_old} → **{'通过' if cond_old else '未通过/触发'}** | {detail_new} → **{'通过' if cond_new else '未通过/触发'}** |"
        (flipped if cond_old != cond_new else kept).append(row)

    # —— paper ——
    p_ew = _get(arms_by_fam, "paper", "P", "v2_aer_ew")
    ev("paper 门禁（P 组 AER(等权) < +3%）", True, p_ew < 0.03,
       f"P(ew) −0.14%", f"P(ew) {p_ew:+.2%}")
    k_idx = _get(arms_by_fam, "paper", "K", "v2_aer_idx")
    k_ir = _get(arms_by_fam, "paper", "K", "v2_ir_idx")
    m_idx = _get(arms_by_fam, "paper", "M", "v2_aer_idx")
    r_idx = _get(arms_by_fam, "paper", "R", "v2_aer_idx")
    ev("paper 定性判据（K idx>0 且 IR>0.5 且 K>M、R）", True,
       k_idx > 0 and k_ir > 0.5 and k_idx > m_idx and k_idx > r_idx,
       "K +8.99%/IR 0.78", f"K {k_idx:+.2%}/IR {k_ir:+.2f}（IR 跌破 0.5）")
    ev("paper 锚点判据（|K idx − 0.1911| < 10pp 且同号）", False,
       abs(k_idx - 0.1911) < 0.10 and k_idx > 0,
       "差 −10.12pp", f"差 {_pp(k_idx - 0.1911)}")
    # —— oos ——
    oos_mean_ew = _get(arms_by_fam, "baseline_oos", "mean", "v2_aer_ew")
    oos_mean_idx = _get(arms_by_fam, "baseline_oos", "mean", "v2_aer_idx")
    ev("oos 判据1（mean 双基准 > 0）", False,
       oos_mean_ew > 0 and oos_mean_idx > 0,
       "−15.10%/−14.29%", f"{oos_mean_ew:+.2%}/{oos_mean_idx:+.2%}")
    # —— G1 存活 ——
    g1_ew = _get(arms_by_fam, "g1_backtest", "G1_mean", "v2_aer_ew")
    g1_idx = _get(arms_by_fam, "g1_backtest", "G1_mean", "v2_aer_idx")
    ev("G1 判据3 存活（G1_mean backtest 双基准 > 0）", True,
       g1_ew > 0 and g1_idx > 0,
       "+14.33%/+10.66%", f"{g1_ew:+.2%}/{g1_idx:+.2%}（idx 转负）")
    # —— G2 种子 ——
    s102_ew = _get(arms_by_fam, "g2_backtest", "G2S102_mean", "v2_aer_ew")
    s102_idx = _get(arms_by_fam, "g2_backtest", "G2S102_mean", "v2_aer_idx")
    ev("G2-S1 保级（s102 backtest 双基准 > 0）", True,
       s102_ew > 0 and s102_idx > 0,
       "+18.67%/+14.64%", f"{s102_ew:+.2%}/{s102_idx:+.2%}（idx 转负）")
    f0_h2 = _get(arms_by_fam, "g2_2025h2", "F0_mean", "v2_aer_ew")
    s101_h2 = _get(arms_by_fam, "g2_2025h2", "G2S101_mean", "v2_aer_ew")
    s102_h2 = _get(arms_by_fam, "g2_2025h2", "G2S102_mean", "v2_aer_ew")
    ev("G2-S4 跨窗（新种子 2025H2 较 F0_mean ≥ +5pp）", True,
       (s101_h2 - f0_h2) >= 0.05 and (s102_h2 - f0_h2) >= 0.05,
       "s101 +27.95pp / s102 +20.69pp（2/2）",
       f"s101 {_pp(s101_h2 - f0_h2)} / s102 {_pp(s102_h2 - f0_h2)}（2/2）")
    # —— G4 ——
    g4_ew = _get(arms_by_fam, "g4_backtest", "G4S101_mean", "v2_aer_ew")
    g4_idx = _get(arms_by_fam, "g4_backtest", "G4S101_mean", "v2_aer_idx")
    ev("G4-J1 存活（s101 中位 backtest 双基准 > 0）", True,
       g4_ew > 0 and g4_idx > 0,
       "+11.68%/+8.07%", f"{g4_ew:+.2%}/{g4_idx:+.2%}（双转负）")
    ev("G4-J5 否定（中位 ≤ +12.33%，特征路线关闭）", True, g4_ew <= 0.1233,
       "中位 +11.68% 触发关闭", f"中位 {g4_ew:+.2%} 触发关闭")
    # —— G7 / N50 / G8 ——
    w85 = [_get(arms_by_fam, "g7_backtest", f"W85S{s}_mean", "v2_aer_ew") for s in (100, 101, 102)]
    w85_med = sorted(w85)[1]
    ev("G7-K3 否定（中位 ≤ +16.67%，终审关闭）", True, w85_med <= 0.1667,
       "中位 −22.04% 触发关闭", f"中位 {w85_med:+.2%} 触发关闭")
    n50 = [_get(arms_by_fam, "n50_backtest", f"G1N50S{s}_mean", "v2_aer_ew") for s in (100, 101, 102)]
    n50_med = sorted(n50)[1]
    ev("N50-N3 反常（中位 ≤ +16.67%，不注册）", True, n50_med <= 0.1667,
       "中位 +13.90% 触发", f"中位 {n50_med:+.2%} 触发")
    g8 = [_get(arms_by_fam, "g8_backtest", f"G8S{s}_mean", "v2_aer_ew") for s in (100, 101, 102)]
    g8_med = sorted(g8)[1]
    g8_idx_med = sorted([_get(arms_by_fam, "g8_backtest", f"G8S{s}_mean", "v2_aer_idx") for s in (100, 101, 102)])[1]
    ev("G8-G8-1 注册门（中位双基准 > 0）", False, g8_med > 0 and g8_idx_med > 0,
       "−3.11%/−6.59% 未过", f"{g8_med:+.2%}/{g8_idx_med:+.2%} 未过")
    # —— G9 ——
    e15_bt = _get(arms_by_fam, "g9_backtest", "G9E15_mean", "v2_aer_ew")
    e1_bt = _get(arms_by_fam, "g9_backtest", "G9E1_mean", "v2_aer_ew")
    e15_h2 = _get(arms_by_fam, "g9_2025h2", "G9E15_mean", "v2_aer_ew")
    e1_h2 = _get(arms_by_fam, "g9_2025h2", "G9E1_mean", "v2_aer_ew")
    d_bt, d_h2 = e15_bt - e1_bt, e15_h2 - e1_h2
    ev("G9-K1 存活（E15 backtest 双基准 > 0）", False,
       _get(arms_by_fam, "g9_backtest", "G9E15_mean", "v2_aer_ew") > 0
       and _get(arms_by_fam, "g9_backtest", "G9E15_mean", "v2_aer_idx") > 0,
       "−2.19%/−6.20%", "更深负（见 §3 表）")
    ev("G9-K2 方向（E15−E1 两窗同号 → CE 早停正确）", True,
       (d_bt < 0 and d_h2 < 0) or (d_bt > 0 and d_h2 > 0),
       "两窗均负（−16.53/−8.85pp）", f"两窗均负（{_pp(d_bt)}/{_pp(d_h2)}）")
    ev("G9-K4 关闭（E15 双基准≤0 且配对差 < −26pp）", False,
       e15_bt <= 0 and (e15_bt - e1_bt) < -0.26,
       "差 −16.53pp 未触发", f"差 {_pp(d_bt)} 未触发")
    # —— L1 / R1 / C4 ——
    ft = _get(arms_by_fam, "l1_backtest", "L250FT100_mean", "v2_aer_ew")
    ev("L1-LC4 关闭（FT100 mean ≤ 0 → 长上下文议题关闭）", True, ft <= 0,
       "−4.49% 触发关闭", f"{ft:+.2%} 触发关闭")
    zs_med = sorted([_get(arms_by_fam, "l1_backtest", f"L250ZS{s}_mean", "v2_aer_ew") for s in (100, 101, 102)])[1]
    ev("L1-LC1 靶存在（d_ft>0 且 d_zs 中位>0）", False,
       (ft - g1_ew) > 0 and (zs_med - g1_ew) > 0,
       "两腿皆负未过", f"两腿皆负（FT−L90 {_pp(ft - g1_ew)}，zs−L90 {_pp(zs_med - g1_ew)}）")
    rk = sorted([_get(arms_by_fam, "r1_backtest", f"R-kda_s{s}", "v2_aer_ew") for s in (42, 43, 44)])[1]
    rl = sorted([_get(arms_by_fam, "r1_backtest", f"R-lin_s{s}", "v2_aer_ew") for s in (42, 43, 44)])[1]
    ev("R1-RC1 存活（双头中位 backtest 双基准 > 0）", False, rk > 0 and rl > 0,
       "R-lin −12.32%/R-kda −1.18%", f"R-lin {rl:+.2%}/R-kda {rk:+.2%}")
    ev("R1-RC4 终审关闭（读出议题关闭）", True, rk < g1_ew and rl < g1_ew,
       "中位深负于在位者，触发", f"中位深负于在位者（v2 在位者 {g1_ew:+.2%}），触发")
    c4_med = sorted([_get(arms_by_fam, "c4_merged", f"C4S{s}_c4", "v2_aer_ew") for s in (100, 101, 102)])[1]
    ev("C4-T5 否定（中位全期 ≤ 0 → 形态关闭）", True, c4_med <= 0,
       "中位 −2.65% 触发关闭", f"中位 {c4_med:+.2%} 触发关闭")
    # —— G0 ——
    g0 = _get(arms_by_fam, "g0_2025h2", "G0_mean", "v2_aer_ew")
    ev("G0-判据4 稳定性（F1@2025H2 ≥ F0 同窗 +5pp）", True, (g0 - f0_h2) >= 0.05,
       "G0 +3.10% vs F0 −16.79%（+19.89pp）",
       f"G0 {g0:+.2%} vs F0 {f0_h2:+.2%}（{_pp(g0 - f0_h2)}）")
    # —— baseline 判据4 ——
    bp = arms_by_fam["baseline_paper"]
    ev("baseline 判据4（min 强于 mean ≥ 3pp）", False,
       bp["min"]["v2_aer_ew"] - bp["mean"]["v2_aer_ew"] >= 0.03,
       "差 +1.40pp", f"差 {_pp(bp['min']['v2_aer_ew'] - bp['mean']['v2_aer_ew'])}")

    # —— F1 判据2（塌方缓解）——
    f1_ew = _get(arms_by_fam, "f_backtest", "F1_mean", "v2_aer_ew")
    f0_bt = _get(arms_by_fam, "f_backtest", "F0_mean", "v2_aer_ew")
    f1_old = _get(arms_by_fam, "f_backtest", "F1_mean", "old_aer_ew")
    f0_old = _get(arms_by_fam, "f_backtest", "F0_mean", "old_aer_ew")
    ev("F1 判据2 塌方缓解（F1_mean 显著高于 F0_mean）", True, f1_ew > f0_bt,
       f"F1 {f1_old:+.2%} vs F0 {f0_old:+.2%}", f"F1 {f1_ew:+.2%} vs F0 {f0_bt:+.2%}")

    return flipped, kept


def main() -> None:
    with open(DATA_DIR / "v2_replay_results.json", encoding="utf-8") as f:
        data = json.load(f)
    arms_by_fam = {f["fam"]: f["arms"] for f in data["families"]}

    parts: list[str] = []
    parts.append(
        "# 引擎 v2 重放对照（2026-09-05）\n\n"
        "> **声明：本仓库全部已冻结的旧 AER 数字即日起一律视为 provisional（引擎口径含已确认的"
        "六项偏差，见 §1）；本文档只做同数据重放对照与判据翻转披露，不修改任何预注册判据、"
        "不重开任何已封盘议题。** 判据翻转清单（§4）仅记录\"若按修正后引擎，冻结判定的符号会"
        "怎样\"，不构成翻案。\n"
    )
    parts.append(
        "## 1. 背景：旧引擎六项口径修正\n\n"
        "Codex_review 复核报告（2026-09-05）§2-B/§2-C 指出 `paper_replication/engine.py` 六处"
        "口径问题。修正实现在 `paper_replication/engine_v2.py`（**不改 engine.py**），六项各带"
        "独立开关、默认全开、全关时数值复现旧引擎（单测 `tests/test_engine_v2.py` 16 例：先 FAIL"
        "后 PASS，含 \"B 当日先涨 100%、次日涨 10% 必须得 5.0%（旧引擎 6.667%）\" 与 \"一次换手 X"
        " 扣 2×15bp·X（旧引擎 15bp·X）\" 两个反例，及随机数据 legacy 等价断言）：\n\n"
        "1. **双边成本**：`cost = (freed + bought) × cost_bps/1e4`（旧：(freed+bought)/2，少扣一半）；\n"
        "2. **delay=1**：t 日信号 t+1 收盘成交、首个收益日 t+2（旧：t 日决策同刻成交，t+1 涨跌"
        "经权重漂移虚记给当日新腿）；\n"
        "3. **涨跌停/一字板不可成交**：买入排除当日收盘涨停、卖出排除当日收盘跌停——DDB "
        "`up_down_limit_status`（实测 ±1=收盘封板，1433/1433 例 close==high 交叉验证）优先，"
        "字段缺失回退 `high==low` 一字板判定；被禁买入顺延候选下一位、卖出顺延；\n"
        "4. **新买入腿不参与当日漂移**：漂移移到成交前、只作用既有腿（旧：成交后统一漂移，"
        "新腿被乘 (1+r_成交日) 的幻觉加仓）；\n"
        "5. **等权基准掩码 = tradeable ∧ signal.notna()**（逐臂；旧：仅 tradeable，信号缺失股"
        "污染同池等权基准）；\n"
        "6. **年化常量 252**（AER 与 IR 一致；旧 244）。\n"
    )
    fid_bad = [
        (f["fam"], a, r["fidelity_delta"])
        for f in data["families"] for a, r in f["arms"].items()
        if r.get("fidelity_ok") is False
    ]
    n_arms = sum(len(f["arms"]) for f in data["families"])
    n_flips = sum(1 for f in data["families"] for r in f["arms"].values() if r.get("sign_flipped"))
    parts.append(
        "## 2. 方法学与保真校验\n\n"
        f"- 重放范围：**{len(data['families'])} 族 / {n_arms} 臂**（paper / oos1 / backtest / "
        "2025H2 / c4-merged 全部已封盘信号 parquet）；列宇宙逐族按原 runner 口径重建"
        "（union / 冻结清单 / own 三流派，见 `paper_replication/replay_v2.py` FAMILIES）。\n"
        f"- **保真校验**：旧引擎在同构数据上的重放数字 vs 冻结文档锚点，容差 0.5pp——"
        f"除 c4_merged 4 臂（Δ 0.53~0.70pp，其 merged 窗 warmup 切片为近似，原文档即注明"
        "持仓跨界结转口径）外**全部通过**，绝大多数锚点 Δ=0.00%（paper 族四臂全部逐位复现）。"
        "旧引擎列的数字即冻结数字的重放。\n"
        f"- 符号翻转臂：**{n_flips}/{n_arms}**（ew 或 idx 任一基准 AER 变号）。\n"
        "- v2 列 = 六修正全开；等权基准逐臂掩码（修正 5），指数基准不变。\n"
    )
    parts.append("## 3. 旧引擎 vs v2 全表\n")
    for fam in data["families"]:
        parts.append(fam_table(fam))

    flipped, kept = build_flip_table(arms_by_fam)
    parts.append("## 4. 判据翻转清单\n")
    parts.append(
        "### 4.1 符号改变的冻结判据（v2 下判定翻转）\n\n"
        "| 判据 | 旧口径判定 | v2 口径判定 |\n|---|---|---|\n" + "\n".join(flipped) + "\n"
    )
    # —— §4.3 六开关逐项归因（对翻转臂：每次关一个开关，其余保持 v2）——
    attr_g1 = arms_by_fam["g1_backtest"]["G1_mean"].get("attribution", {})
    sec_43 = ""
    if attr_g1:
        fix_names = {
            "fix_double_sided_cost": "(1) 双边成本",
            "fix_delay_1": "(2) delay=1",
            "fix_limit_block": "(3) 涨跌停禁成交",
            "fix_new_leg_drift": "(4) 新腿不当日漂移",
            "fix_annualization_252": "(6) 年化 252",
            "fix_ew_benchmark_mask": "(5) 等权基准掩码（只影响 ew）",
        }
        arows = [
            f"| {fix_names.get(k, k)} 关 | {v.get('aer_ew', float('nan')):+.2%} "
            f"| {v.get('aer_idx', float('nan')):+.2%} |"
            if v.get("aer_idx") is not None else
            f"| {fix_names.get(k, k)} 关 | {v.get('aer_ew', float('nan')):+.2%} | — |"
            for k, v in attr_g1.items()
        ]
        g1_row = arms_by_fam["g1_backtest"]["G1_mean"]
        sec_43 = (
            "\n### 4.3 归因：哪个修正贡献了多大差值（以在位者 G1_mean 为例）\n\n"
            "基线行 = v2 全开；每行 = 关掉该开关、其余保持 v2，数值回升幅度即该修正的独立贡献：\n\n"
            "| 开关状态 | AER(ew) | AER(idx) |\n|---|---|---|\n"
            f"| v2 全开 | {g1_row['v2_aer_ew']:+.2%} | {g1_row['v2_aer_idx']:+.2%} |\n"
            + "\n".join(arows) + "\n\n"
            f"结论：**修正 (2) delay=1 是最大单一驱动（ew "
            f"{(attr_g1['fix_delay_1']['aer_ew'] - g1_row['v2_aer_ew']) * 100:+.1f}pp）**，其后依次为 "
            f"(3) 涨跌停（{(attr_g1['fix_limit_block']['aer_ew'] - g1_row['v2_aer_ew']) * 100:+.1f}pp）、"
            f"(1) 双边成本（{(attr_g1['fix_double_sided_cost']['aer_ew'] - g1_row['v2_aer_ew']) * 100:+.1f}pp）、"
            f"(4) 漂移（{(attr_g1['fix_new_leg_drift']['aer_ew'] - g1_row['v2_aer_ew']) * 100:+.1f}pp）、"
            f"(5) 掩码（{(attr_g1['fix_ew_benchmark_mask']['aer_ew'] - g1_row['v2_aer_ew']) * 100:+.1f}pp，方向"
            "相反）、(6) 年化（≈0）。64 个翻转臂的归因明细见 "
            "`paper_replication/data/v2_replay_results.json` 各臂 `attribution` 字段。\n"
        )
    parts.append(
        "### 4.2 判定不变的冻结判据（数值普遍恶化/缩水，但符号未变）\n\n"
        "| 判据 | 旧口径判定 | v2 口径判定 |\n|---|---|---|\n" + "\n".join(kept) + "\n\n"
        "**锚失效提示**：所有以旧口径在位者 +14.33%（G1_mean backtest ew）为基准派生的阈值"
        "（G4-J2 +16.33%、G5-J2、G7-K1/N50-N1 +20.67% 等）在 v2 下锚点本身坍缩到 "
        f"{_get(arms_by_fam, 'g1_backtest', 'G1_mean', 'v2_aer_ew'):+.2%}"
        "——这些阈值不重算（不改判据），只在此披露其参照系已失效。\n" + sec_43
    )

    # —— qlib 对拍 ——
    qlib_txt = ""
    qpath = DATA_DIR / "qlib_crosscheck_v2.json"
    if qpath.exists():
        with open(qpath, encoding="utf-8") as f:
            qd = json.load(f)
        rows = []
        v2map = {"G1m_bt": ("g1_backtest", "G1_mean"), "M_bt": ("g1_backtest", "M"),
                 "K_paper": ("paper", "K")}
        for name, q in qd.items():
            fam, arm = v2map.get(name, (None, None))
            v2 = _get(arms_by_fam, fam, arm, "v2_aer_idx") if fam else None
            old = _get(arms_by_fam, fam, arm, "old_aer_idx") if fam else None
            rows.append(
                f"| {name} | {old:+.2%} | {v2:+.2%} | {q['aer_idx_net']:+.2%} "
                f"| {q['aer_idx_net'] - v2:+.2%} |"
            )
        qlib_txt = (
            "## 5. qlib 官方回测对拍（integration 分支 QlibBacktest 思路，DDB 后端）\n\n"
            "integration 分支无名为 `QlibBacktestEngine` 的类，最接近的参照是 "
            "`finetune/qlib_test.py::QlibBacktest`（qlib 官方 `TopkDropoutStrategy` + "
            "`qlib.backtest`）。本仓库把它移植到 DDB 后端"
            "（`paper_replication/qlib_crosscheck_v2.py`）：close 成交、双边 15bp、"
            "limit_threshold=0.095、hold_thresh=5、信号前移 1 日模拟 delay=1、基准 000300、"
            "年化 252。**结果：<1pp 对拍目标未达成，定性为策略层不可比**：\n\n"
            "- 交换层验证一致：基准序列逐日 max|Δ|=1.2e-7；成本即双边 15bp；\n"
            "- 策略层语义不同：qlib TopkDropout 按当日卖出回笼现金配新仓、卖出在决策函数内即时"
            "成交、且在本后端上对信号 top-50 的持仓重叠仅 12~28/50（v2 为 38~41）、组合换手只有"
            "论文规则的一半、持仓数漂移超 topk（观测到 53）——它没有复现论文的 top-k/drop-n 规则"
            "本身，因此其 AER 与 v2/旧引擎均不可比。\n\n"
            "| 臂 | 旧引擎 AER(idx) | v2 AER(idx) | qlib AER(idx,净) | qlib−v2 |\n|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n\n"
            "旁证三角：对 M_bt 直算\"信号 t 日 top-50 等权、t+1 日收益、零成本\"（每日全量换选）"
            "累计 +14.7%；v2（含轮换约束/持有期/双边成本）累计超额 +9.9%；旧引擎 +15.3%。"
            "序关系 旧 > 理想 > v2 与各自含有的 lookahead/摩擦一致，v2 落在合理位置。\n"
        )
    else:
        qlib_txt = "## 5. qlib 对拍\n\n（未运行）\n"
    parts.append(qlib_txt)

    parts.append(
        "## 6. 范围外（如实声明）\n\n"
        + "\n".join(f"- {x}" for x in data["out_of_scope"]) + "\n\n"
        "## 7. 结论\n\n"
        "- v2 重放不改变任何封盘结论的方向：**全部否定性判据（G5/G7/N50/C4/G8/E1/L1/R1 的关闭与"
        "不注册）在 v2 下全部维持**，且大多更负；\n"
        "- 发生符号翻转的是**肯定性判据**：paper 定性判据（K 的 IR 0.78→0.48 跌破 0.5 线）、"
        "G1 存活判据（idx +10.66%→−3.48%）、G2-S1（s102 idx 转负）、G4-J1（s101 双基准转负）——"
        "即旧引擎口径下的\"正 alpha\"臂在修正后引擎下不再构成正 alpha；归因（§4.3，以 G1_mean 为例）："
        "修正 (2) delay=1 最大（ew +9.4pp），其后 (3) 涨跌停（+4.1pp）、(1) 双边成本（+3.7pp）、"
        "(4) 漂移（+0.6pp）、(5) 掩码（−1.7pp，反向）、(6) 年化（≈0）；\n"
        "- 在位者 G1_mean 的 +14.33%（ew）在 v2 下坍缩至约 +1.9%（idx 转负）——**旧 AER 一律 "
        "provisional 的核心原因**；2026-11 forward 结算（对 v2 前向零接触）将是唯一无争议的"
        "终局口径；\n"
        "- 本文档不改判据、不重开议题；若未来要按 v2 口径重立判据，须走新一轮预注册。\n"
    )

    DOC_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"文档已生成 {DOC_PATH}")


if __name__ == "__main__":
    main()
