"""七臂 + C1/C2/C3 结算执行器（forward结算计划_20260818.md §3/§4 的代入式实现）。

流程（§4 顺序，审计先于任何绩效计算）：
1. 可结算交易日 ≥60 断言（H=10 口径）；
2. 完整性审计（§3.3）+ 登记信号 vs 第二存储抽样对拍（≥5 日 × 3 臂）；
3. 信号装配——登记级：G1 三种子四变体/M/C1/C2/C3（登记列确定性函数，
   C2 另做迟补日剔除敏感性 C2_excl_late）；复算级：F0/F1/B3 重放 + R1 组装；
4. 引擎全表（全数呈现无挑选；判据只在 mean）；
5. §3 判据**一次性**代入 → G1 主判据路由关闭/滚动分支；
6. 双轨措辞文档（决策栏=冻结判据代入；证据栏=±26pp 噪声底折扣评估）。

引擎与数据源全部注入（合成演习 or 真实结算），本模块零真实路径硬编码。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from settlement.audit import audit_registry
from settlement.recompute import generate_arm_signals
from settlement.rules import derive_c1_day, derive_c2_day, derive_c3_day, r1_assemble
from settlement.template import (
    BRANCH_CLOSE,
    BRANCH_ROLLING,
    render_settlement_doc,
)

SEEDS = (100, 101, 102)
VARIANTS = ("last", "mean", "max", "min")
WINDOW = "forward"
MIN_SETTLEABLE_DAYS = 60
CROSS_CHECK_ARMS = ("G1_s100", "G1_s101", "G1_s102", "M")
CROSS_CHECK_N_DAYS = 5

GRADES = {
    **{f"G1_s{s}": "登记级" for s in SEEDS},
    "C1": "登记级", "C2": "登记级", "M": "登记级", "C3": "登记级",
    "F0": "复算级", "F1": "复算级", "B3": "复算级", "R1": "复算级",
}
ARM_ORDER = (
    [f"G1_s{s}" for s in SEEDS] + ["C1", "C2", "C2_excl_late", "C3", "M"]
    + ["F0", "F1", "R1", "B3"]
)


# ---------------------------------------------------------------------------
# 信号装配（登记列确定性函数 + 复算重放）
# ---------------------------------------------------------------------------
def _registered_seed_wides(registry, dates: pd.DatetimeIndex) -> dict[str, dict[str, pd.DataFrame]]:
    """G1 三种子四变体 + M：{arm: {variant: date×code 宽表}}。"""
    out: dict[str, dict[str, pd.DataFrame]] = {f"G1_s{s}": {v: {} for v in VARIANTS}
                                               for s in SEEDS}
    out["M"] = {"mean": {}}
    for d in dates:
        rec = registry.day(d)
        for s in SEEDS:
            for v in VARIANTS:
                out[f"G1_s{s}"][v][d] = rec.wide[f"s{s}_{v}"].dropna()
        out["M"]["mean"][d] = rec.wide["M"].dropna()
    return {a: {v: pd.DataFrame(rows).T.reindex(dates) for v, rows in vs.items()}
            for a, vs in out.items()}


def _combo_wides(registry, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """C1/C2/C2_excl_late/C3：逐日确定性推导（规则冻结于两份预注册文档）。"""
    c1, c2, c2x, c3 = {}, {}, {}, {}
    for d in dates:
        rec = registry.day(d)
        c1[d] = derive_c1_day(rec.wide)
        c2[d] = derive_c2_day(rec.wide, gate=rec.gate)
        if not rec.late:
            c2x[d] = c2[d]
        c3[d] = derive_c3_day(rec.wide)["C3"]
    keep = pd.DatetimeIndex(sorted(c2x)) if c2x else dates  # 全部迟补 → 退化为全窗（如实呈现）
    return {
        "C1": {"mean": pd.DataFrame(c1).T.reindex(dates)},
        "C2": {"mean": pd.DataFrame(c2).T.reindex(dates)},
        "C2_excl_late": {"mean": pd.DataFrame(c2x).T.reindex(keep)},
        "C3": {"mean": pd.DataFrame(c3).T.reindex(dates)},
    }


def _recompute_wides(registry, dates, recompute) -> dict[str, dict[str, pd.DataFrame]]:
    """F0/F1/B3 重放（四变体）+ R1 组装（gate True→M，False→F0_mean）。"""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for arm in ("F0", "F1", "B3"):
        out[arm] = {v: generate_arm_signals(arm, dates, recompute, variant=v)
                    for v in recompute.variants(arm)}
    rows = {}
    for d in dates:
        rec = registry.day(d)
        m = rec.wide["M"].dropna()
        f0 = out["F0"]["mean"].loc[d].dropna()
        rows[d] = r1_assemble(m, f0, gate=rec.gate)
    out["R1"] = {"mean": pd.DataFrame(rows).T.reindex(dates)}
    return out


def _cross_check(registry, dates, secondary) -> dict:
    """§3.3：登记信号 vs 第二存储抽样对拍（≥5 日 × ≥3 臂，max|Δ| 贴实数）。"""
    if secondary is None:
        return {"skipped": True}
    sample_days = list(dates[:: max(1, len(dates) // CROSS_CHECK_N_DAYS)])[:CROSS_CHECK_N_DAYS]
    max_delta, n_pairs = 0.0, 0
    for d in sample_days:
        wide = registry.day(d).wide
        for arm in CROSS_CHECK_ARMS:
            col = "M" if arm == "M" else f"s{arm[-3:]}_mean"
            a = wide[col].dropna().astype(float).sort_index()
            b = secondary(arm, d).astype(float).sort_index()
            assert list(a.index) == list(b.index), f"对拍 {arm}/{d.date()} 池不一致"
            if len(a):
                max_delta = max(max_delta, float((a - b).abs().max()))
                n_pairs += 1
    return {"arms": list(CROSS_CHECK_ARMS), "days": [d.date().isoformat() for d in sample_days],
            "n_pairs": n_pairs, "max_abs_delta": max_delta}


# ---------------------------------------------------------------------------
# 判据（§3 一次性代入——纯函数）
# ---------------------------------------------------------------------------
def judge(full_table: dict) -> dict:
    def ew(arm: str, variant: str = "mean") -> float:
        return full_table[f"{arm}@{WINDOW}"][variant]["aer_ew"]

    def idx(arm: str, variant: str = "mean") -> float:
        return full_table[f"{arm}@{WINDOW}"][variant]["aer_idx"]

    seed_ews = {s: ew(f"G1_s{s}") for s in SEEDS}
    med_seed = sorted(SEEDS, key=lambda s: seed_ews[s])[1]
    med_ew, med_idx = ew(f"G1_s{med_seed}"), idx(f"G1_s{med_seed}")
    g1_pass = med_ew > 0 and med_idx > 0

    def survived(arm: str) -> bool:
        return ew(arm) > 0 and idx(arm) > 0

    r1_ok = survived("R1")
    b3_luck = ew("B3") <= 0
    return {
        "g1_main": {
            "passed": g1_pass,
            "median_seed": f"s{med_seed}",
            "median_seed_aer": {"aer_ew": med_ew, "aer_idx": med_idx},
            "by_seed_aer_ew": {f"s{s}": seed_ews[s] for s in SEEDS},
            "decision_wording": (
                "前向首读通过，进入滚动确认" if g1_pass
                else "触发 Q2 预承诺：项目关闭"
            ),
        },
        "branch": BRANCH_ROLLING if g1_pass else BRANCH_CLOSE,
        "C1": {"survived": survived("C1"), "increment": ew("C1") > ew("G1_s100")},
        "C2": {"survived": survived("C2"), "increment": ew("C2") > max(ew("M"), ew("C1")),
               "late_sensitivity": {"with_late_ew": ew("C2"), "excl_late_ew": ew("C2_excl_late")}},
        "C3": {"survived": survived("C3"), "increment": ew("C3") > max(ew("C1"), ew("M"))},
        "R1": {"survived": r1_ok, "grade": "复算级",
               "wording": "第 2 轮'状态假说'前向存活（复算级）" if r1_ok
                          else "未存活（复算级）"},
        "F1_direction": ew("F1") > ew("F0"),
        "B3": {"seed_luck_confirmed": b3_luck, "grade": "复算级",
               "wording": "'种子运气'定谳（复算级）" if b3_luck
                          else "记录为未解释残差，不复活路线（复算级）"},
    }


def _decision_rows(v: dict, ft: dict) -> list[tuple[str, str, str, str]]:
    med = v["g1_main"]["median_seed_aer"]
    g1 = ("G1 主判据（Q2）",
          "三种子 mean 中位：AER(等权)>0 且 AER(指数)>0",
          f"中位 {v['g1_main']['median_seed']}：{med['aer_ew']:+.2%} / {med['aer_idx']:+.2%}",
          v["g1_main"]["decision_wording"])
    rows = [g1]
    for name, key, defs in (
        ("C1-存活", ("C1", "survived"), "AER 双正"),
        ("C1-增值", ("C1", "increment"), "> G1_s100 同窗"),
        ("C2-存活", ("C2", "survived"), "AER 双正"),
        ("C2-增值", ("C2", "increment"), "> max(纯 M, 纯 C1) 同窗"),
        ("C3-存活", ("C3", "survived"), "AER 双正"),
        ("C3-增值", ("C3", "increment"), "> max(纯 C1, 纯 M) 同窗"),
    ):
        arm, kind = key
        base = {"C1": "C1", "C2": "C2", "C3": "C3"}[arm]
        val = ft[f"{base}@{WINDOW}"]["mean"]["aer_ew"]
        rows.append((name, defs, f"AER(等权)={val:+.2%}",
                     "通过" if v[arm][kind] else "未通过"))
    r1v = ft[f"R1@{WINDOW}"]["mean"]
    rows.append(("R1 原版", "AER(等权) 双基准 > 0",
                 f"{r1v['aer_ew']:+.2%} / {r1v['aer_idx']:+.2%}",
                 "存活" if v["R1"]["survived"] else "未存活"))
    f1v, f0v = ft[f"F1@{WINDOW}"]["mean"], ft[f"F0@{WINDOW}"]["mean"]
    rows.append(("F1 方向", "F1_mean > F0_mean 同窗",
                 f"{f1v['aer_ew']:+.2%} vs {f0v['aer_ew']:+.2%}",
                 "本土化改善方向前向复现（复算级）" if v["F1_direction"]
                 else "未复现（复算级）"))
    b3v = ft[f"B3@{WINDOW}"]["mean"]
    rows.append(("B3 复核", "AER(等权) ≤ 0 → 定谳",
                 f"{b3v['aer_ew']:+.2%}", v["B3"]["wording"]))
    return rows


def _evidence_rows(ft: dict) -> list[tuple[str, float]]:
    def ew(arm: str) -> float:
        return ft[f"{arm}@{WINDOW}"]["mean"]["aer_ew"]

    return [
        ("C1 − G1_s100", (ew("C1") - ew("G1_s100")) * 100),
        ("C2 − max(M, C1)", (ew("C2") - max(ew("M"), ew("C1"))) * 100),
        ("C3 − max(C1, M)", (ew("C3") - max(ew("C1"), ew("M"))) * 100),
        ("F1 − F0", (ew("F1") - ew("F0")) * 100),
    ]


def _full_table_md(ft: dict, grades: dict) -> str:
    lines = ["| 臂 | 级别 | 变体 | AER(等权) | AER(指数) |", "|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        key = f"{arm}@{WINDOW}"
        if key not in ft:
            continue
        grade = grades.get(arm, GRADES.get(arm, "—"))
        for v in ("mean", "min", "max", "last"):
            if v not in ft[key]:
                continue
            perf = ft[key][v]
            lines.append(f"| {arm} | {grade} | {v} | {perf['aer_ew']:+.2%} "
                         f"| {perf['aer_idx']:+.2%} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_settlement(
    *,
    registry,
    dates: pd.DatetimeIndex,
    engine,
    recompute,
    out_dir: Path,
    label: str = "FORWARD SETTLEMENT",
    file_prefix: str = "",
    cross_check_secondary=None,
) -> dict:
    dates = pd.DatetimeIndex(dates)
    assert len(dates) >= MIN_SETTLEABLE_DAYS, (
        f"可结算交易日 {len(dates)} < {MIN_SETTLEABLE_DAYS}（H=10 口径，结算计划 §0）"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # —— §3.3 判读前置：完整性审计 + 抽样对拍（先于任何绩效计算）——
    audit = audit_registry(registry.manifest_path(), registry.dir(),
                           trading_calendar=dates)
    cross_check = _cross_check(registry, dates, cross_check_secondary)
    late_days = list(audit["late_dates"])

    # —— 信号装配 + 引擎全表（判据只在 mean）——
    arms: dict[str, dict[str, pd.DataFrame]] = {}
    arms.update(_registered_seed_wides(registry, dates))
    arms.update(_combo_wides(registry, dates))
    arms.update(_recompute_wides(registry, dates, recompute))

    full_table: dict = {}
    for arm in ARM_ORDER:
        for v, wide in arms.get(arm, {}).items():
            full_table.setdefault(f"{arm}@{WINDOW}", {})[v] = engine.evaluate(wide)

    verdict = judge(full_table)

    results_path = out_dir / f"{file_prefix}results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"window": WINDOW, "n_days": len(dates), "audit": audit,
                   "cross_check": cross_check, "late_days": late_days,
                   "full_table": full_table, "verdict": verdict},
                  f, ensure_ascii=False, indent=2, default=str)

    doc = render_settlement_doc(
        label=label, branch=verdict["branch"], dates=dates, audit=audit,
        cross_check=cross_check, late_days=late_days,
        decision_rows=_decision_rows(verdict, full_table),
        evidence_rows=_evidence_rows(full_table),
        full_table_md=_full_table_md(full_table, GRADES),
    )
    doc_path = out_dir / f"{file_prefix}settlement_doc.md"
    doc_path.write_text(doc, encoding="utf-8")

    return {"root": out_dir, "results_path": str(results_path),
            "doc_path": str(doc_path), "full_table": full_table,
            "verdict": verdict, "grades": GRADES, "audit": audit,
            "cross_check": cross_check, "late_days": late_days}
