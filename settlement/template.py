"""双轨措辞文档模板（滚动再训协议_20260819.md §3/§4 措辞框架的落地）。

双轨纪律：
- **决策栏** = 冻结判据代入（预承诺装置）——forward结算计划_20260818.md §3 是
  唯一来源，本文档模板不得出现任何 §3 之外的通过/失败标准；
- **证据栏** = ±26pp 噪声底折扣后的评估（134 日窗日度 TE 推导的标准误，
  N50 封盘方法论遗产）——噪声底内的差距措辞"不可判"，不得作为决策依据。

"通过"的措辞上限 = "待滚动确认"（60 交易日读数 t 值 ≈ 0.4，只是方向检查点）。
分支路由：G1 主判据负 → §5 关闭（Q2 预承诺）；正 → §6 滚动确认。
"""
from __future__ import annotations

import pandas as pd

BRANCH_CLOSE = "关闭"
BRANCH_ROLLING = "滚动"

NOISE_FLOOR_PP = 26.0  # 滚动再训协议 §3：比较类结论只认超噪声底的差距

TEMPLATE = """# {label}：forward 结算结果（{branch_label}分支）

> 窗口：{date_range}（{n_days} 个可结算交易日，H=10 口径）
> 判据唯一来源：forward结算计划_20260818.md §3（代入式开封，一次判读）
> 噪声框架：滚动再训协议_20260819.md（±{noise_pp}pp 噪声底，比较类结论只认超噪声底）
> {banner}

## 1. 完整性审计（§3.3 判读前置）

{audit_block}

登记信号 vs 复算存储抽样对拍：{cross_check_block}

迟补日清点：{late_block}

## 2. 决策栏（冻结判据代入——预承诺装置，本栏是唯一决策依据）

| 判据 | 冻结定义 | 代入 | 裁决 |
|---|---|---|---|
{decision_rows}

## 3. 证据栏（±{noise_pp}pp 噪声底折扣后的评估——不得作为决策依据）

| 比较 | 差距(pp) | 噪声底判定 |
|---|---|---|
{evidence_rows}

措辞约束：噪声底内的任何差距（无论方向）= "不可判"；本栏与决策栏结论
不一致时，以决策栏为准并如实并列。

## 4. 七臂 + C3 全表（{n_days} 日，全数呈现无挑选；登记级/复算级分级标注）

{full_table_md}

## 5. 分支执行

{branch_section}

## 6. 纪律回执

- 判据代入在完整性审计之后执行；一次开封一次判读；
- 已关闭臂（G4/G5/G7/N50 谱系）不进结算全表；
- 证据级分级：登记级（MANIFEST+git 盖章）/ 复算级（冻结协议确定性重放）；
- 本文档不包含任何冻结判据之外的标准；"通过"上限 = "待滚动确认"。
"""

CLOSE_SECTION = """**{branch_label}分支（forward结算计划 §5）**：G1 主判据首读为负 →
**触发 Q2 预承诺：项目关闭**。Kronos-alpha 主张就此收官：

- [ ] 终版总账 `docs/Kronos项目终版总账_<日期>.md`（8 轮谱系 + 关闭清单 + 方法论产出）
- [ ] G3 登记 cron 停止；仓库归档
- [ ] **不做任何挽救实验**（不参数搜索、不换窗口、不"再等等/再试试"）
"""

ROLLING_SECTION = """**{branch_label}分支（forward结算计划 §6）**：G1 主判据
**前向首读通过，进入滚动确认**（措辞上限即此——首读 ≠ 有效，t 值 ≈ 0.4）：

- [ ] 下一读数 = 再累积 60 个可结算交易日（预计约 {next_reading_hint}），判据沿用 §3 不重设
- [ ] C1/C2/C3 通过增值判据者升格为在位组合候选，G3 登记续跑
- [ ] 滚动再训协议_20260819.md 生效（年度再训/灾难否决/双轨措辞全按协议执行）
"""


def _fmt(x: float) -> str:
    return f"{x:+.2%}" if x == x else "n/a"


def noise_verdict(gap_pp: float) -> str:
    """证据栏措辞：超噪声底才可判方向，噪声底内一律"不可判"。"""
    if abs(gap_pp) > NOISE_FLOOR_PP:
        return f"超噪声底（|{gap_pp:+.1f}pp| > ±{NOISE_FLOOR_PP:.0f}pp），方向可读但仍属半污染窗证据"
    return f"噪声底内（|{gap_pp:+.1f}pp| ≤ ±{NOISE_FLOOR_PP:.0f}pp），不可判"


def render_settlement_doc(
    *,
    label: str,
    branch: str,
    dates: pd.DatetimeIndex,
    audit: dict,
    cross_check: dict,
    late_days: list[str],
    decision_rows: list[tuple[str, str, str, str]],
    evidence_rows: list[tuple[str, float]],
    full_table_md: str,
    next_reading_hint: str = "再累积 60 个交易日后",
) -> str:
    is_close = branch == BRANCH_CLOSE
    branch_label = "关闭" if is_close else "滚动"
    banner = (
        "**SYNTHETIC DRILL——合成演习产物，非真实结算文档；数字无任何真实含义**"
        if label.upper().startswith("SYNTHETIC")
        else "真实结算（开封执行）"
    )
    audit_block = (
        f"MANIFEST 哈希链：{audit['n_days']} 日登记，"
        f"断链 {audit['n_hash_mismatch']} / 缺文件 {audit['n_missing_file']} / "
        f"重复日 {audit['n_duplicate_date']} / 行数不符 {audit['n_nstocks_mismatch']} / "
        f"缺日 {audit['n_missing_days']}（日历：{audit['calendar']}）→ "
        f"{'通过' if audit['passed'] else '未通过（完整性事件，须披露）'}"
    )
    if cross_check.get("skipped"):
        cc_block = "未执行（无第二存储源）"
    else:
        cc_block = (
            f"{len(cross_check['arms'])} 臂 × {len(cross_check['days'])} 日，"
            f"max|Δ|={cross_check['max_abs_delta']:.1e}"
        )
    late_block = (
        f"{len(late_days)} 日（{', '.join(late_days) if late_days else '—'}）；"
        "C2 已做剔除敏感性（C2_excl_late 入全表）"
    )
    decision = "\n".join(
        f"| {c} | {d} | {s} | **{v}** |" for c, d, s, v in decision_rows
    )
    evidence = "\n".join(
        f"| {name} | {gap:+.1f} | {noise_verdict(gap)} |" for name, gap in evidence_rows
    )
    section = (CLOSE_SECTION if is_close else ROLLING_SECTION).format(
        branch_label=branch_label, next_reading_hint=next_reading_hint
    )
    return TEMPLATE.format(
        label=label,
        branch_label=branch_label,
        banner=banner,
        date_range=f"{dates[0].date()}~{dates[-1].date()}",
        n_days=len(dates),
        noise_pp=f"{NOISE_FLOOR_PP:.0f}",
        audit_block=audit_block,
        cross_check_block=cc_block,
        late_block=late_block,
        decision_rows=decision,
        evidence_rows=evidence,
        full_table_md=full_table_md,
        branch_section=section,
    )
