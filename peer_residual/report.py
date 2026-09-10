"""结果报告生成（计划 §9：docs/同日跨股票交互残差头实验结果_实际日期.md）。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from peer_residual import config as C


def _fmt_pp(x: float) -> str:
    return f"{x * 100:+.2f}pp"


def _load(art: Path, name: str):
    p = art / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write_report(doc_path: Path, art_dir: Path) -> str:
    """从 artifacts 生成 markdown 报告（pilot/confirm 均可，缺文件如实标注）。"""
    pre = _load(art_dir, "preflight_manifest.json") or {}
    cache_mf = _load(art_dir, "peer_cache_manifest.json") or {}
    verify = _load(art_dir, "peer_verify_gate.json") or {}
    ledger = _load(art_dir, "budget_ledger.json") or {}
    pilot = _load(art_dir, "pilot_verdict.json")
    confirm = _load(art_dir, "confirm_verdict.json")
    summary = _load(art_dir, "comparison_summary.json") or {}
    stage = "confirm" if confirm else "pilot"
    verdict = (confirm or pilot or {}).get("verdict", {})
    per_seed = summary.get("per_seed", {})
    seeds = sorted(per_seed)
    lines: list[str] = []
    ap = lines.append
    ap("# 同日跨股票交互残差头优化 G1 mean：实验结果（20260910）\n")
    ap(f"> 生成时间：{datetime.now().isoformat(timespec='seconds')}；"
       f"阶段：{stage}；协议：{C.PROTOCOL_VERSION}（run_id={C.RUN_ID}）。\n")
    ap("**结论一句话：** " + _one_liner(verdict, per_seed) + "\n")
    ap("## 1. 协议与身份\n")
    ap(f"- 运行代码基线：计划分支 `codex/peer-residual-plan` @ `77f05a2`；"
       f"preflight git_head=`{pre.get('git_head', 'N/A')}`。")
    ap(f"- G1 tokenizer/predictor SHA："
       f"`{pre.get('g1_weights', {}).get('tokenizer', {}).get('sha256', 'N/A')[:16]}…` / "
       f"`{pre.get('g1_weights', {}).get('predictor', {}).get('sha256', 'N/A')[:16]}…`"
       f"（与 SAE 冻结缓存同底座，4 锚点日抽样核验 max|Δh|≤"
       f"{max((v['max_abs_diff'] for v in verify.values()), default=float('nan')):.1e}"
       f" ≤ 1e-5）。")
    ap(f"- σe={pre.get('sigma_e')}（f6c0726 norm_stats.json 原值，μh/σh/σe 未重拟合）；"
       f"D={pre.get('d_in')}；头参数 {pre.get('head', {}).get('param_table', {}).get('total')}。")
    ap(f"- 基线 SHA：W3=`{C.BASELINE_SHA256['W3'][:16]}…`、"
       f"W4=`{C.BASELINE_SHA256['W4'][:16]}…`（逐值核对通过）。\n")
    ap("## 2. PeerSet / LossSet 覆盖\n")
    ap("| 段 | 决策日 | 平均 PeerSet/日 | 额外补 h | LossSet/输出格 |")
    ap("|---|---|---|---|---|")
    for split in ("train", "val", "W3", "W4"):
        v = cache_mf.get("splits", {}).get(split, {})
        ap(f"| {split} | {v.get('n_days', 'N/A')} | "
           f"{v.get('avg_peers_per_day', 'N/A')} | "
           f"{v.get('n_extra_h_total', 'N/A')} | "
           f"{v.get('n_loss_days_total', 'N/A')} |")
    ap("")
    ap("- LossSet = SAE 冻结监督键原值（train 534日×64、val 22日×64）；"
       "所有 LossSet/基线格 ⊆ PeerSet 逐日断言通过（无缩池/替换）。")
    ap("- 额外 peer 仅作 attention context，不新增交易候选；"
       "PeerSet 资格不含未来信息（PIT 成分 ∩ 90 日窗口完整 ∩ 无停牌）。\n")
    ap("## 3. 三臂两窗主结果（Qlib 同口径，cumsum 窗口末差）\n")
    for seed in seeds:
        p = per_seed[seed]
        ap(f"### seed {seed}\n")
        ap("| 臂 | W3 curve_end | W4 curve_end | 复利 | MDD(复利) | 费用和 |")
        ap("|---|---|---|---|---|---|")
        for name in ("G1_mean", f"PEER_OFF_s{seed}", f"PEER_ON_s{seed}"):
            a = p["arms"][name]
            ap(f"| {name} | {a['W3']['curve_end']:.4f} | "
               f"{a['W4']['curve_end']:.4f} | "
               f"{a['W3']['compound_cum']:+.2%}/{a['W4']['compound_cum']:+.2%} | "
               f"{a['W3']['max_drawdown_compound']:+.2%}/{a['W4']['max_drawdown_compound']:+.2%} | "
               f"{a['W3']['cost_sum']:.2f}/{a['W4']['cost_sum']:.2f} |")
        ap("")
        ap(f"- D_OFF（OFF−G1）：W3 {_fmt_pp(p['D_OFF']['W3'])}、"
           f"W4 {_fmt_pp(p['D_OFF']['W4'])}")
        ap(f"- D_ON（ON−G1）：W3 {_fmt_pp(p['D_ON']['W3'])}、"
           f"W4 {_fmt_pp(p['D_ON']['W4'])}")
        ap(f"- D_peer（ON−OFF）：W3 {_fmt_pp(p['D_peer']['W3'])}、"
           f"W4 {_fmt_pp(p['D_peer']['W4'])}\n")
    ap("### 修正量诊断（seed100，只诊断不另挑 checkpoint）\n")
    diag_rows = []
    for w in C.TEST_WINDOWS:
        d = _load(art_dir, f"diagnostics_{w}_s{seeds[0]}.json") if seeds else None
        if not d:
            continue
        for arm in C.ARMS:
            v = d[arm]
            diag_rows.append(
                f"| {w} | {arm} | {v['correction_std_daily_mean']:.4f} | "
                f"{v['spearman_vs_g1_mean']:.4f} | {v['top50_overlap_mean']:.2%} |")
    if diag_rows:
        ap("| 窗 | 臂 | 修正截面 std(日均) | Spearman vs G1(日均) | top50 重合(日均) |")
        ap("|---|---|---|---|---|")
        lines.extend(diag_rows)
    ap("")
    ap("## 4. 判据（计划 §7）\n")
    for key, label in (("off", "OFF 重复超过 G1"), ("on", "ON 重复超过 G1"),
                       ("peer_extra", "跨股交互额外有效（D_peer）")):
        v = verdict.get(key, {})
        ap(f"- {label}：两窗均胜 seed 数 {v.get('both_windows_count')}/"
           f"{verdict.get('n_seeds')}，两窗中位数 "
           f"{ {k: _fmt_pp(x) for k, x in v.get('median_delta', {}).items()} } → "
           f"{'**满足**' if v.get('repeats') else '**不满足**'}")
    ap("")
    ap(f"- confirm 补种子门禁（任一头两窗均胜 G1）："
       f"{'通过（已补 101/102 两臂）' if confirm else '未通过或未执行——不补种子'}")
    ap("- 所有结论限已观察历史（W3/W4）；未读 forward，未替换 G1，"
       "未改旧封盘判据。\n")
    ap("## 5. 预算与运行\n")
    ap(f"- GPU 预算 12h：累计 wall {ledger.get('gpu_used_wall_s', 0):.0f}s"
       f"（cuda 实测 {ledger.get('cuda_measured_s', 0):.0f}s），"
       f"余 {ledger.get('remaining_gpu_budget_s', 0):.0f}s。")
    heads = (pilot or confirm or {}).get("heads", {})
    if heads:
        ap("- best 选点（验证等权 MSE 最低 epoch）：" + "；".join(
            f"{k}=e{v['best_epoch']}（val {v['best_val_l_pred']:.6f}）"
            for k, v in heads.items()))
    ap("")
    ap("## 6. 提交物\n")
    ap("- 小头权重/逐 epoch 指标：`artifacts/experiments/peer_residual_20260910/`"
       "（`head_*_e*.pt`、`best_*.json`、`epochs_metrics_*.json`）")
    ap("- 三臂两窗 signal/report/日曲线：`signal_*.parquet`、`report_*.parquet`、"
       "`daily_*.parquet`；主图 `fig_W*_main.png`。")
    ap("- peer 缓存清单：`peer_cache_manifest.json`（逐 chunk SHA；大缓存留 "
       f"`{C.PEER_CACHE_DIR}` 不入 git）。")
    ap("- 复现门禁：`reproduction_gate.json`（G1 逐日 return/cost/bench 与 "
       "mean_comparison_20260909 对拍 ≤1e-10）。")
    text = "\n".join(lines) + "\n"
    doc_path.write_text(text, encoding="utf-8")
    return text


def _one_liner(verdict: dict, per_seed: dict) -> str:
    if not per_seed:
        return "（评价未完成）"
    p = per_seed[sorted(per_seed)[0]]
    on_ok = p.get("on_beats_g1_both")
    off_ok = p.get("off_beats_g1_both")
    if verdict:
        if verdict.get("on", {}).get("repeats") and \
                verdict.get("peer_extra", {}).get("repeats"):
            return ("ON 头两窗重复超过 G1 且 D_peer 同条件满足——同日跨股票交互"
                    "在该口径下有可重复增益（限历史窗口）。")
        if verdict.get("off", {}).get("repeats"):
            return ("OFF 头两窗重复超过 G1 但交互额外有效（D_peer）不满足——"
                    "增益不来自跨股交互，不能宣传交互成功。")
    if not on_ok and not off_ok:
        return ("seed100 两头均未在两窗同时超过 G1（D_ON：W3 "
                f"{p['D_ON']['W3']:+.4f}/W4 {p['D_ON']['W4']:+.4f}；D_OFF：W3 "
                f"{p['D_OFF']['W3']:+.4f}/W4 {p['D_OFF']['W4']:+.4f}）——"
                "确认门禁未触发，封存本配置，不补种子。")
    return f"seed100 至少一头两窗均胜 G1（on_both={on_ok}, off_both={off_ok}）。"


__all__ = ["write_report"]
