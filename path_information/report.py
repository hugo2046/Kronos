"""PATH1 结果报告生成（计划 §8：明确与 G1 策略胜出是不同层级）。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _fmt(x: float) -> str:
    return f"{x:+.4f}" if isinstance(x, (int, float)) else str(x)


def write_report(doc_path: Path, art_dir: Path) -> None:
    pre = json.loads((art_dir / "preflight_manifest.json").read_text(
        encoding="utf-8"))
    smoke = json.loads((art_dir / "smoke_report.json").read_text(
        encoding="utf-8"))
    cache_mf = json.loads((art_dir / "cache_manifest.json").read_text(
        encoding="utf-8"))
    train_s = json.loads((art_dir / "train_summary.json").read_text(
        encoding="utf-8"))
    verdict = json.loads((art_dir / "evaluate_verdict.json").read_text(
        encoding="utf-8"))
    budget = json.loads((art_dir / "budget_ledger.json").read_text(
        encoding="utf-8"))
    lines: list[str] = []
    a = lines.append
    a("# Kronos 采样路径信息增量实验结果（2026-09-11）\n")
    a(f"> 执行：Debian zcode（worktree `Kronos-path-information`，计划分支 "
      f"`codex/path-information-plan`，HEAD 含 `d3c1bb3`）。协议 "
      f"`{pre['protocol']}`。本报告是**历史开发检验的诊断结果**，不是策略"
      f"回测，与「G1 策略胜出」是不同层级（计划 §6.6）。\n")
    a("## 1. 结论先行\n")
    a(f"- 唯一开发判据（val 与 dev_eval 的 PATH−MEAN 日均配对 IC 差均>0 且 "
      f"dev_eval 的 PATH−G1>0）：**{'触发' if verdict['criterion_pass'] else '未触发'}**。")
    if not verdict["criterion_pass"]:
        a("- 判据未触发 → 按计划 §6.4 封存本配置：不换 loss、不加门控、"
          "不加 TimesFM、不补种子、不自动扩展实验。")
    a("")
    a("## 2. 三臂与协议\n")
    a("- 三臂：`G1_original`（基线文件原值）、`MEAN_s100`（零 shape 对照）、"
      "`PATH_s100`（真实 20×10 路径形状）；均为 `s_final = s_g1_original + "
      "σe_train·r_hat`，753 参数同构小头、末层零初始化、训练 seed100。")
    a(f"- 时间边界：train {pre['segments']['train']['decision_days'][0]}~"
      f"{pre['segments']['train']['decision_days'][1]}（{pre['segments']['train']['n_days']}日）、"
      f"val {pre['segments']['val']['n_days']}日、dev_eval "
      f"{pre['segments']['dev_eval']['n_days']}日；`calendar[t+10] ≤ 段标签末"
      f"日`排除边界；train/val 标签覆盖 ≥95% 门禁通过。")
    a(f"- 基线/权重身份：W3/W4 SHA 与计划冻结值一致；tokenizer/predictor "
      f"SHA 一致（preflight manifest）。G1 时间证据为配置级（train 至 "
      f"2024-12-31、选模至 2025-06-30），非历史真实时点部署证明；上游预"
      f"训练语料边界未知，如实披露。")
    a(f"- 新采样链与原 G1 mean 之差仅记录（sample_keys.parquet 的 "
      f"`s_g1` vs `s_new_mean`），不替换基线。")
    a("")
    a("## 3. 诊断（逐日 Spearman，平均秩，共同有效标签格）\n")
    a("| 段 | 指标 | G1 | MEAN | PATH |")
    a("|---|---|---:|---:|---:|")
    for split, seg in verdict["segments"].items():
        ic = seg["daily_ic_mean"]
        a(f"| {split} | 日均RankIC | {_fmt(ic['G1'])} | "
          f"{_fmt(ic['MEAN'])} | {_fmt(ic['PATH'])} |")
        mse = seg["mse"]
        a(f"| {split} | 归一化MSE | {_fmt(mse['G1'])} | "
          f"{_fmt(mse['MEAN'])} | {_fmt(mse['PATH'])} |")
        p = seg["paired"]
        a(f"| {split} | 配对IC差 MEAN−G1={_fmt(p['MEAN-G1']['mean'])}、"
          f"PATH−G1={_fmt(p['PATH-G1']['mean'])}、"
          f"PATH−MEAN={_fmt(p['PATH-MEAN']['mean'])}"
          f"（有效日 {p['PATH-MEAN']['n_days']}）| | | |")
    a("")
    a("## 4. 训练与选点\n")
    heads = train_s["heads"]
    for arm in ("MEAN", "PATH"):
        h = heads[arm]
        a(f"- {arm}：best e{h['best_epoch']}，val MSE "
          f"{h['best_val_mse']:.6f}（G1 零残差 {h['g1_zero_val_mse']:.6f}）。")
    a(f"- σe_train={train_s['stats']['sigma_e']:.6f}，训练有效格 "
      f"{train_s['stats']['n_train_cells']}。")
    a("")
    a("## 5. 预算与产物\n")
    a(f"- GPU 累计 {budget['gpu_used_wall_s']}s / {pre['budget']['gpu_s']}s；"
      f"CPU 累计 {budget['cpu_used_wall_s']}s / {pre['budget']['cpu_s']}s。")
    a(f"- smoke 外推（×1.5）{smoke['estimated_full_gpu_wall_s_x1_5']}s；峰值"
      f"显存 {smoke['peak_vram_gb']}GB。")
    a(f"- 缓存 {cache_mf['n_chunks']} chunks / {cache_mf['n_cells']} 格 / "
      f"{cache_mf['cache_total_mib']}MiB（压缩 npz，禁 pickle，随 Git 交付）。")
    a("")
    a("## 6. 边界与披露\n")
    a("- val 参与选点；dev_eval（W4）已被历史研究反复观察；预测标签存在 "
      "10 日重叠；三臂日期等权、64 股抽样诊断不换算 FULL 收益。")
    a("- 即使判据触发，也只称「本历史划分下观察到路径信息增量，需另立复验」。")
    a("- 未读 forward（读取上界 2026-07-24）；预测特征 fetch.end≤t；标签由"
      "独立阶段按段边界读取。")
    a(f"\n生成时间：{datetime.now().isoformat(timespec='seconds')}\n")
    doc_path.write_text("\n".join(lines), encoding="utf-8")
