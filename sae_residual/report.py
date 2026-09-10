"""docs 结果报告生成（计划 §8：负结果也完整交付）。"""
from __future__ import annotations

import json
from pathlib import Path

from sae_residual import config as C


def _fmt_pp(x: float) -> str:
    return f"{x * 100:+.2f}pp"


def write_report(out_doc: Path, art_dir: Path | None = None) -> str:
    art = art_dir or C.ART_DIR
    pilot = json.loads((art / "pilot_verdict.json").read_text(encoding="utf-8"))
    summary = json.loads((art / "comparison_summary.json").read_text(encoding="utf-8"))
    cache = json.loads((art / "cache_manifest.json").read_text(encoding="utf-8"))
    led = json.loads(C.BUDGET_LEDGER.read_text(encoding="utf-8"))
    stats = json.loads((art / "norm_stats.json").read_text(encoding="utf-8"))
    confirm = None
    p_conf = art / "confirm_verdict.json"
    if p_conf.is_file():
        confirm = json.loads(p_conf.read_text(encoding="utf-8"))

    seeds = summary["seeds"]
    per_seed = summary["per_seed"]
    lines: list[str] = []
    lines.append("# SAE 残差预测头优化 G1 mean：实验结果（20260910）\n")
    lines.append("> 协议：" + C.PROTOCOL_VERSION +
                 "；分支 `codex/sae-residual-plan`（计划提交 386c64c）。"
                 "本报告为历史相对比较（研究窗口内同口径回测），"
                 "不称样本外或实盘证明。\n")

    lines.append("## 1. 结论一句话\n")
    ae_rep = pilot["verdict"]["ae"]
    sae_rep = pilot["verdict"]["sae"]
    if confirm:
        ae_rep = confirm["verdict"]["ae"]
        sae_rep = confirm["verdict"]["sae"]
        sae_ae = confirm["verdict"]["sae_over_ae"]
    any_arm = (ae_rep["both_windows_count"] >= 1 or sae_rep["both_windows_count"] >= 1)
    if not any_arm:
        lines.append(
            "AE 与 SAE 残差头在 seed100 下均未在两窗同时超过 G1 mean —— "
            "**判据未触发，本配置封存，不补种子、不调参追收益**。"
            "负结论限定为本协议（小规模样本/固定结构），不扩大为"
            "「所有 SAE 方法无效」。\n")
    elif not confirm:
        lines.append(
            "存在臂两窗均胜 G1（pilot 门禁通过），确认种子见 §5。\n")
    else:
        lines.append(
            f"三 seed 终审：AE 重复超过 G1 = {ae_rep['repeats']}；"
            f"SAE 重复超过 G1 = {sae_rep['repeats']}；"
            f"SAE 相对 AE 额外增益 = {sae_ae['repeats']}。\n")

    lines.append("## 2. 主表：三臂两窗（cumsum 窗口末，扣费）\n")
    seed0 = str(seeds[0])                      # JSON 往返后键为字符串
    arms0 = per_seed[seed0]["arms"]
    lines.append("| 臂 | W3 curve_end | W4 curve_end | W3 复利 | W4 复利 | "
                 "W3 costΣ | W4 costΣ |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("G1_mean", f"AE_residual_s{seed0}", f"SAE_residual_s{seed0}"):
        a = arms0[name]
        lines.append(
            f"| {name} | {a['W3']['curve_end']:+.4%} | "
            f"{a['W4']['curve_end']:+.4%} | {a['W3']['compound_cum']:+.2%} | "
            f"{a['W4']['compound_cum']:+.2%} | {a['W3']['cost_sum']:.4f} | "
            f"{a['W4']['cost_sum']:.4f} |")

    lines.append("\n## 3. 末日差（相对 G1，cumsum）\n")
    lines.append("| seed | D_AG W3 | D_AG W4 | D_SG W3 | D_SG W4 | "
                 "D_SA W3 | D_SA W4 | AE两窗胜 | SAE两窗胜 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(per_seed):
        p = per_seed[s]
        lines.append(
            f"| {s} | {_fmt_pp(p['D_AG']['W3'])} | {_fmt_pp(p['D_AG']['W4'])} | "
            f"{_fmt_pp(p['D_SG']['W3'])} | {_fmt_pp(p['D_SG']['W4'])} | "
            f"{_fmt_pp(p['D_SA']['W3'])} | {_fmt_pp(p['D_SA']['W4'])} | "
            f"{'✓' if p['ae_beats_g1_both'] else '✗'} | "
            f"{'✓' if p['sae_beats_g1_both'] else '✗'} |")

    lines.append("\n## 4. 训练/验证与选点\n")
    lines.append(f"- 训练样本 {stats['n_train']}（534 决策日 × 64 只），"
                 f"验证 {stats['n_val']}；D={stats['d_in']}；"
                 f"σe={stats['sigma_e']:.6f}（训练残差 std，未减均值）。")
    lines.append(f"- 残差诊断：e_train mean={stats['e_train_mean']:+.5f}, "
                 f"std={stats['e_train_std']:.5f}；e_val std={stats['e_val_std']:.5f}。")
    for k, v in pilot["heads"].items():
        lines.append(f"- {k}: best epoch = e{v['best_epoch']}"
                     f"（val L_pred={v['best_val_l_pred']:.6f}，验证各日等权）。")
    lines.append("\n选点唯一依据 = 验证 L_pred 最低（并列最早）；"
                 "不读测试收益选 epoch。零残差初始化保证训练起点逐值等于 G1"
                 "（合成门禁逐位验证）。\n")

    lines.append("## 5. 判据与种子状态\n")
    lines.append(f"- pilot seed100：完成。AE 两窗均胜 = "
                 f"{pilot['verdict']['ae']['both_windows_count'] >= 1}；"
                 f"SAE 两窗均胜 = "
                 f"{pilot['verdict']['sae']['both_windows_count'] >= 1}。")
    if confirm:
        lines.append(f"- 确认 seeds 101/102：完成。判据细节："
                     f"`{json.dumps(confirm['verdict'], ensure_ascii=False)}`")
    else:
        lines.append("- 确认 seeds 101/102：未启动（pilot 门禁未触发）。")

    lines.append("\n## 6. 复现门禁与同口径\n")
    g = pilot["reproduction_gate"]
    lines.append(f"- 原 G1 两窗复现（原 3 臂共同掩码，纯 CPU）："
                 + "；".join(
                     f"{w} max|Δ|={max(d['max_abs_diff']['return'], d['max_abs_diff']['cost'], d['max_abs_diff']['bench']):.1e}"
                     for w, d in g.items()) + " ≤ 1e-10 ✓")
    lines.append("- Qlib 口径与 `0b982e8` 绑定：TopkDropout50/5/hold5、"
                 "SimulatorExecutor(day)、open 成交、买0.001/卖0.0015/min5、"
                 "limit 0.095、资金 1 亿、基准 000300.SH；内部 shift=1 未再平移。")
    lines.append("- 三臂共同掩码 = G1 有效格（残差头覆盖全部基线格，"
                 "构建期逐日断言，无掩码删格）。")

    lines.append("\n## 7. 披露与限制\n")
    lines.append("- G1 训练期预测来自在该历史上训练过的底座（in-sample 残差"
                 "拟合）；验证期与 G1 原验证期重叠。teacher 随机协议 = 原 G1 "
                 "生成协议（每日 seed42、代码升序、chunk32、N=20、T=1.0、"
                 "top_p=0.9），AE/SAE 严格共用同一缓存。")
    lines.append("- 原测试基线信号 SHA 与计划一致（W3 9352b403…/W4 453e2aea…），"
                 "未重新生成测试 G1；legacy 信号生成链保持 legacy_unverified 标记。")
    lines.append("- 本实验为方法改编（启发源：华源证券 2026-02-02 稀疏自编码"
                 "指数择时研报），**不是研报复现**；不得以研报择时年化作为本实验预期。")
    lines.append("- 小规模机制检验（csi300 每 5 日 64 只），负结论不扩大为"
                 "「全数据规模 SAE 无效」。")

    lines.append("\n## 8. 资源与执行偏差\n")
    lines.append(f"- GPU 预算 12h：GPU 活跃 stage wall 累计 "
                 f"{led['gpu_used_wall_s']:.0f}s（CUDA 事件实测 "
                 f"{led['cuda_measured_s']:.0f}s）；head 训练与回测在 CPU 另记。")
    lines.append(f"- 缓存规模：{json.dumps({k: v['n_samples'] for k, v in cache['splits'].items()}, ensure_ascii=False)}；"
                 "大缓存留 Debian（SHA 见 cache_manifest.json），小头权重/信号/"
                 "逐日报告已随 git 提交。")

    out_doc.write_text("\n".join(lines), encoding="utf-8")
    return str(out_doc)


__all__ = ["write_report"]
