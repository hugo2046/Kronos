"""同口径 Qlib 主评价（计划 §7）：三臂两窗、复现门禁、诊断与判据。

主口径 ``net_daily = return − cost`` 的 ``cumsum`` 曲线；窗口末差
``D_OFF=OFF−G1``、``D_ON=ON−G1``、``D_peer=ON−OFF``。复利/回撤/费用只作
附表。Qlib 回测复用 ``qlib_mean_comparison.run``（与 ``0b982e8`` 绑定，
内部 shift=1 不再平移）。三臂同一交易候选集合（= 基线 G1 有效格）、
独立账户；ON/OFF 额外 peer 只影响 attention context，不扩大可交易池。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from peer_residual import config as C
from peer_residual import data as PD
from peer_residual.train import load_best


def infer_arm_signals(days: list, model, arm: str, sigma_e: float,
                      batch_days: int = C.TRAIN["batch_days"]
                      ) -> pd.DataFrame:
    """生产推理：逐日全 PeerSet forward，输出格=SAE 覆盖格（基线格）。

    只缺未来标签的尾部日期同样有预测（测试路径从不读标签）；返回
    date×instrument 宽表，格 = LossSet/输出格（额外 peer 仅 context）。
    """
    import torch

    rows_date, rows_code, rows_val = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(days), batch_days):
            chunk = days[i:i + batch_days]
            x, valid, loss, _ = PD.collate_batch(chunk)
            r_hat = model(x, valid, arm)
            for j, d in enumerate(chunk):
                idx = np.where(loss[j].numpy())[0]
                for k in idx:
                    rows_date.append(d.date)
                    rows_code.append(d.codes[k])
                s_fin = d.s_g1[idx] + sigma_e * r_hat[j].numpy()[idx].astype(
                    np.float64)
                rows_val.extend(s_fin.tolist())
    df = pd.DataFrame({"datetime": pd.to_datetime(rows_date),
                       "instrument": rows_code, "value": rows_val})
    return df.pivot(index="datetime", columns="instrument",
                    values="value").sort_index()


def infer_test_signal(arm: str, seed: int, wname: str, stats: dict,
                      heads_dir: Path, weight_shas: dict | None = None
                      ) -> tuple[pd.DataFrame, dict]:
    """``s_final = s_G1 + σe·r_hat`` → 宽表（格=基线格，CPU 确定性推理）。"""
    days = PD.load_split_days(wname, stats, weight_shas)
    model, best = load_best(heads_dir, arm, seed, int(stats["d_in"]))
    return infer_arm_signals(days, model, arm, stats["sigma_e"]), best


def backtest_arm(signal: pd.Series, start: str, end: str) -> pd.DataFrame:
    """单臂窗回测（复用 0b982e8 的纯函数，口径冻结）。"""
    from qlib_mean_comparison import run as mc

    report, _ = mc.run_qlib_backtest(signal, start, end)
    return report


def load_wide(wname: str) -> pd.DataFrame:
    """基线 G1 宽表（date×instrument，原值不动）。"""
    df = pd.read_parquet(C.BASELINE_SIGNALS[wname])
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.T
    return df.sort_index()


def reproduce_g1_gate(art_dir: Path, tol: float = 1e-10) -> dict:
    """复现门禁：原 3 臂共同掩码下 G1 逐日 return/cost/bench 与旧 report 对拍。

    旧 ``report_W{3,4}_G1_mean.parquet`` 出自 mean_comparison_20260909；
    以同一掩码重跑纯 CPU 回测，误差必须 ≤ 1e-10（不改容差凑过）。
    """
    from qlib_mean_comparison import run as mc

    results = {}
    for w, (start, end) in C.TEST_WINDOWS.items():
        wides = {a: mc.load_wide(a, w) for a in mc.ARMS}
        mask, _ = mc.common_set(wides)
        sig = mc.prepare_signal_series(wides["G1_mean"], mask)
        report = backtest_arm(sig, start, end)
        ref = pd.read_parquet(C.G1_REPORT_REF / f"report_{w}_G1_mean.parquet")
        diffs = {}
        for col in ("return", "cost", "bench"):
            d = float(np.abs(report[col].to_numpy()
                             - ref[col].to_numpy()).max())
            assert report.index.equals(ref.index), f"{w} {col} 日期不一致"
            diffs[col] = d
        worst = max(diffs.values())
        assert worst <= tol, f"{w} 复现失败：max|Δ|={worst:.3e} > {tol:.0e}"
        results[w] = {"max_abs_diff": diffs, "tol": tol,
                      "n_days": int(len(report))}
        logger.info(f"[复现门禁] {w}: max|Δ|={worst:.3e} ≤ {tol:.0e} ✓")
    (art_dir / "reproduction_gate.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def signal_diagnostics(g1_wide: pd.DataFrame, arm_wide: pd.DataFrame,
                       sigma_e: float) -> dict:
    """修正量诊断（§7，只诊断不另挑 checkpoint）：截面 std / 排序变化 /
    top50 重合度——判断是否只学到对全体股票相同的偏置。"""
    from scipy.stats import spearmanr

    corr = (arm_wide - g1_wide).loc[g1_wide.index]
    std_daily = corr.std(axis=1, skipna=True)
    mean_abs = corr.abs().mean(axis=1, skipna=True)
    rhos, overlaps = [], []
    for t in g1_wide.index:
        a = arm_wide.loc[t].dropna()
        g = g1_wide.loc[t].dropna()
        common = a.index.intersection(g.index)
        if len(common) < 10:
            continue
        rho = spearmanr(a[common].to_numpy(), g[common].to_numpy()).statistic
        if np.isfinite(rho):
            rhos.append(rho)
        top_a = set(a[common].nlargest(50).index)
        top_g = set(g[common].nlargest(50).index)
        overlaps.append(len(top_a & top_g) / 50.0)
    return {"sigma_e": sigma_e,
            "correction_std_daily_mean": float(std_daily.mean()),
            "correction_std_daily_median": float(std_daily.median()),
            "correction_abs_mean": float(mean_abs.mean()),
            "spearman_vs_g1_mean": float(np.mean(rhos)),
            "top50_overlap_mean": float(np.mean(overlaps)),
            "n_diag_days": len(rhos)}


def frozen_wide(art_dir: Path, name: str, wname: str) -> pd.DataFrame:
    """只读冻结信号（计划 §6.2）：``PEER_OFF_s100`` → ``signal_W3_OFF_s100``；
    ``G1_mean`` → 基线原文件。回测阶段不得调用 ``infer_test_signal``。"""
    if name == "G1_mean":
        return load_wide(wname)
    arm = name.split("_")[1]                 # PEER_OFF_s100 → OFF
    seed = name.split("_s")[1]
    return pd.read_parquet(art_dir / f"signal_{wname}_{arm}_s{seed}.parquet")


def run_comparison(art_dir: Path, stats: dict, seeds: list[int],
                   wide_provider=None, device_note: str = "cpu") -> dict:
    """三臂两窗主评价：G1/OFF/ON 同格回测 → 曲线/判据/图表/诊断。

    :param wide_provider: ``(name, wname) -> 宽表``；缺省只读冻结信号文件
        （§6.2 顺序：生成全部信号 → 冻结 manifest → 只读回测）。
    """
    from qlib_mean_comparison import run as mc

    provider = wide_provider or (lambda name, w: frozen_wide(art_dir, name, w))
    per_seed: dict[int, dict] = {}
    for seed in seeds:
        arms: dict[str, dict[str, dict]] = {}
        for w, (start, end) in C.TEST_WINDOWS.items():
            g1_wide = load_wide(w)
            wides = {"G1_mean": g1_wide}
            for arm in C.ARMS:
                wide = provider(f"PEER_{arm}_s{seed}", w)
                wide = wide.reindex(index=g1_wide.index, columns=g1_wide.columns)
                assert wide.notna().equals(g1_wide.notna()), \
                    f"{arm} s{seed} {w} 信号格与 G1 不一致（缩池/扩池均禁止）"
                wides[f"PEER_{arm}_s{seed}"] = wide
            mask, _ = mc.common_set(wides)
            for name, wide in wides.items():
                sig = mc.prepare_signal_series(wide, mask)
                report = backtest_arm(sig, start, end)
                curves = mc.compute_curves(report)
                report.to_parquet(art_dir / f"report_{w}_{name}.parquet")
                pd.DataFrame({"net_daily": curves["net_daily"],
                              "curve": curves["curve"],
                              "excess_curve": curves["excess_curve"],
                              "compound_nav": curves["compound_nav"]}
                             ).to_parquet(art_dir / f"daily_{w}_{name}.parquet")
                arms.setdefault(name, {})[w] = {
                    "curve_end": float(curves["curve"].iloc[-1]),
                    "compound_cum": curves["compound_cum"],
                    "max_drawdown_compound": curves["max_drawdown_compound"],
                    "cost_sum": curves["cost_sum"],
                    "turnover_daily_mean": curves["turnover_daily_mean"],
                    "n_days": int(len(report)),
                }
            benches = [pd.read_parquet(art_dir / f"report_{w}_{n}.parquet")["bench"]
                       for n in wides]
            for b in benches[1:]:
                assert benches[0].equals(b), f"{w} 基准不一致"
            # 修正量诊断（只诊断，不另挑 checkpoint）
            diag = {}
            for arm in C.ARMS:
                name = f"PEER_{arm}_s{seed}"
                diag[arm] = signal_diagnostics(g1_wide, wides[name],
                                               stats["sigma_e"])
            (art_dir / f"diagnostics_{w}_s{seed}.json").write_text(
                json.dumps(diag, ensure_ascii=False, indent=2),
                encoding="utf-8")
        off = f"PEER_OFF_s{seed}"
        on = f"PEER_ON_s{seed}"
        per_seed[seed] = {
            "arms": arms,
            "D_OFF": {w: arms[off][w]["curve_end"]
                      - arms["G1_mean"][w]["curve_end"]
                      for w in C.TEST_WINDOWS},
            "D_ON": {w: arms[on][w]["curve_end"]
                     - arms["G1_mean"][w]["curve_end"]
                     for w in C.TEST_WINDOWS},
            "D_peer": {w: arms[on][w]["curve_end"] - arms[off][w]["curve_end"]
                       for w in C.TEST_WINDOWS},
            "off_beats_g1_both": all(
                arms[off][w]["curve_end"] > arms["G1_mean"][w]["curve_end"]
                for w in C.TEST_WINDOWS),
            "on_beats_g1_both": all(
                arms[on][w]["curve_end"] > arms["G1_mean"][w]["curve_end"]
                for w in C.TEST_WINDOWS),
        }
    summary = {"created_at": datetime.now().isoformat(timespec="seconds"),
               "seeds": seeds, "per_seed": per_seed,
               "device_note": device_note}
    (art_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_main(art_dir, per_seed, seeds)
    return summary


def plot_main(art_dir: Path, per_seed: dict, seeds: list[int]) -> None:
    """主图：上=三臂扣费累计（G1/OFF/ON+沪深300），下=三种差与 0 线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK HK", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    seed = seeds[0]                          # 主图 = pilot seed
    off, on = f"PEER_OFF_s{seed}", f"PEER_ON_s{seed}"
    for w in C.TEST_WINDOWS:
        cur = {n: pd.read_parquet(art_dir / f"daily_{w}_{n}.parquet")["curve"]
               for n in ("G1_mean", off, on)}
        bench = pd.read_parquet(art_dir / f"report_{w}_G1_mean.parquet")["bench"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        style = {"G1_mean": ("#12456e", "-"), off: ("#de7c33", "-"),
                 on: ("#3a923a", "-")}
        for name, (c, ls) in style.items():
            axes[0].plot(cur[name].index, cur[name], color=c, linestyle=ls,
                         label=name.replace("_", " "))
        axes[0].plot(bench.index, bench.cumsum(), color="black",
                     linestyle="--", label="沪深300")
        axes[0].set_title(f"{w}：扣费累计收益（Qlib 同口径，open 成交，cumsum）")
        axes[0].set_ylabel("累计日收益（相加）")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        d_off = cur[off] - cur["G1_mean"]
        d_on = cur[on] - cur["G1_mean"]
        d_peer = cur[on] - cur[off]
        axes[1].plot(d_off.index, d_off, color="#de7c33", label="D_OFF = OFF−G1")
        axes[1].plot(d_on.index, d_on, color="#3a923a", label="D_ON = ON−G1")
        axes[1].plot(d_peer.index, d_peer, color="#7a5ba6", linestyle=":",
                     label="D_peer = ON−OFF")
        axes[1].axhline(0, color="grey", linewidth=0.8)
        axes[1].set_ylabel("窗口内逐日差额")
        axes[1].set_title(
            f"末日差：D_OFF {d_off.iloc[-1] * 100:+.2f}pp | "
            f"D_ON {d_on.iloc[-1] * 100:+.2f}pp | "
            f"D_peer {d_peer.iloc[-1] * 100:+.2f}pp")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(art_dir / f"fig_{w}_main.png", dpi=150)
        plt.close(fig)


def judge_confirm_gate(per_seed: dict) -> dict:
    """判据（§7）：某头「重复超过 G1」= ≥2/3 seed 两窗 D>0 且两窗各自三
    seed D 中位数 >0；跨股交互额外有效 = ON 满足且 D_peer 同条件。"""
    out = {"n_seeds": len(per_seed)}
    for arm, key in (("off", "D_OFF"), ("on", "D_ON")):
        both_cnt = sum(1 for s in per_seed
                       if per_seed[s][f"{arm}_beats_g1_both"])
        med = {w: float(np.median([per_seed[s][key][w]
                                   for s in sorted(per_seed)]))
               for w in C.TEST_WINDOWS}
        out[arm] = {"both_windows_count": int(both_cnt),
                    "median_delta": med,
                    "repeats": bool(both_cnt >= 2 and all(
                        v > 0 for v in med.values()))}
    both_cnt = sum(1 for s in per_seed
                   if all(per_seed[s]["D_peer"][w] > 0
                          for w in C.TEST_WINDOWS))
    med_peer = {w: float(np.median([per_seed[s]["D_peer"][w]
                                    for s in sorted(per_seed)]))
                for w in C.TEST_WINDOWS}
    out["peer_extra"] = {"both_windows_count": int(both_cnt),
                         "median_delta": med_peer,
                         "repeats": bool(both_cnt >= 2 and all(
                             v > 0 for v in med_peer.values()))}
    return out


__all__ = ["infer_arm_signals", "infer_test_signal", "backtest_arm",
           "load_wide", "frozen_wide", "reproduce_g1_gate", "run_comparison",
           "plot_main", "judge_confirm_gate", "signal_diagnostics"]
