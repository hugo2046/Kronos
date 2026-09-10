"""同口径 Qlib 主评价（计划 §6）：三臂两窗、复现门禁、判据与图表。

主口径 ``net_daily = return − cost`` 的 ``cumsum`` 曲线；末日差
``D_SG = SAE − G1``、``D_AG = AE − G1``、``D_SA = SAE − AE``。复利与
回撤、费用只作附表。Qlib 回测函数直接复用 ``qlib_mean_comparison.run``
（与 ``0b982e8`` 绑定，内部 shift=1 不再平移）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from sae_residual import config as C
from sae_residual import data as D
from sae_residual.cache_io import load_split_arrays
from sae_residual.train import load_best


def infer_test_signal(arm: str, seed: int, wname: str, stats: dict,
                      heads_dir: Path, expect: dict | None = None
                      ) -> pd.DataFrame:
    """``s_final = s_G1 + σe·r_hat`` → date×instrument 宽表（格=G1 有效格）。

    测试基线值直接来自缓存内嵌的基线信号（构建时从原 parquet 抄录，
    不重新生成 G1 采样路径）。
    """
    import torch

    arrs = load_split_arrays(wname, {**dict(expect or {}), "split": wname})
    model, best = load_best(heads_dir, arm, seed, int(stats["d_in"]))
    x = D.normalize_hidden(arrs["h"], stats)
    model.eval()
    with torch.no_grad():
        r_hat = model(torch.from_numpy(x))[0].numpy().astype(np.float64)
    s_final = arrs["s_g1"] + stats["sigma_e"] * r_hat
    from sae_residual.cache_io import wide_from_arrays

    return wide_from_arrays(arrs["dates"], arrs["instruments"], s_final), best


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

    旧 ``report_W{3,4}_G1_mean.parquet`` 出自 mean_comparison_20260909
    （原 3 臂共同集合）；本函数以同一掩码重跑纯 CPU 回测，误差必须 ≤ 1e-10。
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


def run_comparison(art_dir: Path, heads_dir: Path, stats: dict,
                   seeds: list[int], device_note: str = "cpu",
                   expect: dict | None = None) -> dict:
    """三臂两窗主评价：G1/AE/SAE 同掩码回测 → 曲线/判据/图表/报告。

    AE/SAE 信号格 = G1 有效格（残差头覆盖全部基线格，缓存构建期已断言），
    故三臂共同掩码恒等于 G1 自身掩码——G1 臂即基线原信号回测。
    """
    from qlib_mean_comparison import run as mc

    per_seed: dict[int, dict] = {}
    for seed in seeds:
        arms: dict[str, dict[str, dict]] = {}
        for w, (start, end) in C.TEST_WINDOWS.items():
            g1_wide = load_wide(w)
            sigs = {"G1_mean": g1_wide}
            for arm in C.ARMS:
                wide, best = infer_test_signal(arm, seed, w, stats, heads_dir,
                                               expect)
                # 对齐基线行列顺序后，残差信号格必须与基线逐格一致
                # （无新增/缺失格；pivot 列序为排序序，与生成序不同）
                wide = wide.reindex(index=g1_wide.index, columns=g1_wide.columns)
                assert wide.notna().equals(g1_wide.notna()), \
                    f"{arm} s{seed} {w} 信号格与 G1 不一致"
                sigs[f"{arm}_residual_s{seed}"] = wide
            wides = sigs
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
            # 基准逐值一致断言
            benches = [pd.read_parquet(art_dir / f"report_{w}_{n}.parquet")["bench"]
                       for n in wides]
            for b in benches[1:]:
                assert benches[0].equals(b), f"{w} 基准不一致"
        ae = f"AE_residual_s{seed}"
        sae = f"SAE_residual_s{seed}"
        per_seed[seed] = {
            "arms": arms,
            "D_SG": {w: arms[sae][w]["curve_end"] - arms["G1_mean"][w]["curve_end"]
                     for w in C.TEST_WINDOWS},
            "D_AG": {w: arms[ae][w]["curve_end"] - arms["G1_mean"][w]["curve_end"]
                     for w in C.TEST_WINDOWS},
            "D_SA": {w: arms[sae][w]["curve_end"] - arms[ae][w]["curve_end"]
                     for w in C.TEST_WINDOWS},
            "ae_beats_g1_both": all(
                arms[ae][w]["curve_end"] > arms["G1_mean"][w]["curve_end"]
                for w in C.TEST_WINDOWS),
            "sae_beats_g1_both": all(
                arms[sae][w]["curve_end"] > arms["G1_mean"][w]["curve_end"]
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
    """主图：上=三臂扣费累计（G1/AE/SAE+沪深300 参考），下=三条差额与 0 线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK HK", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    seed = seeds[0]                          # 主图 = pilot seed
    ae, sae = f"AE_residual_s{seed}", f"SAE_residual_s{seed}"
    for w in C.TEST_WINDOWS:
        cur = {n: pd.read_parquet(art_dir / f"daily_{w}_{n}.parquet")["curve"]
               for n in ("G1_mean", ae, sae)}
        bench = pd.read_parquet(art_dir / f"report_{w}_G1_mean.parquet")["bench"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        style = {"G1_mean": ("#12456e", "-"), ae: ("#de7c33", "-"),
                 sae: ("#3a923a", "-")}
        for name, (c, ls) in style.items():
            axes[0].plot(cur[name].index, cur[name], color=c, linestyle=ls,
                         label=name.replace("_", " "))
        axes[0].plot(bench.index, bench.cumsum(), color="black",
                     linestyle="--", label="沪深300")
        axes[0].set_title(f"{w}：扣费累计收益（Qlib 同口径，open 成交，cumsum）")
        axes[0].set_ylabel("累计日收益（相加）")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        d_ag = cur[ae] - cur["G1_mean"]
        d_sg = cur[sae] - cur["G1_mean"]
        d_sa = cur[sae] - cur[ae]
        axes[1].plot(d_ag.index, d_ag, color="#de7c33", label="D_AG = AE−G1")
        axes[1].plot(d_sg.index, d_sg, color="#3a923a", label="D_SG = SAE−G1")
        axes[1].plot(d_sa.index, d_sa, color="#7a5ba6", linestyle=":",
                     label="D_SA = SAE−AE")
        axes[1].axhline(0, color="grey", linewidth=0.8)
        axes[1].set_ylabel("窗口内逐日差额")
        axes[1].set_title(
            f"末日差：D_AG {d_ag.iloc[-1] * 100:+.2f}pp | "
            f"D_SG {d_sg.iloc[-1] * 100:+.2f}pp | "
            f"D_SA {d_sa.iloc[-1] * 100:+.2f}pp")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(art_dir / f"fig_{w}_main.png", dpi=150)
        plt.close(fig)


def judge_confirm_gate(per_seed: dict) -> dict:
    """判据（§6）：某臂「重复超过 G1」= ≥2/3 seed 两窗 D>0 且两窗各自
    三 seed 差的中位数 >0；SAE 额外增益 = SAE 满足且 SAE−AE 同条件。"""
    out = {"n_seeds": len(per_seed)}
    for arm in ("ae", "sae"):
        both_cnt = sum(1 for s in per_seed if per_seed[s][f"{arm}_beats_g1_both"])
        med = {w: float(np.median([per_seed[s][
            "D_SG" if arm == "sae" else "D_AG"][w] for s in sorted(per_seed)]))
            for w in C.TEST_WINDOWS}
        out[arm] = {"both_windows_count": int(both_cnt),
                    "median_delta": med,
                    "repeats": bool(both_cnt >= 2 and all(v > 0 for v in med.values()))}
    # SAE−AE 同条件
    both_cnt = 0
    for s in per_seed:
        p = per_seed[s]
        if all(p["D_SA"][w] > 0 for w in C.TEST_WINDOWS):
            both_cnt += 1
    med_sa = {w: float(np.median([per_seed[s]["D_SA"][w]
                                  for s in sorted(per_seed)]))
        for w in C.TEST_WINDOWS}
    out["sae_over_ae"] = {"both_windows_count": int(both_cnt),
                          "median_delta": med_sa,
                          "repeats": bool(both_cnt >= 2 and all(
                              v > 0 for v in med_sa.values()))}
    return out


__all__ = ["infer_test_signal", "backtest_arm", "load_wide",
           "reproduce_g1_gate", "run_comparison", "plot_main",
           "judge_confirm_gate"]
