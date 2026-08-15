"""G2.3：种子诊断统一封盘判读（计划 §1，20260816 计划）——判据 S1~S4 一次判读。

**预注册判据（跑前冻结，一次判读）**：

| # | 判据 | 冻结定义 |
|---|---|---|
| S1 | 保级 | 三种子（100/101/102）backtest 窗 AER(等权) 的**中位种子**双基准 > 0 |
| S2 | 稳健强度 | 三种子 backtest 窗 AER(等权) 全部 > 0（附注级） |
| S3 | 降级 | 中位种子任一基准 ≤ 0 → G1 降级"疑似种子运气"（与 D1 同标准） |
| S4 | 跨窗一致 | 新种子（101/102）2025H2 较 F0_mean 同窗改善 ≥ +5pp 的复现比例 |

四变体纪律：判据只在 mean；min/max/last 为记录族全表呈现不挑主线。
seed=100 backtest 数字 = 第 5 轮封盘**只读复用**（不重跑）；seed=100 无
2025H2 数据（第 5 轮计划外，S4 只定义在新种子上）。
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PKG_DIR = Path(__file__).resolve().parent
G2_DIR = PKG_DIR / "data" / "g2"
G1_DIR = PKG_DIR / "data" / "g1"
G0_DIR = PKG_DIR / "data" / "g0"
ROUND4_DATA = PKG_DIR / "data"

SEEDS = ("s100", "s101", "s102")
NEW_SEEDS = ("s101", "s102")


def arm_tag(seed: str) -> str:
    """文件/表名里的种子臂标签：s101 → G2S101。"""
    return f"G2{seed.upper()}"


def judge_g2(perf: dict) -> dict:
    """判据 S1~S4（纯函数，tests/test_finetune_g2.py 合成数字单元覆盖）。

    :param perf: 冻结输入结构::

        {
          "backtest": {"s100": {"aer_ew":…, "aer_idx":…}, "s101": …, "s102": …},
          "h2_new_seeds": {"s101": {"aer_ew":…}, "s102": …},
          "f0_mean_h2_aer_ew": float,
        }

    中位种子按三种子 backtest AER(等权) 排序取中。
    """
    bt = perf["backtest"]
    ranked = sorted(SEEDS, key=lambda s: bt[s]["aer_ew"])
    median_seed = ranked[1]
    med = bt[median_seed]

    s1 = med["aer_ew"] > 0 and med["aer_idx"] > 0
    s2 = all(bt[s]["aer_ew"] > 0 for s in SEEDS)
    s3 = not s1  # 冻结定义与 S1 互补：中位种子任一基准 ≤ 0

    f0_h2 = perf["f0_mean_h2_aer_ew"]
    replicated = [
        s for s in NEW_SEEDS
        if perf["h2_new_seeds"][s]["aer_ew"] - f0_h2 >= 0.05
    ]
    ratio = len(replicated) / len(NEW_SEEDS)

    if s1:
        s1_note = (
            f"中位种子 {median_seed} AER(等权)={med['aer_ew']:+.2%} / "
            f"AER(指数)={med['aer_idx']:+.2%} 双基准 > 0 → G1 升级"
            "『多种子存活，待前向确认』"
        )
        s3_note = "未触发（S1 通过）"
    else:
        s1_note = (
            f"中位种子 {median_seed} AER(等权)={med['aer_ew']:+.2%} / "
            f"AER(指数)={med['aer_idx']:+.2%} → 未双双为正"
        )
        s3_note = (
            f"中位种子 {median_seed} 任一基准 ≤ 0 → G1 降级『疑似种子运气』"
            "（与 D1 同标准），G3 登记降为研究性记录并继续"
        )

    return {
        "S1_keep": {
            "median_seed": median_seed,
            "median_aer_ew": med["aer_ew"],
            "median_aer_idx": med["aer_idx"],
            "passed": bool(s1),
            "note": s1_note,
        },
        "S2_all_positive": {
            "aer_ew_by_seed": {s: bt[s]["aer_ew"] for s in SEEDS},
            "passed": bool(s2),
            "note": (
                "三种子 AER(等权) 全部 > 0 → 附注『全种子存活』" if s2 else
                "存在 AER(等权) ≤ 0 的种子 → 不附注全种子存活"
            ),
        },
        "S3_downgrade": {
            "median_seed": median_seed,
            "triggered": bool(s3),
            "note": s3_note,
        },
        "S4_cross_window": {
            "f0_mean_h2_aer_ew": f0_h2,
            "h2_aer_ew_by_new_seed": {
                s: perf["h2_new_seeds"][s]["aer_ew"] for s in NEW_SEEDS
            },
            "improvement_pp": {
                s: perf["h2_new_seeds"][s]["aer_ew"] - f0_h2 for s in NEW_SEEDS
            },
            "replicated_seeds": replicated,
            "ratio": ratio,
            "note": (
                f"新种子 2025H2 较 F0_mean（{f0_h2:+.2%}）改善 ≥+5pp 复现比例 "
                f"{ratio:.0%}（{len(replicated)}/{len(NEW_SEEDS)}）——如实记录，"
                "无通过/失败"
            ),
        },
    }


def parse_epoch_table(console_path: Path) -> tuple[pd.DataFrame, int]:
    """从训练控制台日志解析逐 epoch val loss 与 best epoch（与第 5 轮同款）。"""
    text = console_path.read_text(encoding="utf-8", errors="replace")
    epochs, vals, bests = [], [], []
    cur = None
    for line in text.splitlines():
        m = re.search(r"--- Epoch (\d+)/\d+ Summary ---", line)
        if m:
            cur = int(m.group(1))
            continue
        m = re.search(r"Validation Loss: ([0-9.]+)", line)
        if m and cur is not None and (not epochs or epochs[-1] != cur):
            epochs.append(cur)
            vals.append(float(m.group(1)))
            continue
        if "Best model saved" in line and cur is not None:
            bests.append(cur)
    if not epochs:
        raise ValueError(f"{console_path} 未解析到 epoch 表")
    return pd.DataFrame({"epoch": epochs, "val_loss": vals}), (max(bests) if bests else None)


def main() -> None:
    from baseline_suite.common import VARIANTS, BaselineConfig
    from baseline_suite.pipeline import build_dual_benchmarks, run_group
    from baseline_suite.signal import build_px_tradeable
    from finetune_suite.run_f1_signals import BACKTEST_END, BACKTEST_START
    from kronos_qlib import QlibProvider

    # —— 冻结输入 1：seed=100 backtest = 第 5 轮封盘（只读复用，不重跑）——
    g1 = json.loads((G1_DIR / "g1_backtest_results.json").read_text(encoding="utf-8"))
    bt = {"s100": {
        "aer_ew": g1["groups"]["G1_mean"]["perf_ew"]["aer"],
        "aer_idx": g1["groups"]["G1_mean"]["perf_idx"]["aer"],
    }}
    full_rows: dict[tuple[str, str], dict] = {
        ("s100", "backtest"): {
            v: {
                "aer_ew": g1["groups"][f"G1_{v}"]["perf_ew"]["aer"],
                "aer_idx": g1["groups"][f"G1_{v}"]["perf_idx"]["aer"],
            } for v in VARIANTS
        }
    }

    # —— 冻结输入 2：新种子两窗引擎 + F0/M 对照（每窗同引擎同双基准）——
    for window, start, end, f0_dir, wlabel in (
        ("backtest", BACKTEST_START, BACKTEST_END, ROUND4_DATA, "backtest"),
        ("2025h2", "2025-07-01", "2025-12-31", G0_DIR, "2025h2"),
    ):
        cfg = replace(
            BaselineConfig.load(window="oos"),
            backtest_start=start, backtest_end=end,
        )
        signals: dict[str, pd.DataFrame] = {}
        for s in NEW_SEEDS:
            for v in VARIANTS:
                signals[f"{s}_{v}"] = pd.read_parquet(
                    G2_DIR / s / f"daily_signals_{wlabel}_{arm_tag(s)}_{v}.parquet"
                )
        for v in VARIANTS:
            signals[f"F0_{v}"] = pd.read_parquet(f0_dir / f"daily_signals_{wlabel}_F0_{v}.parquet")
        signals["M"] = pd.read_parquet(f0_dir / f"daily_signals_{wlabel}_M.parquet")

        provider = QlibProvider(cfg.pool, start, end)
        all_cols = sorted(set().union(*[set(df.columns) for df in signals.values()]))
        rebalances = pd.DatetimeIndex(signals["M"].index)
        px, trd = build_px_tradeable(provider, cfg, rebalances, all_cols)
        bench_idx, bench_ew, beta_gap = build_dual_benchmarks(provider, cfg, px, trd)

        for tag, wide in signals.items():
            pi, pe, _, _, _ = run_group(
                wide, px, trd, bench_idx, bench_ew, cfg=cfg, name=f"{window}/{tag}"
            )
            seed, _, variant = tag.partition("_")
            full_rows.setdefault((seed, window), {})[variant] = {
                "aer_ew": pe.aer, "aer_idx": pi.aer,
            }
        print(f"[{window}] 引擎完成：{len(signals)} 组，beta_gap={beta_gap:+.2%}")

        if window == "backtest":
            for s in NEW_SEEDS:
                bt[s] = {
                    "aer_ew": full_rows[(s, "backtest")]["mean"]["aer_ew"],
                    "aer_idx": full_rows[(s, "backtest")]["mean"]["aer_idx"],
                }

    g0 = json.loads((G0_DIR / "g0_backtest_results.json").read_text(encoding="utf-8"))
    perf = {
        "backtest": bt,
        "h2_new_seeds": {
            s: {"aer_ew": full_rows[(s, "2025h2")]["mean"]["aer_ew"]} for s in NEW_SEEDS
        },
        "f0_mean_h2_aer_ew": g0["groups"]["F0_mean"]["perf_ew"]["aer"],
    }
    verdict = judge_g2(perf)

    out = {
        "perf": perf,
        "full_table": {f"{s}@{w}": v for (s, w), v in full_rows.items()},
        "verdict": verdict,
    }
    out_path = G2_DIR / "g2_judge_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    print("=== 预注册判据 S1~S4 封盘判读 ===")
    for k, v in verdict.items():
        head = v.get("passed", v.get("triggered"))
        print(f"[{k}] {'通过/触发' if head else '未通过/未触发'}：{v['note']}")
    print(f"判读落盘：{out_path}")


if __name__ == "__main__":
    main()
