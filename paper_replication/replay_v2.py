"""引擎 v2 全量重放：对全部已封盘信号 parquet 用旧引擎与 v2 同数据重放。

目的（docs/引擎v2重放对照_20260905.md 的数据来源）：

    1. **保真校验**：旧引擎重放数字必须复现已冻结的文档数字（锚点容差
       0.5pp），证明重放的数据构造（窗口/列宇宙/基准）与原运行一致；
    2. **v2 对照**：同份数据上 v2（六修正全开）的 AER/IR 双基准对照表；
    3. **翻转归因**：对符号翻转的臂做六开关逐项关闭归因（每次关一个）。

列宇宙流派（与各实验 runner 一字不差，见各 runner 源码）：

    - ``union``：本族信号 parquet 列并集（paper / baseline / F 族 / G0~G2 /
      G8 / G9 的 A 流派）；
    - ``frozen:backtest`` / ``frozen:2025h2``：G4 起的 UNIVERSE_PARQUETS
      冻结清单（G4/G5/G7/N50/L1/R1/C4 镜像 g5_head）；
    - ``files:<family>``：显式文件清单并集（g2 / g2_supp / g8 的混合清单）；
    - ``own``：单臂自身列（improve 网格 C1~C3，同 run_stage3）。

窗口：paper 2024-07-01~2025-06-30；oos 2025-07-01~2026-07-24；
backtest 2026-01-01~2026-07-24；2025h2 2025-07-01~2025-12-31；
c4 merged 2025-07-01~2026-07-24 剔除前 30 个预热交易日（WARMUP=30）。

不在重放范围（文档中如实声明）：liquidity_strat（非本引擎，外部 qlib_bt +
union 池）、improve R1 规则切换与 S1~S3/R3 分布信号（无独立宽表 parquet 或
非引擎产物）、mamba_head / B3-csi500（csi500 池信号 parquet 未落盘）、
forward 登记 parquet（2026-07-25 后零接触）。

用法::

    python -m paper_replication.replay_v2 [--fam g5_backtest] [--no-attr]

产出 ``paper_replication/data/v2_replay_results.json``。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from paper_replication.benchmark import probe_index_benchmark
from paper_replication.common import DATA_DIR, REPO_ROOT
from paper_replication.engine import (
    EngineConfig,
    attach_benchmark,
    compute_perf,
    run_portfolio,
)
from paper_replication.engine_v2 import (
    EngineConfigV2,
    build_limit_masks,
    build_pool_equal_weight_benchmark_v2,
    compute_perf_v2,
    run_portfolio_v2,
)

V4 = ("last", "mean", "max", "min")
FIX_KEYS = (
    "fix_double_sided_cost",
    "fix_delay_1",
    "fix_limit_block",
    "fix_new_leg_drift",
    "fix_annualization_252",
)

# —— 冻结宇宙清单（B 流派；与 g5_head/run_g5_eval.py::UNIVERSE_PARQUETS 一致）——
_FROZEN_FILES = {
    "backtest": [
        *[f"finetune_suite/data/g1/daily_signals_backtest_G1_{v}.parquet" for v in V4],
        *[f"finetune_suite/data/daily_signals_backtest_{a}_{v}.parquet" for a in ("F1", "F0") for v in V4],
        "finetune_suite/data/daily_signals_backtest_M.parquet",
    ],
    "2025h2": [
        *[f"finetune_suite/data/g0/daily_signals_2025h2_{a}_{v}.parquet" for a in ("G0", "F0") for v in V4],
        "finetune_suite/data/g0/daily_signals_2025h2_M.parquet",
    ],
}

F = "paper_replication/data"
B = "baseline_suite/data"
FS = "finetune_suite/data"
G5 = "g5_head/data"
G4 = "g4_features/data"
G7 = "g7_shortwindow/data"
G9 = "g9_ckpt/data"
N5 = "n50_amplify/data"
L1 = "l1_context/data"
R1 = "r1_objective/data"
C4 = "c4_temporal/data"
IM = "improve_suite/data"


def _v4(dirpath, prefix, suffix):
    return {f"{prefix}_{v}": f"{dirpath}/{prefix}_{v}{suffix}" for v in V4}


FAMILIES: list[dict] = [
    # ============ paper 窗（2024-07-01 ~ 2025-06-30）============
    dict(
        fam="paper", start="2024-07-01", end="2025-06-30", universe="union",
        arms={
            "K": f"{F}/daily_signals_K.parquet",
            "M": f"{F}/daily_signals_M.parquet",
            "P": f"{F}/daily_signals_P.parquet",
            "R": f"{F}/daily_signals_R.parquet",
        },
        anchors={"K": (0.0447, 0.0899), "M": (-0.1169, -0.0786),
                 "R": (-0.0363, 0.0063), "P": (-0.0014, 0.0429)},
    ),
    dict(
        fam="baseline_paper", start="2024-07-01", end="2025-06-30", universe="union",
        arms={
            **_v4(B, "daily_signals_paper", ".parquet"),
            "M": f"{B}/daily_signals_paper_M.parquet",
            "P": f"{B}/daily_signals_paper_P.parquet",
            "R": f"{B}/daily_signals_paper_R.parquet",
        },
        rename={"daily_signals_paper_last": "last", "daily_signals_paper_mean": "mean",
                "daily_signals_paper_max": "max", "daily_signals_paper_min": "min"},
        anchors={"mean": (0.0447, 0.0899), "last": (0.0783, 0.1252),
                 "max": (0.0710, 0.1171), "min": (0.0587, 0.1052),
                 "M": (-0.1169, -0.0786), "R": (-0.0363, 0.0063),
                 "P": (-0.0014, 0.0429)},
    ),
    dict(
        fam="baseline_oos", start="2025-07-01", end="2026-07-24", universe="union",
        arms={
            **_v4(B, "daily_signals_oos", ".parquet"),
            "M": f"{B}/daily_signals_oos_M.parquet",
            "P": f"{B}/daily_signals_oos_P.parquet",
            "R": f"{B}/daily_signals_oos_R.parquet",
        },
        rename={"daily_signals_oos_last": "last", "daily_signals_oos_mean": "mean",
                "daily_signals_oos_max": "max", "daily_signals_oos_min": "min"},
        anchors={"mean": (-0.1510, -0.1429), "last": (-0.1725, -0.1648),
                 "max": (-0.1494, -0.1411), "min": (-0.1770, -0.1688),
                 "M": (0.2140, 0.2328), "R": (-0.1598, -0.1490),
                 "P": (-0.0653, -0.0544)},
    ),
    # ============ improve 网格 C1~C3（own 宇宙，同 run_stage3）============
    dict(
        fam="improve_paper", start="2024-07-01", end="2025-06-30", universe="own",
        arms={
            "C1_L8": f"{IM}/daily_signals_paper_L8_H5_T1.0_mean.parquet",
            "C2_L30": f"{IM}/daily_signals_paper_L30_H5_T1.0_mean.parquet",
            "C3_L90T06": f"{IM}/daily_signals_paper_L90_H10_T0.6_mean.parquet",
        },
        anchors={"C1_L8": (-0.0376, None), "C2_L30": (-0.0558, None),
                 "C3_L90T06": (0.0246, None)},
    ),
    dict(
        fam="improve_oos", start="2025-07-01", end="2026-07-24", universe="own",
        arms={
            "C1_L8": f"{IM}/daily_signals_oos_L8_H5_T1.0_mean.parquet",
            "C2_L30": f"{IM}/daily_signals_oos_L30_H5_T1.0_mean.parquet",
            "C3_L90T06": f"{IM}/daily_signals_oos_L90_H10_T0.6_mean.parquet",
        },
        anchors={"C1_L8": (-0.1135, None), "C2_L30": (-0.1296, None),
                 "C3_L90T06": (-0.1635, None)},
    ),
    # ============ backtest 窗（2026-01-01 ~ 2026-07-24）============
    dict(
        fam="f_backtest", start="2026-01-01", end="2026-07-24", universe="union",
        arms={
            **_v4(FS, "daily_signals_backtest_F1", ".parquet"),
            **_v4(FS, "daily_signals_backtest_F0", ".parquet"),
            "M": f"{FS}/daily_signals_backtest_M.parquet",
        },
        rename={f"daily_signals_backtest_F1_{v}": f"F1_{v}" for v in V4}
        | {f"daily_signals_backtest_F0_{v}": f"F0_{v}" for v in V4},
        anchors={"F1_mean": (-0.0424, None), "F0_mean": (-0.1723, None),
                 "M": (0.3602, 0.3094)},
    ),
    dict(
        fam="g1_backtest", start="2026-01-01", end="2026-07-24", universe="union",
        arms={
            **_v4(f"{FS}/g1", "daily_signals_backtest_G1", ".parquet"),
            **_v4(FS, "daily_signals_backtest_F1", ".parquet"),
            **_v4(FS, "daily_signals_backtest_F0", ".parquet"),
            "M": f"{FS}/daily_signals_backtest_M.parquet",
        },
        rename={f"daily_signals_backtest_G1_{v}": f"G1_{v}" for v in V4}
        | {f"daily_signals_backtest_F1_{v}": f"F1_{v}" for v in V4}
        | {f"daily_signals_backtest_F0_{v}": f"F0_{v}" for v in V4},
        anchors={"G1_mean": (0.1433, 0.1066), "G1_last": (0.1844, None),
                 "G1_max": (0.2928, None), "G1_min": (0.1763, None),
                 "F1_mean": (-0.0424, None), "F0_mean": (-0.1723, None),
                 "M": (0.3602, 0.3094)},
    ),
    dict(
        fam="g2_backtest", start="2026-01-01", end="2026-07-24", universe="files",
        files=[
            *[f"{FS}/g2/s101/daily_signals_backtest_G2S101_{v}.parquet" for v in V4],
            *[f"{FS}/g2/s102/daily_signals_backtest_G2S102_{v}.parquet" for v in V4],
            *[f"{FS}/daily_signals_backtest_F0_{v}.parquet" for v in V4],
            f"{FS}/daily_signals_backtest_M.parquet",
        ],
        arms={
            **_v4(f"{FS}/g2/s101", "daily_signals_backtest_G2S101", ".parquet"),
            **_v4(f"{FS}/g2/s102", "daily_signals_backtest_G2S102", ".parquet"),
            **_v4(FS, "daily_signals_backtest_F0", ".parquet"),
            "M": f"{FS}/daily_signals_backtest_M.parquet",
        },
        rename={f"daily_signals_backtest_G2S101_{v}": f"G2S101_{v}" for v in V4}
        | {f"daily_signals_backtest_G2S102_{v}": f"G2S102_{v}" for v in V4}
        | {f"daily_signals_backtest_F0_{v}": f"F0_{v}" for v in V4},
        anchors={"G2S101_mean": (0.2906, 0.2484), "G2S102_mean": (0.1867, 0.1464),
                 "F0_mean": (-0.1723, None), "M": (0.3602, 0.3094)},
    ),
    dict(
        fam="g2_supp_backtest", start="2026-01-01", end="2026-07-24", universe="files",
        files=[
            *[f"{FS}/g2/s103/daily_signals_backtest_G2S103_{v}.parquet" for v in V4],
            *[f"{FS}/g2/s104/daily_signals_backtest_G2S104_{v}.parquet" for v in V4],
            *[f"{FS}/g2/dtok/daily_signals_backtest_DTOK_{v}.parquet" for v in V4],
            *[f"{FS}/daily_signals_backtest_F0_{v}.parquet" for v in V4],
            f"{FS}/daily_signals_backtest_M.parquet",
        ],
        arms={
            **_v4(f"{FS}/g2/s103", "daily_signals_backtest_G2S103", ".parquet"),
            **_v4(f"{FS}/g2/s104", "daily_signals_backtest_G2S104", ".parquet"),
            **_v4(f"{FS}/g2/dtok", "daily_signals_backtest_DTOK", ".parquet"),
        },
        rename={f"daily_signals_backtest_G2S103_{v}": f"G2S103_{v}" for v in V4}
        | {f"daily_signals_backtest_G2S104_{v}": f"G2S104_{v}" for v in V4}
        | {f"daily_signals_backtest_DTOK_{v}": f"DTOK_{v}" for v in V4},
        anchors={"G2S103_mean": (0.2036, None), "G2S104_mean": (0.1074, None),
                 "DTOK_mean": (0.2665, None)},
    ),
    dict(
        fam="g8_backtest", start="2026-01-01", end="2026-07-24", universe="files",
        files=[
            *[f"{FS}/g8/s10{i}/daily_signals_backtest_G8S10{i}_{v}.parquet" for i in (0, 1, 2) for v in V4],
            *[f"{FS}/daily_signals_backtest_F0_{v}.parquet" for v in V4],
            f"{FS}/daily_signals_backtest_M.parquet",
        ],
        arms={
            **_v4(f"{FS}/g8/s100", "daily_signals_backtest_G8S100", ".parquet"),
            **_v4(f"{FS}/g8/s101", "daily_signals_backtest_G8S101", ".parquet"),
            **_v4(f"{FS}/g8/s102", "daily_signals_backtest_G8S102", ".parquet"),
        },
        rename={f"daily_signals_backtest_G8S10{i}_{v}": f"G8S10{i}_{v}" for i in (0, 1, 2) for v in V4},
        anchors={"G8S100_mean": (-0.0012, -0.0364), "G8S101_mean": (-0.0311, -0.0659),
                 "G8S102_mean": (-0.0367, -0.0709)},
    ),
    dict(
        fam="g9_backtest", start="2026-01-01", end="2026-07-24", universe="union",
        arms={
            **_v4(f"{G9}/e0", "daily_signals_backtest_G9E0", ".parquet"),
            **_v4(f"{G9}/e1", "daily_signals_backtest_G9E1", ".parquet"),
            **_v4(f"{G9}/e5", "daily_signals_backtest_G9E5", ".parquet"),
            **_v4(f"{G9}/e10", "daily_signals_backtest_G9E10", ".parquet"),
            **_v4(f"{G9}/e15", "daily_signals_backtest_G9E15", ".parquet"),
            **_v4(FS, "daily_signals_backtest_F0", ".parquet"),
            "M": f"{FS}/daily_signals_backtest_M.parquet",
        },
        rename={f"daily_signals_backtest_G9E{e}_{v}": f"G9E{e}_{v}" for e in (0, 1, 5, 10, 15) for v in V4}
        | {f"daily_signals_backtest_F0_{v}": f"F0_{v}" for v in V4},
        anchors={"G9E1_mean": (0.1433, 0.1066), "G9E15_mean": (-0.0219, -0.0620),
                 "G9E0_mean": (-0.1163, -0.1592), "G9E5_mean": (-0.0209, None),
                 "G9E10_mean": (-0.0206, None), "F0_mean": (-0.1723, None)},
    ),
    dict(
        fam="g5_backtest", start="2026-01-01", end="2026-07-24", universe="frozen:backtest",
        arms={
            "H-kda_s42": f"{G5}/daily_signals_backtest_H-kda_s42.parquet",
            "H-kda_s43": f"{G5}/daily_signals_backtest_H-kda_s43.parquet",
            "H-kda_s44": f"{G5}/daily_signals_backtest_H-kda_s44.parquet",
            "H-lin_s42": f"{G5}/daily_signals_backtest_H-lin_s42.parquet",
            "H-mamba_s42": f"{G5}/daily_signals_backtest_H-mamba_s42.parquet",
            "H-mamba_s43": f"{G5}/daily_signals_backtest_H-mamba_s43.parquet",
            "H-mamba_s44": f"{G5}/daily_signals_backtest_H-mamba_s44.parquet",
        },
        anchors={"H-kda_s42": (-0.1348, -0.1746), "H-kda_s43": (-0.1432, -0.1831),
                 "H-kda_s44": (-0.1965, -0.2335), "H-lin_s42": (-0.0195, -0.0645),
                 "H-mamba_s42": (-0.2431, -0.2791), "H-mamba_s43": (-0.2147, -0.2520),
                 "H-mamba_s44": (-0.0770, -0.1204)},
    ),
    dict(
        fam="g4_backtest", start="2026-01-01", end="2026-07-24", universe="frozen:backtest",
        arms={
            **_v4(f"{G4}/s100", "daily_signals_backtest_G4S100", ".parquet"),
            **_v4(f"{G4}/s101", "daily_signals_backtest_G4S101", ".parquet"),
            **_v4(f"{G4}/s102", "daily_signals_backtest_G4S102", ".parquet"),
        },
        rename={f"daily_signals_backtest_G4S10{i}_{v}": f"G4S10{i}_{v}" for i in (0, 1, 2) for v in V4},
        anchors={"G4S100_mean": (0.1255, 0.0897), "G4S101_mean": (0.1168, 0.0807),
                 "G4S102_mean": (0.0085, -0.0244)},
    ),
    dict(
        fam="g7_backtest", start="2026-01-01", end="2026-07-24", universe="frozen:backtest",
        arms={
            **_v4(f"{G7}/s100", "daily_signals_backtest_W85S100", ".parquet"),
            **_v4(f"{G7}/s101", "daily_signals_backtest_W85S101", ".parquet"),
            **_v4(f"{G7}/s102", "daily_signals_backtest_W85S102", ".parquet"),
        },
        rename={f"daily_signals_backtest_W85S10{i}_{v}": f"W85S10{i}_{v}" for i in (0, 1, 2) for v in V4},
        anchors={"W85S100_mean": (-0.2204, -0.2530), "W85S101_mean": (-0.1598, -0.1942),
                 "W85S102_mean": (-0.2253, -0.2576)},
    ),
    dict(
        fam="n50_backtest", start="2026-01-01", end="2026-07-24", universe="frozen:backtest",
        arms={
            **_v4(f"{N5}/s100", "daily_signals_backtest_G1N50S100", ".parquet"),
            **_v4(f"{N5}/s101", "daily_signals_backtest_G1N50S101", ".parquet"),
            **_v4(f"{N5}/s102", "daily_signals_backtest_G1N50S102", ".parquet"),
        },
        rename={f"daily_signals_backtest_G1N50S10{i}_{v}": f"G1N50S10{i}_{v}" for i in (0, 1, 2) for v in V4},
        anchors={"G1N50S100_mean": (0.1913, 0.1527), "G1N50S101_mean": (0.1348, 0.0971),
                 "G1N50S102_mean": (0.1390, 0.0998)},
    ),
    dict(
        fam="l1_backtest", start="2026-01-01", end="2026-07-24", universe="frozen:backtest",
        arms={
            **_v4(f"{L1}/L250ZS100", "daily_signals_backtest_L1L250ZS100", ".parquet"),
            **_v4(f"{L1}/L250ZS101", "daily_signals_backtest_L1L250ZS101", ".parquet"),
            **_v4(f"{L1}/L250ZS102", "daily_signals_backtest_L1L250ZS102", ".parquet"),
            **_v4(f"{L1}/L500ZS100", "daily_signals_backtest_L1L500ZS100", ".parquet"),
            **_v4(f"{L1}/L250FT100", "daily_signals_backtest_L1L250FT100", ".parquet"),
        },
        rename={f"daily_signals_backtest_L1{n}_{v}": f"{n}_{v}"
                for n in ("L250ZS100", "L250ZS101", "L250ZS102", "L500ZS100", "L250FT100") for v in V4},
        anchors={"L250ZS100_mean": (0.0480, None), "L250ZS101_mean": (0.0482, None),
                 "L250ZS102_mean": (0.0554, None), "L500ZS100_mean": (-0.1903, None),
                 "L250FT100_mean": (-0.0449, None)},
    ),
    dict(
        fam="r1_backtest", start="2026-01-01", end="2026-07-24", universe="frozen:backtest",
        arms={f"R-{h}_s{s}": f"{R1}/daily_signals_backtest_R-{h}_s{s}.parquet"
              for h in ("lin", "kda") for s in (42, 43, 44)},
        anchors={"R-lin_s44": (-0.1232, -0.1645), "R-kda_s43": (-0.0118, -0.0570)},
    ),
    # ============ 2025H2 窗（2025-07-01 ~ 2025-12-31）============
    dict(
        fam="g0_2025h2", start="2025-07-01", end="2025-12-31", universe="union",
        arms={
            **_v4(f"{FS}/g0", "daily_signals_2025h2_G0", ".parquet"),
            **_v4(f"{FS}/g0", "daily_signals_2025h2_F0", ".parquet"),
            "M": f"{FS}/g0/daily_signals_2025h2_M.parquet",
        },
        rename={f"daily_signals_2025h2_G0_{v}": f"G0_{v}" for v in V4}
        | {f"daily_signals_2025h2_F0_{v}": f"F0_{v}" for v in V4},
        anchors={"G0_mean": (0.0310, 0.1173), "F0_mean": (-0.1679, None)},
    ),
    dict(
        fam="g2_2025h2", start="2025-07-01", end="2025-12-31", universe="files",
        files=[
            *[f"{FS}/g2/s101/daily_signals_2025h2_G2S101_{v}.parquet" for v in V4],
            *[f"{FS}/g2/s102/daily_signals_2025h2_G2S102_{v}.parquet" for v in V4],
            *[f"{FS}/g0/daily_signals_2025h2_F0_{v}.parquet" for v in V4],
            f"{FS}/g0/daily_signals_2025h2_M.parquet",
        ],
        arms={
            **_v4(f"{FS}/g2/s101", "daily_signals_2025h2_G2S101", ".parquet"),
            **_v4(f"{FS}/g2/s102", "daily_signals_2025h2_G2S102", ".parquet"),
            **_v4(f"{FS}/g0", "daily_signals_2025h2_F0", ".parquet"),
            "M": f"{FS}/g0/daily_signals_2025h2_M.parquet",
        },
        rename={f"daily_signals_2025h2_G2S101_{v}": f"G2S101_{v}" for v in V4}
        | {f"daily_signals_2025h2_G2S102_{v}": f"G2S102_{v}" for v in V4}
        | {f"daily_signals_2025h2_F0_{v}": f"F0_{v}" for v in V4},
        anchors={"G2S101_mean": (0.1116, None), "G2S102_mean": (0.0390, None),
                 "F0_mean": (-0.1679, None)},
    ),
    dict(
        fam="g5_2025h2", start="2025-07-01", end="2025-12-31", universe="frozen:2025h2",
        arms={
            **_v4(G5, "daily_signals_2025h2_G1", ".parquet"),
            "H-kda_s42": f"{G5}/daily_signals_2025h2_H-kda_s42.parquet",
            "H-kda_s43": f"{G5}/daily_signals_2025h2_H-kda_s43.parquet",
            "H-kda_s44": f"{G5}/daily_signals_2025h2_H-kda_s44.parquet",
            "H-lin_s42": f"{G5}/daily_signals_2025h2_H-lin_s42.parquet",
            "H-mamba_s42": f"{G5}/daily_signals_2025h2_H-mamba_s42.parquet",
            "H-mamba_s43": f"{G5}/daily_signals_2025h2_H-mamba_s43.parquet",
            "H-mamba_s44": f"{G5}/daily_signals_2025h2_H-mamba_s44.parquet",
        },
        rename={f"daily_signals_2025h2_G1_{v}": f"G1_{v}" for v in V4},
        anchors={"G1_mean": (0.0595, 0.1484)},
    ),
    dict(
        fam="g4_2025h2", start="2025-07-01", end="2025-12-31", universe="frozen:2025h2",
        arms={
            **_v4(f"{G4}/s100", "daily_signals_2025h2_G4S100", ".parquet"),
            **_v4(f"{G4}/s101", "daily_signals_2025h2_G4S101", ".parquet"),
            **_v4(f"{G4}/s102", "daily_signals_2025h2_G4S102", ".parquet"),
        },
        rename={f"daily_signals_2025h2_G4S10{i}_{v}": f"G4S10{i}_{v}" for i in (0, 1, 2) for v in V4},
        anchors={"G4S100_mean": (0.0929, 0.1857), "G4S101_mean": (0.1903, 0.2922),
                 "G4S102_mean": (0.2227, 0.3279)},
    ),
    dict(
        fam="g9_2025h2", start="2025-07-01", end="2025-12-31", universe="union",
        arms={
            **_v4(f"{G9}/e1", "daily_signals_2025h2_G9E1", ".parquet"),
            **_v4(f"{G9}/e15", "daily_signals_2025h2_G9E15", ".parquet"),
            **_v4(f"{FS}/g0", "daily_signals_2025h2_F0", ".parquet"),
            "M": f"{FS}/g0/daily_signals_2025h2_M.parquet",
        },
        rename={f"daily_signals_2025h2_G9E{e}_{v}": f"G9E{e}_{v}" for e in (1, 15) for v in V4}
        | {f"daily_signals_2025h2_F0_{v}": f"F0_{v}" for v in V4},
        anchors={"G9E1_mean": (0.0595, 0.1484), "G9E15_mean": (-0.0290, 0.0519),
                 "F0_mean": (-0.1679, None)},
    ),
    dict(
        fam="l1_2025h2", start="2025-07-01", end="2025-12-31", universe="frozen:2025h2",
        arms={
            **_v4(f"{L1}/L250ZS100", "daily_signals_2025h2_L1L250ZS100", ".parquet"),
            **_v4(f"{L1}/L250ZS101", "daily_signals_2025h2_L1L250ZS101", ".parquet"),
            **_v4(f"{L1}/L250ZS102", "daily_signals_2025h2_L1L250ZS102", ".parquet"),
        },
        rename={f"daily_signals_2025h2_L1L250ZS10{i}_{v}": f"L250ZS10{i}_{v}" for i in (0, 1, 2) for v in V4},
        anchors={"L250ZS100_mean": (0.2718, None), "L250ZS101_mean": (0.1985, None),
                 "L250ZS102_mean": (0.1933, None)},
    ),
    dict(
        fam="r1_2025h2", start="2025-07-01", end="2025-12-31", universe="frozen:2025h2",
        arms={f"R-{h}_s{s}": f"{R1}/daily_signals_2025h2_R-{h}_s{s}.parquet"
              for h in ("lin", "kda") for s in (42, 43, 44)},
        anchors={},  # 文档只给了三种子中位（-11.69/-26.36），不钉单种子锚
    ),
    # ============ C4 merged 窗（2025-07-01 ~ 2026-07-24，剔前 30 交易日）============
    dict(
        fam="c4_merged", start="2025-07-01", end="2026-07-24", universe="frozen:both",
        warmup=30,
        arms={
            "C4S100_c4": f"{C4}/s100/daily_signals_merged_C4S100_c4.parquet",
            "C4S101_c4": f"{C4}/s101/daily_signals_merged_C4S101_c4.parquet",
            "C4S102_c4": f"{C4}/s102/daily_signals_merged_C4S102_c4.parquet",
            # G1 merged 对照（transform.load_g1_mean_merged 同构：2025h2 + backtest 拼接）
            "G1m_s100": f"{G5}/daily_signals_2025h2_G1_mean.parquet|{FS}/g1/daily_signals_backtest_G1_mean.parquet",
            "G1m_s101": f"{FS}/g2/s101/daily_signals_2025h2_G2S101_mean.parquet|{FS}/g2/s101/daily_signals_backtest_G2S101_mean.parquet",
            "G1m_s102": f"{FS}/g2/s102/daily_signals_2025h2_G2S102_mean.parquet|{FS}/g2/s102/daily_signals_backtest_G2S102_mean.parquet",
        },
        anchors={"C4S100_c4": (-0.0033, -0.0072), "C4S101_c4": (-0.0265, -0.0317),
                 "C4S102_c4": (-0.0403, -0.0459), "G1m_s100": (0.1243, None),
                 "G1m_s101": (0.2503, None), "G1m_s102": (0.1486, None)},
    ),
]

# 范围外（文档声明用）
OUT_OF_SCOPE = [
    "liquidity_strat（外部 qlib_bt 引擎 + union 池，非 paper_replication 引擎）",
    "improve R1 规则切换 / S1~S3 / R3 分布信号（无独立日频宽表 parquet 或非引擎产物）",
    "mamba_head T1 / B3-csi500（csi500 池信号 parquet 未落盘）",
    "finetune_suite/registry forward 登记 parquet（2026-07-25 后零接触纪律）",
]


def _load_signal(relpath: str) -> pd.DataFrame:
    if "|" in relpath:  # merged 拼接（c4 的 G1 对照）
        parts = [pd.read_parquet(REPO_ROOT / p) for p in relpath.split("|")]
        return pd.concat(parts).sort_index()
    return pd.read_parquet(REPO_ROOT / relpath)


def fetch_window(provider, cols, start, end):
    """一次拉窗口数据：px / tradeable / up_down_limit_status（同 build_px_tradeable 口径）。"""
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = start
        provider._end_date = end
        provider.instruments_ = list(cols)
        df = provider.fetch(
            ["$close", "$tradestatuscode", "$up_down_limit_status"], freq="day"
        )
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig
    px = df["close"].unstack("instrument").sort_index()
    tsc = df["tradestatuscode"].unstack("instrument").sort_index().reindex_like(px)
    uls = df["up_down_limit_status"].unstack("instrument").sort_index().reindex_like(px)
    if (uls != 0).sum().sum() == 0:
        raise RuntimeError("up_down_limit_status 全零——DDB 字段缺失，涨跌停修正不可用")
    trd = (tsc == -1).fillna(False) & px.notna()
    return px, trd, uls


def _universe_cols(spec, signals):
    mode = spec["universe"]
    if mode == "union":
        return sorted(set().union(*[set(s.columns) for s in signals.values()]))
    if mode == "own":
        return None  # 逐臂各自 fetch
    if mode.startswith("frozen:"):
        key = mode.split(":", 1)[1]
        files = _FROZEN_FILES["backtest"] + _FROZEN_FILES["2025h2"] if key == "both" else _FROZEN_FILES[key]
        return sorted(set().union(*[set(pd.read_parquet(REPO_ROOT / f).columns) for f in files]))
    if mode == "files":
        return sorted(set().union(*[set(pd.read_parquet(REPO_ROOT / f).columns) for f in spec["files"]]))
    raise ValueError(mode)


def _arm_name(spec, key):
    return spec.get("rename", {}).get(key, key)


def run_family(spec, provider, *, attribution=True) -> dict:
    from paper_replication.benchmark import build_pool_equal_weight_benchmark

    fam = spec["fam"]
    raw = {k: _load_signal(v) for k, v in spec["arms"].items()}
    arms = {_arm_name(spec, k): df for k, df in raw.items()}
    anchors = {_arm_name(spec, k): v for k, v in spec.get("anchors", {}).items()}
    warmup = spec.get("warmup", 0)

    uni = _universe_cols(spec, raw)
    own_mode = uni is None
    groups: dict[str, list[str]] = {}  # fetch 组 → 臂
    if own_mode:
        for a in arms:
            groups[f"own:{a}"] = [a]
    else:
        groups["shared"] = list(arms)

    out = {"fam": fam, "window": [spec["start"], spec["end"]], "warmup": warmup,
           "universe": spec["universe"], "arms": {}}
    cache: dict[str, tuple] = {}

    for gname, arm_list in groups.items():
        if gname == "shared":
            px, trd, uls = fetch_window(provider, uni, spec["start"], spec["end"])
            bench_idx = probe_index_benchmark(provider, spec["start"], spec["end"])
            bench_ew_legacy = build_pool_equal_weight_benchmark(px, trd)
            cache[gname] = (px, trd, uls, bench_idx, bench_ew_legacy)
        else:
            a = arm_list[0]
            cols = sorted(set(arms[a].columns))
            px, trd, uls = fetch_window(provider, cols, spec["start"], spec["end"])
            bench_idx = probe_index_benchmark(provider, spec["start"], spec["end"])
            bench_ew_legacy = build_pool_equal_weight_benchmark(px, trd)
            cache[gname] = (px, trd, uls, bench_idx, bench_ew_legacy)

    for gname, arm_list in groups.items():
        px, trd, uls, bench_idx, bench_ew_legacy = cache[gname]
        if warmup:
            px, trd, uls = px.iloc[warmup:], trd.iloc[warmup:], uls.iloc[warmup:]
            bench_ew_legacy = bench_ew_legacy.reindex(px.index).dropna()
        masks = build_limit_masks(uls_wide=uls)
        for a in arm_list:
            sig = arms[a].reindex(index=px.index, columns=px.columns)
            # —— 旧引擎（244 / legacy 基准）——
            old_cfg = EngineConfig()
            old_ret, _, old_trades = run_portfolio(sig, px, trd, cfg=old_cfg)
            p_old_idx = compute_perf(attach_benchmark(old_ret, bench_idx), old_trades, name=a)
            p_old_ew = compute_perf(attach_benchmark(old_ret, bench_ew_legacy), old_trades, name=a)
            # —— v2（六修正全开；等权基准逐臂掩码）——
            v2_ret, v2_trades = run_portfolio_v2(
                sig, px, trd, cfg=EngineConfigV2(),
                buy_blocked=masks[0], sell_blocked=masks[1],
            )
            bench_ew_v2 = build_pool_equal_weight_benchmark_v2(px, trd, sig, fix_mask=True)
            p_v2_idx = compute_perf_v2(_excess(v2_ret, bench_idx), v2_trades, name=a)
            p_v2_ew = compute_perf_v2(_excess(v2_ret, bench_ew_v2), v2_trades, name=a)

            row = {
                "old_aer_idx": p_old_idx.aer, "old_ir_idx": p_old_idx.ir,
                "old_aer_ew": p_old_ew.aer, "old_ir_ew": p_old_ew.ir,
                "v2_aer_idx": p_v2_idx.aer, "v2_ir_idx": p_v2_idx.ir,
                "v2_aer_ew": p_v2_ew.aer, "v2_ir_ew": p_v2_ew.ir,
                "n_days": p_old_idx.n_days,
                "anchor_ew": anchors.get(a, (None, None))[0],
                "anchor_idx": anchors.get(a, (None, None))[1],
                "fidelity_ok": None,
            }
            ew_a, idx_a = row["anchor_ew"], row["anchor_idx"]
            dews = abs(p_old_ew.aer - ew_a) if ew_a is not None else None
            didx = abs(p_old_idx.aer - idx_a) if idx_a is not None else None
            if dews is not None or didx is not None:
                worst = max(x for x in (dews, didx) if x is not None)
                row["fidelity_ok"] = bool(worst < 0.005)
                row["fidelity_delta"] = worst

            # 翻转归因：任一基准 AER 符号改变 → 逐项关一个开关重跑
            flipped = (
                np.sign(row["old_aer_ew"]) != np.sign(row["v2_aer_ew"])
                or np.sign(row["old_aer_idx"]) != np.sign(row["v2_aer_idx"])
            )
            row["sign_flipped"] = bool(flipped)
            row["attribution"] = {}
            if flipped and attribution:
                for fk in FIX_KEYS:
                    kw = {fk: False}
                    cfg_x = EngineConfigV2(**kw)
                    rx, tx = run_portfolio_v2(
                        sig, px, trd, cfg=cfg_x, buy_blocked=masks[0], sell_blocked=masks[1]
                    )
                    b_ew_x = bench_ew_v2 if fk != "fix_ew_benchmark_mask" else bench_ew_legacy
                    p_ew_x = compute_perf_v2(_excess(rx, b_ew_x), tx, name=a)
                    p_idx_x = compute_perf_v2(_excess(rx, bench_idx), tx, name=a)
                    row["attribution"][fk] = {
                        "aer_ew": p_ew_x.aer, "aer_idx": p_idx_x.aer,
                    }
                # (5) 基准掩码单独归因：引擎不动，只换基准
                p_ew_legacy_mask = compute_perf_v2(_excess(v2_ret, bench_ew_legacy), v2_trades, name=a)
                row["attribution"]["fix_ew_benchmark_mask"] = {"aer_ew": p_ew_legacy_mask.aer}
            out["arms"][a] = row
            logger.info(
                f"[{fam}] {a}: old(ew)={p_old_ew.aer:+.2%} v2(ew)={p_v2_ew.aer:+.2%} | "
                f"old(idx)={p_old_idx.aer:+.2%} v2(idx)={p_v2_idx.aer:+.2%}"
                + (f" | 锚Δ={row.get('fidelity_delta', float('nan')):.2%}"
                   if row.get("fidelity_delta") is not None else "")
                + (" | 符号翻转!" if flipped else "")
            )
    return out


def _excess(daily_ret: pd.Series, bench: pd.Series) -> pd.Series:
    common = daily_ret.index.intersection(bench.index)
    return daily_ret.loc[common] - bench.loc[common]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default=None, help="只跑指定族（逗号分隔），默认全部")
    ap.add_argument("--no-attr", action="store_true", help="跳过翻转归因")
    args = ap.parse_args()

    from kronos_qlib import QlibProvider

    specs = FAMILIES
    if args.fam:
        want = set(args.fam.split(","))
        specs = [s for s in FAMILIES if s["fam"] in want]
    provider = QlibProvider("csi300", "2024-07-01", "2026-07-24")

    results = []
    for spec in specs:
        logger.info(f"===== {spec['fam']} [{spec['start']}~{spec['end']}] =====")
        results.append(run_family(spec, provider, attribution=not args.no_attr))

    out_path = DATA_DIR / "v2_replay_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"out_of_scope": OUT_OF_SCOPE, "families": results},
            f, indent=2, ensure_ascii=False, default=float,
        )
    logger.info(f"重放结果落盘 {out_path}")


if __name__ == "__main__":
    main()
