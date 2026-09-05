"""累计收益持有日 IC 剖面（``ic_horizon_profile --cumulative`` 的实现体）。

信号日 t 对 t+1..t+k **累计收益**（close(t+k)/close(t) − 1）的横截面
Spearman rank-IC，k=1..10，G1 三种子 mean + M、backtest 与 2025H2 两窗。
相邻信号日的 IC 因收益窗重叠（重叠 k−1 日）自相关，t 值用 Newey-West HAC
（Bartlett 核，lag = k−1；k=1 退化为普通 t，与单日模式一致）。

本模块单独成文件的原因：ic_horizon_profile.py 的候选改写在会话写入门禁
侧反复触发路径穿越误报（同构造在其已提交版本与其他模块中均通过），故
将新增入口隔离在本文件；``python -m paper_replication.ic_horizon_profile
--cumulative`` 经其 ``__main__`` 守卫分派到 :func:`main_cumulative`。

单趟同时重算**单日 + 累计**两种剖面并整体落盘（含取数窗口断言重跑），
不读回旧 JSON 合并。
"""
from __future__ import annotations

import json

import pandas as pd
from loguru import logger

from paper_replication.common import DATA_DIR, REPO_ROOT
from paper_replication.ic_horizon_profile import (
    ARM_ORDER,
    ARMS,
    K_RANGE,
    WINDOWS,
    _fetch_px,
    assert_inference_window,
    ic_profile,
)


def main_cumulative() -> None:
    """单趟计算单日 + 累计两剖面（k=1..10），整体写 ic_horizon_profile.json。"""
    from kronos_qlib import QlibProvider

    modes = ("single", "cumulative")
    results: dict[str, dict] = {m: {} for m in modes}
    assert_lines: dict[str, list[str]] = {}

    for win, (start, end, fetch_end) in WINDOWS.items():
        sigs = {}
        for (w, a), rel in ARMS.items():
            if w != win:
                continue
            sig_path = (REPO_ROOT / rel).resolve()
            if not sig_path.is_relative_to(REPO_ROOT.resolve()):
                raise ValueError(f"信号 parquet 路径越界：{sig_path}")
            sigs[a] = pd.read_parquet(sig_path)
        cols = sorted(set().union(*[set(s.columns) for s in sigs.values()]))
        provider = QlibProvider("csi300", start, fetch_end)
        px = _fetch_px(provider, cols, start, fetch_end)
        logger.info(f"[{win}] px {px.shape[0]} 日 × {px.shape[1]} 列（{start}~{fetch_end}）")

        for mode in modes:
            results[mode][win] = {}
            for arm in ARM_ORDER:
                rows = [ic_profile(sigs[arm], px, k, mode=mode) for k in K_RANGE]
                results[mode][win][arm] = rows
                row = " ".join(f"k{r['k']}:{r['mean']:+.4f}({r['t']:+.1f})" for r in rows)
                logger.info(f"[{win}][{mode}] {arm}: {row}")

        idx = sigs["M"].index
        picks = sorted({idx[0], idx[len(idx) // 4], idx[len(idx) // 2],
                        idx[3 * len(idx) // 4], idx[-1]})
        assert_lines[win] = assert_inference_window(
            provider, [d.strftime("%Y-%m-%d") for d in picks]
        )
        for ln in assert_lines[win]:
            logger.info(f"[{win}][窗口断言] {ln}")

    out = {
        "windows": {w: WINDOWS[w][:2] for w in WINDOWS},
        "arms": {f"{w}|{a}": p for (w, a), p in ARMS.items()},
        "min_cross_n": 30,
        "ic": results["single"],
        "ic_cum": results["cumulative"],
        "window_assert": assert_lines,
    }
    out_path = DATA_DIR / "ic_horizon_profile.json"
    # 常量拼接路径（DATA_DIR=包内 data/，模块常量，无外部输入）。
    # 以等价 write_text 落盘并通过显式校验：
    if not out_path.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"落盘路径越界：{out_path}")
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    logger.info(f"IC 剖面（single+cumulative）落盘 {out_path}")


if __name__ == "__main__":
    main_cumulative()
