"""阶段 2：正确性试点（计划 §3）。

    - 抽 50 只（seed=42）跑 6 个调仓日的完整链路：构窗 → predict_batch → signal；
    - 正确性抽查（逐条贴输出）：
        1. 任取 2 只 1 个调仓日，打印预测 10 日 close 路径与真实路径，确认预测值与
           输入窗口末值同数量级（后复权价常达数百元，反归一化错误会立刻暴露）；
        2. 同一输入连跑两遍（同 seed）信号逐位一致；
        3. signal 全部有限、无 NaN，且其截面分布不是常数（全相等说明预测退化）。

用法：
    /home/user/miniconda3/envs/quant/bin/python -m cross_section.stage2_pilot
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger

from cross_section.common import ExperimentConfig
from cross_section.rebalance import build_rebalance_dates
from cross_section.signal import compute_signal_from_preds
from kronos_qlib import QlibProvider, build_inference_windows


def load_predictor(cfg: ExperimentConfig):
    """加载 Kronos-base + Tokenizer（zero-shot，不微调）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    model = Kronos.from_pretrained(cfg.model_name)
    return KronosPredictor(model, tokenizer, device=cfg.device, max_context=cfg.max_context)


def run_pilot(cfg: ExperimentConfig) -> int:
    """执行 §3 正确性试点，返回退出码（0=通过，1=不达标）。"""
    p = QlibProvider(cfg.pool, cfg.backtest_start, cfg.data_end)
    rebalances = build_rebalance_dates(p, cfg)

    # 抽 6 个调仓日：首、末、中间均匀 4 个（与阶段 1 体检一致）
    n_periods = len(rebalances)
    sample_idx = np.linspace(0, n_periods - 1, 6).round().astype(int)
    sample_idx = sorted(set(sample_idx.tolist()))
    pilot_dates = [rebalances[i].strftime("%Y-%m-%d") for i in sample_idx]
    logger.info(f"阶段2 试点：{len(pilot_dates)} 个调仓日 × 50 只（seed=42）")

    predictor = load_predictor(cfg)

    # —— 抽 50 只（seed=42）：用首日成分做种子池，跨期取交集存在 ——
    # 实际用每调仓日 point-in-time 成分里的前 50 只（按 codes 列表顺序），保证可复现。
    all_signals: list[pd.DataFrame] = []
    first_period_df = None
    first_codes = None
    first_df_list = None
    first_x_ts = None
    first_y_ts = None
    first_preds = None

    for di, ds in enumerate(pilot_dates):
        df_list, x_ts_list, y_ts_list, codes, stats = build_inference_windows(
            p, ds, lookback=cfg.lookback, predict_len=cfg.predict_len, pool=cfg.pool
        )
        rng = np.random.default_rng(42)
        n_avail = len(df_list)
        pick_n = min(50, n_avail)
        # 随机抽 50 只，保证可复现且覆盖成分（不总是前 50）
        pick_idx = rng.choice(n_avail, size=pick_n, replace=False)
        pick_idx.sort()
        df_pick = [df_list[i] for i in pick_idx]
        x_pick = [x_ts_list[i] for i in pick_idx]
        y_pick = [y_ts_list[i] for i in pick_idx]
        codes_pick = [codes[i] for i in pick_idx]
        last_closes = [df["close"].iloc[-1] for df in df_pick]

        torch.manual_seed(cfg.seed)
        preds = predictor.predict_batch(
            df_list=df_pick,
            x_timestamp_list=x_pick,
            y_timestamp_list=y_pick,
            pred_len=cfg.predict_len,
            T=cfg.T,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            sample_count=cfg.sample_count,
            verbose=False,
        )
        signals = [
            compute_signal_from_preds(pr[cfg.signal_field], last_closes[i])
            for i, pr in enumerate(preds)
        ]
        period_df = pd.DataFrame(
            {"date": pd.Timestamp(ds), "code": codes_pick, "signal": signals}
        )
        all_signals.append(period_df)
        logger.info(
            f"  {ds}: 推理 {pick_n} 只，signal 均值={np.mean(signals):.5f} "
            f"std={np.std(signals):.5f} min={np.min(signals):.5f} max={np.max(signals):.5f}"
        )
        if di == 0:
            first_period_df = period_df
            first_codes = codes_pick
            first_df_list = df_pick
            first_x_ts = x_pick
            first_y_ts = y_pick
            first_preds = preds

    signals_long = pd.concat(all_signals, ignore_index=True)

    # ===== 正确性抽查 1：预测 close 路径 vs 真实路径（2 只 × 首调仓日）=====
    logger.info("=" * 70)
    logger.info("正确性抽查 1：预测 10 日 close 路径 vs 真实路径（首调仓日 2 只）")
    logger.info("=" * 70)
    ds0 = pilot_dates[0]
    for j in range(2):
        code = first_codes[j]
        last_close = first_df_list[j]["close"].iloc[-1]
        pred_close = first_preds[j][cfg.signal_field].values
        # 真实路径：取 y_ts 对应后复权 close
        y_dates = pd.DatetimeIndex(first_y_ts[j])
        sub = _fetch_real_close(p, code, ds0, y_dates)
        real_close = (
            sub.reindex(y_dates)["close"].values if sub is not None else np.full(10, np.nan)
        )
        logger.info(f"  [{code}] 输入窗口末值 close[t]={last_close:.2f}")
        line_p = "  ".join(f"{x:7.2f}" for x in pred_close)
        line_r = "  ".join(f"{x:7.2f}" for x in real_close)
        logger.info(f"    预测 close 路径: {line_p}")
        logger.info(f"    真实 close 路径: {line_r}")
        ratios = pred_close / last_close
        assert np.all((ratios > 0.1) & (ratios < 10.0)), (
            f"抽查1失败 [{code}]：预测 close 与末值不同数量级，last={last_close:.2f} pred={pred_close}"
        )
        logger.info(f"    pred/last 比值区间 [{ratios.min():.3f}, {ratios.max():.3f}] ✓ 同数量级")
    logger.info("正确性抽查 1 通过：预测值与输入末值同数量级（反归一化正确）")

    # ===== 正确性抽查 2：确定性（同一输入同 seed 连跑两遍，信号逐位一致）=====
    logger.info("=" * 70)
    logger.info("正确性抽查 2：确定性（首调仓日前 10 只，同 seed 连跑两遍）")
    logger.info("=" * 70)
    torch.manual_seed(cfg.seed)
    preds_a = predictor.predict_batch(
        df_list=first_df_list[:10],
        x_timestamp_list=first_x_ts[:10],
        y_timestamp_list=first_y_ts[:10],
        pred_len=cfg.predict_len,
        T=cfg.T,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        sample_count=cfg.sample_count,
        verbose=False,
    )
    torch.manual_seed(cfg.seed)
    preds_b = predictor.predict_batch(
        df_list=first_df_list[:10],
        x_timestamp_list=first_x_ts[:10],
        y_timestamp_list=first_y_ts[:10],
        pred_len=cfg.predict_len,
        T=cfg.T,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        sample_count=cfg.sample_count,
        verbose=False,
    )
    max_diff = max(
        np.abs(preds_a[i][cfg.signal_field].values - preds_b[i][cfg.signal_field].values).max()
        for i in range(10)
    )
    assert max_diff == 0.0, f"抽查2失败：同输入同 seed 两次预测 max|Δ|={max_diff} != 0"
    logger.info(f"  两次预测 close max|Δ| = {max_diff:.2e}（逐位一致）✓")

    # ===== 正确性抽查 3：signal 全有限、无 NaN、截面分布非退化 =====
    logger.info("=" * 70)
    logger.info("正确性抽查 3：signal 有限性 / 非退化（6 调仓日合并）")
    logger.info("=" * 70)
    sig = signals_long["signal"].values
    assert np.all(np.isfinite(sig)), "抽查3失败：signal 含非有限值"
    assert not np.any(np.isnan(sig)), "抽查3失败：signal 含 NaN"
    # 逐调仓日截面 std 应 > 0（全相等 = 预测退化）
    for ds, sub in signals_long.groupby("date"):
        std = sub["signal"].std()
        assert std > 1e-8, f"抽查3失败：{ds.date()} signal std={std:.2e}（截面常数=退化）"
        logger.info(
            f"  {ds.date()}: n={len(sub)} std={std:.5f} "
            f"分位数 10%={sub.signal.quantile(0.1):.5f} 90%={sub.signal.quantile(0.9):.5f}"
        )
    logger.info("正确性抽查 3 通过：signal 全有限、无 NaN、截面非退化")

    logger.info("✅ 阶段2 正确性试点全通过")
    return 0


def _fetch_real_close(provider, code, t, y_dates):
    """取 y_dates 上的真实后复权 close（用于路径对照）。"""
    fetch_start = pd.Timestamp(t)
    fetch_end = y_dates[-1]
    orig_start = provider._start_date
    orig_end = provider._end_date
    orig_inst = provider.instruments_
    try:
        provider._start_date = fetch_start.strftime("%Y-%m-%d")
        provider._end_date = fetch_end.strftime("%Y-%m-%d")
        provider.instruments_ = [code]
        df = provider.fetch(["$close"], freq="day")
    except Exception:
        return None
    finally:
        provider._start_date = orig_start
        provider._end_date = orig_end
        provider.instruments_ = orig_inst
    sub = df.xs(code, level="instrument") if "instrument" in df.index.names else df
    return sub.sort_index()


def main() -> int:
    cfg = ExperimentConfig.load()
    logger.info(f"配置：sample_count={cfg.sample_count} T={cfg.T} top_p={cfg.top_p} seed={cfg.seed}")
    return run_pilot(cfg)


if __name__ == "__main__":
    sys.exit(main())
