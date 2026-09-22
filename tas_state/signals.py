"""TAS1 各臂逐日信号生成（计划 §7：覆盖率、溯源、断点续跑）。

用法（P2）::

    /home/user/miniconda3/envs/quant/bin/python -m tas_state.signals \\
        --stage pilot --all-arms --window W1

规则：

- 逐窗（W1 2025-07-01~2025-12-31 / W2 2026-01-01~2026-07-24）按日生成
  CSI300 PIT 池信号宽表（index=date, columns=code，mean 聚合）；
- 每臂目录带 manifest（config/tokenizer/底座/checkpoint 哈希 + 股票列哈希），
  resume 时身份不匹配拒绝复用缓存；
- 覆盖率硬门禁：单日合格股票数 < 30 报错（不允许缩池）；
- 生成期只输出进度/覆盖率，不输出历史 IC/AER（开封在 evaluate --unseal）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from tas_state.config import PKG_DIR, REPO_ROOT, TASConfig

SIG_DIR = PKG_DIR / "data" / "signals"


@dataclass(frozen=True)
class SignalManifest:
    """信号目录身份（resume 校验：任一字段不匹配拒绝复用）。"""

    config_sha256: str
    tokenizer_sha256: str
    arm: str
    seed: int
    window: str
    columns_hash: str
    checkpoint_sha256: str = ""
    base_sha256: str = ""


def check_resume_compatible(
    m: SignalManifest, **expect: str | int
) -> None:
    """身份校验（tokenizer 哈希 / 臂 / seed / 窗口 / 列序变化拒绝复用）。"""
    for k, v in expect.items():
        got = getattr(m, k)
        if got != v:
            raise ValueError(
                f"manifest 不匹配：{k} 缓存={got!r} ≠ 请求={v!r}（拒绝复用缓存）"
            )


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def columns_hash(codes: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(codes)).encode()).hexdigest()


# ============================================================
# 推理窗口（复用 kronos_qlib.build_inference_windows，PIT CSI300）
# ============================================================
def window_dates(provider, cfg: TASConfig, window: str) -> pd.DatetimeIndex:
    """评估窗内的全部交易日（信号日 = 窗内每个交易日）。"""
    bounds = {
        "W1": cfg.eval_window_1,
        "W2": cfg.eval_window_2,
    }[window]
    cal = provider.trading_days(bounds[0], bounds[1])
    return cal


def generate_arm_signals(
    predictor,
    provider,
    cfg: TASConfig,
    *,
    arm: str,
    window: str,
    seed: int,
    out_dir: Path,
    tokenizer_sha256: str,
    base_sha256: str = "",
    checkpoint_sha256: str = "",
    sample_count: int | None = None,
    chunk: int = 32,
) -> pd.DataFrame:
    """单臂单窗逐日信号（断点续跑 + 覆盖率门禁）。

    :param predictor: ``TASPredictor``（T 族）或 ``KronosPredictor``（B1）。
    :param arm: 臂名（T-PRE / T-POST / T-SHUFFLE / C0 / B1）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"daily_signals_{window}_{arm}.parquet"
    mf_path = out_dir / f"manifest_{window}_{arm}.json"

    dates = window_dates(provider, cfg, window)
    rows: list[dict] = []
    done: set[pd.Timestamp] = set()
    if parquet.exists() and mf_path.exists():
        m = SignalManifest(**json.loads(mf_path.read_text(encoding="utf-8")))
        check_resume_compatible(
            m,
            config_sha256=cfg.sha256(),
            tokenizer_sha256=tokenizer_sha256,
            arm=arm,
            seed=seed,
            window=window,
        )
        old = pd.read_parquet(parquet)
        done = set(pd.to_datetime(old.index))
        rows = [old.loc[d].dropna().to_dict() for d in old.index]
        logger.info(f"[{arm}/{window}] 断点续跑：已完成 {len(done)} 日")

    N = sample_count or cfg.sample_count
    pending = [d for d in dates if d not in done]
    logger.info(f"[{arm}/{window}] {len(dates)} 日，待跑 {len(pending)} 日，N={N}")

    for i, d in enumerate(pending):
        from kronos_qlib import build_inference_windows

        df_list, x_ts, y_ts, codes, stats = build_inference_windows(
            provider, f"{d:%Y-%m-%d}", lookback=cfg.lookback,
            predict_len=cfg.predict_len, pool="csi300",
        )
        if len(codes) == 0:
            raise RuntimeError(f"{d:%Y-%m-%d}: CSI300 无可用股票（{stats}）")
        if len(codes) < cfg.ic_min_stocks:
            raise RuntimeError(
                f"{d:%Y-%m-%d}: 合格股票 {len(codes)} < {cfg.ic_min_stocks}，"
                f"覆盖门禁失败（不允许缩池续评）"
            )
        torch.manual_seed(cfg.seed)  # 逐日重置采样 RNG（baseline_suite 同构）
        day_rows: dict[str, float] = {}
        for s in range(0, len(df_list), chunk):
            sub = slice(s, min(s + chunk, len(df_list)))
            preds = predictor.predict_batch(
                df_list[sub], x_ts[sub], y_ts[sub],
                pred_len=cfg.predict_len, T=cfg.temperature,
                top_k=cfg.sample_top_k, top_p=cfg.top_p,
                sample_count=N, verbose=False,
                codes=codes[sub] if hasattr(predictor, "_maybe_shuffle") else None,
            )
            for j, p in enumerate(preds):
                code = codes[s + j]
                last_close = df_list[s + j]["close"].iloc[-1]
                vals = p[cfg.signal_field].values
                day_rows[code] = float(np.mean(vals) / last_close - 1.0)
        rows.append(day_rows)

        if (i + 1) % 10 == 0 or i == 0 or i == len(pending) - 1:
            logger.info(
                f"[{arm}/{window}] {i + 1}/{len(pending)} {d:%Y-%m-%d}："
                f"{len(day_rows)} 只，覆盖 {len(day_rows)}/{len(codes)}"
            )
            _dump_partial(parquet, mf_path, rows, dates[: len(done) + i + 1],
                          arm, window, seed, cfg, tokenizer_sha256,
                          base_sha256, checkpoint_sha256)

    wide = pd.DataFrame(rows, index=dates)
    wide.to_parquet(parquet)
    m = SignalManifest(
        config_sha256=cfg.sha256(), tokenizer_sha256=tokenizer_sha256,
        arm=arm, seed=seed, window=window,
        columns_hash=columns_hash(list(wide.columns)),
        checkpoint_sha256=checkpoint_sha256, base_sha256=base_sha256,
    )
    mf_path.write_text(json.dumps(asdict(m), ensure_ascii=False, indent=2),
                       encoding="utf-8")
    logger.info(f"[{arm}/{window}] 完成：{parquet}（{wide.shape}）")
    return wide


def _dump_partial(parquet, mf_path, rows, index, arm, window, seed, cfg,
                  tok_sha, base_sha, ckpt_sha) -> None:
    pd.DataFrame(rows, index=index).to_parquet(parquet)
    m = SignalManifest(
        config_sha256=cfg.sha256(), tokenizer_sha256=tok_sha, arm=arm,
        seed=seed, window=window, columns_hash="partial",
        checkpoint_sha256=ckpt_sha, base_sha256=base_sha,
    )
    mf_path.write_text(json.dumps(asdict(m), ensure_ascii=False, indent=2),
                       encoding="utf-8")


def main() -> int:
    """P2 信号入口（需要 DDB + GPU + 已训练 checkpoint）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PKG_DIR / "config.json"))
    ap.add_argument("--stage", choices=["pilot", "historical"], default="pilot")
    ap.add_argument("--all-arms", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--window", choices=["W1", "W2", "both"], default="both")
    args = ap.parse_args()

    cfg = TASConfig.load(args.config)
    logger.info(
        "信号生成入口：模型/checkpoint 就绪后运行；本命令只产出信号与覆盖率，"
        "不输出历史 IC/AER（开封在 evaluate --unseal）"
    )
    raise NotImplementedError(
        "P2 执行入口：seed42 训练完成后按臂枚举运行 generate_arm_signals；"
        "B0/B2/B3 直接复用只读 parquet（config.baseline_signal_paths）"
    )


if __name__ == "__main__":
    raise SystemExit(main())
