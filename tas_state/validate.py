"""TAS1 验证选点器（计划 §5.2：三 checkpoint 在 2025H1 自由生成选点）。

用法（P2，训练完成后）::

    /home/user/miniconda3/envs/quant/bin/python -m tas_state.validate \\
        --family dynamic --seed 42

规则（冻结）：

- 验证 = 2025H1 合法决策日（全部未来 10 日目标仍在 2025H1 内），每日 PIT CSI300；
- 每 checkpoint 自由生成 N=20（PRE 布局；T 族只按 PRE 分选点，POST 用
  同一 checkpoint；STATIC 按自己的 PRE 分）；
- 选点 = k=10 截面 RankIC 均值最大，平局到 1e-6 取较早 step；
- 本命令只产出验证分与选点结果，不触碰历史评估窗口（W1/W2）成绩。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from tas_state.config import PKG_DIR, TASConfig
from tas_state.evaluate import daily_rank_ic
from tas_state.model import ConditionalKronos, StateEncoder
from tas_state.signals import generate_arm_signals, _sha_file
from tas_state.train import OUT_DIR, STATIC_Z_PATH, select_checkpoint

VAL_DIR = PKG_DIR / "data" / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)


def validation_dates(provider, cfg: TASConfig) -> pd.DatetimeIndex:
    """2025H1 合法验证决策日（t+10 目标日仍在 2025-06-30 内）。"""
    cal = provider.trading_days(cfg.val_start, cfg.val_target_end)
    end_limit = pd.Timestamp(cfg.val_target_end)
    # 决策日 t：其后第 10 个交易日 ≤ end_limit
    ok = [i for i in range(len(cal) - cfg.predict_len)
          if cal[i + cfg.predict_len] <= end_limit]
    return cal[ok]


def forward_returns(provider, cfg: TASConfig, dates, codes) -> pd.DataFrame:
    """k=10 前向收益宽表（close_{t+10}/close_t − 1；后复权 close）。"""
    fetch_start = f"{dates[0]:%Y-%m-%d}"
    fetch_end = f"{dates[-1] + pd.Timedelta(days=25):%Y-%m-%d}"
    # 显式窗口拉取（provider.fetch 用构造区间；这里按需临时改区间）
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = fetch_start
        provider._end_date = fetch_end
        provider.instruments_ = "csi300"
        px = provider.fetch(["$close"])
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig
    wide = px["close"].unstack("instrument")
    cols = wide.columns
    fwd = pd.DataFrame(index=dates, columns=cols, dtype=float)
    cal = provider.trading_days(fetch_start, fetch_end)
    cal_pos = {d: i for i, d in enumerate(cal)}
    for t in dates:
        tp = cal_pos.get(t)
        if tp is None or tp + cfg.predict_len >= len(cal):
            continue
        t2 = cal[tp + cfg.predict_len]
        fwd.loc[t] = wide.loc[t2] / wide.loc[t] - 1.0
    return fwd


def validate_family(
    family: str,
    seed: int,
    cfg: TASConfig,
    *,
    device: str | None = None,
) -> dict:
    """三 checkpoint 验证选点（只产验证分；选点规则测试锁定）。"""
    from kronos_qlib import QlibProvider
    from model.kronos import Kronos, KronosTokenizer

    out_dir = OUT_DIR / f"{family}_s{seed}"
    ckpts = sorted((out_dir / "checkpoints").glob("step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"无 checkpoint：{out_dir}")
    dev = torch.device(device or cfg.device)

    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    kronos = Kronos.from_pretrained(cfg.model_name)
    encoder = StateEncoder(kronos.d_model, hidden=cfg.state_hidden,
                           n_states=cfg.n_states)
    ck = ConditionalKronos(kronos, tokenizer, encoder, cfg).to(dev)
    tok_sha = _sha_file(
        Path(__file__).resolve().parents[1] / "tas_state" / "config.json"
    )  # 身份锚（真实权重哈希在 manifest_p0）

    provider = QlibProvider("csi300", cfg.val_start, cfg.val_target_end)
    dates = validation_dates(provider, cfg)
    logger.info(f"[{family}/s{seed}] 验证决策日 {len(dates)} 天"
                f"（{dates[0]:%Y-%m-%d}~{dates[-1]:%Y-%m-%d}）")
    fwd = forward_returns(provider, cfg, dates, None)

    z_mean = None
    if family == "static":
        if not STATIC_Z_PATH.exists():
            raise FileNotFoundError("STATIC Z 不存在（先训练 static 族）")
        z_mean = np.load(STATIC_Z_PATH)["z_mean"]

    from tas_state.predictor import TASPredictor

    val_scores: dict[int, float] = {}
    daily_ics: dict[int, pd.Series] = {}
    for cpt in ckpts:
        step = int(cpt.stem.replace("step", ""))
        blob = torch.load(cpt, map_location=dev, weights_only=True)
        if blob.get("config_sha256") != cfg.sha256():
            raise ValueError(f"{cpt.name} config 哈希不匹配（拒绝跨配置选点）")
        encoder.load_state_dict(blob["encoder_state"])

        predictor = TASPredictor(ck, cfg, layout="pre")
        # STATIC 臂：S 来自固定平均 Z（predict_batch 内部走 ck.encode_state，
        # 这里覆盖实例方法使每次推理用同 Z；finally 恢复）
        orig_encode = ck.encode_state

        def _encode_static(x_t, xs_t, _zm=z_mean):
            zt = torch.tensor(_zm, dtype=torch.float32, device=x_t.device)
            zt = zt.unsqueeze(0).expand(x_t.shape[0], -1, -1)
            return ck.encoder(zt), ck.first_pass(x_t, xs_t)[1]

        if family == "static":
            ck.encode_state = _encode_static  # type: ignore[method-assign]
        try:
            sig = generate_arm_signals(
                predictor, provider, cfg,
                arm=f"VAL-{family}-s{seed}", window=f"VAL{step}",
                seed=seed, out_dir=VAL_DIR / f"{family}_s{seed}" / f"step{step}",
                tokenizer_sha256=tok_sha,
                checkpoint_sha256=_sha_file(cpt),
                dates=dates,
            )
        finally:
            ck.encode_state = orig_encode  # type: ignore[method-assign]

        ic = daily_rank_ic(sig, fwd, sig.index)
        val_scores[step] = float(ic.mean())
        daily_ics[step] = ic
        logger.info(f"[{family}/s{seed}] step{step} 验证 RankIC 均值 "
                    f"= {val_scores[step]:+.4f}")

    best = select_checkpoint(val_scores)
    result = {
        "family": family, "seed": seed, "config_sha256": cfg.sha256(),
        "val_scores": val_scores, "selected_step": best,
        "n_val_days": len(dates),
    }
    (out_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    daily = pd.DataFrame(daily_ics)
    daily.to_parquet(out_dir / "validation_daily_ic.parquet")
    logger.info(f"[{family}/s{seed}] 选点 step{best}（验证分 "
                f"{val_scores[best]:+.4f}）→ {out_dir / 'validation.json'}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PKG_DIR / "config.json"))
    ap.add_argument("--family", choices=["dynamic", "static"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = TASConfig.load(args.config)
    validate_family(args.family, args.seed, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
