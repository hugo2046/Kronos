"""G3：前瞻信号登记（计划 §2，20260816 计划）——**只写不读，零判读**。

把 forward 验证从事后回测升级为事前登记：每个交易日收盘后，把三种子
（s100=G1 / s101、s102=G2 重训）四变体当日 csi300 截面信号 + M 动量 +
MA200 门控状态 + 成分与可交易快照**在结果发生前**落盘并 git 盖章，
2026-11 结算时具备实盘级前瞻证据等级。

冻结机制：

- 幂等：重复运行同日不重复追加（manifest 已有该日且 parquet 指纹一致 → 跳过；
  指纹不一致 → 报错拒绝，不许静默重生成）；
- 无前视：一切取数边界 ≤ 决策日（provider 构造 end=决策日；断言当日 K 线
  已收盘——指数当日有收盘价才算"可得"）；
- git 盖章：parquet SHA256 写入 ``registry/MANIFEST.csv``
  （date,file,sha256,n_stocks,created_utc,late）并提交推送——
  **manifest 入库，parquet 不入库**（*.parquet 已被 .gitignore 排除）；
- 节假日/数据滞后：auto 模式解析最新可得日；与上次登记日之间的缺口自动补登
  并标 ``late=true``（迟补日期结算时单独列示）；无新数据静默跳过；
- **零判读**：登记期间不计算任何绩效、不出图、不看收益（计划 §2 冻结）。

用法::

    python finetune_suite/run_registry.py --date auto     # cron 每交易日 16:30 CST
    python finetune_suite/run_registry.py --date 2026-08-14  # 显式补登
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_suite.common import BaselineConfig, VARIANTS
from baseline_suite.signal import compute_variants_from_preds
from paper_replication.signal import predict_batch_chunked

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
REGISTRY_DIR = PKG_DIR / "registry"
DB_PATH = REGISTRY_DIR / "registry.duckdb"
MANIFEST_PATH = REGISTRY_DIR / "MANIFEST.csv"

MANIFEST_HEADER = ["date", "file", "sha256", "n_stocks", "created_utc", "late"]

# 三种子 predictor（s100=G1 封盘权重只读；s101/s102=G2 重训）；tokenizer 共享 G1
SEED_MODEL_PATHS = {
    "s100": "finetune_predictor_g1",
    "s101": "finetune_predictor_g2_s101",
    "s102": "finetune_predictor_g2_s102",
}

CSI300_INDEX = "000300.SH"


# ============================================================================
# 日期解析（节假日跳过 / 数据滞后补登）
# ============================================================================
def resolve_registration_dates(
    trading_days: pd.DatetimeIndex,
    last_registered: pd.Timestamp | None,
    latest_available: pd.Timestamp,
    today: pd.Timestamp,
) -> list[tuple[pd.Timestamp, bool]]:
    """待登记日列表 [(date, late)]。

    - 首日（无登记历史）：只登最新可得日（不回填历史——事前登记自首日起）；
    - 常规：补 (last_registered, latest_available] 内的交易日，当日登记
      late=false、迟补 late=true；
    - 无新数据：空列表（cron 节假日/周末触发 → 静默跳过）。
    """
    if latest_available <= (last_registered or pd.Timestamp.min):
        return []
    pending = trading_days[
        (trading_days > (last_registered or pd.Timestamp.min))
        & (trading_days <= latest_available)
    ]
    if last_registered is None:
        pending = pending[-1:]  # 首日只登最新可得日
    return [(d, bool(d.date() != today.date())) for d in pending]


def _latest_available_date() -> pd.Timestamp:
    """DDB 最新已收盘交易日（以指数当日有收盘价为准——断言 K 线已收盘）。"""
    from kronos_qlib import QlibProvider

    probe_end = (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = QlibProvider([CSI300_INDEX], "2020-01-01", probe_end).fetch(["$close"])
    if len(df) == 0:
        raise RuntimeError(f"指数 {CSI300_INDEX} 无数据——DDB 数据层异常，拒绝登记")
    return pd.Timestamp(df.index.get_level_values("datetime").max())


# ============================================================================
# 数据辅助（无前视：provider 构造边界 end=决策日，测试对拍未来数据不触碰）
# ============================================================================
def _build_provider(d: pd.Timestamp):
    """取数 provider：start 留足 L=90 回看 + M=10 + MA200 缓冲，end 恒 = d。"""
    from kronos_qlib import QlibProvider

    fetch_start = (d - pd.Timedelta(days=460)).strftime("%Y-%m-%d")
    return QlibProvider("csi300", fetch_start, d.strftime("%Y-%m-%d"))


def _build_index_provider(d: pd.Timestamp):
    """指数专用 provider（csi300 池 fetch 不含指数行情）；end 恒 = d。"""
    from kronos_qlib import QlibProvider

    return QlibProvider([CSI300_INDEX], "2020-01-01", d.strftime("%Y-%m-%d"))


def _momentum_signal(provider, d: pd.Timestamp) -> pd.Series:
    """M 动量：close[t]/close[t-10]-1（t=决策日，10 个交易日回看）。"""
    win_start = (d - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    raw = provider.fetch(["$close"])
    # provider 构造已限定 end ≤ d；此处再显式截断一次（双保险）
    raw = raw[raw.index.get_level_values("datetime") <= d]
    px = raw["close"].unstack("instrument")
    px = px.loc[win_start:]
    mom = px.iloc[-1] / px.iloc[-11] - 1.0
    return mom.dropna()


def _ma200_gate(index_provider, d: pd.Timestamp) -> dict:
    """MA200 门控：csi300 指数收盘 vs 其 200 日均线（R1 切换的前瞻记录）。

    :param index_provider: **只含指数**的 provider（``_build_index_provider(d)``，
        构造边界 end=决策日——无前视由调用方构造保证，本函数不再自建 provider）。
    """
    df = index_provider.fetch(["$close"])
    close = df["close"].xs(CSI300_INDEX, level="instrument").sort_index()
    close = close[close.index <= d]
    if len(close) < 200:
        raise RuntimeError(f"MA200 数据不足：指数仅 {len(close)} 行 ≤ {d.date()}")
    ma200 = float(close.iloc[-200:].mean())
    last = float(close.iloc[-1])
    return {"index_close": last, "ma200": ma200, "gate": bool(last > ma200)}


def _tradeable_mask(provider, d: pd.Timestamp) -> pd.Series:
    """当日可交易掩码（tradestatuscode == -1 为正常交易，DDB 真实语义）。"""
    raw = provider.fetch(["$tradestatuscode"])
    raw = raw[raw.index.get_level_values("datetime") <= d]
    tsc = raw["tradestatuscode"].unstack("instrument")
    day = tsc.loc[tsc.index <= d].iloc[-1]
    return (day == -1).fillna(False)


# ============================================================================
# 当日信号（canonical 推理链路，推理 seed 恒 42）
# ============================================================================
def compute_day_signals(d: pd.Timestamp, provider=None) -> tuple[pd.DataFrame, dict]:
    """决策日 d 的登记内容：三种子×四变体 + M + tradeable + 元数据。

    :returns: (wide, meta)——wide 行=code，列 = s{seed}_{variant}×12 + M +
        tradeable；meta 含池快照、MA200 门控与各模型覆盖统计。
    """
    import torch

    from kronos_qlib import build_inference_windows
    from finetune_suite.train_g1 import G1Config
    from finetune_suite.train_g2 import G2Config
    from model import Kronos, KronosPredictor, KronosTokenizer

    cfg = BaselineConfig.load(window="oos")
    provider = provider or _build_provider(d)
    ds = d.strftime("%Y-%m-%d")

    tokenizer = KronosTokenizer.from_pretrained(G1Config().finetuned_tokenizer_path)
    df_list, x_ts, y_ts, codes, stats = build_inference_windows(
        provider, ds, lookback=cfg.lookback, predict_len=cfg.predict_len, pool=cfg.pool
    )
    last_closes = [df["close"].iloc[-1] for df in df_list]

    model_paths = {
        "s100": G1Config().finetuned_predictor_path,
        "s101": G2Config(101).finetuned_predictor_path,
        "s102": G2Config(102).finetuned_predictor_path,
    }
    wide = pd.DataFrame(index=pd.Index(codes, name="code"))
    cover = {}
    for seed, path in model_paths.items():
        model = Kronos.from_pretrained(path)
        predictor = KronosPredictor(
            model, tokenizer, device=cfg.device, max_context=cfg.max_context
        )
        torch.manual_seed(cfg.seed)  # canonical：推理 seed 恒 42，与训练种子无关
        preds = predict_batch_chunked(
            predictor, df_list, x_ts, y_ts,
            pred_len=cfg.predict_len, T=cfg.T, top_k=cfg.sample_top_k,
            top_p=cfg.top_p, sample_count=cfg.sample_count,
        )
        cols = {v: {} for v in VARIANTS}
        for j, pred_df in enumerate(preds):
            variants = compute_variants_from_preds(pred_df[cfg.signal_field], last_closes[j])
            for v in VARIANTS:
                cols[v][codes[j]] = variants[v]
        for v in VARIANTS:
            wide[f"{seed}_{v}"] = pd.Series(cols[v])
        cover[seed] = len(preds)
        del predictor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mom = _momentum_signal(provider, d)
    wide["M"] = mom.reindex(wide.index)
    trd = _tradeable_mask(provider, d)
    wide["tradeable"] = trd.reindex(wide.index).fillna(False).astype(bool)

    gate = _ma200_gate(_build_index_provider(d), d)
    meta = {
        "decision_date": ds,
        "pool_members": json.dumps(sorted(provider.list_pool_at(cfg.pool, ds))),
        "n_pool": stats["n_pool"],
        "n_kept": stats["n_kept"],
        "coverage_by_seed": cover,
        **gate,
    }
    return wide, meta


# ============================================================================
# 落盘：parquet + DuckDB + manifest + git
# ============================================================================
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_rows() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_manifest(date: str, file: str, sha: str, n_stocks: int, late: bool) -> None:
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(MANIFEST_HEADER)
        w.writerow([
            date, file, sha, n_stocks,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "true" if late else "false",
        ])


def _write_duckdb(d: pd.Timestamp, wide: pd.DataFrame, meta: dict) -> None:
    import duckdb

    con = duckdb.connect(str(DB_PATH))
    con.execute(
        "CREATE TABLE IF NOT EXISTS registry "
        "(date DATE, code TEXT, model TEXT, variant TEXT, value DOUBLE)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS registry_meta (date DATE, key TEXT, value TEXT)"
    )
    n = con.execute("SELECT COUNT(*) FROM registry WHERE date=?", [d.date().isoformat()]).fetchone()[0]
    if n:
        con.close()
        raise RuntimeError(f"registry 表已有 {d.date()} 行 {n} 条（幂等冲突，人工核查）")

    rows = []
    for seed in ("s100", "s101", "s102"):
        for v in VARIANTS:
            s = wide[f"{seed}_{v}"].dropna()
            rows += [(d.date().isoformat(), c, seed, v, float(x)) for c, x in s.items()]
    s = wide["M"].dropna()
    rows += [(d.date().isoformat(), c, "M", "-", float(x)) for c, x in s.items()]
    s = wide["tradeable"]
    rows += [(d.date().isoformat(), c, "mask", "tradeable", float(bool(x))) for c, x in s.items()]
    con.executemany("INSERT INTO registry VALUES (?, ?, ?, ?, ?)", rows)

    for k, v in meta.items():
        con.execute(
            "INSERT INTO registry_meta VALUES (?, ?, ?)",
            [d.date().isoformat(), k, str(v)],
        )
    con.close()


def _git_commit_and_push(d: pd.Timestamp) -> bool:
    """manifest 入库盖章：提交并推送（推送失败仅告警，下次成功推送统一冲账）。"""
    date = d.date().isoformat()
    cmds = [
        ["git", "add", str(MANIFEST_PATH.relative_to(REPO_ROOT))],
        ["git", "commit", "-m",
         f"chore(registry): 登记前瞻信号 {date}\n\nCo-Authored-By: Hugo <shen.lan123@gmail.com>"],
        ["git", "push", "origin", "HEAD"],
    ]
    for cmd in cmds[:2]:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    push = subprocess.run(cmds[2], cwd=REPO_ROOT, capture_output=True, text=True)
    if push.returncode != 0:
        logger.warning(f"git push 失败（本地已提交，下次运行统一冲账）：{push.stderr[-300:]}")
        return False
    return True


def register_one(
    d: pd.Timestamp,
    *,
    late: bool,
    do_git: bool = True,
    compute=None,  # None → 调用时解析模块属性 compute_day_signals（可 monkeypatch）
) -> dict:
    """登记单个决策日（幂等）。返回 {"status": "registered"|"already-registered", …}。"""
    if compute is None:
        compute = compute_day_signals
    date = d.date().isoformat()
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    parquet = REGISTRY_DIR / f"signals_{date}.parquet"

    for row in _manifest_rows():
        if row["date"] == date:
            if parquet.exists() and _sha256(parquet) == row["sha256"]:
                logger.info(f"{date} 已登记（manifest 指纹一致）→ 幂等跳过")
                return {"status": "already-registered", "date": date}
            raise RuntimeError(
                f"{date} manifest 已有记录但 parquet 缺失/指纹不一致——拒绝静默重生成，人工核查"
            )

    wide, meta = compute(d)
    wide.to_parquet(parquet)
    sha = _sha256(parquet)
    _write_duckdb(d, wide, meta)
    _append_manifest(date, parquet.name, sha, int(len(wide)), late)
    pushed = _git_commit_and_push(d) if do_git else None
    logger.info(
        f"{date} 登记完成：{len(wide)} 股 × {wide.shape[1]} 列 | "
        f"ma200_gate={meta['gate']} | late={late} | sha256={sha[:12]}…"
    )
    return {"status": "registered", "date": date, "sha256": sha, "pushed": pushed}


# ============================================================================
# 主入口
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="G3 前瞻信号登记（只写不读，零判读）")
    parser.add_argument("--date", default="auto", help="auto=最新可得日（含缺口补登）| YYYY-MM-DD")
    args = parser.parse_args()

    today = pd.Timestamp.now().normalize()
    if args.date == "auto":
        latest = _latest_available_date()
        from kronos_qlib import QlibProvider

        cal = QlibProvider(
            "csi300",
            (latest - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
        ).trading_days()
        rows = _manifest_rows()
        last_registered = (
            max(pd.Timestamp(r["date"]) for r in rows) if rows else None
        )
        pending = resolve_registration_dates(cal, last_registered, latest, today)
        if not pending:
            logger.info(f"无新交易日可登记（最新可得 {latest.date()}，已登记至 "
                        f"{last_registered.date() if last_registered is not None else '无'}）→ 跳过")
            return
        logger.info(f"待登记 {len(pending)} 日：{[(d.date().isoformat(), l) for d, l in pending]}")
        for d, late in pending:
            register_one(d, late=late)
    else:
        d = pd.Timestamp(args.date)
        assert d <= today, f"登记日 {d.date()} 晚于今日（拒绝未来日期）"
        assert d == _latest_available_date() or d < _latest_available_date(), (
            "登记日无已收盘数据（断言当日 K 线已收盘失败）"
        )
        register_one(d, late=bool(d.date() != today.date()))


if __name__ == "__main__":
    main()
