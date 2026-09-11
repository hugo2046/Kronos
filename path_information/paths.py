"""PATH1 路径生成与缓存（计划 §3/§4，可断点续跑、身份校验、标签分离）。

与既有 ``kronos_qlib.windows.build_inference_windows`` 的关键差异（计划
§2/勘误1）：**预测特征 fetch.end ≤ t**——未来时间戳只来自交易日日历，
绝不为构造输入窗口请求 t 之后的数据。标签由独立阶段按段边界读取，不与
预测共用会放宽边界的 provider。

路径生成协议（与原 G1 推理同参的新采样链）：代码升序、chunk32、每决策
日 seed42（Python/numpy/TorchCPU/CUDA 全记录）、N=20、T=1.0、top_p=0.9、
top_k=0、clip5、max_context512、FP32。新链均值与基线 s_g1 的差仅记录，
不替换基线（计划 §4）。
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from path_information import config as C
from path_information import data as D

REQUIRED_COLS = ["open", "high", "low", "close", "volume", "amount"]
AUX_COLS = ["preclose", "tradestatuscode"]
CLOSE_IDX = 3                          # predict_batch 列序中 close 的下标


class CacheMismatchError(RuntimeError):
    """缓存身份不匹配（拒绝复用，不自动重训）。"""


def sha256_file(path: Path) -> str:
    """文件 SHA256（1MiB 分块）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_path_chunk(out_dir: Path, name: str, payload: dict,
                     identity: dict) -> Path:
    """原子写压缩 npz chunk（身份内嵌 JSON 字符串；禁 pickle）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    tmp = out_dir / (name + ".tmp.npz")
    np.savez_compressed(tmp, identity_json=np.array(
        json.dumps(identity, ensure_ascii=False)), **payload)
    tmp.replace(path)
    return path


def read_path_chunk(path: Path, expect: dict) -> dict:
    """加载并校验身份；任一字段不匹配 → :class:`CacheMismatchError`。"""
    with np.load(path, allow_pickle=False) as z:
        if "identity_json" not in z.files:
            raise CacheMismatchError(f"{path.name} 缺身份字段")
        got = json.loads(str(z["identity_json"]))
        for k, v in expect.items():
            if got.get(k) != v:
                raise CacheMismatchError(
                    f"{path.name} 身份字段 {k} 不匹配：{got.get(k)!r}!={v!r}")
        return {k: z[k] for k in z.files if k != "identity_json"}


def path_chunk_name(split: str, t: pd.Timestamp) -> str:
    return f"{split}_{t.strftime('%Y%m%d')}.npz"


def path_chunk_identity(split: str, t: pd.Timestamp, weight_shas: dict) -> dict:
    """逐 chunk 身份：协议/段/日期/权重 SHA/采样协议/种子族。"""
    return {"protocol": C.PROTOCOL_VERSION, "split": split,
            "date": t.strftime("%Y-%m-%d"),
            "seeds": {"python": C.INFERENCE["seed"],
                      "numpy": C.INFERENCE["seed"],
                      "torch_cpu": C.INFERENCE["seed"],
                      "torch_cuda": C.INFERENCE["seed"]},
            "inference": {k: v for k, v in C.INFERENCE.items()
                          if k not in ("seed",)},
            **weight_shas}


# ---------------- 模型装载 ----------------

def load_frozen_g1(device: str = "cuda:0"):
    """装载冻结 G1（tokenizer + predictor，FP32、eval）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    tok = KronosTokenizer.from_pretrained(str(C.G1_TOKENIZER)).to(device)
    kronos = Kronos.from_pretrained(str(C.G1_PREDICTOR)).to(device)
    predictor = KronosPredictor(kronos, tok, device=device,
                                max_context=C.INFERENCE["max_context"],
                                clip=C.INFERENCE["clip"])
    tok.eval()
    kronos.eval()
    return predictor


def g1_weight_shas() -> dict:
    """当前 G1 权重 SHA（只读引用主仓库；缺失即失败）。"""
    shas = {"tokenizer_sha": sha256_file(C.G1_TOKENIZER / "model.safetensors"),
            "predictor_sha": sha256_file(C.G1_PREDICTOR / "model.safetensors")}
    for w, want in (("tokenizer_sha", "5ca43492146cd93da2f9dd5528009aa6"
                                       "bd882f2a448ff4c6bd3c1b0ab08609ee"),
                    ("predictor_sha", "4f2382138138e85ea1c543d43ade2de2"
                                      "373e4168a995e34fd43cbca2895c2fe2")):
        if shas[w] != want:
            raise RuntimeError(f"G1 {w} 与计划冻结值不匹配：{shas[w]}")
    return shas


# ---------------- 窗口构造（fetch.end ≤ t） ----------------

def build_windows_fetch_le_t(provider, rebalance_date: str, *, pool: str,
                             lookback: int = C.LOOKBACK,
                             predict_len: int = C.PREDICT_LEN):
    """PIT 池 + ≤t 的 L 日窗口 + 停牌剔除（与 G1 链同口径），fetch 严格 ≤ t。

    与 ``kronos_qlib/windows.py`` 同语义（行数不足 L 跳过、窗口内
    ``tradestatuscode == 0`` 跳过不前向填充、列序固定、不预归一化），差异
    仅一处：数据请求区间止于 t（未来时间戳只从日历取，用于 y_stamp）。

    :returns: ``(df_list, x_ts_list, y_ts_list, codes, stats)``。
    """
    t = pd.Timestamp(rebalance_date)
    if t > pd.Timestamp(C.FORWARD_CUTOFF):
        raise RuntimeError(
            f"访问数据前拒绝：{rebalance_date} 超封存线 {C.FORWARD_CUTOFF}")
    members = provider.list_pool_at(pool, rebalance_date)
    if len(members) == 0:
        raise ValueError(f"{rebalance_date} 时点 {pool} 成分为空")
    full_cal = provider.trading_days()
    cal_le_t = full_cal[full_cal <= t]
    if len(cal_le_t) == 0:
        raise ValueError(f"日历中无 <= {rebalance_date} 的交易日")
    t_effective = cal_le_t[-1]
    t_pos = len(cal_le_t) - 1
    x_start_pos = max(0, t_pos - lookback + 1)
    x_window_cal = full_cal[x_start_pos: t_pos + 1]
    y_window_cal = full_cal[t_pos + 1: t_pos + 1 + predict_len]
    if len(y_window_cal) < predict_len:
        raise ValueError(f"{rebalance_date} 之后交易日不足 {predict_len} 个")

    fetch_start = x_window_cal[0].strftime("%Y-%m-%d")
    fetch_end = t_effective.strftime("%Y-%m-%d")     # fetch.end ≤ t（核心差异）
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = fetch_start
        provider._end_date = fetch_end
        provider.instruments_ = members
        fields = [f"${c}" for c in REQUIRED_COLS + AUX_COLS]
        raw = provider.fetch(fields, freq="day")
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig
    if "instrument" not in raw.index.names:
        raise ValueError("fetch 返回缺少 instrument 索引层")

    available = raw.index.get_level_values("instrument").unique()
    df_list, x_ts_list, y_ts_list, codes = [], [], [], []
    stats = {"n_pool": len(members), "n_kept": 0, "skipped_short": 0,
             "skipped_halt": 0}
    y_window_idx = pd.DatetimeIndex(y_window_cal)
    for code in members:
        if code not in available:
            stats["skipped_short"] += 1
            continue
        sub = raw.xs(code, level="instrument").loc[:t_effective]
        if len(sub) < lookback:
            stats["skipped_short"] += 1
            continue
        window = sub.iloc[-lookback:]
        if "tradestatuscode" in window.columns and (
                (window["tradestatuscode"] == 0)).any():
            stats["skipped_halt"] += 1
            continue
        window_df = window[REQUIRED_COLS].copy()
        df_list.append(window_df)
        x_ts_list.append(pd.Series(window_df.index))
        y_ts_list.append(pd.Series(y_window_idx))
        codes.append(code)
        stats["n_kept"] += 1
    return df_list, x_ts_list, y_ts_list, codes, stats


# ---------------- 路径生成（新采样链） ----------------

def seed_all(seed: int = C.INFERENCE["seed"]) -> None:
    """每决策日起点：Python/numpy/TorchCPU/CUDA 全部 seed42。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_day_paths(predictor, df_list, x_ts_list, y_ts_list,
                      chunk_size: int = C.INFERENCE["chunk_size"]
                      ) -> np.ndarray:
    """单日 64 股 → ``[n, N, H]`` 真实价格 close 采样路径（代码升序输入）。

    逐 chunk 调 ``predict_batch(return_samples=True)``，两 chunk 按输入顺序
    共享该日起点种子后的随机流；close 列下标 = 3（列序固定）。
    """
    seed_all()
    outs = []
    for s in range(0, len(df_list), chunk_size):
        e = min(s + chunk_size, len(df_list))
        samples = predictor.predict_batch(
            df_list=df_list[s:e], x_timestamp_list=x_ts_list[s:e],
            y_timestamp_list=y_ts_list[s:e], pred_len=C.PREDICT_LEN,
            T=C.INFERENCE["T"], top_k=C.INFERENCE["top_k"],
            top_p=C.INFERENCE["top_p"],
            sample_count=C.INFERENCE["sample_count"], verbose=False,
            return_samples=True)                       # [b, N, H, 6]
        outs.append(samples[:, :, :, CLOSE_IDX].astype(np.float64))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def new_chain_mean_signal(close_paths: np.ndarray,
                          close_t: np.ndarray) -> np.ndarray:
    """新采样链的 mean 信号（仅记录与基线 s_g1 之差，不替换基线）。"""
    p = D.to_relative_paths(close_paths, close_t)
    return p.mean(axis=(1, 2))


# ---------------- GPU 让位（§8：16:30 登记任务） ----------------

def gpu_busy() -> bool:
    """其余进程显存占用探测（只读 nvidia-smi，不改 cron、不读登记内容）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return False
        used = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return bool(used) and max(used) >= C.GPU_BUSY_MEM_MB
    except Exception:                                   # noqa: BLE001 - 探测失败不阻塞
        return False


def yield_for_registry(model_loader, note: str = ""):
    """日边界让位：卸载 nn.Module 等待登记任务完成，再重装恢复。

    :param model_loader: 零参调用返回 predictor（可恢复状态 = 已完成日
        chunk 落盘 + 模型可从主仓库权重随时重装）。
    :returns: predictor（就绪）。
    """
    logger.warning(f"[yield] GPU 被登记任务占用，卸载模型等待（{note}）")
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    while gpu_busy():
        time.sleep(C.GPU_POLL_S)
    logger.info("[yield] 登记任务完成，重装 G1 继续")
    return model_loader()


# ---------------- 段级路径缓存构建 ----------------

def build_segment_paths(provider, model_loader, split: str, cal, out_dir: Path,
                        baseline_wide: pd.DataFrame, weight_shas: dict,
                        limit_days: int | None = None) -> dict:
    """逐决策日生成 64×20×10 路径缓存（身份校验续跑；断点只在完整日边界）。

    资格（唯一权威路径）：PIT csi300 ∩ fetch≤t 的 90 日窗口完整 ∩ 基线
    G1 有限信号 → ``SHA256('PATH1|17|date|code')`` 升序前 64；不以未来
    收益/停牌/标签完整性挑选。先保存路径与样本键，标签由独立阶段读取。
    """
    start, label_end = C.SEGMENTS[split]
    days = D.segment_days(cal, start, label_end)
    if limit_days is not None:
        days = days[:limit_days]
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = model_loader()
    stats = {"split": split, "n_days": len(days), "rebuilt": 0, "reused": 0,
             "wall_s": 0.0, "excluded": []}
    t_all = time.perf_counter()
    for t in days:
        ds = t.strftime("%Y-%m-%d")
        path = out_dir / path_chunk_name(split, t)
        ident = path_chunk_identity(split, t, weight_shas)
        if path.exists():
            try:
                read_path_chunk(path, ident)
                stats["reused"] += 1
                continue
            except CacheMismatchError as exc:
                logger.warning(f"chunk 身份不匹配，重建：{path}（{exc}）")
        if gpu_busy():
            predictor = yield_for_registry(model_loader, ds)
        t0 = time.perf_counter()
        if ds not in baseline_wide.index:
            raise RuntimeError(f"{split} {ds} 不在基线信号日期内——停止排查")
        g1_valid = set(baseline_wide.loc[ds].dropna().index)
        df_list, x_ts_list, y_ts_list, codes, wstats = build_windows_fetch_le_t(
            provider, ds, pool=C.POOL)
        eligible = [c for c in codes if c in g1_valid]
        try:
            picked = D.select_codes(eligible, ds)      # SHA256 升序前 64
        except ValueError as exc:
            raise RuntimeError(f"{split} {ds}：{exc}") from exc
        rows = {c: j for j, c in enumerate(codes) if c in set(picked)}
        sel_codes = sorted(rows)                       # 最终推理按 code 排序
        close_t = np.array([df_list[rows[c]]["close"].iloc[-1] for c in sel_codes],
                           dtype=np.float64)
        s_g1 = np.array([float(baseline_wide.loc[ds, c]) for c in sel_codes],
                        dtype=np.float64)
        paths = predict_day_paths(
            predictor, [df_list[rows[c]] for c in sel_codes],
            [x_ts_list[rows[c]] for c in sel_codes],
            [y_ts_list[rows[c]] for c in sel_codes])
        lds = D.label_dates(cal, t)
        payload = {
            "dates": np.array([ds] * len(sel_codes)),
            "instruments": np.array(sel_codes),
            "close_paths": paths.astype(np.float32),   # [64, 20, 10]
            "close_t": close_t,
            "s_g1": s_g1,
            "s_new_mean": new_chain_mean_signal(paths, close_t),
            "x_start": np.array([str(pd.DatetimeIndex(x_ts_list[rows[c]])
                                     .min().date()) for c in sel_codes]),
            "x_end": np.array([ds] * len(sel_codes)),
            "label_start": np.array([str(lds[0].date())] * len(sel_codes)),
            "label_end": np.array([str(lds[-1].date())] * len(sel_codes)),
        }
        write_path_chunk(out_dir, path_chunk_name(split, t), payload, ident)
        stats["rebuilt"] += 1
        stats["excluded"].append({
            "date": ds, "n_pool": wstats["n_pool"],
            "n_window_kept": wstats["n_kept"], "n_g1_valid": len(g1_valid),
            "n_eligible": len(eligible), "n_selected": len(sel_codes),
            "wall_s": round(time.perf_counter() - t0, 1)})
        logger.info(f"[{split}] {ds}: window={wstats['n_kept']} "
                    f"g1_valid={len(g1_valid)} eligible={len(eligible)} "
                    f"selected={len(sel_codes)} "
                    f"wall={stats['excluded'][-1]['wall_s']}s")
    stats["wall_s"] = round(time.perf_counter() - t_all, 1)
    del predictor
    return stats


# ---------------- 标签独立阶段（按段边界读取） ----------------

def _label_close_wide(provider, codes: list[str], start: str, end: str
                      ) -> pd.DataFrame:
    """标签阶段专用 close 拉取：独立 bounds，end = 段标签最晚观测日。"""
    if pd.Timestamp(end) > pd.Timestamp(C.FORWARD_CUTOFF):
        raise RuntimeError(f"标签读取超封存线：{end} > {C.FORWARD_CUTOFF}")
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = start
        provider._end_date = end
        provider.instruments_ = codes
        df = provider.fetch(["$close"], freq="day")
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig
    return df["close"].unstack("instrument").sort_index()


def build_segment_labels(provider, split: str, cal, cache_dir: Path,
                         weight_shas: dict) -> dict:
    """段标签缓存：逐 (日, 股) ``y_mean``，缺失记 NaN（只屏蔽不重选）。

    只在全部路径 chunk 落盘后调用（计划 §3：先路径与样本键，后标签）。
    分母 ``close_t`` 用路径 chunk 冻结值（同一窗口最后一行，量纲一致），
    不在标签阶段重新读行情。
    """
    seg_dir = cache_dir / split
    chunks = sorted(seg_dir.glob(f"{split}_*.npz"))
    assert chunks, f"{split} 路径缓存不存在，标签阶段拒绝启动"
    parts = []
    for p in chunks:
        parts.append(read_path_chunk(p, {"protocol": C.PROTOCOL_VERSION,
                                         "split": split}))
    dates = np.concatenate([a["dates"] for a in parts])
    codes = np.concatenate([a["instruments"] for a in parts])
    close_t = np.concatenate([a["close_t"] for a in parts])
    _, label_end = C.SEGMENTS[split]
    # numpy2 对 unicode dtype 无 ufunc minimum：用 Python min 取字符串最早值
    start = min(str(a["label_start"][0]) for a in parts)
    close_wide = _label_close_wide(provider, sorted(set(codes)), start,
                                   label_end)
    y_final = np.full(len(dates), np.nan)
    i = 0
    for a in parts:
        for j in range(len(a["instruments"])):
            t = pd.Timestamp(str(a["dates"][j]))
            code = str(a["instruments"][j])
            ldays = D.label_dates(cal, t)
            try:
                fut = close_wide.loc[pd.DatetimeIndex(ldays), code].to_numpy(
                    dtype=np.float64)
                y_final[i] = D.mean_label(float(a["close_t"][j]), fut)
            except (KeyError, ValueError):
                pass                       # 缺失/非有限 → 无标签（只屏蔽）
            i += 1
    ident = {"protocol": C.PROTOCOL_VERSION, "split": split,
             "kind": "labels", **weight_shas}
    write_path_chunk(cache_dir, f"labels_{split}.npz", {
        "dates": dates, "instruments": codes, "y_mean": y_final,
        "close_t": close_t, "valid": D.valid_label_mask(y_final)}, ident)
    n_valid = int(D.valid_label_mask(y_final).sum())
    logger.info(f"[{split}] 标签完成：{n_valid}/{len(y_final)} 有效格")
    return {"split": split, "n_cells": int(len(y_final)),
            "n_valid": n_valid,
            "coverage": round(n_valid / len(y_final), 6)}


def build_dev_eval_labels(provider=None, cal=None, cache_dir: Path | None = None,
                          weight_shas: dict | None = None) -> dict:
    """dev_eval 标签：只在分数冻结校验通过后由 :func:`evaluate.unseal` 调用。"""
    from kronos_qlib import QlibProvider

    from path_information.run import get_calendar

    provider = provider or QlibProvider(C.POOL, "2013-06-01", C.DATA_END)
    cal = cal if cal is not None else get_calendar(provider)
    return build_segment_labels(provider, "dev_eval", cal,
                                cache_dir or C.CACHE_DIR,
                                weight_shas or g1_weight_shas())


__all__ = ["CacheMismatchError", "sha256_file", "write_path_chunk",
           "read_path_chunk", "load_frozen_g1", "g1_weight_shas",
           "build_windows_fetch_le_t", "seed_all", "predict_day_paths",
           "new_chain_mean_signal", "build_segment_paths",
           "build_segment_labels", "build_dev_eval_labels",
           "gpu_busy", "yield_for_registry"]
