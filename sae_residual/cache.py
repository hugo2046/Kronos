"""teacher / hidden 缓存构建与加载（计划 §4，可断点续跑、身份校验）。

每个决策日落一个 chunk（``npz``）：训练/验证段含 ``h``（末步隐状态）+
teacher ``s_G1`` + 标签 ``y_mean``；测试段（W3/W4）只含 ``h`` 与键
（原 G1 信号值直接沿用基线文件，绝不重新生成测试 G1）。

资格判定（head 训练/验证样本的唯一权威路径）：
    PIT csi300 成分 ∩ 90 日窗口完整（``build_inference_windows`` 口径）
    ∩ 10 个精确标签日 close 全部有限 → rng17 无放回选 64 只。

teacher 推理随机协议（与原 G1 测试信号生成一致并披露）：代码升序、
chunk=32、每决策日 ``torch.manual_seed(42)``、N=20、T=1.0、top_p=0.9、
top_k=0、clip=5、max_context=512。
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from sae_residual import config as C
from sae_residual import data as D
from sae_residual.cache_io import (CacheMismatchError, chunk_manifest_path,
                                   identity_of, load_chunk, load_split_arrays,
                                   samples_path, sha256_file, write_chunk)

# head 训练/验证决策日（相对全日历）；测试段日期 = 基线信号文件日期
SPLIT_BOUNDS = {"train": (C.TRAIN_START, C.TRAIN_LABEL_END),
                "val": (C.VAL_START, C.VAL_LABEL_END)}


def sha256_file_(p: Path) -> str:  # 兼容旧引用
    return sha256_file(p)


def load_frozen_g1(device: str = "cuda:0"):
    """装载冻结 G1（tokenizer + predictor 共享同一实例：teacher 与 hidden 同底座）。

    :returns: ``(backbone, predictor, weight_identity)``。
    """
    from cross_section_kda import KronosFrozenBackbone
    from model import Kronos, KronosPredictor, KronosTokenizer

    tok = KronosTokenizer.from_pretrained(str(C.G1_TOKENIZER)).to(device)
    kronos = Kronos.from_pretrained(str(C.G1_PREDICTOR)).to(device)
    predictor = KronosPredictor(kronos, tok, device=device,
                                max_context=C.TEACHER["max_context"])
    backbone = KronosFrozenBackbone(tokenizer=tok, kronos=kronos, device=device)
    backbone.eval()
    identity = {
        "tokenizer_sha": sha256_file(C.G1_TOKENIZER / "model.safetensors"),
        "predictor_sha": sha256_file(C.G1_PREDICTOR / "model.safetensors"),
        "g1_tokenizer_path": str(C.G1_TOKENIZER),
        "g1_predictor_path": str(C.G1_PREDICTOR),
    }
    return backbone, predictor, identity


def _calc_stamps(x_ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    from model.kronos import calc_time_stamps

    # calc_time_stamps 用 Series.dt 访问器，需转 Series（与 predict 路径一致）
    return calc_time_stamps(pd.Series(pd.DatetimeIndex(x_ts))).values.astype(
        np.float32)


def extract_hidden(backbone, df_list, x_ts_list, device: str = "cuda:0"
                   ) -> np.ndarray:
    """批量提取末步隐状态 ``h = backbone.extract(x_norm, stamp)[:, -1, :]``。

    x_norm 与 ``KronosPredictor.predict_batch`` 逐字同口径（逐列 z-score
    ``(x-mean)/(std+1e-5)`` + clip±5）；分块 ``HIDDEN_CHUNK`` 控显存。
    """
    xs, ss = [], []
    for df, x_ts in zip(df_list, x_ts_list):
        xn, _, _ = D.normalize_window(df.values.astype(np.float32))
        xs.append(xn)
        ss.append(_calc_stamps(x_ts))
    x = torch.from_numpy(np.stack(xs))
    s = torch.from_numpy(np.stack(ss))
    hs = []
    with torch.no_grad():
        for i in range(0, len(x), C.HIDDEN_CHUNK):
            hb = backbone.extract(x[i:i + C.HIDDEN_CHUNK].to(device),
                                  s[i:i + C.HIDDEN_CHUNK].to(device))
            hs.append(hb[:, -1, :].float().cpu().numpy())
    return np.concatenate(hs, axis=0)


def run_teacher(predictor, df_list, x_ts_list, y_ts_list) -> list[pd.DataFrame]:
    """teacher 预测：代码升序 + chunk32 + 每日 seed42（与原 G1 协议一致）。"""
    from paper_replication.signal import predict_batch_chunked

    torch.manual_seed(C.TEACHER["seed"])
    return predict_batch_chunked(
        predictor, df_list, x_ts_list, y_ts_list,
        pred_len=C.PREDICT_LEN, T=C.TEACHER["T"], top_k=C.TEACHER["top_k"],
        top_p=C.TEACHER["top_p"], sample_count=C.TEACHER["sample_count"],
        chunk_size=C.TEACHER["chunk_size"])


def _bulk_close_wide(provider, start: str, end: str) -> pd.DataFrame:
    """一次性批量拉取 csi300 全区间后复权 close 宽表（date×instrument）。"""
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = start
        provider._end_date = end
        provider.instruments_ = C.POOL
        df = provider.fetch(["$close"], freq="day")
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig
    wide = df["close"].unstack("instrument").sort_index()
    return wide


def _label_for(close_wide: pd.DataFrame, lds: pd.DatetimeIndex, close_t: float,
               code: str) -> float | None:
    """按精确标签日取 close 算 y_mean；任何缺失/非有限返回 None。"""
    try:
        fut = close_wide.loc[lds, code].to_numpy(dtype=np.float64)
    except KeyError:
        return None
    try:
        return D.mean_label(close_t, fut)
    except ValueError:
        return None


def build_head_split(provider, backbone, predictor, identity: dict,
                     split: str, close_wide: pd.DataFrame,
                     cal: pd.DatetimeIndex,
                     limit_days: int | None = None,
                     subdir: str | None = None) -> dict:
    """构建 head 训练/验证缓存（逐决策日 chunk，可断点续跑）。

    :param limit_days: smoke 用——只跑前 N 个决策日。
    :param subdir: smoke 用——写到 ``CACHE_DIR/<subdir>`` 隔离正式缓存。
        抽样 RNG 按日期派生（seed17⊕日期），smoke 与正式构建同日同选。
    :returns: 统计字典（逐日资格/排除数、耗时、chunk 状态）。
    """
    from kronos_qlib import build_inference_windows

    out_dir = C.CACHE_DIR / (subdir or split)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = D.decision_days(cal, *SPLIT_BOUNDS[split])
    if limit_days is not None:
        days = days[:limit_days]
    stats = {"split": split, "n_days": len(days), "skipped": [],
             "rebuilt": 0, "reused": 0, "wall_s": 0.0, "cuda_s": 0.0}
    ident = dict(protocol=C.PROTOCOL_VERSION, split=split,
                 tokenizer_sha=identity["tokenizer_sha"],
                 predictor_sha=identity["predictor_sha"])
    t_all = time.perf_counter()
    for t in days:
        ds = t.strftime("%Y-%m-%d")
        path = out_dir / f"{split}_{t.strftime('%Y%m%d')}.npz"
        if path.exists():
            try:
                load_chunk(path, ident)
                stats["reused"] += 1
                continue
            except CacheMismatchError as exc:
                logger.warning(f"chunk 身份不匹配，重建：{path}（{exc}）")
        ev0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        ev1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        if ev0 is not None:
            ev0.record()
        df_list, x_ts_list, y_ts_list, codes, wstats = build_inference_windows(
            provider, ds, lookback=C.LOOKBACK, predict_len=C.PREDICT_LEN,
            pool=C.POOL)
        lds = D.label_dates(cal, t)
        assert list(pd.DatetimeIndex(y_ts_list[0])) == list(lds), \
            f"{ds}: y_ts 与标签日历不一致"
        eligible: list[tuple[str, int, float, float]] = []  # code, idx, close_t, y
        for j, code in enumerate(codes):
            close_t = float(df_list[j]["close"].iloc[-1])
            y = _label_for(close_wide, lds, close_t, code)
            if y is not None:
                eligible.append((code, j, close_t, y))
        day_stat = {"date": ds, "n_pool": wstats["n_pool"],
                    "n_window_kept": wstats["n_kept"],
                    "n_label_complete": len(eligible)}
        try:
            picked = D.select_codes([c for c, *_ in eligible],
                                    D.make_day_rng(t))
        except ValueError as exc:
            day_stat["skip_reason"] = str(exc)
            stats["skipped"].append(day_stat)
            logger.warning(f"{split} {ds} 跳过：{exc}")
            continue
        picked_set = set(picked)
        rows = {c: (j, ct, y) for c, j, ct, y in eligible if c in picked_set}
        sel_codes = sorted(rows)                       # teacher 协议：代码升序
        sel_df = [df_list[rows[c][0]] for c in sel_codes]
        sel_x = [x_ts_list[rows[c][0]] for c in sel_codes]
        sel_y = [y_ts_list[rows[c][0]] for c in sel_codes]
        preds = run_teacher(predictor, sel_df, sel_x, sel_y)
        s_g1 = np.array([D.teacher_mean_signal(
            preds[k]["close"].to_numpy(dtype=np.float64), rows[c][1])
            for k, c in enumerate(sel_codes)], dtype=np.float64)
        h = extract_hidden(backbone, sel_df, sel_x)
        close_t_arr = np.array([rows[c][1] for c in sel_codes], dtype=np.float64)
        y_arr = np.array([rows[c][2] for c in sel_codes], dtype=np.float64)
        payload = {
            "dates": np.array([ds] * len(sel_codes)),
            "instruments": np.array(sel_codes),
            "h": h.astype(np.float32),
            "s_g1": s_g1, "y_mean": y_arr, "close_t": close_t_arr,
            "x_start": np.array([str(pd.DatetimeIndex(s).min().date())
                                 for s in sel_x]),
            "x_end": np.array([ds] * len(sel_codes)),
            "label_start": np.array([str(lds[0].date())] * len(sel_codes)),
            "label_end": np.array([str(lds[-1].date())] * len(sel_codes)),
        }
        write_chunk(path, payload, ident)
        if ev1 is not None:
            ev1.record()
            torch.cuda.synchronize()
            stats["cuda_s"] += ev0.elapsed_time(ev1) / 1000.0
        stats["rebuilt"] += 1
        day_stat.update({"n_selected": len(sel_codes),
                         "wall_s": time.perf_counter() - t0})
        logger.info(f"[{split}] {ds}: window_kept={day_stat['n_window_kept']} "
                    f"label_ok={len(eligible)} selected={len(sel_codes)} "
                    f"wall={day_stat['wall_s']:.1f}s")
    stats["wall_s"] = time.perf_counter() - t_all
    return stats


def build_test_split(provider, backbone, identity: dict, wname: str,
                     cal: pd.DatetimeIndex, limit_days: int | None = None,
                     subdir: str | None = None) -> dict:
    """构建测试段（W3/W4）hidden 缓存：候选 = 原 G1 有效信号集合。

    原信号值完全保留（评价时直接读基线文件）；此处只为对应股票日抽一次
    隐状态。任一 G1 有效股票日抽不出对应历史窗口 → 抛错停止（不用共同
    掩码删格令基线悄悄改变）。``subdir`` 用于 smoke 隔离。
    """
    from kronos_qlib import build_inference_windows

    out_dir = C.CACHE_DIR / (subdir or wname)
    out_dir.mkdir(parents=True, exist_ok=True)
    wide = pd.read_parquet(C.BASELINE_SIGNALS[wname])
    if not isinstance(wide.index, pd.DatetimeIndex):
        wide = wide.T
    wide = wide.sort_index()
    days = pd.DatetimeIndex(wide.index)
    if limit_days is not None:
        days = days[:limit_days]
    ident = dict(protocol=C.PROTOCOL_VERSION, split=wname,
                 tokenizer_sha=identity["tokenizer_sha"],
                 predictor_sha=identity["predictor_sha"])
    stats = {"split": wname, "n_days": len(days), "rebuilt": 0, "reused": 0,
             "wall_s": 0.0, "cuda_s": 0.0, "coverage_fail": []}
    t_all = time.perf_counter()
    for t in days:
        ds = t.strftime("%Y-%m-%d")
        path = out_dir / f"{wname}_{t.strftime('%Y%m%d')}.npz"
        if path.exists():
            try:
                load_chunk(path, ident)
                stats["reused"] += 1
                continue
            except CacheMismatchError as exc:
                logger.warning(f"chunk 身份不匹配，重建：{path}（{exc}）")
        t0 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        ev1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if ev0 is not None:
            ev0.record()
        valid = wide.loc[t].dropna()
        valid_codes = sorted(valid.index.tolist())
        df_list, x_ts_list, y_ts_list, codes, wstats = build_inference_windows(
            provider, ds, lookback=C.LOOKBACK, predict_len=C.PREDICT_LEN,
            pool=C.POOL)
        idx = {c: j for j, c in enumerate(codes)}
        missing = [c for c in valid_codes if c not in idx]
        if missing:
            stats["coverage_fail"].append({"date": ds, "missing": missing[:20],
                                           "n_missing": len(missing)})
            raise RuntimeError(
                f"{wname} {ds}：{len(missing)} 只原 G1 有效股票抽不出历史窗口"
                f"（示例 {missing[:5]}）——停止排查，不得用掩码删格")
        sel_df = [df_list[idx[c]] for c in valid_codes]
        sel_x = [x_ts_list[idx[c]] for c in valid_codes]
        h = extract_hidden(backbone, sel_df, sel_x)
        payload = {
            "dates": np.array([ds] * len(valid_codes)),
            "instruments": np.array(valid_codes),
            "h": h.astype(np.float32),
            "s_g1": valid.loc[valid_codes].to_numpy(dtype=np.float64),
            "x_start": np.array([str(pd.DatetimeIndex(s).min().date())
                                 for s in sel_x]),
            "x_end": np.array([ds] * len(valid_codes)),
        }
        write_chunk(path, payload, ident)
        if ev1 is not None:
            ev1.record()
            torch.cuda.synchronize()
            stats["cuda_s"] += ev0.elapsed_time(ev1) / 1000.0
        stats["rebuilt"] += 1
        logger.info(f"[{wname}] {ds}: g1_valid={len(valid_codes)} "
                    f"hidden={h.shape} wall={time.perf_counter() - t0:.1f}s")
    stats["wall_s"] = time.perf_counter() - t_all
    return stats


def freeze_cache(art_dir: Path | None = None) -> dict:
    """缓存完成后冻结：聚合样本键 parquet + chunk 清单 + 全量 SHA。

    大缓存不入 git；manifest 记录逐 chunk SHA256 与规模（跨机复算凭据）。
    """
    art_dir = art_dir or C.ART_DIR
    art_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"created_at": datetime.now().isoformat(timespec="seconds"),
                      "splits": {}}
    for split in ("train", "val", "W3", "W4"):
        d = C.CACHE_DIR / split
        chunks = sorted(p for p in d.glob(f"{split}_*.npz")
                        if not p.name.endswith(".tmp.npz"))
        assert chunks, f"{split} 无缓存 chunk"
        rows = []
        chunk_list = []
        for p in chunks:
            arr = np.load(p, allow_pickle=False)
            rows.append(pd.DataFrame({
                "split": split, "date": arr["dates"],
                "instrument": arr["instruments"],
                "x_start": arr["x_start"], "x_end": arr["x_end"]}))
            chunk_list.append({"file": p.name, "sha256": sha256_file(p),
                               "n": int(arr["h"].shape[0]),
                               "h_dim": int(arr["h"].shape[1])})
        keys = pd.concat(rows, ignore_index=True)
        sfp = samples_path(art_dir, split)
        keys.to_parquet(sfp)
        manifest["splits"][split] = {
            "n_chunks": len(chunks), "n_samples": int(len(keys)),
            "sample_keys_file": sfp.name,
            "sample_keys_sha256": sha256_file(sfp),
            "h_dim": chunk_list[0]["h_dim"],
            "chunks": chunk_list,
        }
    (art_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"缓存冻结：{json.dumps({k: v['n_samples'] for k, v in manifest['splits'].items()})}")
    return manifest


__all__ = ["load_frozen_g1", "build_head_split", "build_test_split",
           "freeze_cache", "extract_hidden", "run_teacher",
           "load_split_arrays", "load_chunk", "write_chunk", "sha256_file"]
