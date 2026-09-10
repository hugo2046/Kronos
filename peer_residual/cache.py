"""PeerSet 补 h 缓存构建（计划 §4：优先零 teacher 重推）。

对 SAE 已有 h（train/val 64 股 LossSet、W3/W4 基线格）**只读复用**；
仅补算当日 PeerSet 其余股票的 ``h``（纯 ``KronosFrozenBackbone.extract``
前向，无采样、无 teacher、无标签）。绝不生成新的 teacher 预测。

资格判定沿用 :func:`kronos_qlib.build_inference_windows`（PIT csi300 成分
∩ 90 日窗口完整 ∩ 窗口内无停牌）——PeerSet 资格不使用未来收益/停牌/
退池/标签完整性；LossSet/基线格必须 ⊆ PeerSet，否则停止排查。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from peer_residual import config as C
from peer_residual import data as PD
from sae_residual.cache import extract_hidden, load_frozen_g1
from sae_residual.cache_io import (CacheMismatchError, identity_of,
                                   load_chunk, sha256_file, write_chunk)

# 复用核验的固定抽样（§4）：train/val 最早日 + W3/W4 首日，各前 3 股
VERIFY_TOP_K = 3
VERIFY_TOL = 1e-5


def build_peer_split(provider, backbone, identity: dict, split: str,
                     limit_days: int | None = None, subdir: str | None = None,
                     splits_dir: Path | None = None) -> dict:
    """逐决策日补齐 PeerSet 缺失 h（可断点续跑，身份校验后复用）。

    :param limit_days: smoke 用——只跑前 N 个决策日。
    :param subdir: smoke 用——写到隔离目录。
    :returns: 统计字典（逐日 PeerSet/额外数、耗时、chunk 状态）。
    """
    from kronos_qlib import build_inference_windows
    from peer_residual.history_provider import PeerHistoryProvider

    # §5 读取边界纠偏：实际 fetch 上界钳制到当前决策日（未来日历仍可用）
    hist = PeerHistoryProvider(provider)
    out_dir = C.PEER_CACHE_DIR / (subdir or split)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = PD.split_dates(split)
    if limit_days is not None:
        dates = dates[:limit_days]
    ident_base = dict(protocol=C.PROTOCOL_VERSION, split=split,
                      tokenizer_sha=identity["tokenizer_sha"],
                      predictor_sha=identity["predictor_sha"])
    stats = {"split": split, "n_days": len(dates), "rebuilt": 0, "reused": 0,
             "wall_s": 0.0, "cuda_s": 0.0, "days": [],
             "subset_fail": [], "skipped": []}
    t_all = time.perf_counter()
    for dstr in dates:
        sp = PD.sae_chunk_path(split, dstr)
        path = PD.peer_chunk_path(split, dstr, subdir)
        try:
            sae = load_chunk(sp, {"protocol": C.SAE_PROTOCOL, "split": split,
                                  "tokenizer_sha": identity["tokenizer_sha"],
                                  "predictor_sha": identity["predictor_sha"]})
        except CacheMismatchError as exc:
            raise RuntimeError(f"SAE chunk 身份不匹配，停止：{sp}（{exc}）")
        day_stat: dict = {"date": dstr}
        if path.exists():
            try:
                got = identity_of(path)
                ok = all(got.get(k) == v for k, v in ident_base.items())
                ok = ok and got.get("sae_chunk_sha256") == sha256_file(sp)
                if ok:
                    stats["reused"] += 1
                    day_stat.update({"n_peers": got.get("n_peers"),
                                     "n_extra": got.get("n_extra"),
                                     "status": "reused"})
                    stats["days"].append(day_stat)
                    continue
            except Exception:  # noqa: BLE001 - 损坏则重建
                logger.warning(f"peer chunk 损坏，重建：{path}")
        t0 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        ev1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if ev0 is not None:
            ev0.record()
        hist.set_decision_bound(dstr)          # t > FORWARD_CUTOFF 拒绝
        df_list, x_ts_list, _, codes, wstats = build_inference_windows(
            hist, dstr, lookback=C.LOOKBACK, predict_len=C.PREDICT_LEN,
            pool=C.POOL)
        sae_codes = {str(c) for c in sae["instruments"]}
        peer_set = set(codes)
        missing = sae_codes - peer_set
        if missing:
            stats["subset_fail"].append({"date": dstr,
                                         "missing": sorted(missing)[:20],
                                         "n_missing": len(missing)})
            raise RuntimeError(
                f"{split} {dstr}: LossSet/基线格 {len(missing)} 只不在 PeerSet"
                f"（示例 {sorted(missing)[:5]}）——停止排查，不得缩池")
        extra = sorted(peer_set - sae_codes)
        idx = {c: j for j, c in enumerate(codes)}
        day_stat.update({"n_pool": wstats["n_pool"],
                         "n_window_kept": wstats["n_kept"],
                         "n_skipped_short": wstats["skipped_short"],
                         "n_skipped_halt": wstats["skipped_halt"],
                         "n_sae": len(sae_codes), "n_peers": len(peer_set),
                         "n_extra": len(extra)})
        if extra:
            h = extract_hidden(backbone,
                               [df_list[idx[c]] for c in extra],
                               [x_ts_list[idx[c]] for c in extra])
            assert np.isfinite(h).all(), f"{dstr}: 补 h 含非有限值"
        else:
            h = np.zeros((0, int(sae["h"].shape[1])), dtype=np.float32)
        payload = {
            "dates": np.array([dstr] * len(extra)),
            "instruments": np.array(extra),
            "h": h.astype(np.float32),
            "peer_codes": np.array(sorted(peer_set)),
            "x_start": np.array([str(pd.DatetimeIndex(
                x_ts_list[idx[c]]).min().date()) for c in extra]),
            "x_end": np.array([dstr] * len(extra)),
        }
        ident = {**ident_base, "date": dstr,
                 "sae_chunk_sha256": sha256_file(sp),
                 "n_peers": len(peer_set), "n_extra": len(extra)}
        write_chunk(path, payload, ident)
        if ev1 is not None:
            ev1.record()
            torch.cuda.synchronize()
            stats["cuda_s"] += ev0.elapsed_time(ev1) / 1000.0
        stats["rebuilt"] += 1
        day_stat.update({"wall_s": round(time.perf_counter() - t0, 2),
                         "status": "rebuilt"})
        stats["days"].append(day_stat)
        logger.info(f"[peer:{split}] {dstr}: peers={len(peer_set)} "
                    f"sae={len(sae_codes)} extra={len(extra)} "
                    f"wall={day_stat['wall_s']}s")
    stats["wall_s"] = round(time.perf_counter() - t_all, 1)
    return stats


def verify_sample_against_g1(provider, backbone, identity: dict,
                             subdir: str | None = None) -> dict:
    """复用前固定抽样核验（§4）：4 个锚点日各前 3 股重提 h 对拍。

    train 最早日、val 最早日、W3/W4 首日；各取代码排序前 3 个可用股，
    用当前同一 G1 重提末步 h 并与 SAE 缓存对拍，FP32 绝对容差 1e-5。
    超差抛错停止（不混合来源、不调宽容差）。
    """
    from kronos_qlib import build_inference_windows
    from peer_residual.history_provider import PeerHistoryProvider

    hist = PeerHistoryProvider(provider)
    results = {}
    for split in ("train", "val", "W3", "W4"):
        dstr = PD.split_dates(split)[0]
        sp = PD.sae_chunk_path(split, dstr)
        sae = load_chunk(sp, {"protocol": C.SAE_PROTOCOL, "split": split,
                              "tokenizer_sha": identity["tokenizer_sha"],
                              "predictor_sha": identity["predictor_sha"]})
        sae_codes = [str(c) for c in sae["instruments"]]
        picks = sorted(sae_codes)[:VERIFY_TOP_K]
        hist.set_decision_bound(dstr)
        df_list, x_ts_list, _, codes, _ = build_inference_windows(
            hist, dstr, lookback=C.LOOKBACK, predict_len=C.PREDICT_LEN,
            pool=C.POOL)
        idx = {c: j for j, c in enumerate(codes)}
        absent = [c for c in picks if c not in idx]
        assert not absent, f"{split} {dstr} 核验股抽不出窗口：{absent}"
        h_new = extract_hidden(backbone, [df_list[idx[c]] for c in picks],
                               [x_ts_list[idx[c]] for c in picks])
        h_old = np.stack([sae["h"][sae_codes.index(c)] for c in picks])
        diff = float(np.abs(h_new - h_old).max())
        assert diff <= VERIFY_TOL, (
            f"{split} {dstr} 复用核验超差：max|h_new-h_old|={diff:.3e} > "
            f"{VERIFY_TOL:.0e}——停止查代码/数据/权重，不得调宽容差")
        results[split] = {"date": dstr, "codes": picks,
                          "max_abs_diff": diff, "tol": VERIFY_TOL}
        logger.info(f"[verify] {split} {dstr}: max|Δh|={diff:.3e} ✓")
    return results


def freeze_peer_cache(art_dir: Path | None = None) -> dict:
    """peer 缓存完成后冻结：逐 chunk SHA 清单 + PeerSet/LossSet 覆盖统计。

    大缓存不入 git；manifest 提供逐 chunk SHA256 与规模（跨机复算凭据）。
    """
    art_dir = art_dir or C.ART_DIR
    art_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"created_at": datetime.now().isoformat(timespec="seconds"),
                      "protocol": C.PROTOCOL_VERSION, "run_id": C.RUN_ID,
                      "splits": {}}
    for split in ("train", "val", "W3", "W4"):
        d = C.PEER_CACHE_DIR / split
        chunks = sorted(p for p in d.glob(f"peer_{split}_*.npz")
                        if not p.name.endswith(".tmp.npz"))
        expected = len(PD.split_dates(split))
        assert len(chunks) == expected, \
            f"{split}: chunk {len(chunks)} != 预期键集合 {expected}"
        rows, n_peers_sum, n_extra_sum, n_loss_sum = [], 0, 0, 0
        for p in chunks:
            got = identity_of(p)
            arr = np.load(p, allow_pickle=False)
            n_extra = int(arr["h"].shape[0])
            rows.append({"file": p.name, "sha256": sha256_file(p),
                         "date": got["date"], "n_peers": got["n_peers"],
                         "n_extra": n_extra,
                         "sae_chunk_sha256": got["sae_chunk_sha256"]})
            n_peers_sum += got["n_peers"]
            n_extra_sum += n_extra
            n_loss_sum += got["n_peers"] - n_extra
        manifest["splits"][split] = {
            "n_chunks": len(chunks), "n_days": len(chunks),
            "n_peer_days_total": n_peers_sum, "n_extra_h_total": n_extra_sum,
            "n_loss_days_total": n_loss_sum,
            "avg_peers_per_day": round(n_peers_sum / len(chunks), 1),
            "chunks": rows,
        }
    (art_dir / "peer_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("peer 缓存冻结：" + json.dumps(
        {k: {"days": v["n_days"], "avg_peers": v["avg_peers_per_day"],
             "extra_h": v["n_extra_h_total"]}
         for k, v in manifest["splits"].items()}, ensure_ascii=False))
    return manifest


__all__ = ["build_peer_split", "verify_sample_against_g1",
           "freeze_peer_cache", "load_frozen_g1", "extract_hidden"]
