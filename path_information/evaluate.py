"""PATH1 开封与唯一诊断判据（计划 §6）。

顺序硬约束（spy 锁定）：先冻结两头全部段分数与来源 SHA（manifest），
之后才允许加载 dev_eval 标签或输出其统计——缺一/改一分数文件，标签
加载 0 次。诊断 = 三臂逐日相同有效标签格的 Spearman(score, y_mean)
（平均秩，常数日记 NaN），配对差用双方均有效的同一日期集合。

唯一"值得另立下一阶段"信号：val 与 dev_eval 的日均配对 IC 差
PATH−MEAN 均 >0，且 dev_eval 的 PATH−G1 日均配对 IC 差 >0。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from path_information import config as C
from path_information import data as PD


def final_scores(s_g1: np.ndarray, r_hat: np.ndarray, stats: dict) -> np.ndarray:
    """``s_final = s_g1_original + sigma_e_train * r_hat``（未训练 ≡ G1）。"""
    return (np.asarray(s_g1, dtype=np.float64)
            + stats["sigma_e"] * np.asarray(r_hat, dtype=np.float64))


def daily_spearman(dates: np.ndarray, y: np.ndarray, mask: np.ndarray,
                   **score_arrays: np.ndarray) -> dict[str, list[float]]:
    """三臂逐日 Spearman：仅当日**共同有效标签格**（三臂分数 + 标签全有限）。

    :param dates: ``[N]`` 决策日（YYYY-MM-DD）。
    :param y: ``[N]`` ``y_mean``（NaN = 无标签）。
    :param mask: ``[N]`` 共同有效格（标签有限；缺标签只屏蔽评价格）。
    :param score_arrays: 臂名 → ``[N]`` 分数（G1 臂传 ``G1=s_g1``）。
    :returns: 臂名 → 逐日 IC 列表（顺序 = 该段决策日升序；常数日记 NaN）。
    """
    df = pd.DataFrame({"date": dates, "y": y, "valid": mask,
                       **{k: v for k, v in score_arrays.items()}})
    out: dict[str, list[float]] = {k: [] for k in score_arrays}
    for _, day in df.groupby("date", sort=True):
        v = (day["valid"].to_numpy(dtype=bool)
             & np.isfinite(day["y"].to_numpy(dtype=np.float64)))
        for k in out:
            arr = day[k].to_numpy(dtype=np.float64)
            vv = v & np.isfinite(arr)
            out[k].append(PD.spearman_avg_rank(arr[vv],
                                               day["y"].to_numpy(np.float64)[vv])
                          if vv.sum() >= 2 else float("nan"))
    return out


def paired_daily_ic(ic: dict[str, list[float]],
                    dates: list[str]) -> dict[str, dict]:
    """配对差：双方均有效的同一日期集合上的日均 IC 差。

    :param ic: 臂名 → 逐日 IC（含 NaN；NaN 日对该臂无效）。
    :param dates: 与 IC 列表等长的决策日标签。
    :returns: ``{"PATH-MEAN": {"dates", "mean", "n_days"}, ...}``（含 MEAN-G1）。
    """
    pairs = (("PATH", "MEAN", "PATH-MEAN"), ("PATH", "G1", "PATH-G1"),
             ("MEAN", "G1", "MEAN-G1"))
    out: dict[str, dict] = {}
    for a, b, name in pairs:
        ds, diffs = [], []
        for d, ia, ib in zip(dates, ic[a], ic[b]):
            if np.isfinite(ia) and np.isfinite(ib):
                ds.append(d)
                diffs.append(ia - ib)
        out[name] = {"dates": ds, "n_days": len(ds),
                     "mean": float(np.mean(diffs)) if diffs else float("nan")}
    return out


def criterion(val_pairs: dict, dev_pairs: dict) -> bool:
    """唯一开发信号：val 与 dev_eval 的 PATH−MEAN 均 >0 且 dev 的 PATH−G1 >0。

    满足也只称"本历史划分下观察到路径信息增量，需另立复验"（计划 §6.5）。
    """
    return bool(val_pairs["PATH-MEAN"]["mean"] > 0
                and dev_pairs["PATH-MEAN"]["mean"] > 0
                and dev_pairs["PATH-G1"]["mean"] > 0)


def arm_mse(dates: np.ndarray, y: np.ndarray, mask: np.ndarray,
            s_g1: np.ndarray, r_hat: np.ndarray, stats: dict) -> float:
    """单臂归一化 MSE（日等权）：``mean_daily(mean_cells((r_hat − r_true)²)``，
    ``r_true = (y − s_g1)/σe``——与训练同一误差目标（G1 臂 r_hat≡0 即零残差 MSE）。"""
    df = pd.DataFrame({"date": dates, "y": y, "valid": mask, "s": s_g1,
                       "r": r_hat})
    per_day = []
    for _, day in df.groupby("date", sort=True):
        v = day["valid"].to_numpy(dtype=bool)
        if not v.any():
            continue
        r_true = ((day["y"].to_numpy(np.float64)
                   - day["s"].to_numpy(np.float64)) / stats["sigma_e"])
        err = (day["r"].to_numpy(np.float64) - r_true) ** 2
        per_day.append(float(err[v].mean()))
    return float(np.mean(per_day)) if per_day else float("nan")


def verify_scores_frozen(art_dir: Path) -> dict:
    """完整六臂段、来源身份、权重和分数全部核验后才允许加载标签。

    :param art_dir: 冻结产物目录。
    :returns: 已验证清单。
    :raises RuntimeError: 任一必需臂段缺失、错身份、改字节或错样本键。
    """
    from path_information import paths as PP
    from path_information import train as PT

    mf = art_dir / "scores_manifest.json"
    if not mf.is_file():
        raise RuntimeError(f"分数 manifest 缺失：{mf}")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    need = {f"{a}:{s}" for a in C.ARMS for s in C.SEGMENTS}
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != need:
        raise RuntimeError("冻结清单必须包含完整且唯一的两臂三段")
    arrays = {}
    for key in sorted(need):
        arm, split = key.split(":")
        rec = files[key]
        name = f"scores_{arm}_{split}.npz"
        if rec.get("file") != name:
            raise RuntimeError(f"{key} 文件名与臂段身份不一致")
        p = art_dir / name
        if not p.is_file() or PP.sha256_file(p) != rec.get("sha256"):
            raise RuntimeError(f"冻结分数缺失或字节 SHA 不一致：{key}")
        arr = PP.read_path_chunk(p, {"protocol": C.PROTOCOL_VERSION,
                                     "arm": arm, "split": split,
                                     "seed": C.SEED, "kind": "scores"})
        if not np.isfinite(arr["s_final"]).all():
            raise RuntimeError(f"冻结分数非有限：{key}")
        arrays[key] = arr
    context = PT.current_context(art_dir)
    if manifest.get("context") != context:
        raise RuntimeError("冻结清单上下文与当前期望不一致，旧记录不自动升级")
    heads = manifest.get("heads")
    if not isinstance(heads, dict) or set(heads) != set(C.ARMS):
        raise RuntimeError("冻结清单缺少完整头身份")
    for arm in C.ARMS:
        _, best = PT.load_best(C.HEADS_DIR, arm, C.SEED, expect_context=context)
        if heads[arm] != {"file": best["best_head_file"],
                          "sha256": best["best_head_sha256"], "epoch": best["best_epoch"]}:
            raise RuntimeError(f"{arm} 冻结头与当前 best 身份不一致")
    for split in C.SEGMENTS:
        dates, codes = [], []
        for p in sorted((art_dir / "cache" / split).glob(f"{split}_*.npz")):
            arr = PP.read_path_chunk(p, {"protocol": C.PROTOCOL_VERSION, "split": split})
            dates.extend(arr["dates"].tolist())
            codes.extend(arr["instruments"].tolist())
        if not dates or len(set(zip(dates, codes))) != len(dates):
            raise RuntimeError(f"{split} 路径样本为空或重复")
        for arm in C.ARMS:
            arr = arrays[f"{arm}:{split}"]
            if (arr["dates"].tolist() != dates or arr["instruments"].tolist() != codes
                    or arr["s_final"].shape != (len(dates),)):
                raise RuntimeError(f"{arm}:{split} 分数与冻结路径键不一致")
    return manifest


def unseal(art_dir: Path, label_loader=None, **label_kwargs) -> dict:
    """spy 锁定编排：分数冻结校验通过后才调用标签加载（否则 0 次调用）。"""
    verify_scores_frozen(art_dir)
    if label_loader is None:
        from path_information import paths as PP

        label_loader = PP.build_dev_eval_labels
    return label_loader(**label_kwargs)


__all__ = ["final_scores", "daily_spearman", "paired_daily_ic", "criterion",
           "arm_mse", "verify_scores_frozen", "unseal"]
