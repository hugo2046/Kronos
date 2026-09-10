"""PEER 残差 CPU 诊断（计划 20260910 §4：G1 零残差基准 + OFF/ON 对照）。

只读 SAE/PEER 历史缓存与已选定小头（固定 OFF e18 / ON e16），在 CPU 上
生成 train/val 诊断表：相同 LossSet、日期等权口径的 MSE 与截面排名对照。
不重训、不加载真实 G1 权重、不初始化 CUDA、不读 forward（读取上界
2026-07-24，本轮只读 train/val 缓存）。

冻结公式（§4.1）::

    e = y_mean − s_G1；r = e/σe；r_hat_G1 = 0；s_arm = s_G1 + σe·r_hat_arm
    每日只在原 SAE 64 个监督格计算，各日先日内 mean 再跨日等权 mean；
    不减残差均值、不加校准缩放、不扫 epoch。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peer_residual import config as C  # noqa: E402
from peer_residual import data as PD  # noqa: E402
from peer_residual.identity import (load_legacy_head,  # noqa: E402
                                    verify_cache_dir)
from sae_residual.cache_io import sha256_file  # noqa: E402

# 固定对照（§2/§4）：旧 best 头与验证 MSE 一致性容差
ARMS_FIXED = {"OFF": 18, "ON": 16}
OLD_VAL_MSE = {"OFF": 0.6580492705106735, "ON": 0.6613516435027122}
CONSISTENCY_TOL = 1e-5
# 标签终点上界（§3）
LABEL_END_BOUND = {"train": "2024-12-31", "val": "2025-06-30"}
EXPECTED_DAYS = {"train": 534, "val": 22}
EXPECTED_CELLS = {"train": 34176, "val": 1408}
EXPECTED_EXTRA_H = {"train": 108882, "val": 5058}
INPUT_LIST_DEFAULT = C.REPO_ROOT / "docs" / "PEER诊断输入身份_20260910.json"


# ---------------- 冻结公式（§4.1，可测纯函数） ----------------

def day_errors(r: np.ndarray, r_hat: np.ndarray,
               sigma_e: float) -> dict[str, float]:
    """单日 LossSet 误差（参数已截取当日监督格）。

    :param r: 归一化残差目标 ``e/σe``。
    :param r_hat: 头预测（G1 零残差臂恒 0）。
    :returns: ``mse_norm``（归一化量纲）、``mse_return = mse_norm·σe²``、
        ``bias_return = mean(error)·σe``。
    """
    error = r - r_hat
    mse_norm = float(np.mean(error ** 2))
    return {"mse_norm": mse_norm,
            "mse_return": mse_norm * sigma_e ** 2,
            "bias_return": float(np.mean(error)) * sigma_e}


def daily_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """平均秩 Spearman；任一侧常数（std=0）→ NaN（不擅自记零）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float(np.mean(ra * rb))


# ---------------- 输入身份（§3） ----------------

def verify_input_list(input_list: Path) -> dict:
    """运行前逐文件校验交接清单；不匹配停止（不重写清单凑过）。"""
    spec = json.loads(input_list.read_text(encoding="utf-8"))
    checked = []
    for f in spec["files"]:
        p = C.REPO_ROOT / f["path"]
        if not p.is_file():
            raise RuntimeError(f"输入缺失：{p}")
        got = sha256_file(p)
        if got != f["sha256"] or p.stat().st_size != f["bytes"]:
            raise RuntimeError(f"输入身份不匹配：{p}（{got[:16]}…）")
        checked.append(f["path"])
    logger.info(f"[input] 清单校验 {len(checked)}/{len(spec['files'])} 通过"
                f"（source={spec['source_commit'][:12]}）")
    return spec


def _sae_manifest(sae_art: Path) -> dict:
    mf = json.loads((sae_art / "cache_manifest.json").read_text(encoding="utf-8"))
    return {"splits": {k: {"chunks": [{"file": c["file"], "sha256": c["sha256"]}
                                      for c in v["chunks"]]}
                       for k, v in mf["splits"].items()}}


def verify_history_caches(sae_cache: Path, peer_cache: Path,
                          peer_manifest_path: Path,
                          sae_art: Path) -> dict:
    """按冻结 manifest 校验 SAE/peer 目录：完整键集 + 逐 chunk SHA。"""
    sae_mf = _sae_manifest(sae_art)
    out = {}
    for split in ("train", "val", "W3", "W4"):
        out[f"sae:{split}"] = verify_cache_dir(sae_cache / split, sae_mf, split)
        out[f"peer:{split}"] = verify_cache_dir(
            peer_cache / split,
            {"splits": {split: json.loads(peer_manifest_path.read_text(
                encoding="utf-8"))["splits"][split]}}, split)
    # 规模核对（§3：train/val 534/22 日、34176/1408 格、额外 h 计数）
    pm = json.loads(peer_manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        v = pm["splits"][split]
        assert v["n_days"] == EXPECTED_DAYS[split], \
            f"{split} 决策日 {v['n_days']} != {EXPECTED_DAYS[split]}"
        assert v["n_loss_days_total"] == EXPECTED_CELLS[split], \
            f"{split} 监督格 {v['n_loss_days_total']} != {EXPECTED_CELLS[split]}"
        assert v["n_extra_h_total"] == EXPECTED_EXTRA_H[split], \
            f"{split} 额外 h {v['n_extra_h_total']} != {EXPECTED_EXTRA_H[split]}"
    logger.info("[cache] SAE/peer 目录按冻结清单校验通过（含规模核对）")
    return out


# ---------------- 诊断主流程 ----------------

def _forward_days(days: list, models: dict, threads_note: dict) -> dict:
    """逐日全 PeerSet CPU 前向（eval/no_grad），返回 {(arm,date): r_hat}。"""
    out: dict[tuple[str, str], np.ndarray] = {}
    for i in range(0, len(days), C.TRAIN["batch_days"]):
        chunk = days[i:i + C.TRAIN["batch_days"]]
        x, valid, loss, _ = PD.collate_batch(chunk)
        for arm, model in models.items():
            with torch.no_grad():
                r_hat = model(x, valid, arm)
            for j, d in enumerate(chunk):
                idx = np.where(loss[j].numpy())[0]
                out[(arm, d.date)] = r_hat[j].numpy()[idx]
    return out


def run_diagnostics(sae_cache: Path, peer_cache: Path, output: Path,
                    input_list: Path = INPUT_LIST_DEFAULT,
                    threads: int = 4) -> dict:
    """执行完整诊断并落盘（输出目录已有完成 manifest 时拒绝覆盖）。"""
    output.mkdir(parents=True, exist_ok=True)
    done_marker = output / "diagnostic_summary.json"
    if done_marker.is_file():
        raise RuntimeError(f"输出目录已有完成 manifest，拒绝覆盖：{done_marker}")
    torch.set_num_threads(threads)
    timing = {"torch_threads": threads,
              "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
              "cuda_initialized": torch.cuda.is_initialized(),
              "stages": {}}
    assert not torch.cuda.is_initialized(), "诊断禁止初始化 CUDA"
    t0 = time.perf_counter()
    spec = verify_input_list(input_list)
    sae_art = C.REPO_ROOT / "artifacts" / "experiments" / "sae_residual_20260910"
    peer_art = C.REPO_ROOT / "artifacts" / "experiments" / "peer_residual_20260910"
    peer_manifest = peer_art / "peer_cache_manifest.json"
    cache_check = verify_history_caches(sae_cache, peer_cache, peer_manifest,
                                        sae_art)
    timing["stages"]["input_verify_s"] = round(time.perf_counter() - t0, 2)

    # 底座权重身份从冻结 manifest 获取（不加载真实 G1 权重）
    pre = json.loads((peer_art / "preflight_manifest.json").read_text(
        encoding="utf-8"))
    weight_shas = {"tokenizer_sha": pre["g1_weights"]["tokenizer"]["sha256"],
                   "predictor_sha": pre["g1_weights"]["predictor"]["sha256"]}
    stats = PD.load_frozen_stats()

    # legacy 头只读路径：仅接收交接清单精确 SHA（§6.1）
    head_paths = {arm: peer_art / f"head_{arm}_s100_e{ARMS_FIXED[arm]:03d}.pt"
                  for arm in ARMS_FIXED}
    head_sha_spec = {Path(f["path"]).name: f["sha256"] for f in spec["files"]}
    models, head_notes = {}, {}
    for arm, p in head_paths.items():
        model, note = load_legacy_head(p, head_sha_spec[p.name], arm, 100,
                                       ARMS_FIXED[arm], int(stats["d_in"]))
        models[arm] = model
        head_notes[arm] = note

    daily_rows, sample_rows = [], []
    t1 = time.perf_counter()
    for split in ("train", "val"):
        days = PD.load_split_days(split, stats, weight_shas,
                                  sae_dir=sae_cache, peer_dir=peer_cache)
        assert len(days) == EXPECTED_DAYS[split]
        for d in days:
            assert d.loss_mask.sum() == 64, \
                f"{d.date}: LossSet {int(d.loss_mask.sum())} != 64"
        r_hat = _forward_days(days, models, timing)
        for d in days:
            # SAE 原值监督格：y_mean/s_G1 直读 chunk（不做任何重建）；
            # 重复日期-股票键与标签终点上界逐日检查（§3）
            arr = np.load(PD.sae_chunk_path(split, d.date, sae_cache),
                          allow_pickle=False)
            insts = [str(c) for c in arr["instruments"]]
            assert len(insts) == len(set(insts)), f"{d.date}: 重复股票键"
            assert str(arr["label_end"][0]) <= LABEL_END_BOUND[split], \
                f"{split} {d.date} 标签终点 {str(arr['label_end'][0])} " \
                f"晚于 {LABEL_END_BOUND[split]}"
            y_map = {c: float(v) for c, v in zip(insts, arr["y_mean"])}
            idx = np.where(d.loss_mask)[0]
            y = np.array([y_map[d.codes[i]] for i in idx])
            s_g1 = d.s_g1[idx]
            r = d.r[idx]
            rh = {arm: r_hat[(arm, d.date)] for arm in ARMS_FIXED}
            arms_val = {"G1": np.zeros(len(idx)),
                        "OFF": rh["OFF"], "ON": rh["ON"]}
            for arm, rv in arms_val.items():
                errs = day_errors(r, rv, stats["sigma_e"])
                daily_rows.append({
                    "split": split, "date": d.date, "arm": arm,
                    **errs,
                    "spearman_y": daily_spearman(
                        s_g1 + stats["sigma_e"] * rv, y),
                    "resid_std_return": float(np.std(stats["sigma_e"] * rv)),
                    "n_valid": int(len(idx))})
            for k, i in enumerate(idx):
                sample_rows.append({
                    "split": split, "date": d.date, "instrument": d.codes[i],
                    "y_mean": float(y[k]), "s_G1": float(s_g1[k]),
                    "r_hat_OFF": float(rh["OFF"][k]),
                    "r_hat_ON": float(rh["ON"][k]),
                    "s_OFF": float(s_g1[k] + stats["sigma_e"] * rh["OFF"][k]),
                    "s_ON": float(s_g1[k] + stats["sigma_e"] * rh["ON"][k])})
    timing["stages"]["forward_s"] = round(time.perf_counter() - t1, 2)

    daily = pd.DataFrame(daily_rows)
    samples = pd.DataFrame(sample_rows)
    assert len(samples) == EXPECTED_CELLS["train"] + EXPECTED_CELLS["val"], \
        f"监督格小表 {len(samples)} 行异常"
    daily.to_parquet(output / "daily_diagnostics.parquet", index=False)
    samples.to_parquet(output / "sample_diagnostics.parquet", index=False)

    summary = _summarize(daily)
    _consistency_gate(summary)
    timing["stages"]["total_s"] = round(time.perf_counter() - t0, 2)
    summary["timing"] = timing
    summary["head_notes"] = head_notes
    summary["inputs"] = {
        "source_commit": spec["source_commit"],
        "sae_cache": str(sae_cache), "peer_cache": str(peer_cache),
        "input_list": str(input_list), "n_verified_files": len(spec["files"]),
        "cache_check": {k: v["n_chunks"] for k, v in cache_check.items()}}
    summary["created_at"] = datetime.now().isoformat(timespec="seconds")
    done_marker.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                      default=float), encoding="utf-8")
    (output / "timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "input_manifest.json").write_text(
        json.dumps({"verified_files": [
            {"path": f["path"], "sha256": f["sha256"]} for f in spec["files"]],
            "sae_cache": str(sae_cache), "peer_cache": str(peer_cache),
            "weight_identity_source": "preflight_manifest（冻结 manifest，"
                                      "未加载真实 G1 权重）",
            "created_at": summary["created_at"]},
            ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[diagnose] 完成：{output}（total "
                f"{timing['stages']['total_s']}s CPU）")
    return summary


def _summarize(daily: pd.DataFrame) -> dict:
    """跨日等权汇总 + 对 G1 差额 + 配对 RankIC 差（双方均有效日期）。"""
    out: dict = {"per_split_arm": {}, "delta_vs_g1": {}, "rankic": {}}
    for split in ("train", "val"):
        sub = daily[daily["split"] == split]
        means = sub.groupby("arm")[["mse_norm", "mse_return",
                                    "bias_return"]].mean()
        # 一致性门禁锚点：OFF/ON 验证等权 MSE
        for arm in ARMS_FIXED:
            if split == "val":
                got = float(means.loc[arm, "mse_norm"])
                ref = OLD_VAL_MSE[arm]
                out.setdefault("consistency", {})[arm] = {
                    "this_run": got, "old": ref,
                    "abs_diff": abs(got - ref)}
        for arm in ("OFF", "ON"):
            rel = (1 - means.loc[arm, "mse_norm"]
                   / means.loc["G1", "mse_norm"]
                   if means.loc["G1", "mse_norm"] != 0 else float("nan"))
            out["per_split_arm"][f"{split}:{arm}"] = {
                k: float(means.loc[arm, k]) for k in means.columns}
            out["per_split_arm"][f"{split}:G1"] = {
                k: float(means.loc["G1", k]) for k in means.columns}
            out["delta_vs_g1"][f"{split}:{arm}"] = {
                "mse_norm": float(means.loc[arm, "mse_norm"]
                                  - means.loc["G1", "mse_norm"]),
                "mse_return": float(means.loc[arm, "mse_return"]
                                    - means.loc["G1", "mse_return"]),
                "relative_improvement": float(rel)}
        # 配对 RankIC（arm−G1，同日双方均有效）
        for arm in ("OFF", "ON"):
            g = sub[sub["arm"] == "G1"].set_index("date")["spearman_y"]
            a = sub[sub["arm"] == arm].set_index("date")["spearman_y"]
            both = pd.concat([g.rename("g"), a.rename("a")], axis=1).dropna()
            out["rankic"][f"{split}:{arm}"] = {
                "valid_days": int(len(both)),
                "invalid_days": int(len(g) + len(a) - 2 * len(both)) // 2,
                "mean_spearman_arm": float(a.dropna().mean()),
                "mean_spearman_g1": float(g.dropna().mean()),
                "paired_diff_mean": float((both["a"] - both["g"]).mean())}
    return out


def _consistency_gate(summary: dict) -> None:
    """OFF/ON 验证 MSE 与旧值对拍 ≤1e-5；超差停止解释结论（§4）。"""
    for arm, ref in OLD_VAL_MSE.items():
        got = summary["consistency"][arm]["this_run"]
        if abs(got - ref) > CONSISTENCY_TOL:
            raise RuntimeError(
                f"{arm} 验证 MSE 与旧值超差：{got!r} vs {ref!r} "
                f"(>|{CONSISTENCY_TOL:.0e}|)——停止解释结论，排查输入/精度/"
                f"装配身份；不调宽容差、不重新选头")
    logger.info("[gate] OFF/ON 验证 MSE 与旧值一致（≤1e-5）✓")


def plot_daily_diff(output: Path) -> None:
    """逐日差图（mse_return：arm−G1），train/val 两面板；无策略净值新图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK HK", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    daily = pd.read_parquet(output / "daily_diagnostics.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, split in zip(axes, ("train", "val")):
        sub = daily[daily["split"] == split].pivot(
            index="date", columns="arm", values="mse_return")
        for arm, color in (("OFF", "#de7c33"), ("ON", "#3a923a")):
            ax.plot(pd.to_datetime(sub.index), sub[arm] - sub["G1"],
                    color=color, label=f"{arm}−G1", linewidth=0.9)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(f"{split}：逐日 mse_return 差（负=误差改善）")
        ax.set_ylabel("收益率²量纲")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output / "fig_daily_mse_diff.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PEER 残差 CPU 诊断")
    parser.add_argument("--sae-cache", required=True, type=Path)
    parser.add_argument("--peer-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-list", type=Path, default=INPUT_LIST_DEFAULT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    summary = run_diagnostics(args.sae_cache, args.peer_cache, args.output,
                              args.input_list, args.threads)
    plot_daily_diff(args.output)
    logger.info("[summary] " + json.dumps(
        {k: summary["delta_vs_g1"][k] for k in summary["delta_vs_g1"]},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
