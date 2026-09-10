"""日级 PeerSet 数据装配（计划 §3：PeerSet 与 LossSet 严格分离）。

数据来源（全部只读复用，§4）：

- SAE 冻结缓存 chunk：train/val 提供 64 股 LossSet 的 ``h/s_g1/y_mean``
  （监督键与残差标签原值）；W3/W4 提供基线有效格 ``h/s_g1``。
- PEER1 新建 peer chunk：补齐当日 PeerSet 其余股票的 ``h``（仅 context，
  无标签、无 teacher、不新增交易候选）。
- ``μh/σh/σe`` 直接复用 ``f6c0726`` 保存的训练期 ``norm_stats.json``，
  不在全 peer 或测试段重新拟合。

装配结果 :class:`DayData` 的不变式：``codes`` = 完整 PeerSet（code 升序，
仅用于确定性存储）；``loss_mask`` = LossSet/输出格（train/val=SAE 64 股，
测试=基线格）；标签只进入 ``r``，绝不进入 ``x``/``codes``——扰动未来标签
不改变 PeerSet 与 forward。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from peer_residual import config as C
from sae_residual.cache_io import identity_of, sha256_file


@dataclass
class DayData:
    """单决策日装配结果。

    :ivar date: 决策日 ``YYYY-MM-DD``。
    :ivar codes: 完整 PeerSet（code 升序）。
    :ivar x: ``[N, D]`` 已按冻结 μh/σh 标准化的隐状态（float32）。
    :ivar loss_mask: ``[N]`` bool——train/val = LossSet；测试 = 基线输出格。
    :ivar r: ``[N]`` float32 残差目标 ``e/σe``（仅 loss_mask 处有效，其余 0）。
    :ivar s_g1: ``[N]`` float64——train/val = teacher 原值；测试 = 基线原值。
    :ivar n_extra: 仅作 context 的额外 peer 数（PeerSet − SAE 覆盖）。
    """

    date: str
    codes: list[str]
    x: np.ndarray
    loss_mask: np.ndarray
    r: np.ndarray
    s_g1: np.ndarray
    n_extra: int = 0


def load_frozen_stats() -> dict:
    """读 SAE ``norm_stats.json``（σe 约 0.0490884），不重拟合。

    :returns: ``{"mu": [D], "sigma": [D], "sigma_e": float, "d_in": int}``。
    """
    p = C.SAE_NORM_STATS
    assert p.is_file(), f"SAE norm_stats 缺失：{p}"
    raw = json.loads(p.read_text(encoding="utf-8"))
    stats = {"mu": np.asarray(raw["mu"], dtype=np.float32),
             "sigma": np.asarray(raw["sigma"], dtype=np.float32),
             "sigma_e": float(raw["sigma_e"]),
             "d_in": int(raw["d_in"]),
             "norm_stats_sha256": sha256_file(p)}
    assert stats["d_in"] == 832, f"d_in 应为 832，实际 {stats['d_in']}"
    assert 0 < stats["sigma_e"] < 1.0
    return stats


def normalize_hidden(h: np.ndarray, stats: dict) -> np.ndarray:
    """``x = (h − μh)/σh``（μh/σh 只来自冻结训练文件）。"""
    arr = np.asarray(h, dtype=np.float32)
    return ((arr - stats["mu"]) / stats["sigma"]).astype(np.float32)


def sae_chunk_path(split: str, date: str, sae_dir: Path | None = None):
    """SAE 缓存 chunk 路径（date 为 ``YYYY-MM-DD``；可显式传缓存目录）。"""
    base = (sae_dir or C.SAE_CACHE_DIR)
    return base / split / f"{split}_{date.replace('-', '')}.npz"


def peer_chunk_path(split: str, date: str, subdir: str | None = None,
                    peer_dir: Path | None = None):
    """PEER1 peer chunk 路径（subdir 用于 smoke 隔离；可显式传缓存目录）。"""
    base = (peer_dir or C.PEER_CACHE_DIR)
    d = base / (subdir or split)
    return d / f"peer_{split}_{date.replace('-', '')}.npz"


def g1_weight_shas() -> dict:
    """当前 G1 权重 SHA（用于 chunk 身份校验）。"""
    return {"tokenizer_sha": sha256_file(C.G1_TOKENIZER / "model.safetensors"),
            "predictor_sha": sha256_file(C.G1_PREDICTOR / "model.safetensors")}


def _load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "identity_json"}


def load_day(split: str, date: str, stats: dict, subdir: str | None = None,
             weight_shas: dict | None = None, sae_dir: Path | None = None,
             peer_dir: Path | None = None) -> DayData:
    """装配单日：SAE chunk ∪ peer chunk → 完整 PeerSet DayData。

    身份门禁：peer chunk 的协议/段名/G1 权重 SHA 必须匹配，且其记录的
    源 SAE chunk SHA 与实存文件一致；LossSet/基线格必须 ⊆ PeerSet。
    """
    sp = sae_chunk_path(split, date, sae_dir)
    pp = peer_chunk_path(split, date, subdir, peer_dir)
    assert sp.is_file(), f"SAE chunk 缺失：{sp}"
    assert pp.is_file(), f"peer chunk 缺失：{pp}"
    ident = identity_of(pp)
    expect = {"protocol": C.PROTOCOL_VERSION, "split": split,
              **(weight_shas or g1_weight_shas())}
    for k, v in expect.items():
        if ident.get(k) != v:
            raise RuntimeError(f"peer chunk 身份不匹配 {k}={ident.get(k)!r}!={v!r}")
    if ident.get("sae_chunk_sha256") != sha256_file(sp):
        raise RuntimeError(f"{date}: peer chunk 源 SAE chunk SHA 不一致")
    sae = _load_npz(sp)
    peer = _load_npz(pp)
    return assemble_day(sae, peer, stats, date, split)


def assemble_day(sae: dict, peer: dict, stats: dict, date: str,
                 split: str) -> DayData:
    """纯装配函数（机制测试直接调用）：SAE 数组 + peer 数组 → DayData。

    :param sae: SAE chunk 数组（dates/instruments/h/s_g1[/y_mean]...）。
    :param peer: peer chunk 数组（instruments/h 为额外 peer；peer_codes 全集）。
    """
    sae_codes = [str(c) for c in sae["instruments"]]
    extra_codes = [str(c) for c in peer["instruments"]]
    full_codes = [str(c) for c in peer["peer_codes"]]
    assert full_codes == sorted(full_codes), f"{date}: peer_codes 未排序"
    sae_set, full_set = set(sae_codes), set(full_codes)
    missing = sae_set - full_set
    assert not missing, (
        f"{date}: LossSet/基线格 {sorted(missing)[:5]} 不在 PeerSet——停止排查，"
        f"不得静默替换或缩池")
    overlap = sae_set & set(extra_codes)
    assert not overlap, f"{date}: peer chunk 与 SAE chunk 重叠 {sorted(overlap)[:5]}"
    h_map = {c: sae["h"][i] for i, c in enumerate(sae_codes)}
    h_map.update({c: peer["h"][j] for j, c in enumerate(extra_codes)})
    h_all = np.stack([h_map[c] for c in full_codes]).astype(np.float32)
    assert h_all.shape[1] == stats["d_in"], \
        f"{date}: D={h_all.shape[1]} != 冻结统计 {stats['d_in']}"
    assert np.isfinite(h_all).all(), f"{date}: h 含非有限值"
    x = normalize_hidden(h_all, stats)
    loss_mask = np.array([c in sae_set for c in full_codes], dtype=bool)
    s_g1_map = {c: float(sae["s_g1"][i]) for i, c in enumerate(sae_codes)}
    s_g1 = np.array([s_g1_map.get(c, 0.0) for c in full_codes], dtype=np.float64)
    assert np.isfinite(s_g1[loss_mask]).all(), f"{date}: s_g1 含非有限值"
    r = np.zeros(len(full_codes), dtype=np.float32)
    if "y_mean" in sae:                       # train/val：LossSet 监督目标
        # 无前视断言：标签窗必须在历史窗之后（未来观测不进当前窗口）
        if "label_start" in sae and "x_end" in sae:
            assert str(sae["label_start"][0]) > str(sae["x_end"][0]), \
                f"{date}: 标签窗与历史窗重叠"
        y_map = {c: float(sae["y_mean"][i]) for i, c in enumerate(sae_codes)}
        idx = np.where(loss_mask)[0]
        e = np.array([y_map[full_codes[i]] - s_g1[i] for i in idx],
                     dtype=np.float64)
        r[idx] = (e / stats["sigma_e"]).astype(np.float32)
    return DayData(date=date, codes=full_codes, x=x, loss_mask=loss_mask,
                   r=r, s_g1=s_g1, n_extra=len(extra_codes))


def split_dates(split: str, sae_dir: Path | None = None) -> list[str]:
    """SAE chunk 的全部决策日（升序，作为本轮唯一日期权威）。"""
    d = (sae_dir or C.SAE_CACHE_DIR) / split
    out = []
    for p in sorted(d.glob(f"{split}_*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        ymd = p.stem.split("_")[1]
        out.append(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
    return out


def load_split_days(split: str, stats: dict, weight_shas: dict | None = None,
                    subdir: str | None = None, sae_dir: Path | None = None,
                    peer_dir: Path | None = None) -> list[DayData]:
    """先核验冻结清单的完整键集和字节，再装配全部决策日。"""
    from peer_residual.identity import verify_cache_dir

    for base, manifest in (
        (sae_dir or C.SAE_CACHE_DIR, C.SAE_CACHE_MANIFEST),
        (peer_dir or C.PEER_CACHE_DIR, C.ART_DIR / "peer_cache_manifest.json"),
    ):
        verify_cache_dir(base / split,
                         json.loads(manifest.read_text(encoding="utf-8")), split)
    days = [load_day(split, dstr, stats, subdir, weight_shas, sae_dir,
                     peer_dir)
            for dstr in split_dates(split, sae_dir)]
    assert days, f"{split} 无决策日"
    return days


def collate_batch(days: list[DayData]):
    """≤ ``batch_days`` 个日期 → padding 批张量（日期维 = batch）。

    :returns: ``(x [B,Nmax,D], valid [B,Nmax], loss [B,Nmax], r [B,Nmax])``，
        全 CPU float32/bool（设备迁移由调用方负责）。
    :raises ValueError: 存在全 padding 日期（非法输入，计划 §5）。
    """
    nmax = max(len(d.codes) for d in days)
    d_in = days[0].x.shape[1]
    b = len(days)
    x = np.zeros((b, nmax, d_in), dtype=np.float32)
    valid = np.zeros((b, nmax), dtype=bool)
    loss = np.zeros((b, nmax), dtype=bool)
    r = np.zeros((b, nmax), dtype=np.float32)
    for i, d in enumerate(days):
        if len(d.codes) == 0:
            raise ValueError(f"{d.date}: 全 padding 日期拒绝输入")
        n = len(d.codes)
        x[i, :n] = d.x
        valid[i, :n] = True
        loss[i, :n] = d.loss_mask
        r[i, :n] = d.r
    return (torch.from_numpy(x), torch.from_numpy(valid),
            torch.from_numpy(loss), torch.from_numpy(r))


def torch_from_numpy(a: np.ndarray):
    return torch.from_numpy(a)


def day_keys_digest(days: list[DayData]) -> str:
    """OFF/ON 一致性摘要：日期/PeerSet/LossSet/目标的指纹。"""
    h = hashlib.sha256()
    for d in days:
        h.update(d.date.encode())
        h.update(",".join(d.codes).encode())
        h.update(d.loss_mask.astype(np.uint8).tobytes())
        h.update(d.r.astype(np.float32).tobytes())
    return h.hexdigest()


def r_log(message: str) -> None:  # pragma: no cover - 日志透传
    logger.info(message)


__all__ = ["DayData", "load_frozen_stats", "normalize_hidden",
           "load_day", "assemble_day", "split_dates", "load_split_days",
           "collate_batch", "day_keys_digest", "g1_weight_shas",
           "sae_chunk_path", "peer_chunk_path"]
