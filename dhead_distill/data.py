"""样本清单、历史窗口/标签与按日 batch（方案 §3 数据与时间契约）。

核心不变式（全部在 tests/test_dhead_data.py 离线验证）：

- 每条样本键 ``(decision_date, code)``；历史输入只到 t，未来价格只用于标签；
- ``y[t,h] = close[t+h] / close[t] - 1``（h=1..10，原始小数收益，不逐日 z-score）；
- z-score 统计量只来自该样本过去 90 日（总体标准差、eps=1e-5、clip=5）；
- 历史缺口/非正价格/非有限值 → 整窗剔除并计数，不跨缺口拼接；
- 训练/验证/诊断的 t+10 标签按**交易日历**推进，不硬编码日历日减 10；
- 封存线 2026-07-25：任何 fetch 边界不得越过（含回放窗 t+10 溢出样本——
  它们保留为可发信号样本但 label_ok=False，绝不取其未来价格）；
- 清单用 SHA256 稳定哈希排序，seed=20260905；内容 hash 不含 mtime；
- 落盘用 JSON + NPZ（安全反序列化，不用 pickle）；产物根来自环境解析
  （config.resolve_env），目录名走白名单校验，写盘用对象方法原子落盘。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from dhead_distill.config import DHeadConfig, protocol_hash

#: 封存线（§3.2）：此日起不加载价格、标签或绩效
SEAL_DATE = pd.Timestamp("2026-07-25")

#: 分块拉数大小：哈希序游走时按块取数，取满 per_day 即停（避免全池拉取）
_FETCH_CHUNK = 256

#: 产物子目录名白名单：字母数字开头，仅字母/数字/下划线/连字符；
#: 不含路径分隔符与父目录引用，从构造上杜绝穿越
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def safe_artifact_dir(name: str) -> Path:
    """产物目录 = 环境解析的产物根 + 白名单子目录名。

    产物根来自 :func:`dhead_distill.config.resolve_env`（``DHEAD_ARTIFACT_ROOT``
    或默认同级 ``Kronos-dhead-artifacts``）；子目录名经 :data:`_SAFE_NAME`
    白名单校验（与 l1_context 同款「常量根 + 校验名」模式）。
    """
    if not _SAFE_NAME.match(name):
        raise ValueError(f"非法产物子目录名：{name!r}（只允许字母数字与 _ -）")
    from dhead_distill.config import resolve_env

    root = resolve_env().artifact_root
    d = (root / name).resolve()
    root_r = root.resolve()
    if d != root_r and root_r not in d.parents:
        raise ValueError(f"产物目录 {d} 越出产物根 {root_r}")
    return d


def stable_hash_key(list_seed: int, date: pd.Timestamp, code: str) -> str:
    """(date, code) 稳定哈希键（SHA256，不用 Python 内置 hash，§3.3）。"""
    return _sha256_hex(f"{list_seed}|{date.strftime('%Y-%m-%d')}|{code}")


def _calc_stamps(dates: pd.DatetimeIndex) -> np.ndarray:
    """5 列时间特征 [T,5]：minute/hour/weekday/day/month（官方 calc_time_stamps 同口径）。"""
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["day"] = df.index.day
    df["month"] = df.index.month
    return df.values.astype(np.float32)


def window_zscore_clip(x: np.ndarray, *, eps: float, clip: float) -> np.ndarray:
    """窗口 z-score + clip（KronosPredictor 同口径：总体标准差、逐列）。"""
    mean = x.mean(axis=0)
    std = x.std(axis=0)  # 总体标准差（ddof=0）
    z = (x - mean) / (std + eps)
    return np.clip(z, -clip, clip).astype(np.float32)


def equal_interval(cands: list[pd.Timestamp], n: int) -> list[pd.Timestamp]:
    """排序候选日上 ``np.linspace(0, n-1, min(n,len), dtype=int)`` 等间隔取日。"""
    if n <= 0 or not cands:
        return []
    idx = np.linspace(0, len(cands) - 1, min(n, len(cands)), dtype=int)
    return [cands[i] for i in idx]


@dataclass(frozen=True)
class Sample:
    """一条冻结清单样本：键 (date, code) + 10 期限真实标签。

    :param y_real: ``[10]`` float32，``y[t,h] = close[t+h]/close[t]-1``；
        label_ok=False 时为 NaN（绝不填充）。
    :param label_ok: 标签路径是否完整（回放窗 t+10 溢出样本可为 False）。
    """

    date: pd.Timestamp
    code: str
    y_real: np.ndarray
    label_ok: bool


@dataclass
class DayManifest:
    """一次 prepare 的冻结产物：清单 + 窗口数组 + 统计 + 内容 hash。

    ``x_raw`` 存**未归一化** OHLCVA（教师 predict 内部自做 z-score）；
    ``x_norm`` 由 :class:`DayDataset` 按需确定性计算。
    """

    split: str
    pool: str
    profile: str
    protocol: str                       # protocol_hash(cfg)
    list_seed: int
    samples: list[Sample] = field(default_factory=list)
    # (date_iso, code) → [90,6] float32 原始窗口 / close_t
    x_raw: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    close_t: dict[tuple[str, str], float] = field(default_factory=dict)
    # date_iso → 窗口/未来日历（stamps 与日历日期一一对应）
    x_stamp: dict[str, np.ndarray] = field(default_factory=dict)   # [90,5]
    y_stamp: dict[str, np.ndarray] = field(default_factory=dict)   # [10,5]
    x_cal: dict[str, np.ndarray] = field(default_factory=dict)     # [90] datetime64[D]
    y_cal: dict[str, np.ndarray] = field(default_factory=dict)     # [10] datetime64[D]
    stats: dict = field(default_factory=dict)
    content_hash: str = ""

    # ------------------------------------------------------------------
    # 安全落盘 / 装载（JSON + NPZ；不用 pickle；目录名白名单 + 原子替换）
    # ------------------------------------------------------------------

    def _stacked_arrays(self) -> dict[str, np.ndarray]:
        """把 dict 形态数组堆叠为可 NPZ 化的有序数组集。"""
        days = sorted(self.x_stamp)
        arrs: dict[str, np.ndarray] = {
            "x_stamp": np.stack([self.x_stamp[k] for k in days]) if days
            else np.zeros((0, 0, 5), np.float32),
            "y_stamp": np.stack([self.y_stamp[k] for k in days]) if days
            else np.zeros((0, 0, 5), np.float32),
            "x_cal": np.stack([self.x_cal[k] for k in days]) if days
            else np.zeros((0, 0), "datetime64[D]"),
            "y_cal": np.stack([self.y_cal[k] for k in days]) if days
            else np.zeros((0, 0), "datetime64[D]"),
        }
        if self.samples:
            arrs["x_raw"] = np.stack(
                [self.x_raw[(s.date.strftime("%Y-%m-%d"), s.code)]
                 for s in self.samples]
            )
            arrs["close_t"] = np.array(
                [self.close_t[(s.date.strftime("%Y-%m-%d"), s.code)]
                 for s in self.samples], dtype=np.float64,
            )
            arrs["y_real"] = np.stack([s.y_real for s in self.samples])
        else:
            arrs["x_raw"] = np.zeros((0, 0, 0), np.float32)
            arrs["close_t"] = np.zeros((0,), np.float64)
            arrs["y_real"] = np.zeros((0, 0), np.float32)
        return arrs

    def save(self, name: str) -> Path:
        """原子落盘：manifest.json（元数据+统计）+ arrays.npz（数组）。

        :param name: 白名单子目录名（CLI 侧拼 ``{profile}_{split}_{hash12}``）。
        :returns: 产物目录绝对路径。
        """
        d = safe_artifact_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        days = sorted(self.x_stamp)
        day_pos = {day: i for i, day in enumerate(days)}
        samples_meta = [
            {"date": s.date.strftime("%Y-%m-%d"), "code": s.code,
             "label_ok": s.label_ok, "day": day_pos[s.date.strftime("%Y-%m-%d")]}
            for s in self.samples
        ]
        meta = {
            "split": self.split, "pool": self.pool, "profile": self.profile,
            "protocol": self.protocol, "list_seed": self.list_seed,
            "days": days, "samples": samples_meta,
            "stats": _jsonable(self.stats), "content_hash": self.content_hash,
        }
        npz_tmp = d / "arrays.tmp.npz"   # 以 .npz 结尾：防 np.savez 再追加后缀
        np.savez_compressed(npz_tmp, **self._stacked_arrays())
        json_tmp = d / "manifest.json.tmp"
        json_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
        npz_tmp.replace(d / "arrays.npz")     # 同目录临时文件 + rename：原子落盘
        json_tmp.replace(d / "manifest.json")
        return d

    @staticmethod
    def load(name: str, *, verify: bool = False) -> "DayManifest":
        """从 save 产物目录装载（JSON 反序列化 + NPZ 数组，安全无 pickle）。

        :param verify: True 时重算内容 hash 与存储值比对（防静默损坏/篡改，
            v1 修复 #4 的一部分；train/evaluate 入口应开启）。
        """
        d = safe_artifact_dir(name)
        meta = json.loads((d / "manifest.json").read_text("utf-8"))
        arrs = dict(np.load(d / "arrays.npz", allow_pickle=False))
        m = DayManifest(
            split=meta["split"], pool=meta["pool"], profile=meta["profile"],
            protocol=meta["protocol"], list_seed=meta["list_seed"],
            stats=meta["stats"], content_hash=meta["content_hash"],
        )
        days: list[str] = meta["days"]
        for i, day in enumerate(days):
            m.x_stamp[day] = arrs["x_stamp"][i]
            m.y_stamp[day] = arrs["y_stamp"][i]
            m.x_cal[day] = arrs["x_cal"][i]
            m.y_cal[day] = arrs["y_cal"][i]
        for i, sm in enumerate(meta["samples"]):
            key = (sm["date"], sm["code"])
            m.x_raw[key] = arrs["x_raw"][i]
            m.close_t[key] = float(arrs["close_t"][i])
            m.samples.append(
                Sample(date=pd.Timestamp(sm["date"]), code=sm["code"],
                       y_real=arrs["y_real"][i], label_ok=sm["label_ok"])
            )
        if verify:
            recomputed = content_hash(m)
            if recomputed != m.content_hash:
                raise RuntimeError(
                    f"清单内容校验失败：存储 {m.content_hash[:12]} ≠ 重算 "
                    f"{recomputed[:12]}（{name}）——拒绝使用可能损坏/被改的清单"
                )
        return m


def _jsonable(o: Any) -> Any:
    """stats dict → JSON 可序列化形式（numpy 标量/时间戳转基本类型）。"""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def _content_bytes(m: DayManifest, *, include_labels: bool) -> bytes:
    """内容序列化（与 mtime 无关）：日期→stamps/日历；样本→窗口/close/标签。"""
    parts: list[bytes] = [
        f"dhead-manifest-v1|{m.split}|{m.pool}|{m.profile}|{m.protocol}"
        f"|{m.list_seed}".encode("utf-8"),
    ]
    for s in m.samples:  # samples 顺序本身就是清单顺序
        d = s.date.strftime("%Y-%m-%d")
        parts.append(f"S|{d}|{s.code}".encode("utf-8"))
        parts.append(m.x_raw[(d, s.code)].tobytes())
        parts.append(np.float64(m.close_t[(d, s.code)]).tobytes())
        if include_labels:
            parts.append(s.y_real.tobytes())
            parts.append(b"L1" if s.label_ok else b"L0")
    for d in sorted(m.x_stamp):
        parts.append(f"D|{d}".encode("utf-8"))
        parts.append(m.x_stamp[d].tobytes())
        parts.append(m.y_stamp[d].tobytes())
        parts.append(m.x_cal[d].tobytes())
        parts.append(m.y_cal[d].tobytes())
    return b"|".join(parts)


def content_hash(m: DayManifest) -> str:
    """完整内容 hash（含标签）。"""
    return hashlib.sha256(_content_bytes(m, include_labels=True)).hexdigest()


def input_digest(m: DayManifest) -> str:
    """仅输入侧内容 hash（不含标签）——未来篡改不得改变该值。"""
    return hashlib.sha256(_content_bytes(m, include_labels=False)).hexdigest()


def _fetch_range(provider, instruments, *, lo, hi, fields) -> pd.DataFrame | None:
    """按 windows.py 约定临时改写 provider 区间拉数（fetch 后恢复原状态）。"""
    orig = (provider._start_date, provider._end_date, provider.instruments_)
    try:
        provider._start_date = lo.strftime("%Y-%m-%d")
        provider._end_date = hi.strftime("%Y-%m-%d")
        provider.instruments_ = instruments
        return provider.fetch([f"${c}" for c in fields], freq="day")
    finally:
        provider._start_date, provider._end_date, provider.instruments_ = orig


def _probe_data_start(provider, pool: str, approx_start: pd.Timestamp) -> pd.Timestamp | None:
    """探测池数据的实际起点（一次有界拉数；避免对无历史日期做整池游走）。

    只拉 ``close`` 一列、区间 [approx_start-400 天, approx_start]，
    不触封存线（approx_start ≤ 各 split 起点 ≤ 2026-01-01）。
    """
    lo = approx_start - pd.Timedelta(days=400)
    raw = _fetch_range(provider, pool, lo=lo, hi=approx_start, fields=["close"])
    if raw is None or len(raw) == 0:
        return None
    return pd.Timestamp(raw.index.get_level_values("datetime").min())


def build_manifest(provider, cfg: DHeadConfig, split: str) -> DayManifest:
    """构造冻结清单（§3.2/§3.3）：候选日 → 等间隔取日 → 当日 PIT + 稳定哈希取股。

    :param provider: duck-typing ``kronos_qlib.QlibProvider``（离线测试用
        FakeProvider；真实运行走 DDB）。
    :raises ValueError: 未知 split（由 ``cfg.split_spec`` 抛出）。
    """
    spec = cfg.split_spec(split)
    full_cal: pd.DatetimeIndex = provider.trading_days()
    start_t, end_t = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"])
    label_end = pd.Timestamp(spec["label_end"])

    data_start = _probe_data_start(provider, spec["pool"], start_t)
    if data_start is None:
        raise RuntimeError(
            f"split={split}：池 {spec['pool']} 在 {start_t.date()} 附近无数据"
        )

    # 候选决策日：窗内 + 90 交易日历史不早于数据地板 + 标签边界（日历推进）
    cands: list[pd.Timestamp] = []
    for i, d in enumerate(full_cal):
        if d < start_t or d > end_t or i < cfg.lookback - 1:
            continue
        if full_cal[i - cfg.lookback + 1] < data_start:
            continue
        if spec["label_required"] and i + cfg.predict_len < len(full_cal) \
                and full_cal[i + cfg.predict_len] > label_end:
            continue
        cands.append(d)
    selected = equal_interval(cands, spec["n_dates"])

    m = DayManifest(
        split=split, pool=spec["pool"], profile=cfg.profile,
        protocol=protocol_hash(cfg), list_seed=cfg.list_seed,
    )
    skipped = {"short_history": 0, "gap": 0, "nonpositive": 0, "bad_label": 0}
    dropped_days: list[dict] = []
    per_day_cov: list[dict] = []

    for t in selected:
        members = provider.list_pool_at(spec["pool"], t.strftime("%Y-%m-%d"))
        ordered = sorted(
            members, key=lambda c: (stable_hash_key(cfg.list_seed, t, c), c)
        )
        t_pos = full_cal.get_loc(t)
        x_cal = full_cal[t_pos - cfg.lookback + 1: t_pos + 1]
        y_cal = full_cal[t_pos + 1: t_pos + cfg.predict_len + 1]
        # 未来日历允许使用；未来**价格** fetch 上界 = min(t+10 日, 标签边界, 封存线-1)
        fetch_hi = min(y_cal[-1], label_end, SEAL_DATE - pd.Timedelta(days=1))
        assert fetch_hi < SEAL_DATE, "封存线保护：fetch 不得触达 2026-07-25 及之后"

        d_iso = t.strftime("%Y-%m-%d")
        m.x_stamp[d_iso] = _calc_stamps(x_cal)
        m.y_stamp[d_iso] = _calc_stamps(y_cal)
        m.x_cal[d_iso] = x_cal.values.astype("datetime64[D]")
        m.y_cal[d_iso] = y_cal.values.astype("datetime64[D]")

        kept: list[Sample] = []
        it = iter(ordered)
        exhausted = False
        while len(kept) < cfg.budget.per_day and not exhausted:
            chunk = [c for _, c in zip(range(_FETCH_CHUNK), it)]
            if not chunk:
                exhausted = True
                break
            raw = _fetch_range(provider, chunk, lo=x_cal[0], hi=fetch_hi,
                               fields=list(cfg.feature_cols))
            if raw is None or len(raw) == 0:
                continue
            avail = raw.index.get_level_values("instrument").unique()
            for code in chunk:
                if len(kept) >= cfg.budget.per_day:
                    break
                if code not in avail:
                    skipped["short_history"] += 1
                    continue
                sub = raw.xs(code, level="instrument").sort_index()
                win = sub.loc[:t]
                if len(win) < cfg.lookback:
                    skipped["short_history"] += 1
                    continue
                win = win.iloc[-cfg.lookback:]
                if not np.array_equal(
                    win.index.values.astype("datetime64[D]"),
                    x_cal.values.astype("datetime64[D]"),
                ):
                    skipped["gap"] += 1  # 行日期与日历窗不严格一致 = 缺口/停牌
                    continue
                x_arr = win[list(cfg.feature_cols)].values.astype(np.float32)
                if (not np.isfinite(x_arr).all()) or (x_arr[:, :4] <= 0).any():
                    skipped["nonpositive"] += 1
                    continue
                close_t = float(win.iloc[-1]["close"])

                # 标签路径：h=1..10 的 close 行须存在且有限为正（仅 fetch 界内可见）
                label_dates = y_cal[y_cal <= fetch_hi]
                y_real = np.full(cfg.predict_len, np.nan, dtype=np.float32)
                label_ok = False
                if len(label_dates) == cfg.predict_len:
                    lab = sub.reindex(label_dates)  # 缺行（停牌）→ NaN，不抛错
                    closes = lab["close"].values.astype(np.float64)
                    if len(closes) == cfg.predict_len and np.isfinite(closes).all() \
                            and (closes > 0).all():
                        y_real = (closes / close_t - 1.0).astype(np.float32)
                        label_ok = True
                if spec["label_required"] and not label_ok:
                    skipped["bad_label"] += 1
                    continue

                key = (d_iso, code)
                m.x_raw[key] = x_arr
                m.close_t[key] = close_t
                kept.append(Sample(date=t, code=code, y_real=y_real, label_ok=label_ok))

        if len(kept) < cfg.budget.min_per_day:
            dropped_days.append({"date": d_iso, "n_kept": len(kept),
                                 "reason": f"< min_per_day={cfg.budget.min_per_day}"})
            for s in kept:  # 整日剔除：回滚已存窗口，防幽灵键
                k = (s.date.strftime("%Y-%m-%d"), s.code)
                m.x_raw.pop(k, None)
                m.close_t.pop(k, None)
            m.x_stamp.pop(d_iso, None); m.y_stamp.pop(d_iso, None)
            m.x_cal.pop(d_iso, None); m.y_cal.pop(d_iso, None)
            continue
        m.samples.extend(kept)
        per_day_cov.append({"date": d_iso, "n_pool": len(ordered), "n_kept": len(kept)})

    m.stats = {
        "split": split, "pool": spec["pool"], "profile": cfg.profile,
        "protocol": m.protocol, "list_seed": cfg.list_seed,
        "n_days": len(per_day_cov), "n_samples": len(m.samples),
        "skipped": skipped, "dropped_days": dropped_days,
        "per_day": per_day_cov,
        "n_candidates": len(cands), "n_selected_dates": len(selected),
        "label_required": spec["label_required"],
        "n_unlabeled": sum(0 if s.label_ok else 1 for s in m.samples),
        "data_start": str(data_start.date()),
    }
    m.content_hash = content_hash(m)
    return m


def day_batches(m: DayManifest) -> Iterator[list[Sample]]:
    """按日聚合的批迭代器（一个 batch 只含同一决策日的样本，§3.3）。"""
    cur_date: pd.Timestamp | None = None
    batch: list[Sample] = []
    for s in m.samples:
        if cur_date is not None and s.date != cur_date:
            yield batch
            batch = []
        cur_date = s.date
        batch.append(s)
    if batch:
        yield batch


class DayDataset(Dataset):
    """冻结清单上的确定性 Dataset（DataLoader 只按索引读，不另选样本）。

    ``__getitem__`` 返回单样本 dict；``x_norm`` 由 ``x_raw`` 现场确定性计算
    （窗口 z-score + clip5，与 KronosPredictor 同口径）。
    """

    def __init__(self, manifest: DayManifest, *, zscore_eps: float = 1e-5,
                 clip: float = 5.0):
        self.m = manifest
        self.m_eps = zscore_eps
        self.m_clip = clip

    def __len__(self) -> int:
        return len(self.m.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.m.samples[i]
        d = s.date.strftime("%Y-%m-%d")
        key = (d, s.code)
        x_raw = self.m.x_raw[key]
        return {
            "date": d,
            "code": s.code,
            "x_raw": x_raw,
            "x_norm": window_zscore_clip(x_raw.astype(np.float64),
                                         eps=self.m_eps, clip=self.m_clip),
            "x_stamp": self.m.x_stamp[d],
            "y_stamp": self.m.y_stamp[d],
            # 日历日期以 int64（epoch 天）返回：可被默认 collate 批处理，
            # 教师侧用 pd.to_datetime(arr, unit="D") 还原 DatetimeIndex。
            "x_cal": self.m.x_cal[d].astype("int64"),
            "y_cal": self.m.y_cal[d].astype("int64"),
            "close_t": np.float32(self.m.close_t[key]),
            "y_real": s.y_real,
            "label_ok": bool(s.label_ok),
        }


__all__ = [
    "SEAL_DATE", "Sample", "DayManifest", "DayDataset",
    "build_manifest", "day_batches", "content_hash", "input_digest",
    "stable_hash_key", "window_zscore_clip", "equal_interval",
    "safe_artifact_dir", "_calc_stamps",
]
