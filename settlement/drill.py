"""合成数据端到端演习（关闭/滚动两分支都走通）——settlement 包的自检装置。

纪律（本模块的隔离契约）：
- 一切合成登记数据 ``synthetic_`` 前缀、只落 ``data_synthetic/`` 目录；
- ``SyntheticOnlyGuard`` 在演习期间拦截任何**真实路径**读取/写入
  （pandas 读写 + duckdb.connect 全部经由守卫）——演习全程零真实
  forward 信号/价格读取由构造保证，测试再作断言；
- 合成世界是确定性 DGP（固定种子）：G1 三种子共享潜因子，M/F0/F1/B3
  为相关/正交合成信号，收益按情景（关闭=负相关 / 滚动=正相关）生成——
  **数字无任何真实含义**，仅用于走通审计→对拍→装配→判据→分支→文档管线。
"""
from __future__ import annotations

import hashlib
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_ROOT_DEFAULT = REPO_ROOT / "settlement" / "data_synthetic"
SYNTHETIC_PREFIX = "synthetic_"
SYNTHETIC_DIR_NAME = "data_synthetic"

MANIFEST_HEADER = ["date", "file", "sha256", "n_stocks", "created_utc", "late"]


def assert_synthetic_path(path: Path) -> None:
    """演习物理隔离门禁：文件必须 synthetic_ 前缀且位于 data_synthetic/ 目录。"""
    path = Path(path)
    if not path.name.startswith(SYNTHETIC_PREFIX):
        raise ValueError(
            f"非 {SYNTHETIC_PREFIX} 前缀文件：{path}（演习物理隔离纪律）"
        )
    if path.parent.name != SYNTHETIC_DIR_NAME:
        raise ValueError(f"演习文件必须位于 {SYNTHETIC_DIR_NAME}/ 目录：{path}")


@contextmanager
def SyntheticOnlyGuard(root: Path):
    """演习守卫：拦截一切 data_synthetic/ 之外路径的 pandas 读写与 duckdb 连接。"""
    import duckdb

    root = Path(root).resolve()
    orig_rp, orig_rc = pd.read_parquet, pd.read_csv
    orig_tp, orig_dc = pd.DataFrame.to_parquet, duckdb.connect

    def _check(target) -> None:
        try:
            p = Path(str(target)).resolve()
        except (OSError, ValueError):
            return  # 非路径参数（缓冲区等）放行
        if p != root and root not in p.parents:
            raise RuntimeError(f"演习守卫：拒绝访问真实路径 {p}（只允许 {root}）")

    def _rp(path, *a, **k):
        _check(path)
        return orig_rp(path, *a, **k)

    def _rc(path, *a, **k):
        _check(path)
        return orig_rc(path, *a, **k)

    def _tp(self, path, *a, **k):
        _check(path)
        return orig_tp(self, path, *a, **k)

    def _dc(target, *a, **k):
        _check(target)
        return orig_dc(target, *a, **k)

    pd.read_parquet, pd.read_csv = _rp, _rc
    pd.DataFrame.to_parquet, duckdb.connect = _tp, _dc
    try:
        yield
    finally:
        pd.read_parquet, pd.read_csv = orig_rp, orig_rc
        pd.DataFrame.to_parquet, duckdb.connect = orig_tp, orig_dc


# ---------------------------------------------------------------------------
# 合成世界（确定性 DGP）
# ---------------------------------------------------------------------------
@dataclass
class SyntheticWorld:
    scenario: str
    dates: pd.DatetimeIndex
    codes: list[str]
    gates: dict                       # iso-date -> bool
    registered: dict                  # Timestamp -> 当日登记宽表（schema 同真实）
    recompute: dict = field(default_factory=dict)   # arm -> Timestamp -> DataFrame
    returns: pd.DataFrame | None = None
    index_ret: pd.Series | None = None

    def day_series(self, arm: str, date: pd.Timestamp) -> pd.Series:
        """对拍第二存储源（与登记 parquet 同源同值——确定性 DGP 的自证）。"""
        wide = self.registered[pd.Timestamp(date)]
        return wide["M"] if arm == "M" else wide[f"s{arm[-3:]}_mean"]


def build_synthetic_world(
    scenario: str, *, n_days: int = 70, n_codes: int = 40, seed: int = 11
) -> SyntheticWorld:
    assert scenario in ("关闭", "滚动"), f"未知情景 {scenario!r}（演习只支持关闭/滚动）"
    rng = np.random.default_rng(seed if scenario == "滚动" else seed + 1)
    dates = pd.bdate_range("2026-08-14", periods=n_days)
    codes = [f"SYN{i:04d}" for i in range(1, n_codes + 1)]

    z = rng.standard_normal((n_days, n_codes))            # G1 公共潜因子
    e = {s: rng.standard_normal((n_days, n_codes)) * 0.35 for s in (100, 101, 102)}
    em = rng.standard_normal((n_days, n_codes)) * 0.8
    e0, e1 = rng.standard_normal((n_days, n_codes)) * 0.6, rng.standard_normal((n_days, n_codes)) * 0.3
    e3 = rng.standard_normal((n_days, n_codes)) * 0.7

    seed_u = {s: z + e[s] for s in (100, 101, 102)}
    c1_u = sum(seed_u.values()) / 3.0
    m_u, f0_u, f1_u, b3_u = -0.3 * z + em, 0.5 * z + e0, 0.95 * z + e1, -0.5 * z + e3

    # 收益 DGP：滚动=与 G1 潜因子正相关（首读通过），关闭=负相关（Q2 触发）；
    # M/F0/F1/B3 各带独立小系数 → 各臂结论由数字代入决定，不预设方向。
    sign = 1.0 if scenario == "滚动" else -1.0
    noise = rng.standard_normal((n_days, n_codes)) * 0.006
    ret = (sign * 0.004 * c1_u + 0.0015 * m_u + 0.0010 * f1_u
           - 0.0005 * f0_u - 0.0010 * b3_u + noise)

    registered, recompute = {}, {"F0": {}, "F1": {}, "B3": {}}
    for i, d in enumerate(dates):
        wide = pd.DataFrame(index=pd.Index(codes, name="code"))
        for s in (100, 101, 102):
            base = 0.01 * seed_u[s][i]
            wide[f"s{s}_mean"] = base
            wide[f"s{s}_last"] = base + 0.0001
            wide[f"s{s}_max"] = base + 0.0010
            wide[f"s{s}_min"] = base - 0.0010
        wide["M"] = 0.01 * m_u[i]
        wide["tradeable"] = True
        registered[d] = wide
        for arm, u in (("F0", f0_u), ("F1", f1_u), ("B3", b3_u)):
            base = 0.01 * u[i]
            recompute[arm][d] = pd.DataFrame(
                {"last": base + 0.0001, "mean": base,
                 "max": base + 0.0010, "min": base - 0.0010},
                index=wide.index,
            )

    returns = pd.DataFrame(ret, index=dates, columns=codes)
    index_ret = 0.3 * returns.mean(axis=1) + 0.7 * returns[codes[:5]].mean(axis=1)
    gates = {d.date().isoformat(): bool(i % 2 == 0) for i, d in enumerate(dates)}
    return SyntheticWorld(
        scenario=scenario, dates=dates, codes=codes, gates=gates,
        registered=registered, recompute=recompute,
        returns=returns, index_ret=index_ret,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_synthetic_registry(
    world: SyntheticWorld, root: Path, *, late_days: tuple = ()
) -> Path:
    """把合成世界落成 synthetic_ 前缀登记（parquet + MANIFEST + meta）。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_rows, meta_rows = [], []
    late_set = {pd.Timestamp(d) for d in late_days}
    for d in world.dates:
        iso = d.date().isoformat()
        wide = world.registered[d]
        p = root / f"synthetic_signals_{iso}.parquet"
        wide.to_parquet(p)
        manifest_rows.append([iso, p.name, _sha256(p), len(wide),
                              "2026-08-19T00:00:00+00:00",
                              "true" if d in late_set else "false"])
        meta_rows.append({"date": iso, "gate": world.gates[iso],
                          "late": d in late_set})
    with open(root / "synthetic_MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
        import csv

        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        w.writerows(manifest_rows)
    pd.DataFrame(meta_rows).to_csv(root / "synthetic_registry_meta.csv", index=False)
    return root


# ---------------------------------------------------------------------------
# 演习入口
# ---------------------------------------------------------------------------
def run_drill(scenario: str, root: Path = SYNTHETIC_ROOT_DEFAULT) -> dict:
    """单分支演习：合成登记 → 守卫 → 结算执行器 → 结果/文档落 synthetic_ 目录。"""
    from settlement.drill import build_synthetic_world  # self-contained
    from settlement.engines import SyntheticEngine
    from settlement.executor import run_settlement
    from settlement.recompute import SyntheticRecomputeSource
    from settlement.registry_sources import SyntheticRegistrySource

    synthetic_root = Path(root) / SYNTHETIC_DIR_NAME
    if synthetic_root.exists():
        shutil.rmtree(synthetic_root)  # 只清理演习自目录
    world = build_synthetic_world(scenario, n_days=70)
    # 演习含 2 个迟补日：C2 剔除敏感性有实数据可代
    write_synthetic_registry(world, synthetic_root,
                             late_days=(world.dates[10], world.dates[33]))

    with SyntheticOnlyGuard(synthetic_root):
        result = run_settlement(
            registry=SyntheticRegistrySource(synthetic_root),
            dates=world.dates,
            engine=SyntheticEngine(world),
            recompute=SyntheticRecomputeSource(world),
            out_dir=synthetic_root,
            label="SYNTHETIC DRILL",
            file_prefix=SYNTHETIC_PREFIX,
            cross_check_secondary=world.day_series,
        )
    return {**result, "root": synthetic_root, "scenario": scenario}
