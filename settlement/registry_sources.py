"""登记表读取源（真实 / 合成同协议）——结算执行器的登记级臂数据入口。

- ``SyntheticRegistrySource``：演习专用，读 ``data_synthetic/`` 下
  ``synthetic_signals_<date>.parquet`` + ``synthetic_registry_meta.csv``；
- ``RealRegistrySource``：2026-11 真实结算用，读
  ``finetune_suite/registry/signals_<date>.parquet``，gate 取自
  registry.duckdb 的 registry_meta（盖章值），late 取自 MANIFEST——
  **只在结算开封后实例化**，演习守卫会拦截其任何读取。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settlement.audit import REAL_REGISTRY_DIR


@dataclass
class DayRecord:
    wide: pd.DataFrame          # index=code；列 = s{seed}_{variant}×12 + M + tradeable
    gate: bool                  # MA200 门控盖章值（C2/R1 切换输入）
    late: bool                  # manifest late=true（C2 剔除敏感性输入）


class SyntheticRegistrySource:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._meta: dict[str, dict] = {}
        for _, r in pd.read_csv(self.root / "synthetic_registry_meta.csv").iterrows():
            self._meta[str(r["date"])] = {"gate": bool(r["gate"]), "late": bool(r["late"])}

    def day(self, date: pd.Timestamp) -> DayRecord:
        iso = pd.Timestamp(date).date().isoformat()
        p = self.root / f"synthetic_signals_{iso}.parquet"
        assert p.exists(), f"合成登记缺失：{p}"
        m = self._meta[iso]
        return DayRecord(wide=pd.read_parquet(p), gate=m["gate"], late=m["late"])

    def manifest_path(self) -> Path:
        return self.root / "synthetic_MANIFEST.csv"

    def dir(self) -> Path:
        return self.root


class RealRegistrySource:  # 2026-11 开封时使用；演习零实例化
    def __init__(self, registry_dir: Path = REAL_REGISTRY_DIR) -> None:
        self.root = Path(registry_dir)

    def _gate_from_duckdb(self, iso: str) -> bool:
        import duckdb

        con = duckdb.connect(str(self.root / "registry.duckdb"), read_only=True)
        try:
            row = con.execute(
                "SELECT value FROM registry_meta WHERE date=? AND key='gate'",
                [iso],
            ).fetchone()
        finally:
            con.close()
        assert row is not None, f"registry_meta 缺 {iso} 的 gate 盖章值"
        return row[0].strip().lower() == "true"

    def _late_from_manifest(self, iso: str) -> bool:
        manifest = pd.read_csv(self.root / "MANIFEST.csv", dtype=str)
        hit = manifest[manifest["date"] == iso]
        assert not hit.empty, f"MANIFEST 缺 {iso}"
        return bool(hit.iloc[0]["late"].strip().lower() == "true")

    def day(self, date: pd.Timestamp) -> DayRecord:
        iso = pd.Timestamp(date).date().isoformat()
        p = self.root / f"signals_{iso}.parquet"
        assert p.exists(), f"登记 parquet 缺失：{p}"
        return DayRecord(
            wide=pd.read_parquet(p),
            gate=self._gate_from_duckdb(iso),
            late=self._late_from_manifest(iso),
        )

    def manifest_path(self) -> Path:
        return self.root / "MANIFEST.csv"

    def dir(self) -> Path:
        return self.root
