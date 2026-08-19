"""完整性审计器（forward结算计划_20260818.md §3.3）——先于任何绩效计算。

MANIFEST 哈希链逐日校验 + 缺日/late 日清点。**只碰哈希与日期**：
- manifest 行（date/file/sha256/n_stocks/late）——csv 逐行读；
- parquet 文件字节——仅为重算 SHA256；
- parquet 行数——仅经 pyarrow 元数据（``metadata.num_rows``），不解载任何信号值。

对真实 MANIFEST 可直接运行（``audit_real_manifest``）；对合成演习数据
（``synthetic_MANIFEST.csv``）同链路复用。审计只读不写。
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REGISTRY_DIR = REPO_ROOT / "finetune_suite" / "registry"

MANIFEST_HEADER = ["date", "file", "sha256", "n_stocks", "created_utc", "late"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_num_rows(path: Path) -> int:
    """仅读 parquet 元数据行数（不解载任何值列——哈希/日期纪律）。"""
    from pyarrow.parquet import ParquetFile

    return ParquetFile(path).metadata.num_rows


def audit_registry(
    manifest_path: Path, registry_dir: Path, *, trading_calendar: pd.DatetimeIndex | None = None
) -> dict:
    """审计一份登记 MANIFEST：哈希链、缺日、late 日、重复日、行数一致性。

    :param manifest_path: MANIFEST csv（真实或 synthetic_ 前缀合成）。
    :param registry_dir: parquet 所在目录（manifest 的 file 字段按裸文件名解析，
        含路径分隔符即视为逃逸 → 记 findings）。
    :param trading_calendar: 交易日历（真实结算传入；None 则不做缺日比对，
        报告标注 ``calendar: "未提供（缺日未校验）"``）。
    :returns: 报告 dict（passed = 无哈希断链/缺文件/重复日/行数不符/缺日）。
    """
    manifest_path, registry_dir = Path(manifest_path), Path(registry_dir)
    assert manifest_path.exists(), f"MANIFEST 缺失：{manifest_path}"

    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == MANIFEST_HEADER, (
            f"MANIFEST 表头漂移：{reader.fieldnames}（期望 {MANIFEST_HEADER}）"
        )
        rows = list(reader)

    seen_dates: dict[str, int] = {}
    hash_mismatch: list[str] = []
    missing_file: list[str] = []
    nstocks_mismatch: list[str] = []
    escape_file: list[str] = []
    late_dates: list[str] = []

    for r in rows:
        date = r["date"]
        seen_dates[date] = seen_dates.get(date, 0) + 1
        if r["late"] == "true":
            late_dates.append(date)
        fname = r["file"]
        if "/" in fname or "\\" in fname:
            escape_file.append(date)
            continue
        p = registry_dir / fname
        if not p.exists():
            missing_file.append(date)
            continue
        if _sha256(p) != r["sha256"]:
            hash_mismatch.append(date)
            continue
        if _parquet_num_rows(p) != int(r["n_stocks"]):
            nstocks_mismatch.append(date)

    duplicates = sorted(d for d, n in seen_dates.items() if n > 1)
    registered = sorted(seen_dates)
    missing_days: list[str] = []
    if trading_calendar is not None and registered:
        lo, hi = pd.Timestamp(registered[0]), pd.Timestamp(registered[-1])
        span = {d.date().isoformat() for d in trading_calendar if lo <= d <= hi}
        missing_days = sorted(span - set(registered))

    passed = not (
        hash_mismatch or missing_file or duplicates or nstocks_mismatch
        or escape_file or missing_days
    )
    return {
        "manifest": str(manifest_path),
        "calendar": "已提供" if trading_calendar is not None else "未提供（缺日未校验）",
        "passed": passed,
        "n_days": len(rows),
        "n_distinct_days": len(registered),
        "n_hash_mismatch": len(hash_mismatch),
        "hash_mismatch_dates": hash_mismatch,
        "n_missing_file": len(missing_file),
        "missing_file_dates": missing_file,
        "n_duplicate_date": len(duplicates),
        "duplicate_dates": duplicates,
        "n_nstocks_mismatch": len(nstocks_mismatch),
        "nstocks_mismatch_dates": nstocks_mismatch,
        "n_path_escape": len(escape_file),
        "n_late": len(late_dates),
        "late_dates": late_dates,
        "n_missing_days": len(missing_days),
        "missing_dates": missing_days,
        "first_date": registered[0] if registered else None,
        "last_date": registered[-1] if registered else None,
    }


def audit_real_manifest() -> dict:
    """真实登记审计入口（结算计划 §3.3 判读前置；只碰哈希与日期）。"""
    return audit_registry(REAL_REGISTRY_DIR / "MANIFEST.csv", REAL_REGISTRY_DIR)
