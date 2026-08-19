"""settlement 包端到端演习门禁（forward结算计划_20260818.md / C3预注册 / 滚动再训协议）。

本测试文件先于实现写成（TDD：模块未建时 import 失败 → FAIL；实现后 → PASS）。

冻结来源与测试的对应：
- 审计器（结算计划 §3.3）：MANIFEST 哈希链逐日校验 + 缺日/late 清点，先于任何
  绩效计算；**只碰哈希与日期**（parquet 仅经 pyarrow 元数据读行数，不读任何值）；
- 规则（组合规则预注册_20260816.md §2 + C3混合规则预注册_20260818.md §1）：
  C1=三种子 mean 集成 / C2=MA200 门控切换（True→M，False→C1）/ C3=0.5·z_C1+
  0.5·z_M（总体方差口径）/ R1 原版=True→M，False→F0_mean；
- 执行器（结算计划 §2/§3）：七臂（G1 三种子、C1、C2、M、F1、F0+R1、B3）+ C3
  同场同次开封，判据只在 mean；G1 主判据绑定 Q2 → 关闭/滚动分支路由；
- 双轨措辞（滚动再训协议_20260819.md §3/§4）：决策栏=冻结判据代入（预承诺装置），
  证据栏=±26pp 噪声底折扣后的评估——两轨分列，不得混写；
- 演习纪律：合成登记数据一律 ``synthetic_`` 前缀、落 ``settlement/data_synthetic/``；
  合成与真实数据路径物理隔离；演习全程零真实 forward 信号/价格读取。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
REAL_REGISTRY_DIR = REPO / "finetune_suite" / "registry"
SYNTHETIC_ROOT_DEFAULT = REPO / "settlement" / "data_synthetic"


# ---------------------------------------------------------------------------
# 物理隔离：synthetic_ 前缀 + 目录互斥 + 真实 MANIFEST 演习后原封不动
# ---------------------------------------------------------------------------
class TestPhysicalIsolation:
    def test_synthetic_prefix_guard_rejects_real_paths(self) -> None:
        from settlement.drill import SYNTHETIC_PREFIX, assert_synthetic_path

        assert SYNTHETIC_PREFIX == "synthetic_"
        ok = SYNTHETIC_ROOT_DEFAULT / "synthetic_signals_2026-08-14.parquet"
        assert_synthetic_path(ok)  # 不抛
        with pytest.raises(ValueError, match="synthetic_"):
            assert_synthetic_path(REAL_REGISTRY_DIR / "signals_2026-08-14.parquet")

    def test_drill_writes_only_synthetic_prefixed_files(self, tmp_path: Path) -> None:
        from settlement.drill import run_drill

        for scenario in ("关闭", "滚动"):
            out = run_drill(scenario=scenario, root=tmp_path)
            for p in out["root"].rglob("*"):
                if p.is_file():
                    assert p.name.startswith("synthetic_"), p

    def test_real_registry_untouched_by_drill(self, tmp_path: Path) -> None:
        from settlement.drill import run_drill

        def _state() -> tuple[str, list[str]]:
            man = REAL_REGISTRY_DIR / "MANIFEST.csv"
            digest = hashlib.sha256(man.read_bytes()).hexdigest() if man.exists() else ""
            return digest, sorted(p.name for p in REAL_REGISTRY_DIR.glob("*"))

        before = _state()
        run_drill(scenario="关闭", root=tmp_path)
        run_drill(scenario="滚动", root=tmp_path)
        assert _state() == before
        # 真实登记目录不得出现任何 synthetic_ 文件（物理隔离双向断言）
        assert not list(REAL_REGISTRY_DIR.glob("synthetic_*"))


# ---------------------------------------------------------------------------
# 零真实读取：演习全程 pandas/duckdb 只触碰 synthetic 根
# ---------------------------------------------------------------------------
class TestZeroRealReads:
    def test_drill_reads_only_synthetic_paths(self, tmp_path: Path, monkeypatch) -> None:
        from settlement.drill import run_drill

        accessed: list[str] = []
        real_read_parquet = pd.read_parquet
        real_read_csv = pd.read_csv

        def _spy_parquet(path, *a, **k):
            accessed.append(str(path))
            return real_read_parquet(path, *a, **k)

        def _spy_csv(path, *a, **k):
            accessed.append(str(path))
            return real_read_csv(path, *a, **k)

        monkeypatch.setattr(pd, "read_parquet", _spy_parquet)
        monkeypatch.setattr(pd, "read_csv", _spy_csv)
        for scenario in ("关闭", "滚动"):
            run_drill(scenario=scenario, root=tmp_path)
        assert accessed, "演习应有文件读取（否则监控失效）"
        for p in accessed:
            assert "data_synthetic" in p and "synthetic_" in Path(p).name, (
                f"演习读取了非合成路径：{p}"
            )

    def test_drill_guard_blocks_real_paths(self, tmp_path: Path) -> None:
        """run_drill 自带守卫：演习中任何真实登记路径读取直接抛错。"""
        from settlement.drill import SyntheticOnlyGuard

        real = REAL_REGISTRY_DIR / "signals_2026-08-14.parquet"
        with SyntheticOnlyGuard(tmp_path):
            with pytest.raises(RuntimeError, match="真实"):
                pd.read_parquet(real)
            with pytest.raises(RuntimeError, match="真实"):
                pd.read_csv(REAL_REGISTRY_DIR / "MANIFEST.csv")
        # 守卫退出后恢复正常读取路径
        if real.exists():
            pd.read_parquet(real)


# ---------------------------------------------------------------------------
# 审计器（结算计划 §3.3）：哈希/日期/late/缺日
# ---------------------------------------------------------------------------
class TestAuditor:
    @staticmethod
    def _clean(tmp_path: Path) -> Path:
        from settlement.drill import write_synthetic_registry, build_synthetic_world

        root = tmp_path / "data_synthetic"
        world = build_synthetic_world(scenario="滚动", n_days=8)
        write_synthetic_registry(world, root)
        return root

    def test_clean_synthetic_manifest_passes(self, tmp_path: Path) -> None:
        from settlement.audit import audit_registry

        self._clean(tmp_path)
        report = audit_registry(
            tmp_path / "data_synthetic" / "synthetic_MANIFEST.csv",
            tmp_path / "data_synthetic",
        )
        assert report["passed"] is True
        assert report["n_days"] == 8
        assert report["n_hash_mismatch"] == 0 and report["n_missing_file"] == 0
        assert report["n_duplicate_date"] == 0

    def test_tampered_parquet_flagged(self, tmp_path: Path) -> None:
        from settlement.audit import audit_registry

        root = self._clean(tmp_path)
        p = next(root.glob("synthetic_signals_*.parquet"))
        df = pd.read_parquet(p)
        df.iloc[0, 0] = float(df.iloc[0, 0]) + 1.0  # 篡改一个值 → 哈希断链
        df.to_parquet(p)
        report = audit_registry(root / "synthetic_MANIFEST.csv", root)
        assert report["passed"] is False
        assert report["n_hash_mismatch"] == 1
        assert report["hash_mismatch_dates"]

    def test_missing_file_and_late_and_gap(self, tmp_path: Path) -> None:
        from settlement.audit import audit_registry
        from settlement.drill import build_synthetic_world, write_synthetic_registry

        root = tmp_path / "data_synthetic"
        world = build_synthetic_world(scenario="关闭", n_days=8)
        write_synthetic_registry(world, root, late_days=(world.dates[2],))
        # victim A：parquet 删但 manifest 行保留 → 缺文件事件
        (root / f"synthetic_signals_{world.dates[5].date()}.parquet").unlink()
        # victim B：parquet 删 + manifest 行删 → 缺日事件（日历内无登记）
        gone = root / f"synthetic_signals_{world.dates[6].date()}.parquet"
        gone.unlink()
        rows = (root / "synthetic_MANIFEST.csv").read_text().splitlines()
        rows = [r for r in rows if world.dates[6].date().isoformat() not in r]
        (root / "synthetic_MANIFEST.csv").write_text("\n".join(rows) + "\n")
        calendar = list(world.dates)
        calendar.remove(world.dates[3])  # 该日不在日历 → 不算缺日
        report = audit_registry(root / "synthetic_MANIFEST.csv", root,
                                trading_calendar=pd.DatetimeIndex(calendar))
        assert report["passed"] is False
        assert report["n_missing_file"] == 1
        assert report["n_late"] == 1
        assert report["late_dates"] == [world.dates[2].date().isoformat()]
        assert world.dates[6].date().isoformat() in report["missing_dates"]
        assert world.dates[3].date().isoformat() not in report["missing_dates"]

    def test_duplicate_manifest_date_flagged(self, tmp_path: Path) -> None:
        from settlement.audit import audit_registry

        root = self._clean(tmp_path)
        with open(root / "synthetic_MANIFEST.csv", "a", encoding="utf-8") as f:
            f.write(f.readline() if False else "")
        rows = (root / "synthetic_MANIFEST.csv").read_text().splitlines()
        rows.append(rows[1])  # 复制首行数据 → 重复日期
        (root / "synthetic_MANIFEST.csv").write_text("\n".join(rows) + "\n")
        report = audit_registry(root / "synthetic_MANIFEST.csv", root)
        assert report["n_duplicate_date"] == 1 and report["passed"] is False

    def test_audit_touches_only_hash_and_dates(self, tmp_path: Path) -> None:
        """审计器不得解载任何信号值：patch read_parquet 为陷阱。"""
        import settlement.audit as A

        root = self._clean(tmp_path)
        orig = A.pd.read_parquet
        A.pd.read_parquet = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("审计器不得 read_parquet（只允许哈希与元数据）"))
        try:
            report = A.audit_registry(root / "synthetic_MANIFEST.csv", root)
            assert report["passed"] is True
        finally:
            A.pd.read_parquet = orig

    def test_real_manifest_smoke(self) -> None:
        from settlement.audit import audit_real_manifest

        if not (REAL_REGISTRY_DIR / "MANIFEST.csv").exists():
            pytest.skip("真实 MANIFEST 不在盘")
        report = audit_real_manifest()
        assert report["passed"] is True, report
        assert report["n_days"] >= 1


# ---------------------------------------------------------------------------
# 组合规则（冻结定义的确定性推导）
# ---------------------------------------------------------------------------
class TestRules:
    @staticmethod
    def _day(nan: tuple[str, str] | None = None) -> pd.DataFrame:
        data = {
            "s100_mean": [0.010, 0.020, -0.030, 0.050, 0.001],
            "s101_mean": [0.012, 0.024, -0.020, 0.052, 0.002],
            "s102_mean": [0.008, 0.013, -0.010, 0.051, -0.004],
            "M": [0.030, -0.010, 0.020, 0.044, 0.050],
            "tradeable": True,
        }
        df = pd.DataFrame(data, index=pd.Index(
            [f"SYN{i:04d}" for i in range(1, 6)], name="code"))
        if nan:
            df.loc[nan[0], nan[1]] = float("nan")
        return df

    def test_c1_is_three_seed_mean_with_nan_drop(self) -> None:
        from settlement.rules import derive_c1_day

        s = derive_c1_day(self._day())
        assert s.loc["SYN0001"] == pytest.approx((0.010 + 0.012 + 0.008) / 3)
        s2 = derive_c1_day(self._day(nan=("SYN0003", "s101_mean")))
        assert "SYN0003" not in s2.index and len(s2) == 4

    def test_c2_switches_on_gate(self) -> None:
        from settlement.rules import derive_c1_day, derive_c2_day

        day = self._day()
        on = derive_c2_day(day, gate=True)
        off = derive_c2_day(day, gate=False)
        pd.testing.assert_series_equal(on.dropna(), day["M"].dropna(), check_names=False)
        pd.testing.assert_series_equal(off.dropna(), derive_c1_day(day), check_names=False)

    def test_c3_zscore_and_frozen_weights(self) -> None:
        from settlement.rules import derive_c3_day

        out = derive_c3_day(self._day(nan=("SYN0005", "s102_mean")))
        assert set(out.index) == {"SYN0001", "SYN0002", "SYN0003", "SYN0004"}
        for col in ("z_C1", "z_M"):
            assert out[col].mean() == pytest.approx(0.0, abs=1e-8)
            assert out[col].var(ddof=0) == pytest.approx(1.0, abs=1e-8)
        assert (out["C3"] == 0.5 * out["z_C1"] + 0.5 * out["z_M"]).all()
        assert list(out.columns) == ["C1", "z_C1", "z_M", "C3"]

    def test_r1_assembly_true_m_false_f0(self) -> None:
        from settlement.rules import r1_assemble

        m = pd.Series({"A": 1.0, "B": 2.0})
        f0 = pd.Series({"A": 10.0, "B": 20.0})
        assert r1_assemble(m, f0, gate=True).tolist() == [1.0, 2.0]
        assert r1_assemble(m, f0, gate=False).tolist() == [10.0, 20.0]


# ---------------------------------------------------------------------------
# 复算级臂（F0/F1/B3/R1）
# ---------------------------------------------------------------------------
class TestRecompute:
    def test_specs_frozen_and_on_disk(self) -> None:
        from settlement.recompute import RECOMPUTE_ARM_SPECS, verify_specs_on_disk

        assert set(RECOMPUTE_ARM_SPECS) == {"F0", "F1", "B3", "R1"}
        assert RECOMPUTE_ARM_SPECS["F0"]["weights"] == "NeoQuasar/Kronos-base"
        assert RECOMPUTE_ARM_SPECS["F0"]["tokenizer"] == "NeoQuasar/Kronos-Tokenizer-base"
        # 协议逐字 canonical（结算计划 §1 复算级定义）
        for arm, spec in RECOMPUTE_ARM_SPECS.items():
            assert spec["protocol"] == (
                "L=90/H=10/N=20/T=1.0/top_p=0.9/seed=42 canonical"), arm
        verify_specs_on_disk()  # F1/B3 路径在盘（R1 组装无权重）

    def test_generate_assembles_wide_deterministic(self, tmp_path: Path) -> None:
        from settlement.drill import build_synthetic_world
        from settlement.recompute import SyntheticRecomputeSource, generate_arm_signals

        world = build_synthetic_world(scenario="滚动", n_days=6)
        src = SyntheticRecomputeSource(world)
        wide = generate_arm_signals("F1", world.dates, src)
        assert isinstance(wide, pd.DataFrame)
        assert list(wide.index) == list(world.dates)
        again = generate_arm_signals("F1", world.dates, src)
        pd.testing.assert_frame_equal(wide, again)  # 确定性重放

    def test_unknown_arm_rejected(self, tmp_path: Path) -> None:
        from settlement.drill import build_synthetic_world
        from settlement.recompute import SyntheticRecomputeSource, generate_arm_signals

        world = build_synthetic_world(scenario="关闭", n_days=4)
        with pytest.raises(KeyError, match="复算臂"):
            generate_arm_signals("G9", world.dates, SyntheticRecomputeSource(world))


# ---------------------------------------------------------------------------
# 执行器：七臂+C3 同场、判据代入、分支路由（关闭/滚动两分支都走通）
# ---------------------------------------------------------------------------
class TestExecutorBothBranches:
    @staticmethod
    def _run(scenario: str, tmp_path: Path) -> dict:
        from settlement.drill import run_drill

        return run_drill(scenario=scenario, root=tmp_path / f"drill_{scenario}")

    def test_close_branch(self, tmp_path: Path) -> None:
        from settlement.template import BRANCH_CLOSE, BRANCH_ROLLING

        out = self._run("关闭", tmp_path)
        v = out["verdict"]
        assert v["g1_main"]["passed"] is False
        assert v["branch"] == BRANCH_CLOSE
        assert "Q2" in v["g1_main"]["decision_wording"]
        med = v["g1_main"]["median_seed_aer"]
        assert med["aer_ew"] <= 0 or med["aer_idx"] <= 0  # 判据代入与数字一致
        doc = Path(out["doc_path"]).read_text(encoding="utf-8")
        assert "项目关闭" in doc and "进入滚动确认" not in doc

    def test_rolling_branch(self, tmp_path: Path) -> None:
        from settlement.template import BRANCH_CLOSE, BRANCH_ROLLING

        out = self._run("滚动", tmp_path)
        v = out["verdict"]
        assert v["g1_main"]["passed"] is True
        assert v["branch"] == BRANCH_ROLLING
        assert v["g1_main"]["decision_wording"] == "前向首读通过，进入滚动确认"
        med = v["g1_main"]["median_seed_aer"]
        assert med["aer_ew"] > 0 and med["aer_idx"] > 0
        doc = Path(out["doc_path"]).read_text(encoding="utf-8")
        assert "进入滚动确认" in doc and "项目关闭" not in doc

    def test_seven_arms_plus_c3_full_table(self, tmp_path: Path) -> None:
        out = self._run("滚动", tmp_path)
        ft = out["full_table"]
        arms = {k.split("@")[0] for k in ft}
        for arm in ("G1_s100", "G1_s101", "G1_s102", "C1", "C2", "M",
                    "F1", "F0", "R1", "B3", "C3"):
            assert arm in arms, f"全表缺 {arm}"
        assert not any(a.startswith(("G4", "G5", "G1N50")) for a in arms), \
            "已关闭臂不得进结算（全表纪律）"
        # 分级标注：登记级/复算级
        assert out["grades"]["C1"] == "登记级" and out["grades"]["F0"] == "复算级"

    def test_criteria_substitution_consistent_with_numbers(self, tmp_path: Path) -> None:
        out = self._run("滚动", tmp_path)
        v, ft = out["verdict"], out["full_table"]

        def ew(arm: str) -> float:
            return ft[f"{arm}@forward"]["mean"]["aer_ew"]

        assert v["C1"]["survived"] == (ew("C1") > 0 and ft["C1@forward"]["mean"]["aer_idx"] > 0)
        assert v["C1"]["increment"] == (ew("C1") > ew("G1_s100"))
        assert v["C2"]["increment"] == (ew("C2") > max(ew("M"), ew("C1")))
        assert v["C3"]["increment"] == (ew("C3") > max(ew("C1"), ew("M")))
        assert v["R1"]["survived"] == (ew("R1") > 0 and ft["R1@forward"]["mean"]["aer_idx"] > 0)
        assert v["F1_direction"] == (ew("F1") > ew("F0"))
        assert v["B3"]["seed_luck_confirmed"] == (ew("B3") <= 0)
        assert v["R1"]["grade"] == "复算级" and v["B3"]["grade"] == "复算级"

    def test_audit_and_crosscheck_precede_perf(self, tmp_path: Path) -> None:
        """§3.3 先于绩效计算：audit/cross_check 结果随结算产物落盘。"""
        out = self._run("关闭", tmp_path)
        assert out["audit"]["passed"] is True
        cc = out["cross_check"]
        assert len(cc["arms"]) >= 3 and len(cc["days"]) >= 5
        assert cc["max_abs_delta"] == 0.0  # 同源确定性 → 逐位一致

    def test_c2_late_day_sensitivity(self, tmp_path: Path) -> None:
        out = self._run("滚动", tmp_path)
        assert "C2_excl_late@forward" in out["full_table"]
        assert out["late_days"]

    def test_settleable_days_assertion(self, tmp_path: Path) -> None:
        from settlement.drill import build_synthetic_world, write_synthetic_registry
        from settlement.engines import SyntheticEngine
        from settlement.executor import run_settlement
        from settlement.recompute import SyntheticRecomputeSource
        from settlement.registry_sources import SyntheticRegistrySource

        world = build_synthetic_world(scenario="滚动", n_days=30)  # < 60
        root = tmp_path / "data_synthetic"
        write_synthetic_registry(world, root)
        with pytest.raises(AssertionError, match="60"):
            run_settlement(
                registry=SyntheticRegistrySource(root),
                dates=world.dates,
                engine=SyntheticEngine(world),
                recompute=SyntheticRecomputeSource(world),
                out_dir=root,
                label="SYNTHETIC DRILL",
            )


# ---------------------------------------------------------------------------
# 双轨措辞模板
# ---------------------------------------------------------------------------
class TestTemplate:
    def test_dual_track_and_noise_floor(self, tmp_path: Path) -> None:
        from settlement.drill import run_drill

        out = run_drill(scenario="滚动", root=tmp_path / "t")
        doc = Path(out["doc_path"]).read_text(encoding="utf-8")
        assert "决策栏" in doc and "证据栏" in doc
        assert "26pp" in doc  # ±26pp 噪声底（滚动协议 §3）
        assert "登记级" in doc and "复算级" in doc
        assert "SYNTHETIC" in doc  # 演习横幅——不得与真实结算文档混淆
        assert "待滚动确认" in doc  # "通过"措辞上限
