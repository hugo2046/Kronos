"""C3 混合规则可推导性门禁（docs/C3混合规则预注册_20260818.md §1）。

零判读纪律：本文件只验证 C3 信号是 G3 登记表已有列的确定性函数、可无歧义复算——
不读价格/收益列、不跑 canonical 引擎、不计算任何绩效数字（AER/IR/秩相关等）。
§1 四步与测试的对应：

1. ``C1 = mean(G1_s100_mean, G1_s101_mean, G1_s102_mean)``——登记表列名为
   ``s100_mean``/``s101_mean``/``s102_mean``（s103/104 诊断种子不进集成，只用 mean
   变体）；缺任一种子值剔除该股当日；
2. 当日截面 zscore：z_C1、z_M 在当日有效股票池上均值 0 方差 1。方差取总体口径
   ddof=0——只有总体方差严格归一，"均值 0 方差 1"才精确成立（无歧义口径）；
3. ``C3 = 0.5·z_C1 + 0.5·z_M``，权重 0.5/0.5 冻结；任一侧缺值剔除该股当日。
   有效股票池 = C1 侧（三种子齐）与 M 侧均在册的股票，步骤 2 在该池上做——
   否则"均值 0 方差 1"在步骤 3 存活集上不成立，推导产生歧义；
4. 排序进 canonical 引擎：本门禁只验证排序键可无歧义产出且同日重算确定
   （两次计算逐位一致、行序无关）；引擎/双基准/成本属结算期行为，此处不触碰。

数据源：登记表已有 parquet（finetune_suite/registry/signals_*.parquet，不入 git，
缺失时真实数据用例自动跳过）+ 同 schema 玩具数据注入缺值。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REGISTRY_DIR = Path(__file__).resolve().parents[1] / "finetune_suite" / "registry"

# 文档记号 G1_sXXX_mean → 登记表列 sXXX_mean
C1_SEED_COLS = ("s100_mean", "s101_mean", "s102_mean")
REQUIRED_COLS = C1_SEED_COLS + ("M",)  # §1 全部输入列
Z_TOL = 1e-8  # zscore 性质容差（M 为 float32，参与运算升 float64 后误差 << 此值）


def derive_c3(day_wide: pd.DataFrame) -> pd.DataFrame:
    """按 §1 步骤 1–3 从单日登记宽表确定性推导 C3。

    返回列恰为 ["C1", "z_C1", "z_M", "C3"]（index=code，只含存活股票）——
    不产出任何绩效/收益列（零判读）。
    """
    missing = [c for c in REQUIRED_COLS if c not in day_wide.columns]
    if missing:
        raise KeyError(f"登记表缺 §1 输入列: {missing}")
    if not day_wide.index.is_unique:
        raise ValueError("code 索引有重复——同日推导无法无歧义进行")

    # 步骤 1：C1 = 三种子 mean，skipna=False → 缺任一种子即 NaN（下游剔除）
    c1 = day_wide[list(C1_SEED_COLS)].mean(axis=1, skipna=False)
    m = day_wide["M"].astype("float64")
    # 步骤 3 前置：有效池 = 两侧均在册的股票（任一侧缺值剔除该股当日）
    valid = c1.notna() & m.notna()
    if int(valid.sum()) < 2:
        raise ValueError("当日有效股票池 < 2，截面 zscore 无法定义")

    def _z(x: pd.Series) -> pd.Series:
        sd = x.std(ddof=0)  # 总体方差口径：均值 0 方差 1 精确成立
        if sd == 0:
            raise ValueError("当日截面标准差为 0，zscore 无法无歧义定义")
        return (x - x.mean()) / sd

    out = pd.DataFrame(
        {"C1": c1[valid], "z_C1": _z(c1[valid]), "z_M": _z(m[valid])}
    )
    out["C3"] = 0.5 * out["z_C1"] + 0.5 * out["z_M"]  # 步骤 3：0.5/0.5 冻结
    return out


# ---------------------------------------------------------------------------
# 玩具数据（与登记表 parquet 同 schema：index=code + 14 列）
# ---------------------------------------------------------------------------
_TOY_CODES = ["000001.SZ", "000002.SZ", "600000.SH", "600196.SH", "601398.SH"]
_TOY_SEED_MEANS = {  # 三种子 mean 值（其余统计量列镜像 mean±常数，仅保 schema 完整）
    "000001.SZ": (0.010, 0.012, 0.008),
    "000002.SZ": (0.020, 0.024, 0.013),
    "600000.SH": (-0.030, -0.020, -0.010),
    "600196.SH": (0.050, 0.052, 0.051),
    "601398.SH": (0.001, 0.002, -0.004),
}
_TOY_M = {
    "000001.SZ": 0.030, "000002.SZ": -0.010, "600000.SH": 0.020,
    "600196.SH": 0.044, "601398.SH": 0.050,
}


def _toy_day(nan_seed: tuple[str, str] | None = None,
             nan_m: str | None = None) -> pd.DataFrame:
    """构造单日登记宽表；nan_seed=(code, 列名) 置 NaN 种子值，nan_m 置 NaN 的 M。"""
    data = {}
    for seed in (100, 101, 102):
        vals = [row[{"100": 0, "101": 1, "102": 2}[str(seed)]] for row in _TOY_SEED_MEANS.values()]
        data[f"s{seed}_mean"] = vals
        for stat, off in (("last", 0.001), ("max", 0.010), ("min", -0.010)):
            data[f"s{seed}_{stat}"] = [v + off for v in vals]
    data["M"] = list(_TOY_M.values())
    data["tradeable"] = True
    df = pd.DataFrame(data, index=pd.Index(_TOY_CODES, name="code"))
    if nan_seed is not None:
        df.loc[nan_seed[0], nan_seed[1]] = np.nan
    if nan_m is not None:
        df.loc[nan_m, "M"] = np.nan
    return df


def _registry_days() -> list[Path]:
    return sorted(REGISTRY_DIR.glob("signals_*.parquet"))


# ---------------------------------------------------------------------------
# 步骤 1：C1 = 三种子 mean；缺任一种子剔除该股当日
# ---------------------------------------------------------------------------
def test_step1_c1_is_seed_mean() -> None:
    out = derive_c3(_toy_day())
    for code in _TOY_CODES:
        expected = float(np.mean(_TOY_SEED_MEANS[code]))
        assert out.loc[code, "C1"] == pytest.approx(expected, abs=1e-12)


def test_step1_missing_any_seed_drops_stock() -> None:
    for col in C1_SEED_COLS:  # 缺三种子的任一个都剔除，M 侧完好也不例外
        out = derive_c3(_toy_day(nan_seed=("601398.SH", col)))
        assert "601398.SH" not in out.index
        assert set(out.index) == set(_TOY_CODES) - {"601398.SH"}


# ---------------------------------------------------------------------------
# 步骤 2：当日 zscore——有效池上均值 0 方差 1（总体方差 ddof=0）
# ---------------------------------------------------------------------------
def test_step2_zscore_mean0_var1_toy() -> None:
    out = derive_c3(_toy_day())
    for col in ("z_C1", "z_M"):
        assert out[col].mean() == pytest.approx(0.0, abs=Z_TOL)
        assert out[col].var(ddof=0) == pytest.approx(1.0, abs=Z_TOL)


def test_step2_zscore_mean0_var1_registry() -> None:
    days = _registry_days()
    if not days:
        pytest.skip("登记表 parquet 不入 git，本机无文件——真实数据用例跳过")
    for path in days:
        out = derive_c3(pd.read_parquet(path))
        for col in ("z_C1", "z_M"):
            assert out[col].mean() == pytest.approx(0.0, abs=Z_TOL), path.name
            assert out[col].var(ddof=0) == pytest.approx(1.0, abs=Z_TOL), path.name


# ---------------------------------------------------------------------------
# 步骤 3：C3 = 0.5·z_C1 + 0.5·z_M；任一侧缺值剔除该股当日
# ---------------------------------------------------------------------------
def test_step3_combination_weights_frozen() -> None:
    out = derive_c3(_toy_day())
    assert np.array_equal(
        out["C3"].to_numpy(), 0.5 * out["z_C1"].to_numpy() + 0.5 * out["z_M"].to_numpy()
    )
    assert not out.isna().any().any()
    # 零判读门禁：产出列恰为四个信号列，不掺任何绩效/收益列
    assert list(out.columns) == ["C1", "z_C1", "z_M", "C3"]


def test_step3_missing_either_side_drops_stock() -> None:
    # M 侧缺值（三种子齐）→ 剔除
    out = derive_c3(_toy_day(nan_m="600196.SH"))
    assert "600196.SH" not in out.index
    # 两侧各有缺值 → 存活集 = 两侧均在册者的交集
    out = derive_c3(_toy_day(nan_seed=("601398.SH", "s101_mean"), nan_m="600196.SH"))
    assert set(out.index) == {"000001.SZ", "000002.SZ", "600000.SH"}
    # 步骤 2 性质在存活集上仍精确成立（有效池口径无歧义）
    for col in ("z_C1", "z_M"):
        assert out[col].mean() == pytest.approx(0.0, abs=Z_TOL)
        assert out[col].var(ddof=0) == pytest.approx(1.0, abs=Z_TOL)


# ---------------------------------------------------------------------------
# 列完整性与输入契约
# ---------------------------------------------------------------------------
def test_registry_parquet_column_completeness() -> None:
    days = _registry_days()
    if not days:
        pytest.skip("登记表 parquet 不入 git，本机无文件——真实数据用例跳过")
    for path in days:
        df = pd.read_parquet(path)
        assert df.index.name == "code", path.name
        assert df.index.is_unique, path.name  # 重复 code 会使同日推导产生歧义
        for col in REQUIRED_COLS:
            assert col in df.columns, f"{path.name} 缺 §1 输入列 {col}"
            assert np.issubdtype(df[col].dtype, np.number), f"{col} 非数值列"


def test_missing_input_column_raises() -> None:
    df = _toy_day().drop(columns=["M"])
    with pytest.raises(KeyError, match="M"):
        derive_c3(df)


def test_duplicate_code_raises() -> None:
    df = pd.concat([_toy_day(), _toy_day().iloc[[0]]])
    with pytest.raises(ValueError, match="重复"):
        derive_c3(df)


# ---------------------------------------------------------------------------
# 步骤 4 前置：排序键同日重算确定（两次计算逐位一致、行序无关、重读落盘一致）
# ---------------------------------------------------------------------------
def test_step4_sorting_key_deterministic() -> None:
    toy = _toy_day()
    first = derive_c3(toy)
    assert first.equals(derive_c3(toy))  # 同输入两次计算逐位一致
    # 行序无关：倒序输入后按 code 对齐，数值一致（浮点求和序差异 << 容差）
    shuffled = derive_c3(toy.iloc[::-1]).sort_index()
    aligned = first.sort_index()
    for col in first.columns:
        assert shuffled[col].to_numpy() == pytest.approx(
            aligned[col].to_numpy(), abs=1e-12
        )


def test_step4_deterministic_on_registry_days() -> None:
    days = _registry_days()
    if not days:
        pytest.skip("登记表 parquet 不入 git，本机无文件——真实数据用例跳过")
    for path in days:
        df = pd.read_parquet(path)
        first = derive_c3(df)
        assert len(first) >= 2, path.name  # 有效池 ≥2，zscore 才有定义
        # 重读落盘再算——逐位一致；倒序行再算——按 code 对齐后数值一致
        # （parquet 行序本就未按 code 排序，对拍两侧必须先对齐）
        assert first.equals(derive_c3(pd.read_parquet(path)))
        shuffled = derive_c3(df.iloc[::-1]).sort_index()
        aligned = first.sort_index()
        for col in first.columns:
            assert shuffled[col].to_numpy() == pytest.approx(
                aligned[col].to_numpy(), abs=1e-12
            ), path.name
