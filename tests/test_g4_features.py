"""g4_features 单测（G4 计划 §2 0.3，20260817）。

先于实现写成：模块未建时 import 失败 → FAIL；实现后 → PASS。
三项计划指定测试 + 三项支撑测试（手术复制完整性 / 市场列因果性 /
9 列封装列序），全部纯 CPU，不触发 DDB / GPU（G1 tokenizer 权重只读加载除外）。

- ``TestDataset9Cols``（计划 §2 0.3 test_dataset_9cols）：列名/顺序/无 NaN/
  市场列全截面同值 + 前 6 列与源逐位一致；
- ``TestWarmstartEquivalence``（计划 §2 0.3 test_warmstart_equivalence，
  **冻结门禁**）：零初始化 9 列 tokenizer 对 6 列输入 + 3 列全零，
  encode 输出 token 与 G1 tokenizer 逐位一致；
- ``TestMA200Warmup``（计划 §2 0.3 test_ma200_warmup）：2013 预热下
  train 首日（2014-01-02）gate 可算；
- ``TestSurgeryCopy``：手术只动 embed/head，其余参数逐位复制，新列权重为零；
- ``TestMarketContextCausality``：市场三列只依赖 ≤d 的指数数据（无前视）；
- ``TestInferWrapperCols``：9 列推理封装的列序与父类 6 列约定衔接正确。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
G1_TOKENIZER_DIR = (
    REPO_ROOT / "finetune_suite" / "outputs" / "models"
    / "finetune_tokenizer_g1" / "checkpoints" / "best_model"
)

BASE_COLS = ["open", "high", "low", "close", "vol", "amt"]
MARKET_COLS = ["idx_ret", "mkt_vol", "ma200_gate"]
NINE_COLS = BASE_COLS + MARKET_COLS


def _synthetic_index(start: str = "2012-06-01", periods: int = 620, seed: int = 11):
    """合成指数收盘序列（约 2.5 年 bday，覆盖 MA200 预热 + 2014 首日）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    close = pd.Series(
        3000.0 * np.exp(np.cumsum(rng.standard_normal(periods) * 0.01)), index=idx
    )
    return close


def _synthetic_stocks(mkt_index: pd.DatetimeIndex, n_symbols: int = 3, seed: int = 5):
    """合成 {symbol: 6 列 DataFrame}，日期含 2014-01-02 起（G1 语料首日）。"""
    rng = np.random.default_rng(seed)
    dates = mkt_index[mkt_index >= pd.Timestamp("2014-01-02")][:40]
    data = {}
    for k in range(n_symbols):
        n = len(dates) - k  # 各股行数不同，右连接后仍须对齐
        df = pd.DataFrame(
            {
                "open": 10.0 + rng.standard_normal(n),
                "high": 11.0 + rng.standard_normal(n),
                "low": 9.0 + rng.standard_normal(n),
                "close": 10.5 + rng.standard_normal(n),
                "vol": 1e6 * (1.0 + rng.standard_normal(n)),
                "amt": 1e7 * (1.0 + rng.standard_normal(n)),
            },
            index=dates[:n],
        )
        data[f"C{k:03d}"] = df
    return data


# ============================================================
# test_dataset_9cols
# ============================================================
class TestDataset9Cols:
    def test_dataset_9cols(self) -> None:
        from g4_features.build_dataset import attach_market_context
        from g4_features.market_context import compute_market_context

        close = _synthetic_index()
        mkt = compute_market_context(close)
        data = _synthetic_stocks(close.index)
        src = {s: df.copy() for s, df in data.items()}

        out = attach_market_context(data, mkt)

        for sym, df in out.items():
            # 列名与顺序（冻结）
            assert list(df.columns) == NINE_COLS, f"{sym} 列序错误: {list(df.columns)}"
            # 无 NaN（含市场列）
            assert not df.isna().any().any(), f"{sym} 含 NaN"
            # 行索引与源一致（右连接不改行集）
            assert df.index.equals(src[sym].index)
            # 前 6 列与源逐位一致（唯一变量 = 新增 3 列）
            pd.testing.assert_frame_equal(df[BASE_COLS], src[sym], check_freq=False)

        # 市场列全截面同值：同日跨股票取值完全一致
        for col in MARKET_COLS:
            by_date: dict[pd.Timestamp, set] = {}
            for df in out.values():
                for d, v in df[col].items():
                    by_date.setdefault(d, set()).add(round(float(v), 12))
            assert all(len(vs) == 1 for vs in by_date.values()), f"{col} 截面不同值"

    def test_market_col_missing_raises(self) -> None:
        """源行日期在市场表覆盖外（含 NaN）→ 显式报错，不静默产出 NaN。"""
        from g4_features.build_dataset import attach_market_context
        from g4_features.market_context import compute_market_context

        close = _synthetic_index(periods=620)
        mkt = compute_market_context(close)
        data = _synthetic_stocks(close.index)
        # 人造一只日期早于市场列首个可算日（MA200 需 2012-06 起 199 个 bday
        # ≈ 2013-03 才可算）的股票 → 必须显式报错
        early = pd.bdate_range("2012-07-01", periods=3)
        data["C999"] = pd.DataFrame(
            {c: 1.0 for c in BASE_COLS}, index=early
        )
        with pytest.raises(ValueError, match="市场上下文列存在 NaN"):
            attach_market_context(data, mkt)


# ============================================================
# test_warmstart_equivalence（冻结门禁）
# ============================================================
class TestWarmstartEquivalence:
    @pytest.mark.skipif(
        not G1_TOKENIZER_DIR.exists(), reason="G1 tokenizer 权重不在本机"
    )
    def test_warmstart_equivalence(self) -> None:
        from model.kronos import KronosTokenizer

        from g4_features.surgery import expand_tokenizer_6to9

        g1 = KronosTokenizer.from_pretrained(str(G1_TOKENIZER_DIR))
        assert g1.d_in == 6, f"G1 tokenizer d_in 应为 6，实际 {g1.d_in}"
        g4 = expand_tokenizer_6to9(g1)
        assert g4.d_in == 9

        g1.eval()
        g4.eval()
        torch.manual_seed(0)
        x6 = torch.randn(2, 30, 6)
        x9 = torch.cat([x6, torch.zeros(2, 30, 3)], dim=-1)

        with torch.no_grad():
            tok_g1 = g1.encode(x6, half=True)
            tok_g4 = g4.encode(x9, half=True)
        # 冻结门禁：token 逐位一致（s1 与 s2 两个 half token 流）
        assert torch.equal(tok_g4[0], tok_g1[0]), "s1 token 不一致"
        assert torch.equal(tok_g4[1], tok_g1[1]), "s2 token 不一致"

        # forward 出口前 6 列亦一致（重构侧 sanity，非门禁本体）
        with torch.no_grad():
            (z_pre_g1, z_g1), _, _, _ = g1(x6)
            (z_pre_g4, z_g4), _, _, _ = g4(x9)
        assert torch.allclose(z_g4[:, :, :6], z_g1, atol=1e-6), "z 前 6 列漂移"
        assert torch.allclose(z_pre_g4[:, :, :6], z_pre_g1, atol=1e-6), "z_pre 前 6 列漂移"
        assert torch.count_nonzero(z_g4[:, :, 6:]) + torch.count_nonzero(z_g4[:, :, 6:]) == 0


# ============================================================
# test_ma200_warmup
# ============================================================
class TestMA200Warmup:
    def test_ma200_warmup(self) -> None:
        """2013 预热指数下，train 首日 2014-01-02 的 gate 可算（非 NaN）。"""
        from g4_features.market_context import compute_market_context

        close = _synthetic_index()  # 2012-06 起 bday，2013 全年 + 2014 在内
        mkt = compute_market_context(close)

        d0 = pd.Timestamp("2014-01-02")
        assert d0 in mkt.index
        gate_at_d0 = mkt.loc[d0, "ma200_gate"]
        assert not pd.isna(gate_at_d0), "train 首日 gate 不可算"
        assert gate_at_d0 in (0.0, 1.0)

        # 口径对拍 G3 登记约定：close > mean(最近 200 个收盘)
        hist = close.loc[:d0].iloc[-200:]
        assert len(hist) == 200
        assert gate_at_d0 == float(close.loc[d0] > hist.mean())

        # mkt_vol 在该日也可算（20 日滚动）
        assert not pd.isna(mkt.loc[d0, "mkt_vol"])


# ============================================================
# 支撑：手术复制完整性
# ============================================================
class TestSurgeryCopy:
    @pytest.mark.skipif(
        not G1_TOKENIZER_DIR.exists(), reason="G1 tokenizer 权重不在本机"
    )
    def test_surgery_copy(self) -> None:
        from model.kronos import KronosTokenizer

        from g4_features.surgery import expand_tokenizer_6to9

        g1 = KronosTokenizer.from_pretrained(str(G1_TOKENIZER_DIR))
        g4 = expand_tokenizer_6to9(g1)

        sd1, sd4 = g1.state_dict(), g4.state_dict()
        assert set(sd1) == set(sd4), "参数名集合变化（不应有）"
        for k in sd1:
            if k.startswith("embed."):
                if sd1[k].dim() == 2:
                    assert torch.equal(sd4[k][:, :6], sd1[k])
                    assert torch.count_nonzero(sd4[k][:, 6:]) == 0, f"{k} 新列非零"
                else:  # bias (d_model,) 整体继承
                    assert torch.equal(sd4[k], sd1[k])
            elif k.startswith("head."):
                if sd1[k].dim() == 2:  # weight: (d_in=9 出口, d_model)
                    assert torch.equal(sd4[k][:6, :], sd1[k])
                    assert torch.count_nonzero(sd4[k][6:, :]) == 0, f"{k} 新行非零"
                else:  # bias: (d_in,)
                    assert torch.equal(sd4[k][:6], sd1[k])
                    assert torch.count_nonzero(sd4[k][6:]) == 0, f"{k} 新位非零"
            else:
                assert torch.equal(sd4[k], sd1[k]), f"{k} 未逐位复制"


# ============================================================
# 支撑：市场列因果性（无前视）
# ============================================================
class TestMarketContextCausality:
    def test_causality(self) -> None:
        from g4_features.market_context import compute_market_context

        close = _synthetic_index()
        mkt = compute_market_context(close)

        # 篡改 d 之后的所有指数值，≤d 的三列必须不变
        d = close.index[len(close) // 2]
        close_perturbed = close.copy()
        close_perturbed.loc[close_perturbed.index > d] *= 1.5
        mkt2 = compute_market_context(close_perturbed)

        prefix = mkt.loc[:d]
        prefix2 = mkt2.loc[:d]
        pd.testing.assert_frame_equal(prefix, prefix2, check_freq=False)


# ============================================================
# 支撑：9 列推理封装列序
# ============================================================
class TestInferWrapperCols:
    def test_infer_wrapper_cols(self) -> None:
        from g4_features.infer import G4Predictor, build_inference_windows_9col

        assert G4Predictor.FEATURE_COLS_9[:4] == ["open", "high", "low", "close"]
        assert G4Predictor.FEATURE_COLS_9[4:] == ["volume", "amount"] + MARKET_COLS
        assert len(G4Predictor.FEATURE_COLS_9) == 9

        # 合成 provider + 市场表：封装产出 9 列窗口且右连接对齐
        close = _synthetic_index()
        from g4_features.market_context import compute_market_context

        mkt = compute_market_context(close)
        t = str(close.index[-11].date())  # 留足 10 个 y 交易日（日历末端约束）

        class FakeProvider:
            _start_date = "2000-01-01"
            _end_date = "2100-01-01"
            instruments_ = ["C000", "C001"]

            def list_pool_at(self, pool, day):
                return ["C000", "C001"]

            def trading_days(self, start=None, end=None):
                return close.index

            def fetch(self, fields, filter_pipe=None, freq="day"):
                # 覆盖 t=index[-11] 的 90 日回看 + 前向缓冲（取 120 日足够）
                days = close.index[-120:]
                rows, midx = [], []
                for code in ("C000", "C001"):
                    for d in days:
                        midx.append((d, code))
                        rows.append(
                            {c: float(i % 7) + 1.0 for i, c in enumerate(
                                ["open", "high", "low", "close", "volume", "amount",
                                 "preclose", "tradestatuscode"])}
                        )
                return pd.DataFrame(
                    rows, index=pd.MultiIndex.from_tuples(midx, names=["datetime", "instrument"])
                )

        df_list, x_ts, y_ts, codes, stats = build_inference_windows_9col(
            FakeProvider(), t, market=mkt, lookback=90, predict_len=10, pool="csi300"
        )
        assert len(df_list) == 2
        for df in df_list:
            assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"] + MARKET_COLS
            assert len(df) == 90
            assert not df.isna().any().any()
            # 市场列逐日与市场表一致
            pd.testing.assert_series_equal(
                df["idx_ret"], mkt.loc[df.index, "idx_ret"], check_freq=False
            )
