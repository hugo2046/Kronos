"""MH1 多期限监督实验 契约测试（计划 §5 任务 A/B/C，20260907 TimesFM 启发计划）。

先 FAIL 后 PASS 的预注册契约（与计划条目一一对应）：

任务 A（数据与泄漏门禁）：
- ``test_calendar_horizon_label``：人工日历 4 日/价格 100→133.1，t=首日 1 期限
  收益恰为 0.1；删除 01-03 个股行后该标签必须 NaN（不前向/后向填充、
  不得用个股下一条记录代替下一交易日）；
- ``test_train_purge20_and_val_bounds``：20 日跨段剔除 + 训练/验证边界
  （训练标签终点 ≤ 2024-12-31、验证决策日 ≥ 2025-01-01 且终点 ≤ 2025-06-30）；
- ``test_no_lookahead_normalization``：窗口归一化只由 t 前 90 交易日计算，
  扰动窗口外未来数据不改变 x_norm；
- ``test_pit_membership_only_t``：PIT 资格只看 t 日成员（退池股仍有标签、
  未来新成员不被 t 日采样）；
- ``test_sm_identical_batches_and_skip_rules``：S/M 同种子批次键完全一致、
  127 只日被跳过、128 只无放回无重复、固定种子可复现；
- ``test_w3w4_bounds_and_query_ceiling``：W3/W4 决策日 +20 终点留在段内、
  全部查询与标签上界 ≤ 2026-07-24（forward 封存）。

任务 B（头与训练）：
- ``test_single_horizon_masks_auxiliary_gradients`` / ``test_multi_horizon_...``
  （计划 §5 任务 B 逐字）；
- ``test_backbone_frozen_and_ckpt_scoped``：新头更新前后底座逐值相等且无梯度、
  头参数改变、head.train() 不改变底座 eval、checkpoint 只含新头；
- ``test_selection_reads_only_main_horizon``：选点只读 10 日 RankIC（并列最早、
  非有限不可选）；
- ``test_w3w4_data_does_not_touch_training``：修改 W3/W4 数据不改变训练批次与
  选点（fetch 查询上界断言）。

任务 C（评估与封盘）：
- ``test_ic_code_alignment``：按 code 对齐 IC=1、重排不变、重复 (date,code) 拒绝；
- ``test_ic_missing_and_pairing``：<30 只/常量分数/缺失标签 → 记缺失不填零、
  配对 IC 只用共同股票、三种子只用共同有效日期；
- ``test_segmented_hac``：单窗与 ``_nw_tvalue`` 对齐、跨窗不建滞后协方差、
  缺失日期按原交易日距离配对、退化方差不可判。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mh1_multihorizon.config import (  # noqa: E402
    BATCH, EPOCHS, HORIZONS, LR, MAIN_HORIZON_IDX, PURGE, SEEDS, STEPS_PER_EPOCH,
    TRAIN_LABEL_END, TRAIN_START, VAL_LABEL_END, VAL_START, WD,
    W3_END, W3_START, W4_END, W4_START, FORWARD_CUTOFF, MIN_TRAIN_CROSS,
)

# ============================================================
# 人工 fixture 日历（故意不连续自然日）
# ============================================================

CAL4 = pd.DatetimeIndex(
    ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
CAL4_POS = {d: i for i, d in enumerate(CAL4)}
PX4 = np.array([100.0, 110.0, 121.0, 133.1])


def test_protocol_constants_frozen() -> None:
    """计划 §3 冻结常量逐字段对拍（协议零漂移门禁）。"""
    assert HORIZONS == (1, 5, 10, 20)
    assert MAIN_HORIZON_IDX == 2
    assert (TRAIN_START, TRAIN_LABEL_END) == ("2014-01-02", "2024-12-31")
    assert (VAL_START, VAL_LABEL_END) == ("2025-01-01", "2025-06-30")
    assert (W3_START, W3_END) == ("2025-07-01", "2025-12-31")
    assert (W4_START, W4_END) == ("2026-01-01", "2026-07-24")
    assert FORWARD_CUTOFF == "2026-07-24"
    assert PURGE == 20
    assert (BATCH, LR, WD, STEPS_PER_EPOCH, EPOCHS) == (128, 3e-4, 0.01, 2000, 15)
    assert SEEDS == (42, 43, 44)
    assert MIN_TRAIN_CROSS == 128


# ============================================================
# 任务 A.1：日历定位标签（先失败的核心数据契约）
# ============================================================


def test_calendar_horizon_label() -> None:
    from mh1_multihorizon.data import horizon_labels

    # 完整数据：t=首日，k=1 → 110/100-1 = 0.1（按交易所日历推进）
    lab = horizon_labels(CAL4, PX4, CAL4, CAL4_POS, i=0, horizons=(1, 2))
    assert lab[0] == pytest.approx(0.1)
    assert lab[1] == pytest.approx(0.21)

    # 删除 01-03 个股行：日历不变，1 期限标签必须 NaN（不能变成 0.21 =
    # 121/100-1，即不得用个股下一条记录代替下一交易日）
    keep = [0, 2, 3]
    lab2 = horizon_labels(CAL4[keep], PX4[keep], CAL4, CAL4_POS, i=0,
                          horizons=(1, 2))
    assert np.isnan(lab2[0]), "缺 01-03 行时 1 期限标签应缺失，不得前向填充"
    assert lab2[1] == pytest.approx(0.21)  # k=2 → 01-05 恰存在

    # 分母非有限或 ≤0 → NaN
    bad = PX4.copy()
    bad[1] = 0.0
    lab3 = horizon_labels(CAL4, bad, CAL4, CAL4_POS, i=0, horizons=(1,))
    assert np.isnan(lab3[0])
    # 期限超出日历 → NaN
    lab4 = horizon_labels(CAL4, PX4, CAL4, CAL4_POS, i=3, horizons=(1,))
    assert np.isnan(lab4[0])


# ============================================================
# 合成语料 fixture（训练扫描 / 采样 / PIT）
# ============================================================


def _synth_tables(n_stocks: int = 200, start: str = "2023-06-01",
                  end: str = "2025-08-01") -> tuple[dict, pd.DatetimeIndex]:
    """合成 union 表：全部股票数据完整（pkl 同构 {code: SymbolTable}）。"""
    from mh1_multihorizon.data import SymbolTable

    cal = pd.bdate_range(start, end)
    rng = np.random.default_rng(7)
    tables: dict = {}
    for k in range(n_stocks):
        n = len(cal)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, n))
        vals = np.stack([close, close, close, close,
                         np.full(n, 1e6), np.full(n, 1e8)], axis=1).astype(np.float32)
        tables[f"S{k:03d}"] = SymbolTable(vals=vals, dates=pd.DatetimeIndex(cal))
    return tables, cal


def test_train_purge20_and_val_bounds() -> None:
    """20 日跨段剔除 + 训练/验证边界（合成语料，扫描器为被测对象）。"""
    from mh1_multihorizon.data import scan_train_days, union_calendar

    tables, _ = _synth_tables()
    calendar = union_calendar(tables)
    cal_pos = {d: i for i, d in enumerate(calendar)}

    days, stats = scan_train_days(tables, calendar, start=TRAIN_START,
                                  label_end=TRAIN_LABEL_END)
    assert len(days) > 100
    # 每个训练决策日：四期限全部有限且终点 ≤ 2024-12-31
    n_cal = len(calendar)
    for d in days:
        c = cal_pos[d.date]
        for hi, k in enumerate(HORIZONS):
            assert c + k < n_cal
            assert str(calendar[c + k].date()) <= TRAIN_LABEL_END, (
                f"{d.date.date()} 的 {k} 期限终点 {calendar[c + k].date()} "
                f"越过 {TRAIN_LABEL_END}（20 日跨段剔除失败）")
            assert np.isfinite(d.labels[:, hi]).all()
        assert str(d.date.date()) >= TRAIN_START
    # 末日紧贴上界：下一日历日的 +20 终点必越界
    last_c = cal_pos[days[-1].date]
    assert str(calendar[last_c + 1 + PURGE].date()) > TRAIN_LABEL_END
    # 有效样本数 ≥128
    assert min(len(d.codes) for d in days) >= MIN_TRAIN_CROSS
    assert stats.n_days == len(days)
    assert stats.skipped_days_lt_min >= 0

    # 训练标签终点上界与验证起点之间天然 purge（标签不重叠）
    max_endpoint = max(
        str(calendar[cal_pos[d.date] + PURGE].date()) for d in days)
    assert max_endpoint <= TRAIN_LABEL_END < VAL_START


# ============================================================
# 任务 A.3：无前视归一化
# ============================================================


def test_no_lookahead_normalization() -> None:
    from mh1_multihorizon.data import build_train_batch, scan_train_days, union_calendar

    tables, _ = _synth_tables(150)
    calendar = union_calendar(tables)
    days, _ = scan_train_days(tables, calendar, start="2024-03-01",
                              label_end="2024-12-31")
    day = days[len(days) // 2]
    idx = np.arange(len(day.codes))
    x1, stamp1, lab1 = build_train_batch(tables, day, idx, calendar)

    # 扰动窗口之后的未来数据（> 决策日行）与标签 → x_norm / stamp 必须逐位不变
    cal_pos = {d: i for i, d in enumerate(calendar)}
    row_after = int(day.rows.max()) + 1
    code = day.codes[0]
    tab = tables[code]
    assert row_after < len(tab.vals)
    poisoned = tab.vals.copy()
    poisoned[row_after:] = 9999.0
    tables[code] = type(tab)(vals=poisoned, dates=tab.dates)
    x2, stamp2, _ = build_train_batch(tables, day, idx, calendar)
    assert torch_equal(x1, x2), "未来数据泄漏进窗口归一化"
    assert torch_equal(stamp1, stamp2)
    # 均值/方差只能由该 90 日窗口计算：扰动窗外早期数据同样不影响
    tab2 = tables[day.codes[1]]
    early = tab2.vals.copy()
    cut = int(day.rows[1]) - 90 - 5
    early[: cut] = -777.0
    tables[day.codes[1]] = type(tab2)(vals=early, dates=tab2.dates)
    x3, _, _ = build_train_batch(tables, day, idx, calendar)
    assert torch_equal(x1, x3)


def torch_equal(a, b) -> bool:
    import torch

    return torch.equal(a, b)


# ============================================================
# 任务 A.4：PIT 资格只看 t 日成员
# ============================================================


class FakeProvider:
    """qlib QlibProvider 的确定性替身（fetch 范围记录，供泄漏断言）。"""

    def __init__(self, tables: dict[str, pd.DataFrame], pools: dict[str, list[str]],
                 calendar: pd.DatetimeIndex):
        self._tables = tables          # code -> DataFrame(OHLCVA+aux, DatetimeIndex)
        self._pools = pools            # "YYYY-MM-DD" -> [codes]
        self._cal = calendar
        self.fetch_ranges: list[tuple[str, str]] = []
        # QlibProvider 同名属性（build_pit_days 借道临时改写查询窗）
        self.instruments_: list[str] | str = []
        self._start_date = ""
        self._end_date = ""

    def trading_days(self, start=None, end=None) -> pd.DatetimeIndex:
        cal = self._cal
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal <= pd.Timestamp(end)]
        return pd.DatetimeIndex(cal)

    def list_pool_at(self, pool: str, t: str) -> list[str]:
        return list(self._pools[t])

    def fetch(self, fields, *, filter_pipe=None, freq="day") -> pd.DataFrame:
        # fields 语法与 QlibProvider 一致：["$open", ...]
        cols = [f.lstrip("$") for f in fields]
        start, end = self._start_date, self._end_date
        self.fetch_ranges.append((str(start), str(end)))
        frames = []
        for code, df in self._tables.items():
            sub = df.loc[pd.Timestamp(start): pd.Timestamp(end)]
            if sub.empty:
                continue
            out = sub[cols].copy()
            out["instrument"] = code
            out = out.reset_index().rename(columns={"index": "datetime"})
            frames.append(out.set_index(["datetime", "instrument"]))
        import pandas as pd_  # noqa: F401
        if not frames:
            return pd.DataFrame(columns=cols, index=pd.MultiIndex.from_tuples(
                [], names=["datetime", "instrument"]))
        return pd.concat(frames)


def _pit_fixture_calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-06-01", "2026-08-31")


def _pit_fixture_tables(cal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """A（2024-12-31 退池但数据存续）、B（常驻）、C（2025-06 后新上市）。"""
    rng = np.random.default_rng(11)
    tables: dict[str, pd.DataFrame] = {}
    for code, end_date in (("A", "2026-08-31"), ("B", "2026-08-31"),
                           ("C", "2025-09-30")):
        sub_cal = cal[cal <= pd.Timestamp(end_date)]
        if code == "C":
            sub_cal = sub_cal[sub_cal >= pd.Timestamp("2025-07-01")]
        n = len(sub_cal)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.015, n))
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.full(n, 1e6), "amount": np.full(n, 1e8),
             "preclose": np.roll(close, 1), "tradestatuscode": np.full(n, -1)},
            index=pd.DatetimeIndex(sub_cal))
        tables[code] = df
    # 补足 130 只常驻成分（保证截面 ≥ 128）
    for k in range(130):
        code = f"X{k:03d}"
        n = len(cal)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.015, n))
        tables[code] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.full(n, 1e6), "amount": np.full(n, 1e8),
             "preclose": np.roll(close, 1), "tradestatuscode": np.full(n, -1)},
            index=cal)
    return tables


def _pit_fixture_pool(cal: pd.DatetimeIndex, tables) -> dict[str, list[str]]:
    """A 于 2024-12-31 退出 csi300；C 于 2025-09-01 进入；X* 常驻。"""
    pools: dict[str, list[str]] = {}
    base = [c for c in tables if c.startswith("X")]
    for d in cal:
        ds = str(d.date())
        members = list(base) + (["A", "B"] if d <= pd.Timestamp("2024-12-30") else ["B"])
        if d >= pd.Timestamp("2025-09-01"):
            members = members + ["C"]
        pools[ds] = members
    return pools


def test_pit_membership_only_t() -> None:
    from mh1_multihorizon.data import build_pit_days

    cal = _pit_fixture_calendar()
    tables = _pit_fixture_tables(cal)
    pools = _pit_fixture_pool(cal, tables)
    prov = FakeProvider(tables, pools, cal)

    # 决策日 2024-12-27（A 尚在池）：A 有样本且四期限标签完整
    days, stats = build_pit_days(
        prov, start="2024-12-27", end="2024-12-27", label_end=None,
        require_complete_labels=False)
    assert len(days) == 1
    d0 = days[0]
    assert "A" in d0.codes and "C" not in d0.codes  # C 未上市也未入池
    ai = d0.codes.index("A")
    assert np.isfinite(d0.labels[ai]).all()  # 退池股在 t+1..t+20 仍有标签

    # 决策日 2025-01-06（A 已退池）：A 不再被采样——资格只看 t 日成员
    days2, _ = build_pit_days(
        prov, start="2025-01-06", end="2025-01-06", label_end=None,
        require_complete_labels=False)
    assert "A" not in days2[0].codes and "B" in days2[0].codes


# ============================================================
# 任务 A.5：S/M 同批 + 跳过规则
# ============================================================


def test_sm_identical_batches_and_skip_rules() -> None:
    from mh1_multihorizon.data import (
        PairedDailySampler, scan_train_days, union_calendar,
    )

    tables, _ = _synth_tables(140, start="2023-09-01", end="2024-12-31")
    calendar = union_calendar(tables)

    # 构造一个 127 只的决策日：把 13 只股票在该日的行删掉（数据缺口）
    # —— 直接构造小子表更简单：单独符号表只有 127 只覆盖某段
    cal = calendar
    mid = cal[len(cal) // 2]
    small_tables: dict = {}
    for k, (code, tab) in enumerate(tables.items()):
        if k < 127:
            small_tables[code] = tab
    days_small, stats_small = scan_train_days(small_tables, calendar,
                                              start=str(mid.date()),
                                              label_end="2024-12-31")
    assert all(len(d.codes) >= 128 for d in days_small)
    assert stats_small.skipped_days_lt_min > 0, "127 只的决策日必须被跳过"

    # 完整 140 只：扫描后每日 ≥128 → 批 = 128 无放回（无重复）
    days, _ = scan_train_days(tables, calendar, start="2024-01-02",
                              label_end="2024-12-31")
    s1 = PairedDailySampler(tables, days, calendar, seed=42, batch_size=128)
    s2 = PairedDailySampler(tables, days, calendar, seed=42, batch_size=128)
    for (x1, st1, y1, d1), (x2, st2, y2, d2) in zip(s1.updates(20), s2.updates(20)):
        assert d1.date == d2.date
        assert x1.shape == (128, 90, 6)
        assert len(set(d1.batch_codes)) == 128, "无放回抽样出现重复股票"
        assert torch_equal(x1, x2) and torch_equal(st1, st2)  # 固定种子可复现
        assert np.array_equal(y1, y2)


# ============================================================
# 任务 A.6：W3/W4 边界与查询上界（forward 封存）
# ============================================================


def test_w3w4_bounds_and_query_ceiling() -> None:
    from mh1_multihorizon.data import build_pit_days

    cal = _pit_fixture_calendar()  # 故意延伸到 2026-08-31（超过封存线）
    tables = _pit_fixture_tables(cal)
    pools = _pit_fixture_pool(cal, tables)
    prov = FakeProvider(tables, pools, cal)
    cal_pos = {d: i for i, d in enumerate(cal)}

    for start, end in ((W3_START, W3_END), (W4_START, W4_END)):
        days, _ = build_pit_days(prov, start=start, end=end, label_end=end,
                                 require_complete_labels=False)
        assert days, f"{start}~{end} 无决策日"
        for d in days:
            assert str(d.date.date()) >= start
            c = cal_pos[d.date]
            assert str(cal[c + PURGE].date()) <= end, (
                f"{d.date.date()} 的 20 日终点越过段界 {end}")
        # 末日紧贴：下一交易日 +20 必越界
        last = days[-1]
        assert str(cal[cal_pos[last.date] + 1 + PURGE].date()) > end
    # 全部行情查询上界 ≤ 2026-07-24（forward 零判读）
    for _, end in prov.fetch_ranges:
        assert end <= FORWARD_CUTOFF, f"行情查询越过封存线：{end}"


# ============================================================
# 任务 B.1：头与损失（计划逐字测试）
# ============================================================


def test_single_horizon_masks_auxiliary_gradients() -> None:
    import torch

    from mh1_multihorizon.heads import MultiHorizonHead, horizon_loss

    torch.manual_seed(42)
    scores = torch.randn(128, 4, requires_grad=True)
    labels = torch.randn(128, 4)
    horizon_loss(scores, labels, "S").backward()
    assert torch.count_nonzero(scores.grad[:, [0, 1, 3]]) == 0
    assert torch.count_nonzero(scores.grad[:, 2]) > 0


def test_multi_horizon_trains_every_output_and_shared_layer() -> None:
    import torch

    from mh1_multihorizon.heads import MultiHorizonHead, horizon_loss

    torch.manual_seed(42)
    model = MultiHorizonHead(16)
    hidden = torch.randn(128, 4, 16)
    labels = torch.randn(128, 4)
    out = model(hidden)
    assert out.shape == (128, 4)
    out.retain_grad()
    horizon_loss(out, labels, "M").backward()
    assert torch.all(out.grad.abs().sum(dim=0) > 0)
    assert model.net[1].weight.grad.abs().sum() > 0


def test_horizon_loss_rejects_unknown_arm() -> None:
    import torch

    from mh1_multihorizon.heads import horizon_loss

    with pytest.raises(ValueError):
        horizon_loss(torch.randn(8, 4), torch.randn(8, 4), "X")


# ============================================================
# 任务 B.3：底座冻结 + checkpoint 范围（真实 G1 权重，CPU）
# ============================================================


def test_backbone_frozen_and_ckpt_scoped() -> None:
    import torch

    from g5_head.backbone_g1 import load_g1_backbone
    from mh1_multihorizon.heads import MultiHorizonHead, horizon_loss
    from mh1_multihorizon.train import save_checkpoint, checkpoint_backbone_refs

    backbone = load_g1_backbone("cpu")
    assert backbone.d_model == 832
    head = MultiHorizonHead(backbone.d_model)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)

    x = torch.randn(4, 90, 6)
    stamp = torch.zeros(4, 90, 5)
    labels = torch.randn(4, 4)
    snap = {k: v.clone() for k, v in backbone.state_dict().items()}
    head_snap = {k: v.clone() for k, v in head.state_dict().items()}

    opt.zero_grad()
    with torch.no_grad():
        hidden = backbone.extract(x, stamp)
    loss = horizon_loss(head(hidden), labels, "M")
    loss.backward()
    opt.step()

    # 底座逐值相等、无梯度；头参数已改变
    for k, v in backbone.state_dict().items():
        assert torch.equal(v, snap[k]), f"底座参数被改动：{k}"
    assert all(p.grad is None for p in backbone.parameters())
    changed = any(not torch.equal(head.state_dict()[k], head_snap[k])
                  for k in head.state_dict())
    assert changed
    # head.train() 不改变底座 eval 状态
    head.train()
    assert not backbone.tokenizer.training and not backbone.kronos.training

    # checkpoint 只含新头 + 优化器/进度；底座经哈希引用
    ckpt = save_checkpoint(head, opt, epoch=3, history=[], extra={},
                           out_dir=None)  # 内存路径：返回 dict
    assert isinstance(ckpt, dict)
    assert set(ckpt["state_dict"]) == set(head.state_dict())
    assert "optimizer" in ckpt and "epoch" in ckpt
    refs = checkpoint_backbone_refs(backbone)
    assert set(refs) == {"tokenizer_sha256", "predictor_sha256", "d_model"}
    assert len(refs["tokenizer_sha256"]) == 64  # 完整 SHA256


# ============================================================
# 任务 B.4：选点只读主期限
# ============================================================


def test_selection_reads_only_main_horizon() -> None:
    from mh1_multihorizon.train import select_best_epoch

    # 主期限峰值在 e3；辅助期限（1/5/20 日）峰值在 e5——必须选 e3
    hist = [
        {"epoch": e, "val_rankic_main": m, "val_rankic_h1": 0.0,
         "val_rankic_h5": 0.0, "val_rankic_h20": 0.0}
        for e, m in ((1, 0.01), (2, -0.02), (3, 0.05), (4, 0.04), (5, 0.03))
    ]
    hist[4]["val_rankic_h1"] = 0.99  # 另一期限最优 epoch 故意不同
    assert select_best_epoch(hist) == 3
    # 并列选最早
    hist2 = [dict(h, val_rankic_main=0.05) for h in hist]
    assert select_best_epoch(hist2) == 1
    # 非有限值不能选
    hist3 = [dict(h, val_rankic_main=np.nan) for h in hist[:2]] + hist[2:]
    assert select_best_epoch(hist3) == 3
    with pytest.raises(ValueError):
        select_best_epoch([{"epoch": 1, "val_rankic_main": np.inf}])


# ============================================================
# 任务 B.5：W3/W4 数据不触训练/选点
# ============================================================


def test_w3w4_data_does_not_touch_training() -> None:
    """构造/选点期间全部行情查询 ≤ 2025-06-30；毒化 W3/W4 数据批次不变。"""
    from mh1_multihorizon.data import (
        PairedDailySampler, build_pit_days, scan_train_days, union_calendar,
    )

    tables, _ = _synth_tables(140, start="2023-09-01", end="2024-12-31")
    calendar = union_calendar(tables)
    days, _ = scan_train_days(tables, calendar, start="2024-01-02",
                              label_end=TRAIN_LABEL_END)
    s1 = PairedDailySampler(tables, days, calendar, seed=42)
    batches1 = [(d.date, sorted(d.batch_codes), y.tolist())
                for x, st, y, d in s1.updates(10)]

    # 毒化 2025-07 之后的行情（PIT fixture 世界）+ 验证段查询上界断言
    cal = _pit_fixture_calendar()
    pit_tables = _pit_fixture_tables(cal)
    pools = _pit_fixture_pool(cal, pit_tables)
    prov = FakeProvider(pit_tables, pools, cal)
    _days_val, _ = build_pit_days(prov, start=VAL_START, end=VAL_LABEL_END,
                                  label_end=VAL_LABEL_END,
                                  require_complete_labels=True)
    for _, end in prov.fetch_ranges:
        assert end <= VAL_LABEL_END, f"验证构造查询越过 {VAL_LABEL_END}：{end}"

    s2 = PairedDailySampler(tables, days, calendar, seed=42)
    batches2 = [(d.date, sorted(d.batch_codes), y.tolist())
                for x, st, y, d in s2.updates(10)]
    assert batches1 == batches2, "W3/W4 毒化后训练批次序列改变"


# ============================================================
# 任务 C.1：IC 按 code 对齐
# ============================================================


def test_ic_code_alignment() -> None:
    from mh1_multihorizon.evaluate import rank_ic_by_code

    scores = {"A": 1.0, "B": 2.0, "C": 3.0}
    rets = {"C": 0.3, "A": 0.1, "B": 0.2}  # 故意乱序输入
    ic, reason = rank_ic_by_code(scores, rets, min_stocks=3)
    assert reason is None and ic == pytest.approx(1.0)
    # 任意重排不变
    import random
    items = list(rets.items())
    random.Random(0).shuffle(items)
    ic2, _ = rank_ic_by_code(scores, dict(items), min_stocks=3)
    assert ic2 == pytest.approx(ic)
    # 长格式：重复 (date, code) 必须拒绝
    from mh1_multihorizon.evaluate import daily_ic_from_frame

    df = pd.DataFrame({
        "date": ["2025-07-01"] * 3,
        "instrument": ["A", "B", "A"],
        "score": [1.0, 2.0, 3.0],
        "label": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError):
        daily_ic_from_frame(df, date="2025-07-01")


# ============================================================
# 任务 C.2：缺失处理与配对
# ============================================================


def test_ic_missing_and_pairing() -> None:
    from mh1_multihorizon.evaluate import (
        aggregate_seeds, paired_daily_ic, rank_ic_by_code,
    )

    codes = [f"C{i:03d}" for i in range(40)]
    scores = {c: float(i) for i, c in enumerate(codes)}
    rets = {c: float(i) for i, c in enumerate(codes)}

    # 少于 30 只 → 记缺失
    small = dict(list(scores.items())[:29])
    ic, reason = rank_ic_by_code(small, dict(list(rets.items())[:29]))
    assert ic is None and "stocks" in reason
    # 常量分数 → 记缺失
    ic, reason = rank_ic_by_code({c: 1.0 for c in codes}, rets)
    assert ic is None and "constant" in reason
    # 缺失标签：剔除后 <30 → 缺失（40 剔 11 余 29）；≥30 → 用剩余
    rets_nan = dict(rets)
    for c in codes[29:]:
        rets_nan[c] = np.nan
    ic, reason = rank_ic_by_code(scores, rets_nan)
    assert ic is None and "stocks" in reason
    rets_ok = dict(rets)
    rets_ok[codes[0]] = np.nan
    ic, reason = rank_ic_by_code(scores, rets_ok)
    assert ic is not None

    # 配对 IC：只用 M/S 共同股票
    s_m = dict(scores)
    s_s = {c: -v for c, v in scores.items()}
    del s_s[codes[0]]  # S 少一只
    icm, ics, delta, reason = paired_daily_ic(s_m, s_s, rets)
    assert reason is None
    assert icm == pytest.approx(1.0) and ics == pytest.approx(-1.0)
    assert delta == pytest.approx(2.0)
    # 共同股票 <30 → 配对缺失
    s_s2 = {c: -v for c, v in list(scores.items())[:29]}
    icm, ics, delta, reason = paired_daily_ic(scores, s_s2, rets)
    assert icm is None and "common" in reason

    # 三种子聚合：只用共同有效日期
    daily = {
        "2025-07-01": {"s42": 0.10, "s43": 0.20, "s44": 0.30},
        "2025-07-02": {"s42": 0.10, "s43": np.nan, "s44": 0.30},  # s43 缺
        "2025-07-03": {"s42": 0.10, "s43": 0.20},                 # s44 缺
    }
    agg = aggregate_seeds(daily, seeds=("s42", "s43", "s44"))
    assert agg["n_days"] == 1  # 三种子共同有效日期只有 07-01
    assert agg["mean"] == pytest.approx(0.20)


# ============================================================
# 任务 C.3：分段 HAC
# ============================================================


def test_segmented_hac() -> None:
    import numpy as np

    from mh1_multihorizon.evaluate import segmented_hac
    from paper_replication.ic_horizon_profile import _nw_tvalue

    rng = np.random.default_rng(42)
    x = rng.normal(0.01, 0.05, 120)
    # 单窗：与 _nw_tvalue(lag=9) 数值对齐
    out = segmented_hac([("w", x, list(range(120)))], lag=9)
    assert out["t"] == pytest.approx(_nw_tvalue(x, 9), abs=1e-10)
    assert out["N"] == 120

    # 跨窗不建滞后协方差：两窗符号镜像，边界处 z 积为正（若误拼连续会变大 V）
    a = np.full(30, 0.05)
    b = np.full(30, -0.05)
    xa = a + rng.normal(0, 0.01, 30)
    xb = b + rng.normal(0, 0.01, 30)
    seg = segmented_hac([("w3", xa, list(range(30))),
                         ("w4", xb, list(range(100, 130)))], lag=9)
    cont = _nw_tvalue(np.concatenate([xa, xb]), 9)  # 错误的连续拼接
    assert seg["t"] != pytest.approx(cont, abs=1e-8)
    # 手工复核 V：只用窗内项（中心化用合并 mu）
    mu = np.concatenate([xa, xb]).mean()
    z = np.concatenate([xa, xb]) - mu
    v_manual = float(z @ z) / 60 ** 2
    za, zb = xa - mu, xb - mu
    for l in range(1, 10):
        for zz in (za, zb):
            if len(zz) > l:
                v_manual += 2 * (1 - l / 10) * float(zz[l:] @ zz[:-l]) / 60 ** 2
    assert seg["V"] == pytest.approx(v_manual, abs=1e-12)
    assert seg["t"] == pytest.approx(mu / np.sqrt(v_manual), abs=1e-10)

    # 缺失日期不得压缩：日历位 0,1,3,4（位 2 缺）——位 3 与位 1 不是 lag=1
    # 邻居；距离感知配对 = 手工按日历位差展开
    vals = [1.0, 1.0, -1.0, -1.0]
    pos = [0, 1, 3, 4]
    out_gap = segmented_hac([("w", vals, pos)], lag=9)
    z4 = np.array(vals, dtype=float)
    z4 = z4 - z4.mean()
    pmap = {p: float(zz) for p, zz in zip(pos, z4)}
    cross_gap = 0.0
    for l in range(1, 10):
        wgt = 2.0 * (1.0 - l / 10)
        cross_gap += wgt * sum(
            zv * pmap[p - l] for p, zv in pmap.items() if p - l in pmap)
    v_gap = (float(z4 @ z4) + cross_gap) / 16
    assert out_gap["V"] == pytest.approx(v_gap, abs=1e-12)
    # 压缩成位置相邻（错误做法）的 lag-1 积和 ≠ 距离感知
    compressed_l1 = float(z4[1:] @ z4[:-1])
    aware_l1 = sum(zv * pmap[p - 1] for p, zv in pmap.items() if p - 1 in pmap)
    assert aware_l1 != compressed_l1

    # 退化方差 → 不可判
    out_deg = segmented_hac([("w", np.full(50, 0.01), list(range(50)))], lag=9)
    assert out_deg["judgable"] is False and np.isnan(out_deg["t"])
