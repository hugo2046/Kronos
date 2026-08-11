"""五条预注册判读（计划 §3，跑之前冻结，按顺序，先否决项）。

消费信号层（``analysis_signal_layer.json``）+ 组合层（``analysis_portfolio_layer.json``），
对主口径 ``exst`` 轨逐条核对，输出每条的 pass/fail + 证据。

判读顺序与阈值（计划 §3 原文）：
1. **门禁**：P 组 lo5 的 Q10−Q1 spread 与对档等权超额均 ≈ 0（|年化|<3%）——不过则
   管线有 bug，停止；
2. **信号存在性**：lo5 的 K Q10−Q1（10 期）spread 年化 > 0 且 t > 2；
3. **相对基线**：K 的档内 spread 与对档等权超额须**同时超过 R（反转）同口径**；
4. **单调性**：``lo5 → lo5_10 → mid45_55 → csi300`` 四点上 Kronos 指标单调递减
   （两段单调不全则记"部分成立"）；
5. **经济性**：lo5 经涨跌停 + 停牌 + 成本后的对档等权超额 > 0 才可写"有可实现价值"。

全部判据跑完一次封盘；不做参数搜索。
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from liquidity_strat.common import (
    BUCKETS,
    DATA_DIR,
    HIGH_LIQ_BUCKET,
    SIGNAL_KRONOS,
    SIGNAL_PLACEHOLDER,
    SIGNAL_REV,
    ST_TRACK_MAIN,
)

GATE_THRESHOLD = 0.03  # |年化| < 3%
SIG_T_THRESHOLD = 2.0  # t > 2


def _key(bucket: str, track: str, sig: str) -> str:
    return f"{bucket}|{track}|{sig}"


def _signal_metric(sig_layer: dict, bucket: str, track: str, sig: str) -> dict | None:
    rec = sig_layer.get(_key(bucket, track, sig), {})
    if not rec.get("ok"):
        return None
    return rec["period_10"]


def _portfolio_metric(port_layer: dict, bucket: str, track: str, sig: str) -> dict | None:
    rec = port_layer.get(_key(bucket, track, sig), {})
    if not rec.get("ok"):
        return None
    return rec


def judge(sig_layer: dict, port_layer: dict, *, track: str = ST_TRACK_MAIN) -> dict:
    """跑五条判读，返回结构化结论。"""
    verdicts: dict = {}

    # —— 1. 门禁：P 组 lo5 ——
    p_sig = _signal_metric(sig_layer, "lo5", track, SIGNAL_PLACEHOLDER)
    p_port = _portfolio_metric(port_layer, "lo5", track, SIGNAL_PLACEHOLDER)
    gate_sig_ok = p_sig is not None and abs(p_sig["spread_annualized"]) < GATE_THRESHOLD
    p_excess_ann = p_port["excess_vs_bucket_eq"].get("annual_return") if p_port else None
    gate_port_ok = p_excess_ann is not None and abs(p_excess_ann) < GATE_THRESHOLD
    verdicts["1_gate"] = {
        "desc": "P 组 lo5 的 spread 与对档等权超额均 |年化|<3%",
        "evidence": {
            "p_spread_ann": p_sig["spread_annualized"] if p_sig else None,
            "p_excess_ann": p_excess_ann,
        },
        "pass": bool(gate_sig_ok and gate_port_ok),
        "block_on_fail": True,  # 不过则管线有 bug，停止
    }

    # —— 2. 信号存在性：lo5 K ——
    k_sig = _signal_metric(sig_layer, "lo5", track, SIGNAL_KRONOS)
    if k_sig is None:
        verdicts["2_signal_existence"] = {"pass": False, "reason": "lo5 K 信号层缺失"}
    else:
        exist_ok = k_sig["spread_annualized"] > 0 and k_sig["spread_t_nw"] > SIG_T_THRESHOLD
        verdicts["2_signal_existence"] = {
            "desc": "lo5 K 的 Q10−Q1（10期）spread 年化>0 且 t>2",
            "evidence": {
                "k_spread_ann": k_sig["spread_annualized"],
                "k_spread_t_nw": k_sig["spread_t_nw"],
            },
            "pass": bool(exist_ok),
        }

    # —— 3. 相对基线：K vs R（同 lo5）——
    r_sig = _signal_metric(sig_layer, "lo5", track, SIGNAL_REV)
    k_port = _portfolio_metric(port_layer, "lo5", track, SIGNAL_KRONOS)
    r_port = _portfolio_metric(port_layer, "lo5", track, SIGNAL_REV)
    k_excess = k_port["excess_vs_bucket_eq"].get("annual_return") if k_port else None
    r_excess = r_port["excess_vs_bucket_eq"].get("annual_return") if r_port else None
    k_sa = k_sig["spread_annualized"] if k_sig else None
    r_sa = r_sig["spread_annualized"] if r_sig else None
    rel_ok = (
        k_sa is not None and r_sa is not None and k_excess is not None and r_excess is not None
        and k_sa > r_sa and k_excess > r_excess
    )
    verdicts["3_vs_reversal"] = {
        "desc": "lo5 K 的 spread 与对档等权超额同时 > R（反转）同口径",
        "evidence": {
            "k_spread_ann": k_sa, "r_spread_ann": r_sa,
            "k_excess_ann": k_excess, "r_excess_ann": r_excess,
        },
        "pass": bool(rel_ok),
    }

    # —— 4. 单调性：lo5 → lo5_10 → mid45_55 → csi300 ——
    order = ["lo5", "lo5_10", "mid45_55", HIGH_LIQ_BUCKET]
    spread_series = []
    excess_series = []
    for b in order:
        if b == HIGH_LIQ_BUCKET:
            s = _signal_metric(sig_layer, b, track, SIGNAL_KRONOS)
            p = _portfolio_metric(port_layer, b, track, SIGNAL_KRONOS)
        else:
            s = _signal_metric(sig_layer, b, track, SIGNAL_KRONOS)
            p = _portfolio_metric(port_layer, b, track, SIGNAL_KRONOS)
        spread_series.append(s["spread_annualized"] if s else None)
        excess_series.append(p["excess_vs_bucket_eq"].get("annual_return") if p else None)
    # 严格单调递减判定（忽略 None）
    def _mono_dec(xs):
        vals = [x for x in xs if x is not None]
        if len(vals) < 2:
            return False, 0
        pairs = sum(1 for a, b in zip(vals, vals[1:]) if a > b)
        return pairs == len(vals) - 1, pairs

    sd_full, sd_pairs = _mono_dec(spread_series)
    ed_full, ed_pairs = _mono_dec(excess_series)
    # 两段单调（lo5>lo5_10, mid45_55>csi300）不全则"部分成立"
    def _two_segment(xs):
        a = [x for x in xs if x is not None]
        seg1 = a[0] > a[1] if len(a) > 1 else False
        seg2 = a[2] > a[3] if len(a) > 3 else False
        return seg1, seg2
    s1, s2 = _two_segment(spread_series)
    e1, e2 = _two_segment(excess_series)
    partial = (s1 or s2) or (e1 or e2)
    verdicts["4_monotonicity"] = {
        "desc": "lo5→lo5_10→mid45_55→csi300 上 K 指标单调递减",
        "evidence": {
            "buckets": order,
            "spread_ann": spread_series,
            "excess_ann": excess_series,
        },
        "pass": bool(sd_full or ed_full),
        "partial": bool(partial and not (sd_full or ed_full)),
    }

    # —— 5. 经济性：lo5 K 对档等权超额 > 0 ——
    verdicts["5_economic"] = {
        "desc": "lo5 K 经涨跌停+停牌+成本后的对档等权超额 > 0",
        "evidence": {"k_excess_ann": k_excess},
        "pass": bool(k_excess is not None and k_excess > 0),
    }

    return verdicts


def run_judge() -> dict:
    """读两份 JSON，跑判读，落盘。"""
    sig_path = DATA_DIR / "analysis_signal_layer.json"
    port_path = DATA_DIR / "analysis_portfolio_layer.json"
    if not sig_path.exists() or not port_path.exists():
        raise FileNotFoundError(f"缺分析 JSON：{sig_path} / {port_path}")
    sig_layer = json.loads(sig_path.read_text(encoding="utf-8"))
    port_layer = json.loads(port_path.read_text(encoding="utf-8"))
    verdicts = judge(sig_layer, port_layer)
    out = DATA_DIR / "judgements.json"
    out.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"五条判读落盘：{out}")
    for k, v in verdicts.items():
        status = "PASS" if v.get("pass") else ("部分" if v.get("partial") else "FAIL")
        logger.info(f"  {k}: {status} — {v.get('desc','')}")
    return verdicts


if __name__ == "__main__":
    run_judge()
