"""G10 工程收尾只读审计 CLI（20260909 计划 §4/§7/§8）。

零训练、零推理、零 GPU、不读 forward；只读源 run 目录的协议/历史/checkpoint/
信号/日志，输出独立 closeout 目录：资产 manifest、状态×窗口证据表（区分
checkpoint 文件 SHA 与规范化 state_dict 哈希；生成链证据等级）、A0 两轮
对比、预算 ledger 与对账、研究证据矩阵与去留建议、audit_summary。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from loguru import logger

from g10_head_pilot.runtime_state import normalized_state_hash, sha256_file

STATES = ("A0_bestCE", "A0_e15", "A1_bestCE", "A1_e15")
WINDOWS = ("W3", "W4")


def _ckpt_info(path: Path) -> dict:
    """checkpoint 文件 SHA + 规范化 state_dict 数值哈希（CPU 逐份加载释放）。"""
    if not path.is_file():
        return {"file": str(path), "status": "missing"}
    info = {"file": str(path), "file_sha256": sha256_file(path),
            "size": path.stat().st_size}
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        sd = ck.get("predictor", ck)
        info["state_dict_sha256"] = normalized_state_hash(sd)
        info["epoch_meta"] = ck.get("meta", {}).get("epoch",
                                                    ck.get("epoch"))
    except Exception as e:                     # noqa: BLE001 — 审计如实记录
        info["state_dict_sha256"] = None
        info["load_error"] = str(e)[:120]
    return info


def audit(source: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    assert source.is_dir() and source != out and out not in source.parents

    # —— 任务 A：资产 manifest（前后哈希对拍基准）——
    assets: dict[str, dict] = {}
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".log", ".parquet"}:
            assets[str(p.relative_to(source))] = {
                "sha256": sha256_file(p), "size": p.stat().st_size}
    for arm in ("A0", "A1", "A0_round1_invalid"):
        for f in ("final.pt", "best.pt", "resume.pt"):
            p = source / arm / f
            if p.is_file():
                assets[f"{arm}/{f}"] = {
                    "sha256": sha256_file(p), "size": p.stat().st_size}

    # —— 状态×窗口证据表 + 生成链等级 ——
    ev_rows = []
    for state in STATES:
        arm, kind = state.split("_")
        ep = {"A0_bestCE": 1, "A0_e15": 15, "A1_bestCE": 15, "A1_e15": 15}[state]
        ck = _ckpt_info(source / arm / ("best.pt" if kind == "bestCE"
                                        and ep != 15 else "final.pt"))
        for w in WINDOWS:
            sig = source / "signals" / f"{w}_{state}.parquet"
            ev_rows.append({
                "state": state, "window": w, "label_epoch": ep,
                "checkpoint": ck,
                "signal_sha256": sha256_file(sig) if sig.is_file() else None,
                # 旧日志无"权重哈希→信号哈希"链 → 只能标未验证（不追认）
                "generation_link": "legacy_unverified",
                "alias": state == "A1_bestCE",
            })
    # —— A0 两轮对比（近似数值复现核对）——
    def _hist(arm: str) -> dict:
        return json.loads((source / arm / "history.json").read_text(
            encoding="utf-8"))
    h1 = _hist("A0_round1_invalid")
    h2 = _hist("A0")
    ce_diffs = [abs(a["val_ce"] - b["val_ce"])
                for a, b in zip(h1["history"], h2["history"])]
    tl_diffs = [abs(a["train_loss"] - b["train_loss"])
                for a, b in zip(h1["history"], h2["history"])]
    a0_cmp = {
        "max_abs_val_ce_diff": max(ce_diffs),
        "max_abs_train_loss_diff": max(tl_diffs),
        "batch_digests_identical": [a["batch_digest"] for a in h1["history"]]
        == [b["batch_digest"] for b in h2["history"]],
        "lr_sha_identical": h1["lr_sha"] == h2["lr_sha"],
        "wording": "近似数值复现（非逐位）：max CE 差 "
                   f"{max(ce_diffs):.3e}；逐位相同的仅为批次摘要/LR 轨迹",
        "round1_ckpt_state_hash": "unknown（round1 仅存 e15 权重，"
                                  "e1 best 已丢失）",
    }

    # —— 预算对账 ——
    budget = json.loads((source / "gpu_budget.json").read_text(encoding="utf-8"))
    ledger = {
        "counter_seconds": budget["compute_seconds"],
        "counter_hours": round(budget["compute_seconds"] / 3600, 4),
        "coverage": "smoke+首轮两臂训练+首轮全部推理+A0 重训+A0_bestCE 重推+CPU 收尾（累计器口径）",
        "caveat": "累计器混入 CPU 评价计时；仅墙钟日志可证——不称已独立核验 GPU 实计",
        "guard_wait_excluded": "16:30 守卫等待未计入（按设计）",
    }

    audit_summary = {
        "historical_protocol_conformance": False,
        "reason": "RNG 隔离位置不符合预注册（epoch 开头保存+重置，验证前未独立"
                  "重置，未保存/恢复 CUDA RNG；checkpoint 缺随机状态）——已修复"
                  "并由合成回归测试锁定，历史数字不重算不改判",
        "assets_complete": None,   # 由 manifest 行数陈述，不给单一布尔
        "n_assets_hashed": len(assets),
        "generation_links": "全部 legacy_unverified（日志无权重哈希→信号哈希链）",
        "budget_status": "累计 10.9447h 有账，完整占用未独立核验",
        "engineering_fixes": "runtime-v2：验证边界 RNG 隔离/版本化 epoch+best "
                             "checkpoint（原子写+完整随机状态）/cache 身份门禁/"
                             "协议失效关闭；29 项测试通过",
    }
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": str(source),
        "assets": assets, "evidence_table": ev_rows,
        "a0_two_round_compare": a0_cmp, "budget": budget, "ledger": ledger,
        "audit_summary": audit_summary,
    }
    (out / "audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    logger.info(f"[closeout] {len(assets)} 件资产哈希；{len(ev_rows)} 行证据表；"
                f"A0 两轮 max CE 差 {max(ce_diffs):.3e}；累计器 "
                f"{budget['compute_seconds'] / 3600:.4f}h")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="G10 只读收尾审计")
    parser.add_argument("--source-run", required=True, type=Path)
    args = parser.parse_args()
    src = args.source_run.resolve()
    if not src.is_dir():
        raise RuntimeError(f"源 run 目录不存在：{src}")
    out = Path(__file__).resolve().parent / "data" / "closeout_20260909"
    logger.add(str(out / "closeout.log"), enqueue=False)
    audit(src, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
