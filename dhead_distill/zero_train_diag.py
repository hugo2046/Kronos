"""零训练归因诊断（docs/DHead_rev2复核与零训练诊断交接_20260906.md §3，口径冻结）。

回答且只回答：B 臂相比**自身未训练头（B_init）**与 **z=0 仿射基线（B0）**
到底新增了多少教师拟合能力——分离"接口自带的确定性信号"与"训练增量"。

六个预测器（同 8 日/512 样本、同 scale、同 eval 教师 replica0/1/2，
按日等权，统一原始收益率尺度）：

- G   全局逐期限均值（只从 8 日 replica0 计算）；
- Z   恒零收益（价格持平参照）；
- B0  标准化 close 预测恒 0 → 收益恒为 b_i（逐股变化、逐 horizon 相同）；
- B_init seed42 与原 B 完全相同的初始化权重，零梯度步（不加载旧 ckpt，
  构造顺序与 e5f10c3 一致；报告 state_dict hash）；
- B_final 原 B 第 199 epoch checkpoint（末点）；
- A_final 原 A 第 199 epoch checkpoint。

纪律：只读既有资产（run/teacher/manifest 原样）；不调用 trainer.fit；
不重建教师；常量零方差相关记 NA；配对差为事后描述，无显著性检验；
诊断代码 hash 与历史训练代码 hash 分别记录，不冒充。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy import stats as sps

from dhead_distill.config import DHeadConfig, resolve_env
from dhead_distill.data import (
    DayManifest, affine_restore_params, day_batches, safe_artifact_dir,
)
from dhead_distill.evaluate import _nmse
from dhead_distill.minimal_fit import N_DAYS, SEED, _first_n_day_submanifest
from dhead_distill.train import _materialize_days, _package_code_hash

LAST_EPOCH = 199  # 200 epoch 的末点（冻结诊断点，与原试验一致）


def _state_hash(sd: dict) -> str:
    """state_dict 内容 hash（B_init 重建的可复核指纹）。"""
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode("utf-8"))
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _verify_historical_run(run_dir: Path, arm: str, sub: DayManifest,
                           teacher: np.ndarray) -> dict:
    """§4.1：显式 run 路径严格归属校验——须属于指定试验 manifest。

    run 身份中的 ``teacher_hash`` 是**教师目标数组 hash**（UnifiedTrainer
    按 ``train+val 目标字节拼接`` 计算；minimal_fit 中 train=val=子清单，
    即 ``sha256(arr+arr)``），不是 G1 权重指纹——两者语义不同，此处按
    数组口径重算比对；G1 权重绑定由教师分片 run manifest 的 weight_hash
    承担（run_diag 层另行核验）。
    """
    res_path = run_dir / "result.json"
    if not res_path.exists():
        raise FileNotFoundError(f"run 缺 result.json：{run_dir}")
    res = json.loads(res_path.read_text("utf-8"))
    ident = res.get("identity", {})
    arr_hash = hashlib.sha256(
        teacher.tobytes() + teacher.tobytes()).hexdigest()
    expect = {
        "arm": "D0",  # 两臂均以 D0（纯蒸馏）损失语义训练（minimal_fit 固定）
        "seed": SEED,
        "train_hash": sub.content_hash,
        "teacher_hash": arr_hash,
    }
    for k, v in expect.items():
        if ident.get(k) != v:
            raise RuntimeError(
                f"历史 run 归属校验失败（{run_dir.name}，{k}："
                f"{ident.get(k)!r} ≠ {v!r}）——不属于指定试验 manifest"
            )
    want_os = "normalized_close_affine_return" if arm == "B" else "raw_return"
    if ident.get("output_space") != want_os:
        raise RuntimeError(
            f"历史 run 输出语义不符：{ident.get('output_space')!r} ≠ {want_os!r}"
        )
    return {
        "run_dir": run_dir.name,
        "historical_code_hash": ident.get("code_hash"),
        "epochs": len(res.get("history", [])),
        "best_epoch": res.get("best_epoch"),
        "output_space": ident.get("output_space"),
    }


def _metrics_for_predictor(
    name: str, pred_by_idx: dict[int, np.ndarray], days: list[dict],
    scale: np.ndarray,
) -> dict:
    """单预测器全指标：D/E1/E2 + 对 r0/r1/r2 逐日 Spearman + 截面方差。

    常量（零方差）相关记 NA 且不计入有效日——不能记成有效 0。
    """
    d_list, e1_list, e2_list, var_list = [], [], [], []
    sp = {"r0": [], "r1": [], "r2": []}
    per_day = []
    for day in days:
        idxs = day["idxs"]
        p = np.stack([pred_by_idx[i] for i in idxs]).astype(np.float64)
        r0, r1, r2 = day["teacher"][0], day["teacher"][1], day["teacher"][2]
        d_ = _nmse(p, r0, scale)
        e1_ = _nmse(p, r1, scale)
        e2_ = _nmse(p, r2, scale)
        d_list.append(d_)
        e1_list.append(e1_)
        e2_list.append(e2_)
        sig = p.mean(axis=-1)
        var_list.append(float(sig.var()))
        day_sp = {}
        for tag, rep in (("r0", r0), ("r1", r1), ("r2", r2)):
            t_sig = rep.mean(axis=-1)
            if float(sig.std()) < 1e-12 or float(t_sig.std()) < 1e-12:
                day_sp[tag] = None  # NA：常量输入，不计有效日
            else:
                v = float(sps.spearmanr(sig, t_sig).statistic)
                day_sp[tag] = v
                sp[tag].append(v)
        per_day.append({"date": day["date"], "D": d_, "E1": e1_, "E2": e2_,
                        "spearman": day_sp, "signal_var": float(sig.var())})

    def _m(x):
        return float(np.mean(x)) if x else None

    return {
        "predictor": name,
        "D": _m(d_list), "E1": _m(e1_list), "E2": _m(e2_list),
        "signal_cross_var": _m(var_list),
        "spearman_r0": _m(sp["r0"]), "spearman_r1": _m(sp["r1"]),
        "spearman_r2": _m(sp["r2"]),
        "valid_days_r0": len(sp["r0"]), "valid_days_r1": len(sp["r1"]),
        "valid_days_r2": len(sp["r2"]), "n_days": len(days),
        "per_day": per_day,
    }


def _fp32_tolerance_check(sub: DayManifest, teacher: np.ndarray,
                          n_check: int = 64) -> dict:
    """§4.3：FP32 真实链路容差（数值诊断，不改历史数值）。

    1.7e-16 是 **float64 公式自洽**精度；KronosPredictor 内部张量为
    float32、教师分片以 float32 落盘，学生侧 ``r=a·z+b`` 亦在 float32
    链路计算。本检查用**真实教师输出的等效 z 值**（由 r0 反解，float64）
    与真实 a/b，量化 float32 还原 vs float64 公式的舍入差。
    """
    max_abs, max_rel, checked = 0.0, 0.0, 0
    for i, s in enumerate(sub.samples[:n_check]):
        k = (s.date.strftime("%Y-%m-%d"), s.code)
        close = sub.x_raw[k][:, 3].astype(np.float64)
        mu, denom, ct = close.mean(), close.std() + 1e-5, float(close[-1])
        a64, b64 = affine_restore_params(sub.x_raw[k])
        y0 = teacher[i, 0].astype(np.float64)  # [H] 教师真实输出（float32 存储）
        z = ((y0 + 1.0) * ct - mu) / denom     # 等效标准化 close（float64）
        r64 = a64 * z + b64                     # float64 参照
        r32 = (torch.tensor(a64, dtype=torch.float32)
               * torch.tensor(z, dtype=torch.float32)
               + torch.tensor(b64, dtype=torch.float32)).numpy().astype(np.float64)
        diff = np.abs(r32 - r64)
        rel = diff / np.maximum(np.abs(r64), 1e-12)
        max_abs = max(max_abs, float(diff.max()))
        max_rel = max(max_rel, float(rel.max()))
        checked += 1
    return {
        "n_samples_checked": checked,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "note": "float32 仿射还原 vs float64 公式（真实教师输出等效 z）舍入差",
    }


def run_diag(teacher_namespace: str, run_a: str, run_b: str,
             summary_name: str) -> int:
    """执行零训练归因诊断（只读既有 run；产物写 diag namespace 目录）。"""
    from dhead_distill.cli import _load_manifest, _train_scale
    from dhead_distill.head import MultiHorizonHead
    from dhead_distill.teacher import TeacherRunner

    t0 = time.time()
    env = resolve_env()
    cfg = DHeadConfig.with_profile("pilot")
    diag_code_hash = _package_code_hash()

    # —— 数据（严格 rev2 同源：同 8 日/512 样本/scale/教师分片）——
    train_m = _load_manifest("pilot", "train", cfg)
    sub = _first_n_day_submanifest(train_m, N_DAYS)
    teacher = TeacherRunner.load_verified(
        sub, replicas=cfg.teacher_replicas, predict_len=cfg.predict_len,
        namespace=teacher_namespace, n_paths=cfg.teacher_n_paths,
        teacher_T=cfg.teacher_T, teacher_top_p=cfg.teacher_top_p,
        teacher_top_k=cfg.teacher_top_k,
    ).load_targets_array()[0]  # [N,R,H]
    scale = _train_scale(train_m, cfg)

    # —— 历史试验归属校验（§4.1：显式路径 + 严格身份）——
    summary_dir = safe_artifact_dir(summary_name)
    exp = json.loads((summary_dir / "summary.json").read_text("utf-8"))
    run_dirs = {"A": safe_artifact_dir(run_a), "B": safe_artifact_dir(run_b)}
    provenance = {
        arm: _verify_historical_run(d, arm, sub, teacher)
        for arm, d in run_dirs.items()
    }
    # 教师分片 run manifest 的 G1 权重指纹须与试验 summary 绑定一致
    teacher_run = json.loads(
        (safe_artifact_dir(
            f"teacher-{teacher_namespace}-pilot-train-{sub.content_hash[:12]}")
         / "teacher_run.json").read_text("utf-8"))
    if teacher_run.get("weight_hash") != exp.get("teacher_weight_hash"):
        raise RuntimeError(
            "教师分片权重指纹与试验 summary 不一致——拒绝诊断（防混配）"
        )
    ck_paths = {arm: d / f"epoch-{LAST_EPOCH}.pt" for arm, d in run_dirs.items()}
    for arm, p in ck_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"末点 checkpoint 缺失：{p}")
    ck_hashes = {arm: _file_hash(p) for arm, p in ck_paths.items()}

    # —— 逐日结构（与 manifest 顺序一致；day_batches 产出 Sample 列表）——
    days: list[dict] = []
    ptr = 0
    for batch in day_batches(sub):
        n_b = len(batch)
        days.append({
            "date": batch[0].date.strftime("%Y-%m-%d"),
            "idxs": list(range(ptr, ptr + n_b)),
            "teacher": np.stack(
                [teacher[ptr: ptr + n_b, r, :] for r in range(3)]),
        })
        ptr += n_b

    # —— 共享底座隐状态（eval 前向，一次算好三头复用）——
    from dhead_distill.backbone import load_g1_student

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    backbone = load_g1_student(env, device)
    batches = _materialize_days(sub, teacher)
    with torch.no_grad():
        hidden_by_day = [
            backbone.extract(b.x_norm.to(device), b.x_stamp.to(device))
            for b in batches
        ]
    del backbone
    torch.cuda.empty_cache()

    def _mk_head(output_space: str) -> "MultiHorizonHead":
        return MultiHorizonHead(
            d_model=cfg.d_model, head_dim=cfg.head_dim,
            n_heads=cfg.head_n_heads, n_horizons=cfg.n_horizons,
            calendar_cardinalities=cfg.calendar_cardinalities,
            output_space=output_space,
        ).to(device)

    def _forward_head(head, affine: bool) -> dict[int, np.ndarray]:
        out, sample_ptr, day_ptr = {}, 0, 0
        head.eval()
        with torch.no_grad():
            for b in batches:
                ys = b.y_stamp.to(device)
                p = (head(hidden_by_day[day_ptr], ys,
                          b.a.to(device), b.b.to(device))
                     if affine else head(hidden_by_day[day_ptr], ys))
                arr = p.cpu().numpy()
                for j in range(len(b.y_real)):
                    out[sample_ptr + j] = arr[j]
                sample_ptr += len(b.y_real)  # 样本偏移，非批序号
                day_ptr += 1
        return out

    n = len(sub.samples)
    H = cfg.predict_len
    # —— 六预测器（冻结口径）——
    g_mean = teacher[:, 0, :].mean(axis=0)             # G：全局逐期限均值
    preds: dict[str, dict[int, np.ndarray]] = {
        "G": {i: g_mean.copy() for i in range(n)},
        "Z": {i: np.zeros(H) for i in range(n)},       # Z：恒零收益
    }
    b0 = {}
    for i, s in enumerate(sub.samples):                # B0：z=0 → r=b_i
        _, b_i = affine_restore_params(
            sub.x_raw[(s.date.strftime("%Y-%m-%d"), s.code)])
        b0[i] = b_i
    preds["B0"] = {i: np.full(H, b0[i]) for i in range(n)}
    # B_init：seed42 与原 B 同构造顺序重建（不加载任何旧 ckpt）
    torch.manual_seed(SEED)
    head_b_init = _mk_head("normalized_close_affine_return")
    b_init_hash = _state_hash(head_b_init.state_dict())
    preds["B_init"] = _forward_head(head_b_init, affine=True)
    # B_final / A_final：末点 checkpoint 只读装载
    head_b = _mk_head("normalized_close_affine_return")
    head_b.load_state_dict(
        torch.load(ck_paths["B"], weights_only=True)["head"])
    preds["B_final"] = _forward_head(head_b, affine=True)
    head_a = _mk_head("raw_return")
    head_a.load_state_dict(
        torch.load(ck_paths["A"], weights_only=True)["head"])
    preds["A_final"] = _forward_head(head_a, affine=False)

    # —— 指标与配对差（事后描述，无检验）——
    metrics = {
        name: _metrics_for_predictor(name, pr, days, scale)
        for name, pr in preds.items()
    }
    paired = {}
    for x, y in (("B_final", "B_init"), ("B_final", "B0")):
        d_diff = [metrics[x]["per_day"][k]["D"] - metrics[y]["per_day"][k]["D"]
                  for k in range(len(days))]
        e1_diff = [metrics[x]["per_day"][k]["E1"] - metrics[y]["per_day"][k]["E1"]
                   for k in range(len(days))]
        paired[f"{x}-{y}"] = {
            "D_diff_mean": float(np.mean(d_diff)),
            "E1_diff_mean": float(np.mean(e1_diff)),
            "D_diff_per_day": [float(v) for v in d_diff],
            "E1_diff_per_day": [float(v) for v in e1_diff],
            "note": "负差=改善；事后描述，无显著性检验",
        }

    # —— R_train（照原报告：同日 r1/r2 按日等权）——
    R_train = float(np.mean([
        _nmse(teacher[d["idxs"], 1], teacher[d["idxs"], 2], scale)
        for d in days
    ]))

    doc = {
        "experiment": "DHead rev2 零训练归因诊断",
        "requirements_ref": "docs/DHead_rev2复核与零训练诊断交接_20260906.md",
        "frozen_protocol": "同 rev2 8日/512样本/scale/教师replica；只读；无训练",
        "diag_code_hash": diag_code_hash,
        "historical_code_hash": provenance["A"]["historical_code_hash"],
        "provenance": provenance,
        "selected_ckpt_sha256": ck_hashes,
        "b_init_state_dict_sha256": b_init_hash,
        "teacher_namespace": teacher_namespace,
        "sub_manifest_hash": sub.content_hash,
        "R_train": R_train,
        "predictors": {k: {kk: vv for kk, vv in v.items() if kk != "per_day"}
                       for k, v in metrics.items()},
        "per_day_detail": {k: v["per_day"] for k, v in metrics.items()},
        "paired_diffs": paired,
        "fp32_tolerance_check": _fp32_tolerance_check(sub, teacher),
        "wall_seconds": round(time.time() - t0, 1),
        "trainer_fit_called": False,
    }
    out_dir = safe_artifact_dir(f"diag-zero-train-{teacher_namespace}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(out_dir / "summary.json")
    logger.info(f"[diag] 六预测器完成（wall {doc['wall_seconds']}s）→ "
                f"{out_dir.name}/summary.json")
    for k in ("G", "Z", "B0", "B_init", "B_final", "A_final"):
        m = metrics[k]
        logger.info(
            f"[diag] {k:8s} D={m['D']:.4f} E1={m['E1']:.4f} "
            f"sp_r0={m['spearman_r0']} var={m['signal_cross_var']:.2e}"
        )
    return 0


__all__ = ["run_diag", "LAST_EPOCH"]
