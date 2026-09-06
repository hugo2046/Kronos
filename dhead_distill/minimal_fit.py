"""v1.1 rev2 最小拟合试验（docs/DHead复核与最小试验修订要求_20260906.md §3）。

范围冻结：pilot 训练清单前 8 个决策日（≤512 样本，只读复用，不取新行情），
eval 态教师 3-replica 重生成（新 namespace，不动 v1 目录），两可解释对照臂：

- A：eval 教师 + 旧 raw-return 预测接口；
- B：同一 eval 教师 + R2 仿射还原接口（output_space=
  ``normalized_close_affine_return``）。

两臂骨架/初始化种子/日序/优化器一致，各 200 epoch 无早停、纯蒸馏 D；
**末 epoch 为冻结诊断点**；记录 epoch0（优化前）与逐 epoch 指标。
预算上限 1 GPU 时。允许结论见方案 rev2 §3.3（工程诊断，非统计显著性）。
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from loguru import logger

from dhead_distill.config import DHeadConfig, protocol_hash, resolve_env
from dhead_distill.data import DayManifest, content_hash, safe_artifact_dir
from dhead_distill.evaluate import _nmse
from dhead_distill.train import UnifiedTrainer, _package_code_hash

N_DAYS = 8
MAX_EPOCHS = 200
SEED = 42


def _first_n_day_submanifest(m: DayManifest, n_days: int) -> DayManifest:
    """只读切片：前 n_days 个决策日（保持清单顺序与内容 hash 重算）。"""
    days = []
    for s in m.samples:
        d = s.date.strftime("%Y-%m-%d")
        if d not in days:
            days.append(d)
        if len(days) > n_days:
            break
    keep = set(days[:n_days])
    sub = DayManifest(split=m.split, pool=m.pool, profile=m.profile,
                      protocol=m.protocol, list_seed=m.list_seed)
    for s in m.samples:
        d = s.date.strftime("%Y-%m-%d")
        if d not in keep:
            continue
        k = (d, s.code)
        sub.x_raw[k] = m.x_raw[k]
        sub.close_t[k] = m.close_t[k]
        sub.samples.append(s)
    for d in keep:
        for attr in ("x_stamp", "y_stamp", "x_cal", "y_cal"):
            getattr(sub, attr)[d] = getattr(m, attr)[d]
    sub.content_hash = content_hash(sub)
    return sub


def _day_metrics(trainer: UnifiedTrainer) -> dict:
    """按日等权诊断：train D(r0) / E(vs replica1) / 截面方差 / Spearman / 非有限。

    E 用教师 replica1（与 r0 生成独立，DayTensor.y_teacher_r1）；
    R_train 由调用方按同尺度另算。
    """
    from scipy import stats as sps

    from dhead_distill.losses import normalized_mse

    d_list, e_list, var_list, sp_list, nonfinite = [], [], [], [], 0
    for batch in trainer.train_days:
        with torch.no_grad():
            losses = trainer._day_losses(batch)
        pred = losses.pop("pred").detach()
        nonfinite += int(losses["nonfinite"])
        d_list.append(float(losses["d"]))
        e_list.append(float(normalized_mse(
            pred, batch.y_teacher_r1.to(pred.device), trainer.scale_t)))
        sig = pred.mean(axis=-1)
        var_list.append(float(sig.var()))
        t_sig = batch.y_teacher.to(pred.device).mean(axis=-1)
        if float(sig.std()) > 1e-12 and float(t_sig.std()) > 1e-12:
            sp_list.append(float(sps.spearmanr(
                sig.cpu().numpy(), t_sig.cpu().numpy()).statistic))
    n = max(len(d_list), 1)
    return {
        "train_D_r0": float(np.mean(d_list)),
        "E_replica1": float(np.mean(e_list)),
        "signal_cross_var": float(np.mean(var_list)),
        "mean_daily_spearman_vs_r0": float(np.mean(sp_list)) if sp_list else None,
        "n_days_valid_spearman": len(sp_list),
        "n_days": len(d_list),
        "nonfinite": nonfinite,
    }


def run(namespace: str) -> int:
    """执行最小试验（教师重生成 → A/B 两臂 → 指标与结论文档数据落盘）。"""
    from dhead_distill.cli import _g1_weight_hash, _load_manifest, _load_predictor
    from dhead_distill.teacher import TeacherRunner, ensure_teacher_eval

    t_start = time.time()
    env = resolve_env()
    base_cfg = DHeadConfig.with_profile("pilot")
    train_m = _load_manifest("pilot", "train", base_cfg)
    sub = _first_n_day_submanifest(train_m, N_DAYS)
    n_samples = len(sub.samples)
    logger.info(f"[v11rev2] 子清单：{N_DAYS} 日 × {n_samples} 样本 "
                f"hash={sub.content_hash[:12]}（只读自 pilot train）")

    # —— 教师：eval 态重生成（新 namespace，v1 目录零改动）——
    predictor, w_hash = _load_predictor(env)
    ensure_teacher_eval(predictor)
    runner = TeacherRunner(
        manifest=sub, predict_fn=predictor.predict, weight_hash=w_hash,
        n_paths=base_cfg.teacher_n_paths, replicas=base_cfg.teacher_replicas,
        teacher_T=base_cfg.teacher_T, teacher_top_p=base_cfg.teacher_top_p,
        teacher_top_k=base_cfg.teacher_top_k, predict_len=base_cfg.predict_len,
        fork_devices=["cuda:0"], namespace=namespace, model_eval_verified=True,
    )
    runner.run()
    teacher_arr, _ = runner.load_targets_array()  # [N,R,H]
    del predictor  # 释放教师显存
    torch.cuda.empty_cache()

    # —— 冻结尺度：既有 pilot 训练清单 scale（两臂一致，rev2 §3.1）——
    from dhead_distill.cli import _train_scale

    scale = _train_scale(train_m, base_cfg)

    # —— R_train：同 8 日样本 replica1/2 按日等权 MSE_norm ——
    from dhead_distill.data import day_batches

    r_list = []
    idx_ptr = 0
    for batch in day_batches(sub):
        n_b = len(batch)
        r1 = teacher_arr[idx_ptr: idx_ptr + n_b, 1]
        r2 = teacher_arr[idx_ptr: idx_ptr + n_b, 2]
        r_list.append(_nmse(r1, r2, scale))
        idx_ptr += n_b
    R_train = float(np.mean(r_list))
    logger.info(f"[v11rev2] R_train（replica1 vs 2，按日等权）= {R_train:.6f}")

    # —— 全局逐期限均值基线（仅由 8 日训练 target 计算，rev2 §3.3）——
    global_mean = teacher_arr[:, 0, :].mean(axis=0)  # [H]
    gm_days = []
    idx_ptr = 0
    for batch in day_batches(sub):
        n_b = len(batch)
        r1 = teacher_arr[idx_ptr: idx_ptr + n_b, 1]
        gm = np.tile(global_mean, (n_b, 1))
        gm_days.append(_nmse(gm, r1, scale))
        idx_ptr += n_b
    E_global_mean = float(np.mean(gm_days))

    # —— replica 两两相关性（按日等权 Spearman，信号=期限均值）——
    from scipy import stats as sps

    pair_corr = {"r0r1": [], "r0r2": [], "r1r2": []}
    idx_ptr = 0
    for batch in day_batches(sub):
        n_b = len(batch)
        r = teacher_arr[idx_ptr: idx_ptr + n_b]
        idx_ptr += n_b
        for name, (x, y) in (("r0r1", (r[:, 0], r[:, 1])),
                             ("r0r2", (r[:, 0], r[:, 2])),
                             ("r1r2", (r[:, 1], r[:, 2]))):
            a_, b_ = x.mean(-1), y.mean(-1)
            if float(a_.std()) > 1e-12 and float(b_.std()) > 1e-12:
                pair_corr[name].append(
                    float(sps.spearmanr(a_, b_).statistic))
    pair_corr = {k: float(np.mean(v)) for k, v in pair_corr.items()}

    # —— 两臂 ——
    from dhead_distill.backbone import load_g1_student
    from dhead_distill.head import MultiHorizonHead

    results = {}
    for arm_name, output_space in (("A", "raw_return"),
                                   ("B", "normalized_close_affine_return")):
        import dataclasses

        cfg = dataclasses.replace(base_cfg, output_space=output_space)
        device = "cuda:0"
        backbone = load_g1_student(env, device)
        torch.manual_seed(SEED)  # 两臂同种子 → 同初始化
        head = MultiHorizonHead(
            d_model=cfg.d_model, head_dim=cfg.head_dim,
            n_heads=cfg.head_n_heads, n_horizons=cfg.n_horizons,
            calendar_cardinalities=cfg.calendar_cardinalities,
            output_space=output_space,
        ).to(device)
        trainer = UnifiedTrainer(
            arm="D0", cfg=cfg, backbone=backbone, head=head, scale=scale,
            train_manifest=sub, val_manifest=sub,   # 同 8 日：过拟合检验
            train_teacher=teacher_arr, val_teacher=teacher_arr,
            run_name=f"v11rev2-{arm_name}-s{SEED}", seed=SEED, device=device,
            max_epochs_override=MAX_EPOCHS, disable_early_stop=True,
            output_space=output_space, backbone_weight_hash=w_hash,
        )
        epoch0 = _day_metrics(trainer)   # 优化前
        t0 = time.time()
        res = trainer.fit()
        dt = time.time() - t0
        last = res.history[-1]
        final = _day_metrics(trainer)
        results[arm_name] = {
            "output_space": output_space,
            "protocol": protocol_hash(cfg),
            "epochs_run": len(res.history),
            "epoch0": epoch0,
            "final": final,
            "final_train_loss": last["train_loss"],
            "final_grad_norm": last.get("grad_norm"),
            "nonfinite_total": int(sum(h.get("nonfinite", 0)
                                       for h in res.history)),
            "seconds": round(dt, 1),
            "run_dir": str(trainer.run_dir),
            "history": res.history,   # 逐 epoch 全指标
        }
        logger.info(
            f"[v11rev2/{arm_name}] {len(res.history)} epochs {dt:.0f}s | "
            f"epoch0 D={epoch0['train_D_r0']:.4f} → 末 D={final['train_D_r0']:.4f} "
            f"| E(rep1)={final['E_replica1']:.4f} | R_train={R_train:.4f} | "
            f"D≤R_train: {'YES' if final['train_D_r0'] <= R_train else 'NO'}"
        )
        del backbone, head, trainer
        torch.cuda.empty_cache()

    # —— 汇总落盘（允许结论按 rev2 §3.3 措辞，工程诊断非显著性）——
    eng = {
        a: {
            "last_epoch_D_le_R_train": bool(
                results[a]["final"]["train_D_r0"] <= R_train),
            "D": results[a]["final"]["train_D_r0"],
        } for a in results
    }
    doc = {
        "experiment": "v1.1 rev2 minimal fit",
        "requirements_ref": "docs/DHead复核与最小试验修订要求_20260906.md",
        "namespace": namespace,
        "n_days": N_DAYS, "n_samples": n_samples, "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "sub_manifest_hash": sub.content_hash,
        "pilot_manifest_hash": train_m.content_hash,
        "teacher_weight_hash": w_hash,
        "code_hash": _package_code_hash(),
        "scale": scale.tolist(),
        "R_train": R_train,
        "E_global_mean_baseline": E_global_mean,
        "replica_pair_spearman": pair_corr,
        "engineering_check": eng,
        "arms": {a: {k: v for k, v in r.items() if k != "history"}
                 for a, r in results.items()},
        "history_A": results["A"]["history"],
        "history_B": results["B"]["history"],
        "gpu_seconds_total": round(time.time() - t_start, 1),
    }
    out_dir = safe_artifact_dir(f"v11rev2-minimal-fit-{namespace}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(out_dir / "summary.json")
    logger.info(f"[v11rev2] 汇总 → {out_dir.name}/summary.json "
                f"(GPU {doc['gpu_seconds_total']:.0f}s / 上限 3600s)")
    return 0


__all__ = ["run", "N_DAYS", "MAX_EPOCHS", "SEED"]
