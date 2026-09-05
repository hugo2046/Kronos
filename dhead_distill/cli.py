"""DHead v1 统一 CLI 与阶段门禁（方案 §6/§8）。

命令（均支持 ``--help``；未知臂/非法阶段显式报错）::

    python -m dhead_distill.cli preflight
    python -m dhead_distill.cli prepare --profile pilot [--splits train,val]
    python -m dhead_distill.cli teacher --profile pilot [--splits train,val]
    python -m dhead_distill.cli train --profile pilot --arm D0 --seed 42
    python -m dhead_distill.cli evaluate --profile pilot --stage fidelity
    python -m dhead_distill.cli evaluate --profile main --stage prediction
    python -m dhead_distill.cli evaluate --profile main --stage economic

阶段门禁（§8）：D2/D1-cont 须先通过 ``evaluate --stage fidelity`` 写出的
解锁文件；economic 在引擎独立复核完成前拒绝输出可交易结论。
产物统一写外置 ``DHEAD_ARTIFACT_ROOT``；仓库只保存代码/方案/脱敏摘要。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from dhead_distill.config import (
    DHeadConfig, PROFILES, load_base_env, protocol_hash, resolve_env,
)


# ======================================================================
# 通用工具
# ======================================================================


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """文件 SHA256（分块读；G1 权重指纹）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_json(path: Path, doc: dict) -> None:
    """原子写 JSON（同目录临时文件 + rename）。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(path)


def _artifact_root() -> Path:
    env = resolve_env()
    env.artifact_root.mkdir(parents=True, exist_ok=True)
    return env.artifact_root


def _load_manifest(profile: str, split: str) -> "object":
    """按 profile/split 找最新 prepare 产物并装载（协议校验在身份环节）。"""
    from dhead_distill.data import DayManifest

    root = _artifact_root()
    cands = sorted(
        p for p in root.glob(f"prepare-{profile}-{split}-*") if p.is_dir()
    )
    if not cands:
        raise FileNotFoundError(
            f"未找到 prepare 产物：prepare-{profile}-{split}-*（先运行 prepare）"
        )
    return DayManifest.load(cands[-1].name)


def _require_gpu() -> str:
    """GPU 预检（§8.1：不得在仅 CPU 的机器启动完整实验）。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：训练/教师命令拒绝在 CPU 机运行（§8.1）")
    return "cuda:0"


# ======================================================================
# preflight
# ======================================================================


def cmd_preflight(args: argparse.Namespace) -> int:
    """资产与环境预检（只读；DDB 探测仅用 2024 历史一小段，不触封存线）。"""
    env = resolve_env()
    report: dict = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_repo": str(env.base_repo),
        "worktree": str(env.worktree),
        "artifact_root": str(env.artifact_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    problems: list[str] = []

    for name, p in (("g1_tokenizer", env.g1_tokenizer),
                    ("g1_predictor", env.g1_predictor)):
        st = p.is_dir()
        entry = {"path": str(p), "exists": st}
        if st:
            safetensors = p / "model.safetensors"
            entry["model_safetensors"] = {
                "exists": safetensors.is_file(),
                "sha256": _sha256_file(safetensors) if safetensors.is_file() else None,
                "size_bytes": safetensors.stat().st_size if safetensors.is_file() else 0,
            }
        else:
            problems.append(f"{name} 缺失：{p}")
        report[name] = entry

    try:
        env.artifact_root.mkdir(parents=True, exist_ok=True)
        du = shutil.disk_usage(env.artifact_root)
        report["artifact_root_free_gb"] = round(du.free / 1e9, 1)
    except OSError as e:
        problems.append(f"产物根不可创建/不可写：{e}")

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        else:
            problems.append("CUDA 不可用：训练/教师命令将被拒绝")
    except Exception as e:  # pragma: no cover
        problems.append(f"torch 不可用：{e}")

    try:
        load_base_env(env)
        from kronos_qlib.provider import QlibProvider

        probe = QlibProvider("csi300", "2024-01-02", "2024-01-31")
        df = probe.fetch(["$close"], freq="day")
        report["ddb_adapter"] = {"ok": len(df) > 0, "probe_rows": int(len(df))}
        if len(df) == 0:
            problems.append("DDB 探测返回空数据")
    except Exception as e:
        report["ddb_adapter"] = {"ok": False, "error": str(e)[:200]}
        problems.append(f"DDB 适配器不可用：{e}")

    report["problems"] = problems
    report["ok"] = not problems
    root = _artifact_root()
    out = root / f"preflight-{time.strftime('%Y%m%dT%H%M%S')}.json"
    _write_json(out, report)
    logger.info(f"preflight {'OK' if report['ok'] else 'FAIL'} → {out.name}")
    for p in problems:
        logger.warning(f"preflight 问题：{p}")
    return 0 if report["ok"] else 2


# ======================================================================
# prepare
# ======================================================================


def cmd_prepare(args: argparse.Namespace) -> int:
    """构造冻结清单（train/val/diag/replay），落盘并回读校验内容 hash。"""
    if args.profile not in PROFILES:
        logger.error(f"未知 profile：{args.profile}")
        return 2
    cfg = DHeadConfig.with_profile(args.profile)
    env = resolve_env()
    load_base_env(env)
    from kronos_qlib.provider import QlibProvider

    provider = QlibProvider("csi300", "2014-01-02", "2026-07-24")
    splits = args.splits.split(",") if args.splits else \
        ["train", "val", "diag", "replay"]
    for split in splits:
        from dhead_distill.data import DayManifest, build_manifest, content_hash

        m = build_manifest(provider, cfg, split)
        name = f"prepare-{cfg.profile}-{split}-{m.content_hash[:12]}"
        d = m.save(name)
        m2 = DayManifest.load(name)
        ok = content_hash(m2) == m.content_hash
        logger.info(
            f"prepare[{split}] 样本={m.stats['n_samples']} 日={m.stats['n_days']} "
            f"剔除={m.stats['skipped']} 弃日={len(m.stats['dropped_days'])} "
            f"hash={m.content_hash[:12]} 回读={'OK' if ok else 'FAIL'} → {d.name}"
        )
        if not ok:
            return 3
    return 0


# ======================================================================
# teacher
# ======================================================================


def _load_predictor(env) -> tuple:
    """装载 G1 tokenizer+predictor（只读；返回 (predictor, 联合权重hash)）。"""
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G1 tokenizer（只读）：{env.g1_tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(str(env.g1_tokenizer))
    logger.info(f"加载 G1 predictor（只读）：{env.g1_predictor}")
    model = Kronos.from_pretrained(str(env.g1_predictor))
    device = _require_gpu()
    predictor = KronosPredictor(
        model.to(device), tokenizer.to(device), device=device, max_context=512,
    )
    w_hash = hashlib.sha256(
        _sha256_file(env.g1_tokenizer / "model.safetensors").encode()
        + _sha256_file(env.g1_predictor / "model.safetensors").encode()
    ).hexdigest()
    return predictor, w_hash


def cmd_teacher(args: argparse.Namespace) -> int:
    """教师 3-replica 生成（真实 G1 只在本命令内加载）。"""
    cfg = DHeadConfig.with_profile(args.profile)
    env = resolve_env()
    predictor, w_hash = _load_predictor(env)
    splits = args.splits.split(",") if args.splits else ["train", "val"]
    for split in splits:
        from dhead_distill.teacher import TeacherRunner

        m = _load_manifest(args.profile, split)
        runner = TeacherRunner(
            manifest=m,
            predict_fn=predictor.predict,
            weight_hash=w_hash,
            n_paths=cfg.teacher_n_paths,
            replicas=cfg.teacher_replicas,
            teacher_T=cfg.teacher_T,
            teacher_top_p=cfg.teacher_top_p,
            teacher_top_k=cfg.teacher_top_k,
            predict_len=cfg.predict_len,
            fork_devices=["cuda:0"],
        )
        t0 = time.time()
        runner.run()
        dt = time.time() - t0
        n = len(m.samples)
        logger.info(
            f"teacher[{split}] {n} 样本×{cfg.teacher_replicas} replica 完成，"
            f"{dt / 60:.1f} 分钟（{n * cfg.teacher_replicas / max(dt, 1):.2f} 次/s）"
        )
    return 0


# ======================================================================
# train
# ======================================================================


def _train_scale(train_manifest) -> np.ndarray:
    """scale[h] = max(std_train(y[:,h]), 0.01)（§4.3，冻结）。"""
    y = np.stack([s.y_real for s in train_manifest.samples])  # [N,10]
    return np.maximum(y.std(axis=0), 0.01).astype(np.float32)


def _train_run_name(profile: str, arm: str, seed: int) -> str:
    return f"train-{profile}-{arm}-s{seed}"


def _load_head_ckpt(profile: str, arm: str, seed: int, epoch: int):
    """读指定臂 best epoch 的 head state_dict（dhead_distill.train 产物）。"""
    import torch

    from dhead_distill.data import safe_artifact_dir

    d = safe_artifact_dir(_train_run_name(profile, arm, seed))
    ck = torch.load(d / f"epoch-{epoch}.pt", weights_only=True)
    return ck["head"]


def _arm_epochs(profile: str, arm: str, seed: int) -> int:
    """某臂实际跑的 epoch 数（= history 长度）。"""
    from dhead_distill.data import safe_artifact_dir

    d = safe_artifact_dir(_train_run_name(profile, arm, seed))
    return len(json.loads((d / "result.json").read_text("utf-8"))["history"])


def _best_epoch_of(profile: str, arm: str, seed: int) -> int:
    from dhead_distill.data import safe_artifact_dir

    d = safe_artifact_dir(_train_run_name(profile, arm, seed))
    return json.loads((d / "result.json").read_text("utf-8"))["best_epoch"]


def cmd_train(args: argparse.Namespace) -> int:
    """训练指定臂（含初始化链 D0→D1→D2 与预算匹配对照）。"""
    import torch

    from dhead_distill.backbone import StudentBackbone, inject_lora
    from dhead_distill.data import safe_artifact_dir
    from dhead_distill.head import MultiHorizonHead
    from dhead_distill.teacher import TeacherRunner
    from dhead_distill.train import TRAIN_ARMS, UnifiedTrainer

    arm, seed, profile = args.arm, args.seed, args.profile
    if arm not in TRAIN_ARMS:
        logger.error(f"未知臂：{arm}（可选 {TRAIN_ARMS}；T 为教师基线不训练）")
        return 2
    cfg = DHeadConfig.with_profile(profile)
    env = resolve_env()
    device = _require_gpu()

    train_m = _load_manifest(profile, "train")
    val_m = _load_manifest(profile, "val")

    def _targets(m) -> np.ndarray:
        """只读已生成的教师分片（身份由 teacher 命令写入的 run manifest 保证）。"""
        from dhead_distill.data import safe_artifact_dir as sad
        from dhead_distill.teacher import TeacherRunner as TR

        r = TR.__new__(TR)
        r.manifest = m
        r.replicas = cfg.teacher_replicas
        r.predict_len = cfg.predict_len
        r.run_dir = sad(f"teacher-{profile}-{m.split}-{m.content_hash[:12]}")
        return r.load_targets_array()[0]

    train_teacher = _targets(train_m)
    val_teacher = _targets(val_m)
    scale = _train_scale(train_m)

    # —— 底座与头 ——
    from dhead_distill.backbone import load_g1_student

    backbone = load_g1_student(env, device)
    torch.manual_seed(seed)
    head = MultiHorizonHead(
        d_model=cfg.d_model, head_dim=cfg.head_dim, n_heads=cfg.head_n_heads,
        n_horizons=cfg.n_horizons,
        calendar_cardinalities=cfg.calendar_cardinalities,
    ).to(device)

    lora_named = None
    max_epochs_override = None
    disable_early_stop = False
    if arm == "D1":
        head.load_state_dict(_load_head_ckpt(profile, "D0", seed,
                                             _best_epoch_of(profile, "D0", seed)))
        logger.info(f"D1 ← D0 best-D checkpoint（epoch {_best_epoch_of(profile, 'D0', seed)}）")
    elif arm == "D2":
        _require_d2_gate(profile)
        head.load_state_dict(_load_head_ckpt(profile, "D1", seed,
                                             _best_epoch_of(profile, "D1", seed)))
        lora_named = inject_lora(
            backbone, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout, targets=cfg.lora_targets)
        logger.info("D2 ← D1 best-task checkpoint + 末两层 q/v LoRA(r=8)")
    elif arm == "S-long":
        n_total = _arm_epochs(profile, "D0", seed) + _arm_epochs(profile, "D1", seed)
        max_epochs_override = n_total
        disable_early_stop = True
        logger.info(f"S-long ← S 轨迹延长至 {n_total} epoch（D0+D1 实际步数）")
    elif arm == "D1-cont":
        if not (safe_artifact_dir(_train_run_name(profile, "D2", seed))
                / "result.json").exists():
            logger.error("D1-cont 须在 D2 之后运行（步数匹配对象）")
            return 2
        head.load_state_dict(_load_head_ckpt(profile, "D1", seed,
                                             _best_epoch_of(profile, "D1", seed)))
        max_epochs_override = _arm_epochs(profile, "D2", seed)
        disable_early_stop = True
        logger.info(f"D1-cont ← D1 best-task 起点，延长 {max_epochs_override} epoch")

    trainer = UnifiedTrainer(
        arm=arm, cfg=cfg, backbone=backbone, head=head, scale=scale,
        train_manifest=train_m, val_manifest=val_m,
        train_teacher=train_teacher, val_teacher=val_teacher,
        run_name=_train_run_name(profile, arm, seed), seed=seed, device=device,
        lora_named=lora_named,
        max_epochs_override=max_epochs_override,
        disable_early_stop=disable_early_stop,
    )
    res = trainer.fit()
    logger.info(
        f"train[{arm} s{seed}] 完成：best_epoch={res.best_epoch} "
        f"({res.criterion})，epochs={len(res.history)}"
    )
    return 0


def _require_d2_gate(profile: str) -> None:
    """D2 解锁核验（§8.3：由验证数据的保真/改进条件决定）。"""
    from dhead_distill.data import safe_artifact_dir

    gate_path = safe_artifact_dir(f"eval-{profile}-fidelity") / "summary.json"
    if not gate_path.exists():
        raise RuntimeError(
            "D2 未解锁：先运行 evaluate --stage fidelity（§8.3 条件未核验）"
        )
    doc = json.loads(gate_path.read_text("utf-8"))
    if not doc.get("d2_unlocked", False):
        raise RuntimeError(
            f"D2 未解锁：保真/改进条件未满足——{doc.get('d2_reason', '未知原因')}"
        )


# ======================================================================
# evaluate
# ======================================================================


def _predict_arm(profile: str, arm: str, seed: int, manifest, cfg) -> np.ndarray:
    """用某臂 best checkpoint 在清单上推理，返回 [N,10]（按 samples 顺序）。"""
    import torch

    from dhead_distill.backbone import inject_lora, load_g1_student
    from dhead_distill.head import MultiHorizonHead
    from dhead_distill.train import _materialize_days

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = resolve_env()
    backbone = load_g1_student(env, device)
    head = MultiHorizonHead(
        d_model=cfg.d_model, head_dim=cfg.head_dim, n_heads=cfg.head_n_heads,
        n_horizons=cfg.n_horizons,
        calendar_cardinalities=cfg.calendar_cardinalities,
    ).to(device)
    head.load_state_dict(_load_head_ckpt(profile, arm, seed,
                                         _best_epoch_of(profile, arm, seed)))
    head.eval()
    if arm == "D2":
        inject_lora(backbone, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout, targets=cfg.lora_targets)
        ck = torch.load(_load_ckpt_path(profile, arm, seed), weights_only=True)
        if "lora" in ck:
            # inject_lora 的名字相对 kronos 前缀，补 "kronos." 后整体装载
            lora_state = {f"kronos.{k}": v for k, v in ck["lora"].items()}
            backbone.load_state_dict(lora_state, strict=False)

    days = _materialize_days(manifest, np.zeros((len(manifest.samples),
                                                 cfg.teacher_replicas, 10),
                                                dtype=np.float32))
    preds = []
    with torch.no_grad():
        for b in days:
            h = backbone.extract(b.x_norm.to(device), b.x_stamp.to(device))
            preds.append(head(h, b.y_stamp.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _load_ckpt_path(profile: str, arm: str, seed: int) -> Path:
    from dhead_distill.data import safe_artifact_dir

    d = safe_artifact_dir(_train_run_name(profile, arm, seed))
    p = d / f"epoch-{_best_epoch_of(profile, arm, seed)}.pt"
    return p


def _teacher_array_for(profile: str, manifest, cfg) -> np.ndarray:
    """装载教师 [N,R,10] 分片（与 manifest 对齐）。"""
    from dhead_distill.teacher import TeacherRunner

    r = TeacherRunner.__new__(TeacherRunner)
    r.manifest = manifest
    r.replicas = cfg.teacher_replicas
    r.predict_len = cfg.predict_len
    from dhead_distill.data import safe_artifact_dir as sad

    r.run_dir = sad(f"teacher-{profile}-{manifest.split}-{manifest.content_hash[:12]}")
    return r.load_targets_array()[0]


def _days_for_eval(manifest, teacher_arr, cfg) -> list[dict]:
    """把清单+教师分片转成 evaluate 用的逐日截面结构。"""
    idx_by_date: dict[str, list[int]] = {}
    for i, s in enumerate(manifest.samples):
        idx_by_date.setdefault(s.date.strftime("%Y-%m-%d"), []).append(i)
    days = []
    for d_iso, idxs in idx_by_date.items():
        y = np.stack([manifest.samples[i].y_real for i in idxs])
        rep = np.stack([teacher_arr[idxs, r, :] for r in range(cfg.teacher_replicas)])
        days.append({"date": d_iso, "y": y, "teacher": rep, "idxs": idxs})
    return days


def cmd_evaluate(args: argparse.Namespace) -> int:
    """评价与阶段门禁（fidelity / prediction / economic）。"""
    cfg = DHeadConfig.with_profile(args.profile)
    seed = args.seed
    if args.stage == "fidelity":
        return _eval_fidelity(args.profile, seed, cfg)
    if args.stage == "prediction":
        return _eval_prediction(args.profile, seed, cfg)
    return _eval_economic(args.profile, cfg)


def _eval_fidelity(profile: str, seed: int, cfg) -> int:
    """保真门禁：val 清单上逐臂 E/R/Spearman + D2 解锁判定（§8.2/§8.3）。"""
    from dhead_distill.data import safe_artifact_dir
    from dhead_distill.evaluate import fidelity_gate, fidelity_metrics

    val_m = _load_manifest(profile, "val")
    val_teacher = _teacher_array_for(profile, val_m, cfg)
    days = _days_for_eval(val_m, val_teacher, cfg)
    scale = np.asarray(_train_scale(_load_manifest(profile, "train")))

    summary: dict = {"stage": "fidelity", "profile": profile, "seed": seed,
                     "split": "val", "arms": {}}
    for arm in ("D0", "D1", "D2"):
        from dhead_distill.data import safe_artifact_dir as sad

        if not (sad(_train_run_name(profile, arm, seed)) / "result.json").exists():
            continue
        pred_all = _predict_arm(profile, arm, seed, val_m, cfg)
        pred_by_idx = {i: pred_all[k] for k, i in enumerate(range(len(pred_all)))}
        met = fidelity_metrics(
            days=days,
            student=lambda day: np.stack(
                [pred_by_idx[i] for i in day["idxs"]]),
            scale=scale,
        )
        met["gate"] = fidelity_gate(met, ratio_max=2.0, spearman_min=0.8,
                                    valid_frac_min=0.8)
        summary["arms"][arm] = met
        logger.info(
            f"fidelity[{arm}] E={met['E']:.4f} R={met['R']:.4f} "
            f"ratio={met['ratio']:.3f} spearman={met['mean_spearman_valid']} "
            f"gate={'PASS' if met['gate']['passed'] else 'FAIL'}"
        )

    # D2 解锁（§8.3）：D0 验证保真通过 ∧ D1 验证 task 优于 D0 ∧ D1 E/R≤2
    d0, d1 = summary["arms"].get("D0"), summary["arms"].get("D1")
    unlocked, reason = False, "D0/D1 结果不齐"
    if d0 and d1:
        from dhead_distill.data import safe_artifact_dir as sad

        d0_task = json.loads((sad(_train_run_name(profile, "D0", seed))
                              / "result.json").read_text("utf-8"))
        d1_task = json.loads((sad(_train_run_name(profile, "D1", seed))
                              / "result.json").read_text("utf-8"))
        best_d0 = min(h["val_loss"] for h in d0_task["history"])
        best_d1 = min(h["val_loss"] for h in d1_task["history"])
        conds = {
            "d0_fidelity_passed": bool(d0["gate"]["passed"]),
            "d1_beats_d0_val": bool(best_d1 < best_d0),
            "d1_ratio_le_2": bool(d1["ratio"] <= 2.0),
        }
        unlocked = all(conds.values())
        reason = f"条件 {conds}"
    summary["d2_unlocked"] = unlocked
    summary["d2_reason"] = reason

    out_dir = safe_artifact_dir(f"eval-{profile}-fidelity")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    logger.info(f"fidelity 摘要 → {out_dir.name}/summary.json（d2_unlocked={unlocked}）")
    return 0


def _eval_prediction(profile: str, seed: int, cfg) -> int:
    """真实标签指标 + 配对 bootstrap（诊断清单，§8.4）。"""
    from dhead_distill.data import safe_artifact_dir
    from dhead_distill.evaluate import (
        block_bootstrap_paired_diff, daily_rank_ic, holm_correction,
    )

    diag_m = _load_manifest(profile, "diag")
    teacher = _teacher_array_for(profile, diag_m, cfg)
    days = _days_for_eval(diag_m, teacher, cfg)

    arms = [a for a in ("D0", "S", "D1", "D2", "S-long", "D1-cont")
            if (safe_artifact_dir(_train_run_name(profile, a, seed))
                / "result.json").exists()]
    preds: dict[str, dict[int, np.ndarray]] = {}
    summary: dict = {"stage": "prediction", "profile": profile, "seed": seed,
                     "split": "diag", "arms": {}}
    for arm in arms:
        p_all = _predict_arm(profile, arm, seed, diag_m, cfg)
        preds[arm] = {k: v for k, v in enumerate(p_all)}
        met = daily_rank_ic(
            days=days,
            student=lambda day, _p=preds[arm]: np.stack([_p[i] for i in day["idxs"]]),
        )
        summary["arms"][arm] = met
        logger.info(f"prediction[{arm}] meanRankIC={met['mean_rank_ic']} "
                    f"valid={met['n_days_valid']}/{met['n_days']}")

    # 教师基线 T：replica0 作为“预测”
    t_met = daily_rank_ic(
        days=days, student=lambda day: day["teacher"][0])
    summary["arms"]["T"] = t_met
    logger.info(f"prediction[T] meanRankIC={t_met['mean_rank_ic']}")

    # 配对差（按日期整体）：D1 − S-long、D2 − D1-cont（日 RankIC 差）
    def _ic_by_day(pred_map) -> dict:
        from scipy import stats as sps

        out = {}
        for day in days:
            p = np.stack([pred_map[i] for i in day["idxs"]]).mean(axis=-1)
            yv = day["y"].mean(axis=-1)
            if np.std(p) > 1e-12 and np.std(yv) > 1e-12:
                out[day["date"]] = float(sps.spearmanr(p, yv).statistic)
        return out

    pairs = {}
    for a, b in (("D1", "S-long"), ("D2", "D1-cont")):
        if a in preds and b in preds:
            ia, ib = _ic_by_day(preds[a]), _ic_by_day(preds[b])
            common = sorted(set(ia) & set(ib))
            diff = np.array([ia[d] - ib[d] for d in common])
            ci = block_bootstrap_paired_diff(diff, block=10, n_boot=2000,
                                             seed=20260905)
            pairs[f"{a}-{b}"] = ci
            logger.info(f"paired {a}−{b}: 点估计 {ci['point']:+.4f} "
                        f"95%CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
    if len(pairs) == 2:
        # 两项均检验 → Holm 校正（用平价正态近似 p 值示意，报告原区间为主）
        from scipy import stats as sps

        pvals = [2 * (1 - sps.norm.cdf(abs(ci["point"] /
                     max(np.sqrt(max(ci["hi"] - ci["lo"], 1e-12) ** 2 / 4 / 10), 1e-9))))
                 for ci in pairs.values()]
        summary["holm_adjusted_p"] = holm_correction(pvals)
    summary["paired"] = pairs

    out_dir = safe_artifact_dir(f"eval-{profile}-prediction")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    return 0


def _eval_economic(profile: str, cfg) -> int:
    """经济门禁：引擎独立复核完成前拒绝输出可交易结论（§8.4）。"""
    from dhead_distill.data import safe_artifact_dir
    from dhead_distill.evaluate import evaluate_economic_gate

    verdict = evaluate_economic_gate(
        engine_verified=False, executor_available=False)
    out_dir = safe_artifact_dir(f"eval-{profile}-economic")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "gate.json", verdict)
    logger.warning(f"economic：{verdict['reason']}")
    return 0


# ======================================================================
# 入口
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dhead_distill", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preflight", help="只读资产/环境预检")

    pp = sub.add_parser("prepare", help="构造冻结清单")
    pp.add_argument("--profile", required=True, choices=sorted(PROFILES))
    pp.add_argument("--splits", default="",
                    help="逗号分隔：train,val,diag,replay（默认全建）")

    tp = sub.add_parser("teacher", help="教师 3-replica 生成")
    tp.add_argument("--profile", required=True, choices=sorted(PROFILES))
    tp.add_argument("--splits", default="", help="默认 train,val")

    tr = sub.add_parser("train", help="训练臂")
    tr.add_argument("--profile", required=True, choices=sorted(PROFILES))
    tr.add_argument("--arm", required=True)
    tr.add_argument("--seed", type=int, default=42)

    ev = sub.add_parser("evaluate", help="评价与阶段门禁")
    ev.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ev.add_argument("--stage", required=True,
                    choices=["fidelity", "prediction", "economic"])
    ev.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "teacher":
        return cmd_teacher(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "evaluate":
        return cmd_evaluate(args)
    logger.error(f"未知命令：{args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
