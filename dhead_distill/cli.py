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


def _load_manifest(profile: str, split: str, cfg: DHeadConfig | None = None) -> "DayManifest":
    """按 profile/split 装载 prepare 产物（v1 修复 #4 + R3：数据集级协议过滤）。

    过滤用 :func:`dataset_protocol_hash(cfg)`——学生级字段（输出语义/损失
    尺度口径）不影响清单选取（A/B 两臂共用同一冻结清单），任何数据侧字段
    变化仍失配；装载时重算内容 hash 防静默损坏。
    """
    from dhead_distill.config import dataset_protocol_hash
    from dhead_distill.data import DayManifest

    root = _artifact_root()
    cands = [
        p for p in root.glob(f"prepare-{profile}-{split}-*") if p.is_dir()
    ]
    if not cands:
        raise FileNotFoundError(
            f"未找到 prepare 产物：prepare-{profile}-{split}-*（先运行 prepare）"
        )
    want_protocol = dataset_protocol_hash(cfg) if cfg is not None else None
    matched = []
    for p in cands:
        try:
            meta = json.loads((p / "manifest.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("profile") != profile or meta.get("split") != split:
            continue
        if want_protocol is not None and meta.get("protocol") != want_protocol:
            continue
        matched.append(p)
    if not matched:
        raise FileNotFoundError(
            f"prepare 产物与当前数据集协议不匹配：prepare-{profile}-{split}-* "
            f"(dataset_protocol={str(want_protocol)[:12]})——请用当前配置重跑 prepare"
        )
    matched.sort(key=lambda p: p.stat().st_mtime)
    return DayManifest.load(matched[-1].name, verify=True)


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
    """装载 G1 tokenizer+predictor（只读；返回 (predictor, 联合权重hash)）。

    v1 修复 #1：装载后强制 eval——G1 ffn/resid dropout=0.2，train 态生成
    会把 dropout 噪声注入教师目标与全部 replica（v1 教师分片因此作废，
    由 teacher identity 的 model_eval 字段隔离）。
    """
    from model import Kronos, KronosPredictor, KronosTokenizer

    logger.info(f"加载 G1 tokenizer（只读）：{env.g1_tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(str(env.g1_tokenizer))
    logger.info(f"加载 G1 predictor（只读）：{env.g1_predictor}")
    model = Kronos.from_pretrained(str(env.g1_predictor))
    device = _require_gpu()
    predictor = KronosPredictor(
        model.to(device), tokenizer.to(device), device=device, max_context=512,
    )
    from dhead_distill.teacher import ensure_teacher_eval

    ensure_teacher_eval(predictor)
    w_hash = hashlib.sha256(
        _sha256_file(env.g1_tokenizer / "model.safetensors").encode()
        + _sha256_file(env.g1_predictor / "model.safetensors").encode()
    ).hexdigest()
    return predictor, w_hash


def cmd_teacher(args: argparse.Namespace) -> int:
    """教师 3-replica 生成（真实 G1 只在本命令内加载；R3 支持 namespace）。"""
    from dhead_distill.teacher import ensure_teacher_eval

    _require_profile_gate(args.profile)  # R4：main teacher 入口实检 pilot 门禁
    cfg = DHeadConfig.with_profile(args.profile)
    env = resolve_env()
    predictor, w_hash = _load_predictor(env)
    ensure_teacher_eval(predictor)  # 生成前实际断言（R3：不只信写死标记）
    splits = args.splits.split(",") if args.splits else ["train", "val"]
    for split in splits:
        from dhead_distill.teacher import TeacherRunner

        m = _load_manifest(args.profile, split, cfg)
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
            namespace=args.namespace,
            model_eval_verified=True,
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


def _g1_weight_hash(env) -> str:
    """当前 G1 权重联合指纹（train/evaluate 装载教师分片时比对，R3）。"""
    return hashlib.sha256(
        _sha256_file(env.g1_tokenizer / "model.safetensors").encode()
        + _sha256_file(env.g1_predictor / "model.safetensors").encode()
    ).hexdigest()


def _train_scale(train_manifest, cfg: DHeadConfig | None = None,
                 teacher_targets: np.ndarray | None = None) -> np.ndarray:
    """期限尺度 scale[h]（§4.3 冻结；v1.1 方案 A：口径可配置）。

    - ``train_real_std``（默认，= v1 行为）：``max(std_train(y_real[:,h]),0.01)``；
    - ``teacher_r0_std``：蒸馏目标 replica0 的 ``max(std, 0.01)``（需传
      teacher_targets [N,R,10]；v1.1 仅注册接口，本轮实验不使用）。
    """
    source = cfg.scale_source if cfg is not None else "train_real_std"
    if source == "train_real_std":
        y = np.stack([s.y_real for s in train_manifest.samples])  # [N,10]
        return np.maximum(y.std(axis=0), 0.01).astype(np.float32)
    if source == "teacher_r0_std":
        if teacher_targets is None:
            raise ValueError("scale_source=teacher_r0_std 需要教师目标数组")
        return np.maximum(
            teacher_targets[:, 0, :].std(axis=0), 0.01).astype(np.float32)
    raise ValueError(f"未知 scale_source：{source}")


def _train_run_name(profile: str, arm: str, seed: int) -> str:
    return f"train-{profile}-{arm}-s{seed}"


def _load_head_ckpt(profile: str, arm: str, seed: int, epoch: int,
                    run_prefix: str = "train-", *,
                    output_space: str = "raw_return"):
    """读指定臂 best epoch 的 head state_dict（含 R2 语义守卫）。

    checkpoint 身份的 ``output_space`` 必须与当前请求一致；v1 旧 ckpt
    （无该字段）不得当作新语义（affine）的起点。
    """
    import torch

    from dhead_distill.data import safe_artifact_dir

    d = safe_artifact_dir(f"{run_prefix}{profile}-{arm}-s{seed}")
    ck = torch.load(d / f"epoch-{epoch}.pt", weights_only=True)
    ck_os = ck.get("identity", {}).get("output_space")
    if ck_os != output_space:
        raise RuntimeError(
            f"checkpoint 输出语义不匹配：{arm} epoch{epoch} ckpt 为 "
            f"{ck_os!r}，当前请求 {output_space!r}——旧语义权重禁止作为"
            f"新语义起点（R2）"
        )
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
    _require_profile_gate(profile)   # R4：main/D1 扩大入口实检 pilot 门禁
    cfg = DHeadConfig.with_profile(profile)
    env = resolve_env()
    device = _require_gpu()

    train_m = _load_manifest(profile, "train", cfg)
    val_m = _load_manifest(profile, "val", cfg)

    def _targets(m) -> np.ndarray:
        """只读教师分片（R3 闭环：权重指纹 + 完整教师协议核验）。"""
        from dhead_distill.teacher import TeacherRunner

        r = TeacherRunner.load_verified(
            m, replicas=cfg.teacher_replicas, predict_len=cfg.predict_len,
            expected_weight_hash=_g1_weight_hash(env),
            namespace=args.namespace,
            n_paths=cfg.teacher_n_paths, teacher_T=cfg.teacher_T,
            teacher_top_p=cfg.teacher_top_p, teacher_top_k=cfg.teacher_top_k,
        )
        return r.load_targets_array()[0]

    train_teacher = _targets(train_m)
    val_teacher = _targets(val_m)
    scale = _train_scale(train_m, cfg,
                         train_teacher if cfg.scale_source == "teacher_r0_std"
                         else None)

    # —— 底座与头 ——
    from dhead_distill.backbone import load_g1_student

    backbone = load_g1_student(env, device)
    torch.manual_seed(seed)
    head = MultiHorizonHead(
        d_model=cfg.d_model, head_dim=cfg.head_dim, n_heads=cfg.head_n_heads,
        n_horizons=cfg.n_horizons,
        calendar_cardinalities=cfg.calendar_cardinalities,
        output_space=cfg.output_space,
    ).to(device)

    lora_named = None
    max_epochs_override = None
    disable_early_stop = False
    _require_arm_ready(profile, arm, seed)   # v1 修复 #5：前置臂显式门禁
    if arm == "D1":
        head.load_state_dict(
            _load_head_ckpt(profile, "D0", seed,
                            _best_epoch_of(profile, "D0", seed),
                            output_space=cfg.output_space))
        logger.info(f"D1 ← D0 best-D checkpoint（epoch {_best_epoch_of(profile, 'D0', seed)}）")
    elif arm == "D2":
        _require_d2_gate(profile, seed, cfg)
        head.load_state_dict(
            _load_head_ckpt(profile, "D1", seed,
                            _best_epoch_of(profile, "D1", seed),
                            output_space=cfg.output_space))
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
        head.load_state_dict(
            _load_head_ckpt(profile, "D1", seed,
                            _best_epoch_of(profile, "D1", seed),
                            output_space=cfg.output_space))
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
        output_space=cfg.output_space,
        backbone_weight_hash=_g1_weight_hash(env),
    )
    res = trainer.fit()
    logger.info(
        f"train[{arm} s{seed}] 完成：best_epoch={res.best_epoch} "
        f"({res.criterion})，epochs={len(res.history)}"
    )
    return 0


def _require_d2_gate(profile: str, seed: int, cfg: DHeadConfig) -> None:
    """D2 解锁核验（§8.3；R4：profile/seed/协议/数据/选中点全身份比对）。"""
    from dhead_distill.config import dataset_protocol_hash, protocol_hash
    from dhead_distill.data import safe_artifact_dir

    gate_path = safe_artifact_dir(f"eval-{profile}-fidelity") / "summary.json"
    if not gate_path.exists():
        raise RuntimeError(
            "D2 未解锁：先运行 evaluate --stage fidelity（§8.3 条件未核验）"
        )
    doc = json.loads(gate_path.read_text("utf-8"))
    checks = {
        "seed": (doc.get("seed"), seed),
        "protocol": (doc.get("protocol"), protocol_hash(cfg)),
        "dataset_protocol": (
            doc.get("dataset_protocol"), dataset_protocol_hash(cfg)),
        "output_space": (doc.get("output_space"), cfg.output_space),
    }
    for k, (got, want) in checks.items():
        if got != want:
            raise RuntimeError(
                f"D2 门禁 {k} 不匹配：门禁文件 {got!r} ≠ 当前 {want!r}——"
                f"协议/数据/语义变化后须重新过 fidelity 门禁（R4）"
            )
    if not doc.get("d2_unlocked", False):
        raise RuntimeError(
            f"D2 未解锁：保真/改进条件未满足——{doc.get('d2_reason', '未知原因')}"
        )


def _require_profile_gate(profile: str) -> None:
    """R4：main 侧 teacher/train（含 D1 扩大）必须实检 pilot 门禁。

    前置 result.json 存在 ≠ 门禁通过——main 入口要求 pilot fidelity 摘要
    存在且 D0 保真门禁 PASS（§8.2 冻结规则）。pilot 门禁未过时 main 全锁。
    """
    from dhead_distill.data import safe_artifact_dir

    if profile != "main":
        return
    gate_path = safe_artifact_dir("eval-pilot-fidelity") / "summary.json"
    if not gate_path.exists():
        raise RuntimeError(
            "main 入口被拒：pilot fidelity 门禁未运行（先 evaluate --profile "
            "pilot --stage fidelity）——R4：不得以'存在结果文件'替代门禁"
        )
    doc = json.loads(gate_path.read_text("utf-8"))
    d0 = (doc.get("arms") or {}).get("D0") or {}
    if not (d0.get("gate") or {}).get("passed", False):
        raise RuntimeError(
            "main 入口被拒：pilot D0 保真门禁未通过（§8.2 冻结规则，"
            f"gate={d0.get('gate')}）"
        )


def _require_arm_ready(profile: str, arm: str, seed: int) -> None:
    """臂前置检查（v1 修复 #5：初始化链依赖显式报错，不裸抛 FileNotFoundError）。

    D1←D0、D2←D1（另需门禁）、S-long←D0+D1、D1-cont←D1+D2。
    """
    from dhead_distill.data import safe_artifact_dir

    def _has(a: str) -> bool:
        return (safe_artifact_dir(_train_run_name(profile, a, seed))
                / "result.json").exists()

    deps = {
        "D1": ["D0"],
        "D2": ["D1"],
        "S-long": ["D0", "D1"],
        "D1-cont": ["D1", "D2"],
    }.get(arm, [])
    missing = [a for a in deps if not _has(a)]
    if missing:
        raise RuntimeError(
            f"臂 {arm} 的前置结果缺失：{missing}（seed={seed}）——"
            f"请先按链路顺序训练"
        )


# ======================================================================
# evaluate
# ======================================================================


def _predict_arm(profile: str, arm: str, seed: int, manifest, cfg,
                 run_prefix: str = "train-") -> np.ndarray:
    """用某臂 best checkpoint 在清单上推理，返回 [N,10]（按 samples 顺序）。

    R2：按 checkpoint 的输出语义调用（affine 需逐样本 a/b，由
    ``_materialize_days`` 从历史 90 日算出）。
    """
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
        output_space=cfg.output_space,
    ).to(device)
    head.load_state_dict(
        _load_head_ckpt(profile, arm, seed, _best_epoch_of(profile, arm, seed),
                        run_prefix=run_prefix, output_space=cfg.output_space))
    head.eval()
    if arm == "D2":
        inject_lora(backbone, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout, targets=cfg.lora_targets)
        from dhead_distill.data import safe_artifact_dir

        d = safe_artifact_dir(f"{run_prefix}{profile}-{arm}-s{seed}")
        ck = torch.load(d / f"epoch-{_best_epoch_of(profile, arm, seed)}.pt",
                        weights_only=True)
        if "lora" in ck:
            # inject_lora 的名字相对 kronos 前缀，补 "kronos." 后整体装载
            lora_state = {f"kronos.{k}": v for k, v in ck["lora"].items()}
            backbone.load_state_dict(lora_state, strict=False)

    days = _materialize_days(manifest, np.zeros((len(manifest.samples),
                                                 cfg.teacher_replicas, 10),
                                                dtype=np.float32))
    affine = cfg.output_space == "normalized_close_affine_return"
    preds = []
    with torch.no_grad():
        for b in days:
            h = backbone.extract(b.x_norm.to(device), b.x_stamp.to(device))
            y_stamp = b.y_stamp.to(device)
            if affine:
                p = head(h, y_stamp, b.a.to(device), b.b.to(device))
            else:
                p = head(h, y_stamp)
            preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0)


def _load_ckpt_path(profile: str, arm: str, seed: int) -> Path:
    from dhead_distill.data import safe_artifact_dir

    d = safe_artifact_dir(_train_run_name(profile, arm, seed))
    p = d / f"epoch-{_best_epoch_of(profile, arm, seed)}.pt"
    return p


def _teacher_array_for(profile: str, manifest, cfg,
                       namespace: str = "v1") -> np.ndarray:
    """装载教师 [N,R,10] 分片（R3：评价路径同样核验权重指纹+完整协议）。"""
    from dhead_distill.teacher import TeacherRunner

    r = TeacherRunner.load_verified(
        manifest, replicas=cfg.teacher_replicas, predict_len=cfg.predict_len,
        expected_weight_hash=_g1_weight_hash(resolve_env()),
        namespace=namespace,
        n_paths=cfg.teacher_n_paths, teacher_T=cfg.teacher_T,
        teacher_top_p=cfg.teacher_top_p, teacher_top_k=cfg.teacher_top_k,
    )
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
        return _eval_fidelity(args.profile, seed, cfg, args.namespace)
    if args.stage == "prediction":
        return _eval_prediction(args.profile, seed, cfg, args.namespace)
    return _eval_economic(args.profile, cfg)


def _eval_fidelity(profile: str, seed: int, cfg, namespace: str = "v1") -> int:
    """保真门禁：val 清单上逐臂 E/R/Spearman + D2 解锁判定（§8.2/§8.3 + R1/R3/R4）。"""
    from dhead_distill.config import dataset_protocol_hash, protocol_hash
    from dhead_distill.data import safe_artifact_dir
    from dhead_distill.evaluate import fidelity_gate, fidelity_metrics
    from dhead_distill.train import _package_code_hash

    val_m = _load_manifest(profile, "val", cfg)
    train_m = _load_manifest(profile, "train", cfg)
    val_teacher = _teacher_array_for(profile, val_m, cfg, namespace)
    days = _days_for_eval(val_m, val_teacher, cfg)
    scale = np.asarray(_train_scale(train_m, cfg))

    summary: dict = {
        "stage": "fidelity", "profile": profile, "seed": seed,
        "split": "val", "arms": {},
        # R3/R4：summary 绑定代码/权重/清单/协议/输出语义——D2 门禁逐项比对
        "protocol": protocol_hash(cfg),
        "dataset_protocol": dataset_protocol_hash(cfg),
        "output_space": cfg.output_space,
        "code_hash": _package_code_hash(),
        "weight_hash": _g1_weight_hash(resolve_env()),
        "train_manifest_hash": train_m.content_hash,
        "val_manifest_hash": val_m.content_hash,
        "scale_hash": __import__("hashlib").sha256(
            np.asarray(scale, dtype=np.float32).tobytes()).hexdigest(),
    }
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

    # D2 解锁（§8.3；R1：按实际选中 checkpoint 比较，经 evaluate.d2_unlock_condition）
    d0, d1 = summary["arms"].get("D0"), summary["arms"].get("D1")
    unlocked, reason = False, "D0/D1 结果不齐"
    if d0 and d1:
        from dhead_distill.data import safe_artifact_dir as sad
        from dhead_distill.evaluate import d2_unlock_condition

        d0_res = json.loads((sad(_train_run_name(profile, "D0", seed))
                             / "result.json").read_text("utf-8"))
        d1_res = json.loads((sad(_train_run_name(profile, "D1", seed))
                             / "result.json").read_text("utf-8"))
        verdict = d2_unlock_condition(d0_res, d1_res, d0, d1)
        unlocked = verdict["unlocked"]
        reason = (f"条件 {verdict['conditions']} "
                  f"(D0 选中 e{verdict['d0_selected_epoch']} "
                  f"val_task={verdict['d0_selected_val_task']:.4f}，"
                  f"D1 选中 e{verdict['d1_selected_epoch']} "
                  f"val_task={verdict['d1_selected_val_task']:.4f})")
        summary["d2_unlock_detail"] = verdict
        # R1：fidelity 分数绑定同一 checkpoint 身份
        summary["arms"]["D0"]["selected_epoch"] = verdict["d0_selected_epoch"]
        summary["arms"]["D1"]["selected_epoch"] = verdict["d1_selected_epoch"]
    summary["d2_unlocked"] = unlocked
    summary["d2_reason"] = reason

    out_dir = safe_artifact_dir(f"eval-{profile}-fidelity")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    logger.info(f"fidelity 摘要 → {out_dir.name}/summary.json（d2_unlocked={unlocked}）")
    return 0


def _eval_prediction(profile: str, seed: int, cfg,
                     namespace: str = "v1") -> int:
    """真实标签指标 + 配对 bootstrap（诊断清单，§8.4；R4：p 值显式未实现）。"""
    from dhead_distill.data import safe_artifact_dir
    from dhead_distill.evaluate import (
        block_bootstrap_paired_diff, daily_rank_ic,
    )

    diag_m = _load_manifest(profile, "diag", cfg)
    teacher = _teacher_array_for(profile, diag_m, cfg, namespace)
    days = _days_for_eval(diag_m, teacher, cfg)

    arms = [a for a in ("D0", "S", "D1", "D2", "S-long", "D1-cont")
            if (safe_artifact_dir(_train_run_name(profile, a, seed))
                / "result.json").exists()]
    preds: dict[str, dict[int, np.ndarray]] = {}
    summary: dict = {"stage": "prediction", "profile": profile, "seed": seed,
                     "split": "diag", "arms": {},
                     "namespace": namespace}
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
    # R4：显著性检验（p 值 / Holm）显式标为未实现——旧"示意公式"是错误统计，
    # 已删除；实现时须配数学独立测试。bootstrap 区间含抽样单位声明。
    summary["p_values"] = None
    summary["p_value_status"] = "not_implemented（复核 20260906 R4 禁用示意公式）"
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
    tp.add_argument("--namespace", default="v1",
                    help="run 命名空间（R3：新旧产物共存，如 v11-rev2）")

    tr = sub.add_parser("train", help="训练臂")
    tr.add_argument("--profile", required=True, choices=sorted(PROFILES))
    tr.add_argument("--arm", required=True)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--namespace", default="v1",
                    help="教师分片命名空间（须与 teacher 生成一致）")

    ev = sub.add_parser("evaluate", help="评价与阶段门禁")
    ev.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ev.add_argument("--stage", required=True,
                    choices=["fidelity", "prediction", "economic"])
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--namespace", default="v1",
                    help="教师分片命名空间（须与 teacher 生成一致）")

    mf = sub.add_parser("minimal-fit", help="v1.1 rev2 最小拟合试验（A/B 两臂）")
    mf.add_argument("--namespace", default="v11-rev2")
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
    if args.cmd == "minimal-fit":
        from dhead_distill.minimal_fit import run

        return run(args.namespace)
    logger.error(f"未知命令：{args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
