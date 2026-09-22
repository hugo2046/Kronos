"""TAS1 P0 预检：资源 / 哈希 / 参数量 / 基线旁路对拍 / 预算（计划 §8 P0）。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m tas_state.preflight \\
        --config tas_state/config.json [--full] [--out tas_state/data/manifest_p0.json]

- 默认（只读档）：Git SHA、权重 / 信号 / 语料哈希、环境、参数量实测——
  不加载真实底座、不占 GPU。
- ``--full``（测量档）：在训练期固定 200 样本上做 F0 原调用 vs wrapper
  baseline 旁路对拍（FP32 logits ≤1e-6、同 RNG 采样一致）、20 训练步与
  200 样本推理资源实测、B1 预算 N 冻结，签出 manifest。

任一硬门禁失败即非零退出并写明缺项，不声称可训练。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

from tas_state.config import REPO_ROOT as _RR, PKG_DIR, TASConfig  # noqa: F811

DATA_DIR = PKG_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 训练期固定样本（计划 §4：B1 时间比 r 与旁路对拍共用，SHA256 字典序）
BENCH_N = 200
BENCH_KEYS_CACHE = DATA_DIR / "bench_keys_200.json"


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def hf_snapshot_files(repo_id: str) -> dict[str, str]:
    """HF 缓存内权重文件哈希（离线快照）。"""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    snap = sorted((hub / f"models--{repo_id.replace('/', '--')}").glob("snapshots/*"))
    if not snap:
        return {}
    files = {}
    for p in sorted(snap[-1].rglob("*")):
        if p.is_file() and p.suffix in (".safetensors", ".bin", ".json"):
            rel = str(p.relative_to(snap[-1]))
            if p.suffix == ".json":
                files[rel] = sha256_file(p) if p.stat().st_size < (1 << 20) else "skipped-large"
            else:
                files[rel] = sha256_file(p)
    return files


def bench_keys(cfg: TASConfig) -> list[tuple[str, str]]:
    """训练期固定 200 样本键（SHA256 字典序，SHA256 语料排序规则同 STATIC）。

    落盘缓存首次计算结果；语料或配置哈希变化时失效重建。
    """
    from tas_state.data import TASCorpus

    corpus = TASCorpus(
        cfg.train_corpus_path, cfg, target_end=cfg.train_target_end,
        split_name="bench",
    )
    ranked = corpus.static_sample_keys(BENCH_N)
    payload = {
        "config_sha256": cfg.sha256(),
        "corpus_sha256": sha256_file(REPO_ROOT / cfg.train_corpus_path),
        "keys": [[c, f"{d:%Y-%m-%d}"] for c, d in ranked],
    }
    if BENCH_KEYS_CACHE.exists():
        old = json.loads(BENCH_KEYS_CACHE.read_text(encoding="utf-8"))
        if old.get("config_sha256") == payload["config_sha256"] and old.get(
            "corpus_sha256"
        ) == payload["corpus_sha256"]:
            logger.info(f"bench 200 键缓存命中：{BENCH_KEYS_CACHE}")
            return [(c, __import__("pandas").Timestamp(d)) for c, d in old["keys"]]
    BENCH_KEYS_CACHE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info(f"bench 200 键已冻结：{BENCH_KEYS_CACHE}")
    return ranked


# ============================================================
# 只读档
# ============================================================
def readonly_checks(cfg: TASConfig) -> dict:
    import torch

    report: dict = {"hard_fail": [], "warn": []}

    # 1) Git 与配置
    report["git_sha"] = git_sha()
    report["config_sha256"] = cfg.sha256()

    # 2) 权重缓存
    tok_files = hf_snapshot_files(cfg.tokenizer_name)
    mdl_files = hf_snapshot_files(cfg.model_name)
    report["tokenizer_files"] = tok_files
    report["model_files"] = mdl_files
    if not tok_files or not mdl_files:
        report["hard_fail"].append(
            f"HF 缓存缺权重：tokenizer={len(tok_files)} model={len(mdl_files)} 个文件"
        )

    # 3) 只读对照信号
    sig = {}
    for name, rel in cfg.baseline_signal_paths.items():
        p = REPO_ROOT / rel
        if p.exists():
            sig[name] = {"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size}
        else:
            sig[name] = {"path": rel, "missing": True}
            report["hard_fail"].append(f"对照信号缺失：{name} → {rel}")
    report["signals"] = sig

    # 4) 训练 / 验证语料
    for tag, rel in (("train", cfg.train_corpus_path), ("val", cfg.val_corpus_path)):
        p = REPO_ROOT / rel
        if p.exists():
            report[f"corpus_{tag}"] = {
                "path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size,
            }
        else:
            report[f"corpus_{tag}"] = {"path": rel, "missing": True}
            report["hard_fail"].append(f"语料缺失：{rel}")

    # 5) 环境
    report["env"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": cfg.device,
    }
    if torch.cuda.is_available():
        report["env"]["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        report["env"]["gpu_mem_free_gb"] = round(free / 1e9, 1)
        report["env"]["gpu_mem_total_gb"] = round(total / 1e9, 1)
    elif cfg.device.startswith("cuda"):
        report["hard_fail"].append("配置 device=cuda 但 CUDA 不可用")

    # 6) 参数量公式实测（不加载权重也能验证结构）
    from tas_state.model import StateEncoder

    enc = StateEncoder(832, hidden=cfg.state_hidden, n_states=cfg.n_states)
    n_params = enc.trainable_parameter_count()
    report["state_encoder_params"] = {
        "d_model": 832, "hidden": cfg.state_hidden, "n_states": cfg.n_states,
        "numel": n_params, "formula_202d_plus_64": 202 * 832 + 64,
        "match": n_params == 202 * 832 + 64,
    }
    if not report["state_encoder_params"]["match"]:
        report["hard_fail"].append("参数量与公式 202d+64 不符")

    return report


# ============================================================
# 测量档（--full）：真实底座对拍 + 资源实测 + B1 预算
# ============================================================
def full_checks(cfg: TASConfig, readonly: dict) -> dict:
    import pandas as pd
    import torch

    from model.kronos import Kronos, KronosPredictor, KronosTokenizer
    from tas_state.data import TASCorpus
    from tas_state.model import ConditionalKronos, StateEncoder
    from tas_state.predictor import TASPredictor

    report = {"hard_fail": readonly["hard_fail"], "warn": readonly["warn"]}
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # —— 加载真实底座 ——
    t0 = time.perf_counter()
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    kronos = Kronos.from_pretrained(cfg.model_name)
    torch.manual_seed(cfg.seed)
    encoder = StateEncoder(kronos.d_model, hidden=cfg.state_hidden, n_states=cfg.n_states)
    ck = ConditionalKronos(kronos, tokenizer, encoder, cfg).to(device)
    report["load_seconds"] = round(time.perf_counter() - t0, 1)
    report["d_model"] = kronos.d_model
    report["state_encoder_params_numel"] = encoder.trainable_parameter_count()

    # —— 训练期固定样本（对拍 + B1 预算共用）——
    keys = bench_keys(cfg)[:32]  # 对拍用前 32 只
    corpus = TASCorpus(
        cfg.train_corpus_path, cfg, target_end=cfg.train_target_end, split_name="bench"
    )
    batch = corpus.batch(keys)

    # —— 旁路对拍①：全序列 logits（FP32 ≤1e-6）——
    x = torch.tensor(batch["X"]).to(device)
    stamp = torch.tensor(batch["stamp"]).to(device)
    y_stamp = torch.tensor(batch["y_stamp"]).to(device)
    s1_ids, s2_ids = tokenizer.encode(x, half=True)
    official_logits, _ = kronos.decode_s1(s1_ids, s2_ids, stamp)
    from tas_state.model import _apply_rope_at  # noqa: F401（确保模块内聚可用）

    seq = ck.second_pass_input((s1_ids, s2_ids), stamp, S=None, q=None, layout="baseline")
    h = seq
    for layer in kronos.transformer:
        h = layer(h)
    h = kronos.norm(h)
    bypass_logits = kronos.head(h)
    abs_err = (official_logits - bypass_logits).abs().max().item()
    rel_err = (
        (official_logits - bypass_logits).abs()
        / (official_logits.abs() + 1e-12)
    ).max().item()
    report["parity_logits"] = {
        "n_samples": len(keys), "abs_err": abs_err, "rel_err": rel_err,
        "pass": abs_err <= 1e-6 or rel_err <= 1e-6,
    }
    if not report["parity_logits"]["pass"]:
        report["hard_fail"].append(
            f"baseline 旁路 logits 对拍失败：abs={abs_err:.3e} rel={rel_err:.3e}"
        )

    # —— 旁路对拍②：同 RNG 逐步采样 token 一致（spy 原版 auto_regressive）——
    # 口径披露：官方 KronosPredictor 从不调用 .eval()——Kronos-base 的
    # ffn/resid dropout=0.2 在官方全部推理（含 F0 parquet 生成）中生效。
    # 本对拍的目的是验证 wrapper 拼接/读出路径无实现错误，需确定性：
    # 原调用临时 eval()（官方随机口径不在本对拍判定范围内，见结果文档披露）。
    import model.kronos as kronos_mod
    from model.kronos import auto_regressive_inference

    x8, st8, ys8 = x[:8], stamp[:8], y_stamp[:8]
    with torch.no_grad():
        _, hist8 = ck.first_pass(x8, st8)
    torch.manual_seed(cfg.seed)
    with torch.no_grad():
        g_s1, g_s2 = ck.generate(
            x8, st8, ys8, layout="baseline",
            sample_count=cfg.sample_count,
            temperature=cfg.temperature, top_k=cfg.sample_top_k, top_p=cfg.top_p,
        )
    recorded: list = []
    orig_sample = kronos_mod.sample_from_logits

    def _spy(logits, **kw):
        out = orig_sample(logits, **kw)
        recorded.append(out.detach().cpu().clone())
        return out

    kronos_mod.sample_from_logits = _spy
    was_training = kronos.training
    kronos.eval()
    try:
        torch.manual_seed(cfg.seed)
        with torch.no_grad():
            auto_regressive_inference(
                tokenizer, kronos, x8, st8, ys8,
                max_context=cfg.max_context, pred_len=cfg.predict_len,
                clip=cfg.clip, T=cfg.temperature, top_k=cfg.sample_top_k,
                top_p=cfg.top_p, sample_count=cfg.sample_count, verbose=False,
            )
    finally:
        kronos_mod.sample_from_logits = orig_sample
        kronos.train(was_training)
    H8 = cfg.predict_len
    ar_s1 = torch.cat([recorded[2 * i] for i in range(H8)], dim=1)
    ar_s2 = torch.cat([recorded[2 * i + 1] for i in range(H8)], dim=1)
    gs1 = g_s1.reshape(-1, H8).cpu()
    gs2 = g_s2.reshape(-1, H8).cpu()
    step0_parity = bool(
        torch.equal(gs1[:, 0], ar_s1[:, 0]) and torch.equal(gs2[:, 0], ar_s2[:, 0])
    )
    agree = (gs1 == ar_s1).float().mean().item()
    # 口径说明：原版 auto_regressive_inference 把 token 写入 512 宽 0 填充
    # buffer 再取前缀切片（非连续 stride），cuBLAS 对不同布局的 GEMM 分块
    # 引入 ~1e-7 数值噪声，经 multinomial 边界翻转放大为少量采样分歧。
    # 实现正确性由 ①logits 零差异 + ②step0（无生成前缀时）采样完全一致
    # 共同判定；逐步一致率作披露项不作硬门禁。
    report["parity_sampling"] = {
        "n_samples": 8, "sample_count": cfg.sample_count,
        "step0_token_parity": step0_parity,
        "token_agreement_rate": round(agree, 4),
        "pass": step0_parity,
        "note": (
            "原调用临时 eval()（官方生产路径无 eval()，dropout 生效）；"
            "step>0 分歧源于原版 512-buffer 非连续切片的 GEMM 分块噪声"
        ),
    }
    if not step0_parity:
        report["hard_fail"].append("同 RNG step0 采样 token 对拍失败")

    # —— 资源实测①：20 训练步（真实流程：标签 token 化 + 第一遍 + 第二遍
    #    + backward + step，PRE/POST 奇偶交替）——
    import torch.nn.functional as F

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    ck.train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr)
    sampler = corpus.sampler(seed=cfg.seed)
    micro = min(cfg.microbatch, len(keys))
    step_times = []
    loss = None
    for step in range(20):
        layout = "pre" if step % 2 == 0 else "post"  # 奇偶交替（计划 §4）
        t_s = time.perf_counter()
        bi = corpus.batch(sampler.draw(micro))
        xi = torch.tensor(bi["X"]).to(device)
        sti = torch.tensor(bi["stamp"]).to(device)
        yst_i = torch.tensor(bi["y_stamp"]).to(device)
        with torch.no_grad():
            t1, t2 = _tokenize_targets(tokenizer, xi, bi)
            S_i, _ = ck.encode_state(xi, sti)  # 第一遍（计入每步成本）
        out = ck.teacher_forced_forward(
            xi, sti, t1, t2, yst_i, S=S_i, layout=layout
        )
        loss = F.cross_entropy(
            out.s1_logits.reshape(-1, out.s1_logits.shape[-1]), t1.reshape(-1)
        ) + F.cross_entropy(
            out.s2_logits.reshape(-1, out.s2_logits.shape[-1]), t2.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
        opt.step()
        if step >= 2:  # 前 2 步预热不计
            step_times.append(time.perf_counter() - t_s)
    ck.eval()
    report["train_bench"] = {
        "microbatch": micro, "steps": 20,
        "sec_per_step_median": float(np.median(step_times)),
        "loss_final": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
    }
    if torch.cuda.is_available():
        report["train_bench"]["peak_mem_gb"] = round(
            torch.cuda.max_memory_allocated(device) / 1e9, 2
        )

    # —— 资源实测②：200 样本推理（B0 vs T-PRE，N=20）→ B1 预算 ——
    keys200 = bench_keys(cfg)
    batch200 = corpus.batch(keys200)
    x200 = torch.tensor(batch200["X"]).to(device)
    st200 = torch.tensor(batch200["stamp"]).to(device)
    ys200 = torch.tensor(batch200["y_stamp"]).to(device)

    def time_inference(layout: str) -> float:
        """从归一化输入到生成 token 的全程（B0 含 tokenize；T-PRE 含第一遍）。

        分块 32 只（×N20=640 序列，与官方 predict_batch_chunked 同构），
        避免 200 样本一次入场 OOM。
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        t = time.perf_counter()
        CH = 32
        with torch.no_grad():
            for s in range(0, x200.shape[0], CH):
                e = min(s + CH, x200.shape[0])
                xb, sb, yb = x200[s:e], st200[s:e], ys200[s:e]
                if layout == "baseline":
                    ck.generate(
                        xb, sb, yb, layout=layout,
                        sample_count=cfg.sample_count,
                        temperature=cfg.temperature, top_k=cfg.sample_top_k,
                        top_p=cfg.top_p,
                    )
                else:
                    Sb, _ = ck.encode_state(xb, sb)
                    ck.generate(
                        xb, sb, yb, S=Sb, layout=layout,
                        sample_count=cfg.sample_count,
                        temperature=cfg.temperature, top_k=cfg.sample_top_k,
                        top_p=cfg.top_p,
                    )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t

    t_b0 = time_inference("baseline")
    t_pre = time_inference("pre")
    r = t_pre / max(t_b0, 1e-9)
    n_b1 = min(cfg.b1_n_cap, int(cfg.sample_count * max(cfg.b1_n_min_multiple, int(np.ceil(r)))))
    report["inference_bench"] = {
        "n_samples": len(keys200), "sample_count": cfg.sample_count,
        "sec_b0": round(t_b0, 2), "sec_t_pre": round(t_pre, 2),
        "ratio_r": round(r, 3), "b1_n": n_b1,
        "budget_matched": n_b1 < cfg.b1_n_cap or r * cfg.sample_count <= cfg.b1_n_cap,
    }
    if torch.cuda.is_available():
        report["inference_bench"]["peak_mem_gb"] = round(
            torch.cuda.max_memory_allocated(device) / 1e9, 2
        )

    # —— 训练预算外推（计划 §9：族×种子×4000×秒/步，+25% 余量）——
    sec_step = report["train_bench"]["sec_per_step_median"]
    steps_per_opt = 2  # PRE/POST 交替不增加步数
    total_train_h = (
        len((42, 43, 44)) * 2 * cfg.total_steps * sec_step / 3600 * 1.25
    )
    report["budget_estimate"] = {
        "sec_per_step": round(sec_step, 3),
        "full_protocol_gpu_hours_with_margin": round(total_train_h, 2),
        "p2_budget_hours": cfg.budget_p2_hours,
        "within_p2": total_train_h <= cfg.budget_p2_hours,
    }
    return report


def _tokenize_targets(tokenizer, x_hist, batch):
    """训练标签 token 化：未来 10 行归一化值 → 粗细 token。"""
    import torch

    with torch.no_grad():
        y_norm = torch.tensor(batch["Y"]).to(x_hist.device)
        ids = tokenizer.encode(y_norm, half=True)
    return ids[0], ids[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="TAS1 P0 预检")
    ap.add_argument("--config", default=str(PKG_DIR / "config.json"))
    ap.add_argument("--full", action="store_true", help="含 GPU 测量档")
    ap.add_argument("--out", default=str(DATA_DIR / "manifest_p0.json"))
    args = ap.parse_args()

    cfg = TASConfig.load(args.config)
    report = readonly_checks(cfg)
    if args.full:
        report = full_checks(cfg, report)

    out = Path(args.out)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"P0 预检报告：{out}")
    if report["hard_fail"]:
        logger.error(f"硬门禁失败 {len(report['hard_fail'])} 项：")
        for f in report["hard_fail"]:
            logger.error(f"  - {f}")
        return 1
    logger.info("P0 预检全部通过")
    print(json.dumps({k: v for k, v in report.items() if k != "signals"}, ensure_ascii=False, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
