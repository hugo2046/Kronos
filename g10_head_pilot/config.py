"""G10H-pilot 冻结协议、路径与工具（计划 §2/§3/§4）。

只读数据经绝对路径指向 Debian 主工作树；输出全部落本包
``data/<run_id>/``（run_id 含批次时间，不覆盖）。上界纪律：任何真实
行情/收益/特征读取 ≤ ``FORWARD_CUTOFF``（2026-07-24）；完整逐日推理需要
未来时间戳时只允许交易日历日期。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

SOURCE_ROOT = Path("/home/user/workspace/Kronos")   # 原始数据只读根（主工作树）
REPO_ROOT = Path(__file__).resolve().parent.parent

ARMS = ("A0", "A1")
SEED = 100
EPOCHS = 15
BATCH = 50
N_TRAIN_ITER = 2000 * BATCH          # 100,000 样本/epoch（n_train_iter 语义）
N_VAL_ITER = 400 * BATCH             # 20,000 样本/epoch
N_VAL_BATCHES_PER_EPOCH = 400        # 验证批数封顶（n_val_iter 语义）
LR = 4e-5
ADAM_BETAS = (0.9, 0.95)
ADAM_WD = 0.1
GRAD_CLIP = 3.0
PCT_START, DIV_FACTOR = 0.03, 10
NUM_WORKERS = 2

# A1 允许更新的参数（计划 §3 精确清单；禁止 tokenizer.head）
A1_ALLOWED = {
    "head.proj_s1.weight", "head.proj_s1.bias",
    "head.proj_s2.weight", "head.proj_s2.bias",
}

# 推理（计划 §4 固定）
INFERENCE = {
    "lookback": 90, "pred_len": 10, "sample_count": 20, "T": 1.0,
    "top_p": 0.9, "top_k": 0, "clip": 5, "max_context": 512,
    "variant": "mean", "seed": 100,
}

# 评价窗口（FULL 逐日；W4 上界=封存线）
W3_START, W3_END = "2025-07-01", "2025-12-31"
W4_START, W4_END = "2026-01-01", "2026-07-24"
FORWARD_CUTOFF = "2026-07-24"
WINDOW_BOUNDS = {"W3": (W3_START, W3_END), "W4": (W4_START, W4_END)}
IC_HORIZON = 10
NW_LAG = 9
IC_MIN_STOCKS = 30

# 资源与调度
GPU_BUDGET_HOURS = 12.0
REGISTRY_GUARD = ("16:25", "16:45")

# 资产路径
G1_TOKENIZER = (SOURCE_ROOT / "finetune_suite" / "outputs" / "models" /
                "finetune_tokenizer_g1" / "checkpoints" / "best_model")
OFFICIAL_PREDICTOR = "NeoQuasar/Kronos-base"      # 与 G1 训练起点一致（本地 HF 缓存）
ASHARES_DATA = SOURCE_ROOT / "finetune_suite" / "data" / "ashares"
G1_FULL_SIGNALS = {   # 在位参照：原完整逐日 mean（非 MH1 裁剪版）
    "W3": SOURCE_ROOT / "g5_head" / "data" / "daily_signals_2025h2_G1_mean.parquet",
    "W4": SOURCE_ROOT / "finetune_suite" / "data" / "g1" /
          "daily_signals_backtest_G1_mean.parquet",
}
# FULL 复现门禁参照（git 内，f3e9807 产物）
GRID_AUDIT_FULL_DAILY = (REPO_ROOT / "mh1_multihorizon" / "data" /
                         "grid_audit_20260908" / "daily")

def _resolve_run_dir() -> Path:
    """run 目录跨进程稳定：取最新含 protocol.json 的 g10h_*；无则新建。"""
    data_root = Path(__file__).resolve().parent / "data"
    if data_root.is_dir():
        cands = sorted(p for p in data_root.glob("g10h_*") if p.is_dir()
                       and (p / "protocol.json").is_file())
        if cands:
            return cands[-1]
    return data_root / f"g10h_{datetime.now().strftime('%Y%m%d_%H%M')}"


RUN_DIR = _resolve_run_dir()
RUN_ID = RUN_DIR.name
PROTOCOL_PATH = RUN_DIR / "protocol.json"


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def state_hash(state: dict) -> str:
    """state_dict 稳定哈希（两臂起点一致性对拍）。"""
    import numpy as np

    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def write_protocol(record: dict) -> str:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def load_protocol() -> tuple[dict, str]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("protocol.json 缺失：先运行 --stage preflight")
    record = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    sha = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    frozen = (RUN_DIR / "protocol.json.sha256")
    if frozen.is_file():
        expect = frozen.read_text(encoding="utf-8").strip()
        if sha != expect:
            raise RuntimeError("协议哈希漂移：不允许训练后改协议")
    else:
        frozen.write_text(sha + "\n", encoding="utf-8")
    return record, sha


def in_guard(now: datetime | None = None) -> bool:
    t = (now or datetime.now()).time().strftime("%H:%M")
    return REGISTRY_GUARD[0] <= t < REGISTRY_GUARD[1]


__all__ = [
    "SOURCE_ROOT", "ARMS", "SEED", "EPOCHS", "BATCH", "N_TRAIN_ITER",
    "N_VAL_ITER", "N_VAL_BATCHES_PER_EPOCH", "LR", "ADAM_BETAS", "ADAM_WD",
    "GRAD_CLIP", "A1_ALLOWED",
    "INFERENCE", "WINDOW_BOUNDS", "FORWARD_CUTOFF", "IC_HORIZON", "NW_LAG",
    "IC_MIN_STOCKS", "GPU_BUDGET_HOURS", "REGISTRY_GUARD", "G1_TOKENIZER",
    "OFFICIAL_PREDICTOR", "ASHARES_DATA", "G1_FULL_SIGNALS",
    "GRID_AUDIT_FULL_DAILY", "RUN_ID", "RUN_DIR", "PROTOCOL_PATH",
    "sha256_file", "state_hash", "write_protocol", "load_protocol", "in_guard",
]
