"""MH1 冻结常量、路径配置与协议序列化（计划 §3/§7，跑前冻结）。

全部常量在正式训练开始前冻结进 ``protocol.json`` 并记录 SHA256；后续阶段
（smoke/train/evaluate）必须校验哈希一致，任何改动视同协议变更。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ============================================================
# 实验设计（计划 §3 冻结表）
# ============================================================

HORIZONS: tuple[int, ...] = (1, 5, 10, 20)   # 期限（交易日），固定顺序
MAIN_HORIZON_IDX: int = 2                    # 主期限 = 10 日
POOL: str = "csi300"
LOOKBACK: int = 90
CLIP: float = 5.0
HEAD_HIDDEN: int = 128

# 数据窗（计划 §3 / §4）
TRAIN_START: str = "2014-01-02"      # 训练决策日起点
TRAIN_LABEL_END: str = "2024-12-31"  # 训练四期限标签终点上界
VAL_START: str = "2025-01-01"        # 验证（选点）决策日下界
VAL_LABEL_END: str = "2025-06-30"    # 验证标签终点上界
W3_START, W3_END = "2025-07-01", "2025-12-31"   # 研究评价窗 3
W4_START, W4_END = "2026-01-01", "2026-07-24"   # 研究评价窗 4
FORWARD_CUTOFF: str = "2026-07-24"   # forward 封存线：任何查询/标签不得越过
PURGE: int = 20                      # 统一 purge（最长期限）

# 优化（计划 §3）
BATCH: int = 128
LR: float = 3e-4
WD: float = 0.01
STEPS_PER_EPOCH: int = 2000
EPOCHS: int = 15                     # 固定 15 epochs，两臂均不早停
MIN_TRAIN_CROSS: int = 128           # 训练日最少有效股票（batch 无放回下限）

# 种子与编排（计划 §3：S/M 成对同初始化/同日序/同抽样）
SEEDS: tuple[int, ...] = (42, 43, 44)
ARMS: tuple[str, ...] = ("S", "M")
RUN_ORDER: tuple[str, ...] = ("S42", "M42", "S43", "M43", "S44", "M44")

# 评估（计划 §6）
IC_MIN_STOCKS: int = 30              # 截面有效股票 <30 → IC 记缺失
NW_LAG: int = 9                      # NW(lag=k−1)，k=10

# 资源与调度
GPU_BUDGET_HOURS: float = 12.0       # 本次 GPU 实际计算上限
REGISTRY_GUARD: tuple[str, str] = ("16:25", "16:45")  # 16:30 登记 cron 错峰窗

# 依赖路径（只读绝对路径复用；输出放本包目录）
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
PKG_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PKG_DIR / "data"
LOG_DIR: Path = DATA_DIR / "logs"
PROTOCOL_PATH: Path = DATA_DIR / "protocol.json"
PROTOCOL_SHA_PATH: Path = DATA_DIR / "protocol.json.sha256"

# 精度（首轮 FP32；不切 AMP/量化）
PRECISION: str = "fp32"


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """计算文件 SHA256（流式，大权重文件安全）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_protocol(record: dict) -> tuple[Path, str]:
    """冻结协议：写 protocol.json + 其 SHA256，返回 (路径, 哈希)。

    协议内容不含自身哈希（自引用无意义）；后续阶段用 :func:`verify_protocol`
    重算比对。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    PROTOCOL_SHA_PATH.write_text(digest + "\n", encoding="utf-8")
    return PROTOCOL_PATH, digest


def verify_protocol() -> tuple[dict, str]:
    """校验协议哈希一致并返回 (协议记录, 实测哈希)。

    :raises RuntimeError: 协议缺失或哈希不一致（协议变更须重跑 preflight 并
        披露，不得静默继续）。
    """
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SHA_PATH.is_file():
        raise RuntimeError("protocol.json 缺失：先运行 --stage preflight 冻结协议")
    record = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    frozen = PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip()
    if digest != frozen:
        raise RuntimeError(
            f"协议哈希不一致：冻结 {frozen[:16]}… vs 实测 {digest[:16]}…"
            "（协议已变更，须重跑 preflight 并披露）")
    return record, digest


__all__ = [
    "HORIZONS", "MAIN_HORIZON_IDX", "POOL", "LOOKBACK", "CLIP", "HEAD_HIDDEN",
    "TRAIN_START", "TRAIN_LABEL_END", "VAL_START", "VAL_LABEL_END",
    "W3_START", "W3_END", "W4_START", "W4_END", "FORWARD_CUTOFF", "PURGE",
    "BATCH", "LR", "WD", "STEPS_PER_EPOCH", "EPOCHS", "MIN_TRAIN_CROSS",
    "SEEDS", "ARMS", "RUN_ORDER", "IC_MIN_STOCKS", "NW_LAG",
    "GPU_BUDGET_HOURS", "REGISTRY_GUARD", "PRECISION",
    "REPO_ROOT", "PKG_DIR", "DATA_DIR", "LOG_DIR",
    "PROTOCOL_PATH", "PROTOCOL_SHA_PATH",
    "sha256_file", "write_protocol", "verify_protocol",
]
