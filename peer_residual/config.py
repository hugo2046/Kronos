"""PEER1 同日跨股票交互残差头：冻结协议配置（计划 20260910 §1~§7）。

全部实验自由度在本模块一次性冻结：PeerSet/LossSet 语义、头结构、训练
预算、Qlib 回测口径。运行期只读，不允许下游改写。与 SAE 实验的唯一
差异 = attention 是否允许访问同日其他股票（OFF 对角 / ON 全可见）。
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
# 提交目录（协议/信号/报告/图表/best 头权重入库；大缓存留 peer_residual/data）
ART_DIR = REPO_ROOT / "artifacts" / "experiments" / "peer_residual_20260910"
RUN_ID = "peer1_v1"
PEER_CACHE_DIR = PKG_DIR / "data" / RUN_ID
HEADS_DIR = PKG_DIR / "data" / "heads"

# —— 数据边界（计划 §1/§4）——
FORWARD_CUTOFF = "2026-07-24"          # 真实行情/特征/标签读取上界
DATA_END = "2026-08-07"                # DDB 真实数据末日（calendar 边界用）

# —— PeerSet / LossSet（§3）——
POOL = "csi300"
LOOKBACK = 90                          # L=90（与 G1 输入窗口同）
PREDICT_LEN = 10                       # H=10（仅窗口构造参数，PeerSet 不读标签）
TEST_WINDOWS: dict[str, tuple[str, str]] = {
    "W3": ("2025-07-01", "2025-12-31"),
    "W4": ("2026-01-01", "2026-07-24"),
}

# —— 隐状态提取（与 SAE 冻结缓存同口径，§3.1）——
HIDDEN_CHUNK = 256                     # 隐状态提取单批股票数（纯前向，无采样）

# —— 头结构 / 训练（§5，全部不搜索）——
HEAD = dict(d_model=64, n_heads=4, ffn_dim=128)
TRAIN = dict(lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01, batch_days=4,
             grad_clip=3.0, epochs=100)
ARMS = ("OFF", "ON")                   # 唯一实验变量 = 交互掩码
PILOT_SEED = 100
CONFIRM_SEEDS = (101, 102)

EXPERIMENT_R = "kronos-peer-residual-20260910"
PROTOCOL_VERSION = "peer-residual-protocol-v1"

# —— SAE 只读资产引用（§3.3/§4）——
SAE_ART = REPO_ROOT / "artifacts" / "experiments" / "sae_residual_20260910"
SAE_CACHE_DIR = REPO_ROOT / "sae_residual" / "data" / "cache"
SAE_NORM_STATS = SAE_ART / "norm_stats.json"
SAE_CACHE_MANIFEST = SAE_ART / "cache_manifest.json"
SAE_PROTOCOL = "sae-residual-protocol-v1"

# —— 基线身份（与 SAE 计划同冻结；路径以 Debian 实存为准）——
BASELINE_SIGNALS: dict[str, Path] = {
    "W3": REPO_ROOT / "g5_head" / "data" / "daily_signals_2025h2_G1_mean.parquet",
    "W4": REPO_ROOT / "finetune_suite" / "data" / "g1" /
          "daily_signals_backtest_G1_mean.parquet",
}
BASELINE_SHA256: dict[str, str] = {
    "W3": "9352b40302eb34c714bfebd808481b625a21d2f7f4f668ccca15cda0cfe21d66",
    "W4": "453e2aeae7fe4bee8ce8fa62a9908da0843d5ad8b52777cd3b299a84cc4a3e36",
}
G1_TOKENIZER = (REPO_ROOT / "finetune_suite" / "outputs" / "models" /
                "finetune_tokenizer_g1" / "checkpoints" / "best_model")
G1_PREDICTOR = (REPO_ROOT / "finetune_suite" / "outputs" / "models" /
                "finetune_predictor_g1" / "checkpoints" / "best_model")
G1_REPORT_REF = REPO_ROOT / "artifacts" / "experiments" / "mean_comparison_20260909"

# —— GPU 预算（§1：12 小时总预算含缓存/失败尝试）——
GPU_BUDGET_S = 12 * 3600
BUDGET_LEDGER = ART_DIR / "budget_ledger.json"

__all__ = ["ART_DIR", "PEER_CACHE_DIR", "HEADS_DIR", "RUN_ID", "ARMS",
           "PILOT_SEED", "CONFIRM_SEEDS", "TEST_WINDOWS", "BASELINE_SIGNALS",
           "POOL", "SAE_ART", "SAE_CACHE_DIR", "SAE_NORM_STATS", "HEADS_DIR"]
