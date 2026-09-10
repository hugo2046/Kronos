"""SAE 残差预测头优化 G1 mean：冻结协议配置（计划 20260910 §2~§6）。

全部实验自由度在本模块一次性冻结：样本规则、teacher 推理协议、头结构、
训练预算、Qlib 回测口径。运行期只读，不允许下游改写。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
# 提交目录（manifest/信号/报告/图表/小头权重入库；大缓存留 sae_residual/data）
ART_DIR = REPO_ROOT / "artifacts" / "experiments" / "sae_residual_20260910"
CACHE_DIR = PKG_DIR / "data" / "cache"

# —— 数据边界（计划 §1/§4）——
FORWARD_CUTOFF = "2026-07-24"          # 真实行情/特征/标签读取上界
DATA_END = "2026-08-07"                # DDB 真实数据末日（calendar 边界用）

# —— 固定样本（§4）——
POOL = "csi300"
LOOKBACK = 90                          # L=90
PREDICT_LEN = 10                       # H=10
TRAIN_START = "2014-01-02"
TRAIN_LABEL_END = "2024-12-31"         # 10 日标签终点上界（训练）
VAL_START = "2025-01-01"
VAL_LABEL_END = "2025-06-30"
STRIDE = 5                             # 每 5 个交易日一个决策日
PER_DAY = 64                           # 每日无放回抽样股票数
DATA_RNG_SEED = 17                     # 抽样 RNG（与 head seed 无关）

# 测试窗（候选=原 G1 有效信号集合，§4）
TEST_WINDOWS: dict[str, tuple[str, str]] = {
    "W3": ("2025-07-01", "2025-12-31"),
    "W4": ("2026-01-01", "2026-07-24"),
}

# —— teacher / hidden 推理协议（与原 G1 信号生成同参，§3/§4）——
TEACHER = dict(T=1.0, top_p=0.9, top_k=0, sample_count=20, clip=5,
               max_context=512, seed=42, chunk_size=32, sort_codes=True)
HIDDEN_CHUNK = 256                     # 隐状态提取单批股票数（纯前向，无采样）

# —— 监督目标（§3）——
EPS_STD = 1e-5                         # predict_batch 归一化分母 (std + 1e-5)
NORM_CLIP = 5.0
STATS_FLOOR = 1e-6                     # μh/σh/σe 的 max(σ, floor) 地板

# —— 头结构 / 损失 / 训练（§5，全部不搜索）——
HEAD = dict(enc_dim=64, z_dim=16)
LOSS = dict(recon_weight=0.1, rho=0.05, beta_ae=0.0, beta_sae=1e-3)
TRAIN = dict(lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01, batch_size=256,
             grad_clip=3.0, epochs=100, dtype="float32")
ARMS = ("AE", "SAE")                   # beta=0 / beta=0.001
PILOT_SEED = 100
CONFIRM_SEEDS = (101, 102)

# —— Qlib 主评价（§6，与 0b982e8 绑定；实际 kwargs 复用 qlib_mean_comparison）——
EXPERIMENT_R = "kronos-sae-residual-20260910"

# —— 基线身份（§2 冻结；路径以 Debian 实存为准）——
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
# 原 G1 两窗 report（复现门禁参照，mean_comparison 20260909 产物）
G1_REPORT_REF = REPO_ROOT / "artifacts" / "experiments" / "mean_comparison_20260909"

# —— GPU 预算（§1：12 小时总预算含缓存/失败尝试）——
GPU_BUDGET_S = 12 * 3600
BUDGET_LEDGER = ART_DIR / "budget_ledger.json"

PROTOCOL_VERSION = "sae-residual-protocol-v1"


@dataclass(frozen=True)
class CacheIdentity:
    """缓存身份（计划 §4：身份不匹配拒绝复用）。

    :ivar protocol: 协议版本串。
    :ivar tokenizer_sha: G1 tokenizer ``model.safetensors`` SHA256。
    :ivar predictor_sha: G1 predictor ``model.safetensors`` SHA256。
    :ivar sample_sha: 冻结样本键 parquet 的 SHA256（无则空串）。
    :ivar split: 数据段名（train/val/W3/W4）。
    """

    protocol: str = PROTOCOL_VERSION
    tokenizer_sha: str = ""
    predictor_sha: str = ""
    sample_sha: str = ""
    split: str = ""

    def to_dict(self) -> dict:
        return dict(protocol=self.protocol, tokenizer_sha=self.tokenizer_sha,
                    predictor_sha=self.predictor_sha, sample_sha=self.sample_sha,
                    split=self.split)

    @classmethod
    def from_dict(cls, d: dict) -> "CacheIdentity":
        return cls(**{k: d.get(k, "") for k in
                      ("protocol", "tokenizer_sha", "predictor_sha",
                       "sample_sha", "split")})


__all__ = ["ART_DIR", "CACHE_DIR", "CacheIdentity", "ARMS", "PILOT_SEED",
           "CONFIRM_SEEDS", "TEST_WINDOWS", "BASELINE_SIGNALS", "POOL"]
