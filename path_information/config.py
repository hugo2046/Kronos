"""PATH1 采样路径信息增量最小实验：冻结协议配置（计划 20260911 §1~§8）。

全部实验自由度在本模块一次性冻结：三臂定义、时间边界、选股规则、采样
协议、头结构、训练预算。运行期只读，不允许下游改写。与 SAE/PEER 的
关键差异 = 输入是 20 条采样路径的形状（平均前的样本维信息），不是隐
状态；时间划分不复用 SAE 旧 train/val（2025-07 起三段），监督键也不同。
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
# Debian 交接（计划 §9）：真实旧权重/基线**只读引用主仓库目录**，
# 缓存与新输出全部落本 worktree；不复制凭据/权重入本仓库。
MAIN_REPO = REPO_ROOT.parent / "Kronos"

# 提交目录（协议/路径缓存/标签/小头/分数/诊断全部入库，目标 ≤50MiB）
ART_DIR = REPO_ROOT / "artifacts" / "experiments" / "path_information_20260911"
CACHE_DIR = ART_DIR / "cache"
HEADS_DIR = ART_DIR / "heads"
RUN_ID = "path1_v1"
PROTOCOL_VERSION = "path-information-protocol-v1"

# —— 数据边界（计划 §2）——
FORWARD_CUTOFF = "2026-07-24"          # 真实行情/特征/标签读取上界
DATA_END = "2026-08-07"                # DDB 真实数据末日（仅日历边界参照）

# —— 三段时间边界（§2 表；决策日候选区间 = [start, label_end]）——
SEGMENTS: dict[str, tuple[str, str]] = {
    "train": ("2025-07-01", "2025-09-30"),    # 唯一梯度更新 + 拟合标准化
    "val": ("2025-10-01", "2025-12-31"),      # 唯一 epoch 选点
    "dev_eval": ("2026-01-01", "2026-07-24"), # 一次固定诊断（已观察历史）
}
# 两窗来源关系：train/val 的原 G1 mean 取 W3 基线，dev_eval 取 W4
SEGMENT_WINDOW = {"train": "W3", "val": "W3", "dev_eval": "W4"}

# —— 样本规则（§3）——
POOL = "csi300"
LOOKBACK = 90                            # L=90（与 G1 输入窗口同）
PREDICT_LEN = 10                         # H=10
PER_DAY = 64                             # 每日 SHA256 排序前 64
SELECT_KEY = "PATH1|17|{date}|{code}"    # SHA256 升序键（17=数据 RNG 族）

# —— 路径生成协议（§4：与原 G1 推理同参，新采样链）——
INFERENCE = dict(T=1.0, top_p=0.9, top_k=0, sample_count=20, clip=5,
                 max_context=512, seed=42, chunk_size=32, dtype="float32")

# —— 监督目标 / 标准化（§5）——
EPS_STD = 1e-5                           # predict_batch 归一化分母 (std + 1e-5)
NORM_CLIP = 5.0
STATS_FLOOR = 1e-6                       # σe 数值地板（触地板停止）

# —— 头结构 / 训练（§5，共 753 参数，不搜索）——
HEAD = dict(path_dim=10, hidden=16, mix_in=17)
TRAIN = dict(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
             batch_days=4, grad_clip=3.0, epochs=50)
ARMS = ("MEAN", "PATH")                  # 零 shape 对照 / 真实 shape
SEED = 100                               # 固定训练 seed100

# —— 预算（§1：GPU 2 小时 / CPU 1 小时，失败尝试计入）——
GPU_BUDGET_S = 2 * 3600
CPU_BUDGET_S = 1 * 3600
BUDGET_LEDGER = ART_DIR / "budget_ledger.json"
# GPU 让位（§8）：工作日 16:30 登记任务；日边界探测显存占用
GPU_YIELD_AFTER = "16:25"
GPU_BUSY_MEM_MB = 2000                   # 其余进程显存占用超过此值视为忙
GPU_POLL_S = 300

# —— 基线身份（§3 冻结；路径只读引用主仓库）——
BASELINE_SIGNALS: dict[str, Path] = {
    "W3": MAIN_REPO / "g5_head" / "data" / "daily_signals_2025h2_G1_mean.parquet",
    "W4": MAIN_REPO / "finetune_suite" / "data" / "g1" /
          "daily_signals_backtest_G1_mean.parquet",
}
BASELINE_SHA256: dict[str, str] = {
    "W3": "9352b40302eb34c714bfebd808481b625a21d2f7f4f668ccca15cda0cfe21d66",
    "W4": "453e2aeae7fe4bee8ce8fa62a9908da0843d5ad8b52777cd3b299a84cc4a3e36",
}
G1_TOKENIZER = (MAIN_REPO / "finetune_suite" / "outputs" / "models" /
                "finetune_tokenizer_g1" / "checkpoints" / "best_model")
G1_PREDICTOR = (MAIN_REPO / "finetune_suite" / "outputs" / "models" /
                "finetune_predictor_g1" / "checkpoints" / "best_model")

REPORT_DOC = REPO_ROOT / "docs" / "Kronos采样路径信息增量实验结果_20260911.md"

__all__ = ["ART_DIR", "CACHE_DIR", "HEADS_DIR", "RUN_ID", "SEGMENTS",
           "SEGMENT_WINDOW", "BASELINE_SIGNALS", "ARMS", "SEED", "POOL",
           "INFERENCE", "PROTOCOL_VERSION", "MAIN_REPO"]
