"""G4 配置（G4 计划 §1/§6：特征集/种子冻结）。

继承 ``finetune_suite.config.Config``（两阶段协议/超参/epochs/batch/每 epoch
步数/lr 全部零改动），仅改：

1. ``feature_list``：6 → 9 列（前 6 列顺序逐字不变 + 市场三列，冻结）；
2. 数据路径 → ``g4_features/data``（9 列语料，不覆盖既有 pkl）；
3. 输出目录 → ``g4_features/outputs/models/finetune_{tokenizer,predictor}_g4_s{seed}``；
4. ``seed`` = 100/101/102（三种子完整跑）。

warm-start 源（冻结映射，跑前定案）：tokenizer 三种子全部 ←
``finetune_tokenizer_g1``（G1 族唯一 tokenizer，G2 共享条款；D-tok 为诊断臂
不属 G1 族）；predictor ← 对应种子之 G1 族权重
（s100=``finetune_predictor_g1``，s101=``finetune_predictor_g2_s101``，
s102=``finetune_predictor_g2_s102``）。
"""
from __future__ import annotations

from pathlib import Path

from finetune_suite.config import Config

SEEDS = (100, 101, 102)

_PKG_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _PKG_DIR / "outputs" / "models"

# G1 族 warm-start 源（只读；predictor 按种子映射，tokenizer 三种子共享）
G1_TOKENIZER_DIR = (
    _PKG_DIR.parent / "finetune_suite" / "outputs" / "models"
    / "finetune_tokenizer_g1" / "checkpoints" / "best_model"
)
G1_PREDICTOR_DIRS = {
    100: _PKG_DIR.parent / "finetune_suite" / "outputs" / "models"
         / "finetune_predictor_g1" / "checkpoints" / "best_model",
    101: _PKG_DIR.parent / "finetune_suite" / "outputs" / "models"
         / "finetune_predictor_g2_s101" / "checkpoints" / "best_model",
    102: _PKG_DIR.parent / "finetune_suite" / "outputs" / "models"
         / "finetune_predictor_g2_s102" / "checkpoints" / "best_model",
}

MARKET_COLS = ["idx_ret", "mkt_vol", "ma200_gate"]


class G4Config(Config):
    """G4 = 9 列特征 + 三种子，协议超参逐字继承 Config。"""

    def __init__(self, seed: int = 100):
        super().__init__()
        assert seed in SEEDS, f"G4 冻结：seed ∈ {SEEDS}"
        self.seed = seed

        self.feature_list = list(self.feature_list) + list(MARKET_COLS)  # 6→9 列

        self.dataset_path = str(_PKG_DIR / "data")
        self.save_path = str(_MODELS_DIR)
        self.tokenizer_save_folder_name = f"finetune_tokenizer_g4_s{seed}"
        self.predictor_save_folder_name = f"finetune_predictor_g4_s{seed}"
        self.comet_tag = self.comet_name = f"finetune_suite_g4_s{seed}"
        # 派生路径按新目录名重算
        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )
