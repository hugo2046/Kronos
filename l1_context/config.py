"""L1 臂表 / 窗口 / 冻结锚（计划 §1，臂冻结）。

臂（计划 §1 表，逐字）：

| 臂        | 权重                     | 推理 L | 种子      | 窗口                     |
|-----------|--------------------------|--------|-----------|--------------------------|
| L250ZS100 | G1 s100（不动）          | 250    | 100       | backtest + 2025H2        |
| L250ZS101 | G1 s101（G2 重训，不动） | 250    | 101       | backtest + 2025H2        |
| L250ZS102 | G1 s102（G2 重训，不动） | 250    | 102       | backtest + 2025H2        |
| L500ZS100 | G1 s100（不动）          | 500    | 100       | backtest                 |
| L250FT100 | L250-ft 重训 predictor   | 250    | 100       | backtest                 |

其余逐字 canonical（H=10/N=20/T=1.0/top_p=0.9/推理 seed=42）、csi300。权重路径
与 ``finetune_suite.run_registry.SEED_MODEL_PATHS`` 同源（s101/s102 = G2 重训，
tokenizer 恒 G1 s100 冻结共享）。L90 锚信号既有 parquet 只读复用（§0：本轮不重跑）。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from baseline_suite.common import BaselineConfig

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
FINETUNE_OUTPUTS = REPO_ROOT / "finetune_suite" / "outputs" / "models"
R4_DATA = REPO_ROOT / "finetune_suite" / "data"
G5_DATA = REPO_ROOT / "g5_head" / "data"

# —— 窗口（§0/§1 冻结）——
WINDOW_DEFS: dict[str, tuple[str, str]] = {
    "backtest": ("2026-01-01", "2026-07-24"),
    "2025h2": ("2025-07-01", "2025-12-31"),
}

# —— G1 三种子权重（与 run_registry.SEED_MODEL_PATHS 逐字同源，只读）——
_G1_TOKENIZER = str(FINETUNE_OUTPUTS / "finetune_tokenizer_g1" / "checkpoints" / "best_model")
_SEED_PREDICTOR = {
    "s100": FINETUNE_OUTPUTS / "finetune_predictor_g1" / "checkpoints" / "best_model",
    "s101": FINETUNE_OUTPUTS / "finetune_predictor_g2_s101" / "checkpoints" / "best_model",
    "s102": FINETUNE_OUTPUTS / "finetune_predictor_g2_s102" / "checkpoints" / "best_model",
}

# —— L250-ft（4.3 产物：本包内重训，tokenizer 冻结共享 G1 s100）——
L250FT_CKPT_DIR = PKG_DIR / "outputs" / "models" / "finetune_predictor_l250ft" / "checkpoints"
L250FT_PREDICTOR_PATH = L250FT_CKPT_DIR / "best_model"
L250FT_DATASET_DIR = PKG_DIR / "data" / "ashares_lb250"

# —— 臂表（冻结；parquet/引擎/DuckDB 标签 = L1 + tag）——
ARMS: dict[str, dict] = {
    "L250ZS100": {"lookback": 250, "seed": "s100", "kind": "zs",
                  "windows": ("backtest", "2025h2")},
    "L250ZS101": {"lookback": 250, "seed": "s101", "kind": "zs",
                  "windows": ("backtest", "2025h2")},
    "L250ZS102": {"lookback": 250, "seed": "s102", "kind": "zs",
                  "windows": ("backtest", "2025h2")},
    "L500ZS100": {"lookback": 500, "seed": "s100", "kind": "zs",
                  "windows": ("backtest",)},
    "L250FT100": {"lookback": 250, "seed": "l250ft", "kind": "ft",
                  "windows": ("backtest",)},
}
for _tag, _spec in ARMS.items():
    _spec["model_path"] = (
        str(L250FT_PREDICTOR_PATH) if _spec["kind"] == "ft"
        else str(_SEED_PREDICTOR[_spec["seed"]])
    )
    _spec["tokenizer_path"] = _G1_TOKENIZER  # 全臂冻结共享 G1 s100 tokenizer

# —— 在位者 L90 锚（§0：mean 变体，既有信号只读，冻结 AER 等权）——
L90_ANCHOR_PARQUETS: dict[str, dict[str, Path]] = {
    "backtest": {
        "s100": R4_DATA / "g1" / "daily_signals_backtest_G1_mean.parquet",
        "s101": R4_DATA / "g2" / "s101" / "daily_signals_backtest_G2S101_mean.parquet",
        "s102": R4_DATA / "g2" / "s102" / "daily_signals_backtest_G2S102_mean.parquet",
    },
    "2025h2": {
        "s100": G5_DATA / "daily_signals_2025h2_G1_mean.parquet",
        "s101": R4_DATA / "g2" / "s101" / "daily_signals_2025h2_G2S101_mean.parquet",
        "s102": R4_DATA / "g2" / "s102" / "daily_signals_2025h2_G2S102_mean.parquet",
    },
}
# docs/G2实验结果（g2_judge_results.json 逐位）：三种子 backtest AER(等权)
L90_FROZEN_AER_EW: dict[str, float] = {
    "s100": 0.14333,
    "s101": 0.29059,
    "s102": 0.18671,
}


def arm_tag(tag: str) -> str:
    """DuckDB / 文件名臂标签：L250ZS100 → L1L250ZS100。"""
    return f"L1{tag}"


def build_arm_config(tag: str, window: str) -> BaselineConfig:
    """canonical(oos) + 窗口 + **唯一变量 = lookback 与权重路径**（§1）。"""
    spec = ARMS[tag]
    start, end = WINDOW_DEFS[window]
    return replace(
        BaselineConfig.load(window="oos"),
        window=f"{window}_{arm_tag(tag)}",  # 断点续跑文件名即最终名
        backtest_start=start,
        backtest_end=end,
        lookback=spec["lookback"],
        model_name=spec["model_path"],
        tokenizer_name=spec["tokenizer_path"],
    )


def _make_l250ft_config():
    """L250-ft 配方：G1 配方仅 lookback_window=250 + 数据/落盘/标签三处。"""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from finetune_suite.train_g1 import G1Config

    class _L250FTConfig(G1Config):
        """G1 配方逐字继承；唯一训练变量 = lookback_window 250（几何对齐）。

        - 语料 = 全 A（ashares）同源重建（min_len = 250+10+1 剔除阈值随窗重算）；
        - tokenizer 冻结共享 G1 s100（阶段 2.1 产物只读）；
        - predictor 从官方 Kronos-base 起训（G1 同款起点）、epochs=15、CE 早停、
          seed=100、AdamW 4e-5 / OneCycle——全部继承 Config 不改；
        - **最小声明修正（计划内不可行性，先于任何训练/评估反馈发现，非搜索——
          G5 ffn 512→256 同款裁决规则）**：官方 val_time_range（2025-01-01~
          2025-06-30，约 117 交易日）< window=261，lookback=250 下 val 可采样
          窗口数为 0、CE 早停退化为 epoch1 定型。将 val_time_range 最小扩展为
          2024-01-02~2025-06-30（约 355 交易日 ≥ 261+94 窗），val CE 口径从
          "2025H1 标签窗"变为"2024~2025H1 标签窗"；train 切分/语料/其余超参
          逐字不动。修正理由与影响如实写入结果文档。
        """

        def __init__(self):
            super().__init__()
            self.lookback_window = 250
            self.val_time_range = ["2024-01-02", "2025-06-30"]
            self.dataset_path = str(L250FT_DATASET_DIR)
            self.save_path = str(PKG_DIR / "outputs" / "models")
            self.predictor_save_folder_name = "finetune_predictor_l250ft"
            self.comet_tag = self.comet_name = "l1_context_l250ft"
            self.finetuned_predictor_path = (
                f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
            )

    return _L250FTConfig


def __getattr__(name: str):
    """惰性导出 L250FTConfig（避免模块导入期触发 finetune/ 路径注入副作用）。"""
    if name == "L250FTConfig":
        return _make_l250ft_config()
    raise AttributeError(name)


__all__ = [
    "ARMS", "WINDOW_DEFS", "L90_ANCHOR_PARQUETS", "L90_FROZEN_AER_EW",
    "L250FT_PREDICTOR_PATH", "L250FT_DATASET_DIR", "L250FT_CKPT_DIR",
    "arm_tag", "build_arm_config", "L250FTConfig",
]
