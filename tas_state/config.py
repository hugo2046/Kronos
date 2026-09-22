"""TAS1 不可变实验配置（计划 §1/§4/§5/§9，P0 冻结）。

配置一经 P0 签出即冻结：训练 / 评估 / 登记全部读取同一份 ``config.json``
及其 SHA256，任何字段变更都必须显式重签 manifest，不允许运行期漂移。

口径来源：

- 推理 / 采样 / 组合参数逐字复用 ``paper_replication/config.yaml``
  （canonical F0_mean 同源，禁止漂移）；
- TAS1 专属字段（状态段切分、训练配方、日期分割、STATIC 样本规则、
  B1 预算规则）来自计划 §3~§6，本模块固化。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

# tas_state/ 包目录与仓库根（manifest / config.json 的默认落盘位置）
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent

# canonical 推理口径来源（baseline_suite/common.py 同源约束）
PAPER_CONFIG_PATH = REPO_ROOT / "paper_replication" / "config.yaml"

# 臂名（计划 §4；顺序固定，下游表格一致）
ARMS: tuple[str, ...] = (
    "B0",  # 原始 F0_mean, N=20 —— canonical 基准
    "B1",  # 原始 F0, 较大 N 预算参照
    "B2",  # 既有 G1_s100/s101/s102 mean 集合参照（只读）
    "B3",  # 原始 10 日动量 M（免费参照，只读）
    "C0",  # STATIC：训练集固定平均 Z（同结构 MLP + q）
    "T-PRE",  # 动态状态前置 —— 唯一主候选
    "T-POST",  # 同 checkpoint 同 S，后置（位置效应）
    "T-SHUFFLE",  # 同 checkpoint，同日跨股票状态错配一位（破坏诊断）
)

# 模型族（计划 §4：只训练这两个族，各 42/43/44 三种子，第一阶段仅 42）
FAMILIES: tuple[str, ...] = ("dynamic", "static")
MAIN_SEED: int = 42
EXTRA_SEEDS: tuple[int, ...] = (43, 44)
ALL_SEEDS: tuple[int, ...] = (42, 43, 44)


@dataclass(frozen=True)
class TASConfig:
    """TAS1 全口径冻结配置（不可变 dataclass）。

    推理 / 采样字段与 ``paper_replication/config.yaml`` 逐字一致；
    ``from_paper_yaml`` 在构造时断言这一同源关系，防手工漂移。
    """

    # —— 底座与推理（canonical 同源，计划 §1.1）——
    model_name: str = "NeoQuasar/Kronos-base"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    lookback: int = 90
    predict_len: int = 10
    sample_count: int = 20
    temperature: float = 1.0
    top_p: float = 0.9
    sample_top_k: int = 0
    seed: int = 42
    clip: float = 5.0
    max_context: int = 512
    device: str = "cuda:0"
    signal_field: str = "close"
    # —— 组合引擎（计划 §1.1，engine_v2 消费）——
    top_k: int = 50
    drop_n: int = 5
    min_hold: int = 5
    cost_bps: int = 15

    # —— TAS1 架构（计划 §3）——
    # 90 个位置顺序切四段 [0:23] [23:46] [46:68] [68:90]；Z_j = concat(mean, H[89])
    segments: tuple[tuple[int, int], ...] = (
        (0, 23), (23, 46), (46, 68), (68, 90),
    )
    n_states: int = 4  # 状态 token 数（= 段数）
    state_hidden: int = 64  # 共享 MLP 隐层宽度（2d → 64 → d）

    # —— 数据分割（计划 §5.1）——
    corpus_start: str = "2014-01-02"  # 现有全 A 语料起点
    train_target_end: str = "2024-12-31"  # 所有训练目标日 ≤ 此日
    val_start: str = "2025-01-01"  # 验证 = 2025H1，目标日全在 H1 内
    val_target_end: str = "2025-06-30"
    eval_window_1: tuple[str, str] = ("2025-07-01", "2025-12-31")
    eval_window_2: tuple[str, str] = ("2026-01-01", "2026-07-24")
    # 原 forward 封存边界：2026-07-25 起既有登记按原协议封存，不读不写
    forward_seal_start: str = "2026-07-25"
    data_end: str = "2026-08-07"  # DDB 真实数据末日（baseline 同源）
    # 训练语料（G1 同源全 A pickle，只读）
    train_corpus_path: str = "finetune_suite/data/ashares/train_data.pkl"
    val_corpus_path: str = "finetune_suite/data/ashares/val_data.pkl"

    # —— 训练配方（计划 §5.2，预算选择非经验最优）——
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    effective_batch: int = 64
    total_steps: int = 4000
    ckpt_steps: tuple[int, ...] = (1000, 2000, 4000)
    # 训练时按 optimizer step 奇偶交替 PRE/POST（1:1，计划 §4）
    microbatch: int = 16  # P0 资源实测后冻结；OOM 时降并梯度累积保持 64

    # —— STATIC 固定平均 Z（计划 §4）——
    static_sample_count: int = 4096  # 合法键 SHA256 字典序前 4096；不足用全部并记录

    # —— B1 预算参照 N 规则（计划 §4）——
    b1_n_cap: int = 80
    b1_n_min_multiple: int = 2  # N = 20 × max(2, ceil(r))，上限 b1_n_cap

    # —— 判据（计划 §6）——
    ic_min_stocks: int = 30  # 每天至少 30 只
    hac_lag_main: int = 9  # Newey-West 主 lag（k=10 − 1）
    hac_lag_sensitivity: int = 20  # 敏感性附表
    paired_ic_threshold: float = 0.005  # 阶段 B 合并提升最低实际增量

    # —— 预算（计划 §9，GPU 小时）——
    budget_p0_p1_hours: float = 2.0
    budget_p2_hours: float = 24.0
    budget_p3_hours: float = 48.0

    # —— 只读对照信号（P0 记录哈希，计划 §7；仓库相对路径）——
    baseline_signal_paths: dict[str, str] = field(default_factory=lambda: {
        # F0_mean / M：W3 + W4（ic_multiwindow 同源路径）
        "F0_W3": "finetune_suite/data/g0/daily_signals_2025h2_F0_mean.parquet",
        "F0_W4": "finetune_suite/data/daily_signals_backtest_F0_mean.parquet",
        "M_W3": "finetune_suite/data/g0/daily_signals_2025h2_M.parquet",
        "M_W4": "finetune_suite/data/daily_signals_backtest_M.parquet",
        # G1 三种子（B2 参照，只读）
        "G1_s100_W3": "g5_head/data/daily_signals_2025h2_G1_mean.parquet",
        "G1_s100_W4": "finetune_suite/data/g1/daily_signals_backtest_G1_mean.parquet",
        "G1_s101_W3": "finetune_suite/data/g2/s101/daily_signals_2025h2_G2S101_mean.parquet",
        "G1_s101_W4": "finetune_suite/data/g2/s101/daily_signals_backtest_G2S101_mean.parquet",
        "G1_s102_W3": "finetune_suite/data/g2/s102/daily_signals_2025h2_G2S102_mean.parquet",
        "G1_s102_W4": "finetune_suite/data/g2/s102/daily_signals_backtest_G2S102_mean.parquet",
    })

    def __post_init__(self) -> None:
        """同源断言：推理 / 采样字段与 paper_replication/config.yaml 一致。"""
        raw = _load_paper_yaml()
        pairs = [
            (self.model_name, raw["inference"]["model_name"], "model_name"),
            (self.tokenizer_name, raw["inference"]["tokenizer_name"], "tokenizer_name"),
            (self.sample_count, raw["inference"]["sample_count"], "sample_count"),
            (self.temperature, raw["inference"]["T"], "temperature"),
            (self.top_p, raw["inference"]["top_p"], "top_p"),
            (self.sample_top_k, raw["inference"]["top_k"], "sample_top_k"),
            (self.seed, raw["inference"]["seed"], "seed"),
            (self.lookback, raw["data"]["lookback"], "lookback"),
            (self.predict_len, raw["data"]["predict_len"], "predict_len"),
            (self.top_k, raw["portfolio"]["top_k"], "top_k"),
            (self.drop_n, raw["portfolio"]["drop_n"], "drop_n"),
            (self.min_hold, raw["portfolio"]["min_hold"], "min_hold"),
            (self.cost_bps, raw["portfolio"]["cost_bps"], "cost_bps"),
        ]
        for ours, theirs, name in pairs:
            if ours != theirs:
                raise ValueError(
                    f"TASConfig.{name}={ours!r} 与 paper_replication/config.yaml "
                    f"的 {theirs!r} 不一致：canonical 口径禁止漂移"
                )
        # 段切分校验：四段顺序覆盖 [0, lookback)，无重叠无缝隙
        segs = list(self.segments)
        if segs[0][0] != 0 or segs[-1][1] != self.lookback:
            raise ValueError(f"segments 必须覆盖 [0, {self.lookback})：{segs}")
        for (_, hi), (lo2, _) in zip(segs, segs[1:]):
            if hi != lo2:
                raise ValueError(f"segments 必须顺序无缝：{segs}")
        if len(segs) != self.n_states:
            raise ValueError(f"段数 {len(segs)} 必须等于 n_states={self.n_states}")
        # 最长第二遍输入 4+90+1+9=104（生成后校验 105）< max_context
        max_len = self.n_states + self.lookback + 1 + (self.predict_len - 1)
        if max_len + 1 >= self.max_context:
            raise ValueError(f"第二遍最长输入 {max_len + 1} 须 < max_context")

    # —— 序列化 ——
    def to_json(self) -> str:
        """序列化为稳定排序 JSON（嵌套 tuple 转 list，可 round-trip）。"""
        d = asdict(self)
        d["segments"] = [list(s) for s in self.segments]
        d["ckpt_steps"] = list(self.ckpt_steps)
        d["eval_window_1"] = list(self.eval_window_1)
        d["eval_window_2"] = list(self.eval_window_2)
        return json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True)

    def sha256(self) -> str:
        """配置内容 SHA256（manifest 锚点；与落盘文件字节一致）。"""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "TASConfig":
        """从 JSON 恢复（tuple 字段显式还原，保证 sha256 round-trip）。"""
        d = json.loads(text)
        d["segments"] = tuple(tuple(s) for s in d["segments"])
        d["ckpt_steps"] = tuple(d["ckpt_steps"])
        d["eval_window_1"] = tuple(d["eval_window_1"])
        d["eval_window_2"] = tuple(d["eval_window_2"])
        d["baseline_signal_paths"] = dict(d["baseline_signal_paths"])
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TASConfig":
        """从 ``config.json`` 加载并校验哈希一致。

        :param path: 默认 ``tas_state/config.json``（随实现提交，计划 P0-3）。
        """
        p = Path(path) if path is not None else PKG_DIR / "config.json"
        text = p.read_text(encoding="utf-8")
        cfg = cls.from_json(text)
        if cfg.to_json() != text.rstrip("\n") and cfg.to_json() != text:
            # 允许尾部换行差异；其余任何字节差异都拒绝（防手改漂移）
            logger.warning(f"config.json 与 TASConfig 规范序列化存在字节差异：{p}")
        return cfg


def _load_paper_yaml() -> dict:
    """读 paper_replication/config.yaml（canonical 口径真源）。"""
    import yaml

    with open(PAPER_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_default_config(path: str | Path | None = None) -> Path:
    """把默认 TASConfig 落盘为 ``config.json``（P0 冻结动作，幂等）。"""
    p = Path(path) if path is not None else PKG_DIR / "config.json"
    cfg = TASConfig()
    p.write_text(cfg.to_json() + "\n", encoding="utf-8")
    logger.info(f"TAS1 配置已冻结：{p}（sha256={cfg.sha256()[:16]}…）")
    return p


if __name__ == "__main__":  # python -m tas_state.config → 冻结 / 校验
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "write":
        write_default_config()
    else:
        cfg = TASConfig.load()
        print(json.dumps({
            "sha256": cfg.sha256(),
            "arms": list(ARMS),
            "windows": [list(cfg.eval_window_1), list(cfg.eval_window_2)],
        }, ensure_ascii=False, indent=2))
