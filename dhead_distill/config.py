"""DHead v1 冻结协议配置与路径解析（方案 §3~§5、§10）。

协议字段（几何/时间窗/预算/种子/教师/学生/训练/LoRA）全部进
:func:`protocol_hash`——任何一项变动即新协议，run 目录按 hash 隔离（§6）。
路径字段（基仓、产物根、G1 权重）**不入**协议 hash：它们由远程 Debian 的
环境变量 ``DHEAD_BASE_REPO`` / ``DHEAD_ARTIFACT_ROOT`` 决定，与协议无关。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# ======================================================================
# 预算规格（方案 §3.3 / §8.1）
# ======================================================================


@dataclass(frozen=True)
class BudgetSpec:
    """采样预算：pilot 与 main 的唯一区别（§8.1）。

    :param train_dates: 训练等间隔决策日上限（main=512，pilot=64）。
    :param val_dates: 验证（2025H1）等间隔决策日上限（main=32，pilot=16）。
    :param diag_dates: 历史诊断（2025H2）等间隔决策日上限（两档均 64）。
    :param replay_dates: 历史回放（2026 窗）等间隔决策日上限；
        方案 §3.3 未给回放日数，v1 固定 64 并在结果文档披露。
    :param per_day: 每日稳定哈希选取的样本上限（main=128，pilot=64）。
    :param min_per_day: 低于该数整日剔除并披露（main=32，pilot=16）。
    """

    train_dates: int = 512
    val_dates: int = 32
    diag_dates: int = 64
    replay_dates: int = 64
    per_day: int = 128
    min_per_day: int = 32


PILOT_BUDGET = BudgetSpec(
    train_dates=64, val_dates=16, diag_dates=64, replay_dates=64,
    per_day=64, min_per_day=16,
)
MAIN_BUDGET = BudgetSpec()

#: profile → 预算（§8.1）；未知 profile 在 CLI 层显式报错
PROFILES: dict[str, BudgetSpec] = {"pilot": PILOT_BUDGET, "main": MAIN_BUDGET}


# ======================================================================
# 冻结协议配置
# ======================================================================


@dataclass(frozen=True)
class DHeadConfig:
    """DHead v1 全协议字段（方案 §3 数据、§4 结构与教师、§5 训练）。

    全部字段参与 :func:`protocol_hash`；修改任一字段 = 新协议。
    """

    # —— 几何与特征（§3.1）——
    lookback: int = 90
    predict_len: int = 10
    feature_cols: tuple[str, ...] = (
        "open", "high", "low", "close", "volume", "amount",
    )
    clip: float = 5.0
    zscore_eps: float = 1e-5
    # 标签与统计量：原始小数收益，不逐日 z-score（§3.1）
    # —— 时间窗（§3.2，决策日边界；purge 用交易日历推进）——
    train_start: str = "2014-01-02"
    train_end: str = "2024-12-31"      # 训练 t+10 交易日不晚于此
    val_start: str = "2025-01-01"
    val_end: str = "2025-06-30"        # 验证全部标签落在 H1
    diag_start: str = "2025-07-01"
    diag_end: str = "2025-12-31"       # 诊断全部标签落在 H2
    replay_start: str = "2026-01-01"
    replay_end: str = "2026-07-24"     # 有标签指标只算 t+10≤末日
    seal_date: str = "2026-07-25"      # 封存线：不加载此日及之后的价格/标签
    # —— 池（§3.3：训练全 A PIT，验证/诊断/回放 csi300 PIT）——
    train_pool: str = "ashares"
    eval_pool: str = "csi300"
    # —— 采样预算与种子（§3.3）——
    profile: str = "main"
    budget: BudgetSpec = MAIN_BUDGET
    list_seed: int = 20260905
    train_seed: int = 42
    confirm_seeds: tuple[int, ...] = (43, 44)
    # —— 教师协议（§4.2，复现 G1 历史生成行为）——
    teacher_T: float = 1.0
    teacher_top_p: float = 0.9
    teacher_top_k: int = 0
    teacher_n_paths: int = 20
    teacher_replicas: int = 3
    # —— 学生头（§4.1）——
    d_model: int = 832                 # 构造时校验 G1 d_model（§4.1）
    head_dim: int = 128
    head_n_heads: int = 4
    calendar_cardinalities: tuple[int, ...] = (60, 24, 7, 32, 13)
    n_horizons: int = 10
    # —— 训练（§5）——
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_epochs: int = 15
    patience: int = 3
    loss_w_s: float = 0.5
    loss_w_d: float = 0.5
    loss_w_i: float = 0.05
    # —— LoRA（§5，条件 D2）——
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_last_layers: int = 2          # 仅末两层
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    lora_lr: float = 1e-5

    # ------------------------------------------------------------------
    # 便捷构造
    # ------------------------------------------------------------------

    @staticmethod
    def with_profile(profile: str) -> "DHeadConfig":
        """按 profile 名构造（pilot / main），未知名抛错。"""
        if profile not in PROFILES:
            raise ValueError(f"未知 profile：{profile}（可选 {sorted(PROFILES)}）")
        return DHeadConfig(profile=profile, budget=PROFILES[profile])

    @staticmethod
    def with_list_seed(cfg: "DHeadConfig", seed: int) -> "DHeadConfig":
        """更换清单 seed（仅测试/种子族诊断用）。"""
        return replace(cfg, list_seed=seed)

    def split_spec(self, split: str) -> dict[str, Any]:
        """split → {start, end, pool, n_dates, label_required}（§3.2/§3.3）。"""
        specs = {
            "train": dict(
                start=self.train_start, end=self.train_end, pool=self.train_pool,
                n_dates=self.budget.train_dates, label_required=True,
                label_end=self.train_end,
            ),
            "val": dict(
                start=self.val_start, end=self.val_end, pool=self.eval_pool,
                n_dates=self.budget.val_dates, label_required=True,
                label_end=self.val_end,
            ),
            "diag": dict(
                start=self.diag_start, end=self.diag_end, pool=self.eval_pool,
                n_dates=self.budget.diag_dates, label_required=True,
                label_end=self.diag_end,
            ),
            "replay": dict(
                start=self.replay_start, end=self.replay_end, pool=self.eval_pool,
                n_dates=self.budget.replay_dates, label_required=False,
                label_end=self.replay_end,
            ),
        }
        if split not in specs:
            raise ValueError(f"未知 split：{split}（可选 {sorted(specs)}）")
        return specs[split]


def _canonical(obj: Any) -> Any:
    """协议对象的 canonical 形式（tuple→list，dataclass→dict，float 原样）。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"协议 hash 不支持类型 {type(obj)}")


def protocol_hash(cfg: DHeadConfig) -> str:
    """协议字段的 SHA256（canonical JSON，排序键；路径/环境不入 hash）。"""
    payload = json.dumps(_canonical(cfg), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def replace_profile_budget(
    cfg: DHeadConfig,
    *,
    train_dates: int | None = None,
    val_dates: int | None = None,
    diag_dates: int | None = None,
    replay_dates: int | None = None,
    per_day: int | None = None,
    min_per_day: int | None = None,
) -> DHeadConfig:
    """仅替换预算字段（离线测试用缩小预算；协议其余字段不动）。"""
    fields = {
        "train_dates": train_dates, "val_dates": val_dates,
        "diag_dates": diag_dates, "replay_dates": replay_dates,
        "per_day": per_day, "min_per_day": min_per_day,
    }
    new_budget = replace(
        cfg.budget, **{k: v for k, v in fields.items() if v is not None},
    )
    return replace(cfg, budget=new_budget)


# ======================================================================
# 路径与环境（§10；不入协议 hash）
# ======================================================================


@dataclass(frozen=True)
class EnvPaths:
    """远程 Debian 资产映射（§10.2）。

    :param base_repo: 原研究仓库绝对路径（``DHEAD_BASE_REPO``），
        G1 权重 / .env / 历史产物的只读来源。
    :param artifact_root: 本实验产物根（``DHEAD_ARTIFACT_ROOT``，
        默认工作树同级 ``Kronos-dhead-artifacts``）。
    :param worktree: 当前代码工作树根。
    :param g1_tokenizer: G1 微调 tokenizer 权重目录（只读）。
    :param g1_predictor: G1 微调 predictor 权重目录（只读）。
    """

    base_repo: Path
    artifact_root: Path
    worktree: Path
    g1_tokenizer: Path
    g1_predictor: Path


def resolve_env() -> EnvPaths:
    """解析环境变量 → 路径映射；缺 ``DHEAD_BASE_REPO`` 时显式报错。

    ``preflight`` 验证 base_repo 与当前工作树不同且确有目标资产（§10.1）。
    """
    base = os.environ.get("DHEAD_BASE_REPO", "").strip()
    if not base:
        raise RuntimeError(
            "DHEAD_BASE_REPO 未设置：请指向远程 Debian 上的原研究仓库"
            "（如 /home/user/workspace/Kronos），G1 权重与 .env 从该处只读解析。"
        )
    base_repo = Path(base).resolve()
    worktree = Path(__file__).resolve().parents[1]
    if base_repo == worktree:
        raise RuntimeError(
            f"DHEAD_BASE_REPO（{base_repo}）不得等于当前代码工作树："
            "新工作树不含被 git 忽略的 checkpoint/数据，误把自身当基仓会触发"
            "HF 误下载或重训（方案 §1）。"
        )
    art = os.environ.get("DHEAD_ARTIFACT_ROOT", "").strip()
    artifact_root = (
        Path(art).resolve() if art
        else (worktree.parent / "Kronos-dhead-artifacts").resolve()
    )
    # G1 权重映射：复用 G1Config 的字段语义（finetune_suite/train_g1.py），
    # 但显式按基仓绝对路径解析，不用新工作树算出的不存在路径（§10.2）。
    models = base_repo / "finetune_suite" / "outputs" / "models"
    return EnvPaths(
        base_repo=base_repo,
        artifact_root=artifact_root,
        worktree=worktree,
        g1_tokenizer=models / "finetune_tokenizer_g1" / "checkpoints" / "best_model",
        g1_predictor=models / "finetune_predictor_g1" / "checkpoints" / "best_model",
    )


def load_base_env(env: EnvPaths) -> None:
    """把基仓 ``.env`` 只读加载进进程环境（在 import qlib 之前生效，§10.2）。

    不覆盖已有变量；新工作树自身没有 .env，DDB 连接串必须来自基仓。
    """
    from dotenv import load_dotenv

    env_file = env.base_repo / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


__all__ = [
    "BudgetSpec",
    "DHeadConfig",
    "EnvPaths",
    "PROFILES",
    "protocol_hash",
    "replace_profile_budget",
    "resolve_env",
    "load_base_env",
]
