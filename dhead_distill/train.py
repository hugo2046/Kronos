"""共用训练循环、臂初始化与 checkpoint 选择（方案 §5）。

要点：

- 六训练臂：``D0``（纯蒸馏）、``S``（直接监督）、``D1``（蒸馏后适配）、
  ``D2``（头 + 末两层 q/v LoRA 受限联合）、``S-long`` / ``D1-cont``
  （预算匹配对照，不受 15 epoch 上限与独立早停截断）；
- 梯度步按日 batch 等权（每 epoch 每日恰访问一次，shuffle 只变日序）；
  验证 loss 先按日计算再对日平均；
- checkpoint 选择只访问 train/val 指标（白名单校验，诊断/回放收益拒之门外），
  tie 取更早 epoch；D0 按验证 D 最小，其余按验证 ``S+0.05I`` 最小；
- 断点恢复：checkpoint 含模型/优化器/epoch/身份 hash；日顺序采用
  SHA256 派生的确定性 Fisher-Yates（每轮 ``hash(seed|epoch|i)`` 取混洗步）——
  恢复后后续 epoch 与连续训练逐位一致，无需携带 RNG 内部状态；
- checkpoint 用 ``weights_only=True`` 装载（载荷仅张量/基本类型，无任意对象）；
- D2 禁止读取冻结末层 hidden 缓存（构造期显式拒绝）。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from loguru import logger

from dhead_distill.backbone import StudentBackbone
from dhead_distill.config import DHeadConfig, protocol_hash
from dhead_distill.data import (
    DayManifest, affine_restore_params, safe_artifact_dir, window_zscore_clip,
)
from dhead_distill.head import MultiHorizonHead
from dhead_distill.losses import daily_ic_loss, normalized_mse

#: 合法训练臂（T 是教师基线，不训练）
TRAIN_ARMS = ("D0", "S", "D1", "D2", "S-long", "D1-cont")

#: checkpoint 选择指标白名单：只许 train/val + 诊断计数（不依据诊断/回放收益选 epoch）
_ALLOWED_METRIC_KEYS = frozenset(
    {"epoch", "train_loss", "val_loss", "val_task", "val_d", "val_s", "val_i",
     "seconds", "grad_norm", "nonfinite"}
)


def arm_loss(arm: str, s: torch.Tensor, d: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """按臂组合任务损失（§4.3 权重冻结，不搜索系数）。

    :param s: 真实监督 S（normalized_mse vs y_real）。
    :param d: 蒸馏损失 D（normalized_mse vs y_teacher_replica0）。
    :param i: 排序损失 I（daily_ic_loss vs y_real）。
    """
    if arm == "D0":
        return d
    if arm in ("S", "S-long"):
        return s + 0.05 * i
    if arm in ("D1", "D2", "D1-cont"):
        return 0.5 * s + 0.5 * d + 0.05 * i
    raise ValueError(f"未知臂：{arm}（可选 {TRAIN_ARMS}）")


def select_best_epoch(history: list[dict]) -> int:
    """按验证指标选 epoch：val_loss 最小，tie 取更早。

    history 项的键必须 ⊆ 白名单——混入回放/诊断收益直接报错（防泄漏选点）。
    """
    for h in history:
        bad = set(h) - _ALLOWED_METRIC_KEYS
        if bad:
            raise ValueError(f"非法指标键 {sorted(bad)}：checkpoint 选择只能访问 train/val")
    best = min(history, key=lambda h: (h["val_loss"], h["epoch"]))
    return int(best["epoch"])


def _val_criterion_arm(arm: str) -> str:
    """D0 按验证 D；其余按验证 S+0.05I（§5）。"""
    return "val_d" if arm == "D0" else "val_task"


def _package_code_hash() -> str:
    """dhead_distill 包源码指纹（R3：学生身份绑定代码 hash）。"""
    h = hashlib.sha256()
    pkg_dir = Path(__file__).resolve().parent
    for f in sorted(pkg_dir.glob("*.py")):
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()


def day_order(seed: int, epoch: int, n: int) -> list[int]:
    """确定性日序：SHA256 派生的 Fisher-Yates 混洗（可复现实验排序用）。

    每个混洗步 ``j = hash(seed|epoch|i) mod (i+1)``——同 seed/epoch 序列
    逐位可复现，断点恢复无需携带 RNG 内部状态。
    """
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        digest = hashlib.sha256(f"{seed}|{epoch}|{i}".encode("utf-8")).digest()
        j = int.from_bytes(digest[:8], "big") % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


@dataclass
class DayTensor:
    """一个决策日的物化批（清单顺序即批内顺序）。

    ``a``/``b``：R2 仿射还原系数 ``[B]``，由各样本历史 90 日原始 close 算出
    （:func:`dhead_distill.data.affine_restore_params`）——只在
    ``output_space="normalized_close_affine_return"`` 时被消费，**不来自
    未来标签**，经显式张量字段传递（无隐藏全局状态）。
    """

    date_iso: str
    x_norm: torch.Tensor      # [B,90,6]
    x_stamp: torch.Tensor     # [B,90,5]
    y_stamp: torch.Tensor     # [B,10,5]
    y_real: torch.Tensor      # [B,10]
    y_teacher: torch.Tensor   # [B,10]（replica 0）
    y_teacher_r1: torch.Tensor  # [B,10]（replica 1，独立保真诊断用）
    a: torch.Tensor           # [B] 仿射还原系数（R2）
    b: torch.Tensor           # [B]


def _materialize_days(m: DayManifest, teacher: np.ndarray) -> list[DayTensor]:
    """按日物化批张量（teacher 数组按 manifest.samples 顺序对齐）。"""
    days: list[DayTensor] = []
    idx_by_date: dict[str, list[int]] = {}
    for i, s in enumerate(m.samples):
        idx_by_date.setdefault(s.date.strftime("%Y-%m-%d"), []).append(i)
    for d_iso, idxs in idx_by_date.items():
        xs, ys, yt, yt1, aa, bb = [], [], [], [], [], []
        for i in idxs:
            s = m.samples[i]
            key = (d_iso, s.code)
            xs.append(window_zscore_clip(m.x_raw[key].astype(np.float64),
                                         eps=1e-5, clip=5.0))
            ys.append(s.y_real)
            yt.append(teacher[i, 0])
            yt1.append(teacher[i, 1])
            a_i, b_i = affine_restore_params(m.x_raw[key])  # 只用历史 90 行
            aa.append(a_i)
            bb.append(b_i)
        days.append(
            DayTensor(
                date_iso=d_iso,
                x_norm=torch.from_numpy(np.stack(xs)),
                x_stamp=torch.from_numpy(
                    np.tile(m.x_stamp[d_iso][None], (len(idxs), 1, 1))),
                y_stamp=torch.from_numpy(
                    np.tile(m.y_stamp[d_iso][None], (len(idxs), 1, 1))),
                y_real=torch.from_numpy(np.stack(ys)),
                y_teacher=torch.from_numpy(np.stack(yt)),
                y_teacher_r1=torch.from_numpy(np.stack(yt1)),
                a=torch.tensor(aa, dtype=torch.float32),
                b=torch.tensor(bb, dtype=torch.float32),
            )
        )
    return days


@dataclass
class TrainResult:
    """训练结果：逐 epoch 历史 + 最优点 + 产物目录。"""

    best_epoch: int
    history: list[dict]
    run_dir: Path
    criterion: str
    arm: str


class UnifiedTrainer:
    """统一训练器（六臂共用；教师 T 不训练）。"""

    def __init__(
        self,
        *,
        arm: str,
        cfg: DHeadConfig,
        backbone: StudentBackbone,
        head: MultiHorizonHead,
        scale: np.ndarray,
        train_manifest: DayManifest,
        val_manifest: DayManifest,
        train_teacher: np.ndarray,
        val_teacher: np.ndarray,
        run_name: str,
        seed: int,
        device: str = "cpu",
        lora_named: Optional[Sequence] = None,
        max_epochs_override: Optional[int] = None,
        disable_early_stop: bool = False,
        hidden_cache: Optional[dict] = None,
        output_space: str = "raw_return",
        backbone_weight_hash: str = "",
    ):
        if arm not in TRAIN_ARMS:
            raise ValueError(f"未知臂：{arm}（可选 {TRAIN_ARMS}；T 为教师基线不训练）")
        if arm == "D2" and hidden_cache is not None:
            raise ValueError(
                "D2 禁止读取冻结末层 hidden 缓存：末两层参与梯度，缓存隐状态与"
                "LoRA 路径不一致（§5）"
            )
        if arm == "D2" and lora_named is None:
            raise ValueError("D2 须传入 inject_lora 返回的 (name, param) LoRA 参数")
        self.arm = arm
        self.cfg = cfg
        self.backbone = backbone
        self.head = head
        self.device = device
        self.scale_t = torch.from_numpy(
            np.asarray(scale, dtype=np.float32)).to(device)
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.train_teacher = train_teacher
        self.val_teacher = val_teacher
        self.run_name = run_name
        self.seed = seed
        self.max_epochs = int(max_epochs_override or cfg.max_epochs)
        self.disable_early_stop = disable_early_stop
        self.hidden_cache = hidden_cache
        self.criterion = _val_criterion_arm(arm)

        self.train_days = _materialize_days(train_manifest, train_teacher)
        self.val_days = _materialize_days(val_manifest, val_teacher)
        self._teacher_hash = hashlib.sha256(
            train_teacher.tobytes() + val_teacher.tobytes()
        ).hexdigest()

        self._lora_named = dict(lora_named) if lora_named else {}
        params: list[dict] = [
            {"params": list(head.parameters()), "lr": cfg.lr,
             "weight_decay": cfg.weight_decay},
        ]
        if self._lora_named:
            params.append({"params": list(self._lora_named.values()),
                           "lr": cfg.lora_lr, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr,
                                           weight_decay=cfg.weight_decay)

        self.identity = {
            "arm": arm, "seed": seed, "protocol": protocol_hash(cfg),
            "train_hash": train_manifest.content_hash,
            "val_hash": val_manifest.content_hash,
            "teacher_hash": self._teacher_hash,
            "scale_hash": hashlib.sha256(
                np.asarray(scale, dtype=np.float32).tobytes()).hexdigest(),
            "lora": bool(self._lora_named),
            # R3：学生身份绑定——输出语义 / 底座权重指纹 / 代码指纹；
            # 旧 checkpoint（无这些字段）不得当作新语义起点（装载层校验）。
            "output_space": output_space,
            "backbone_weight_hash": backbone_weight_hash,
            "code_hash": _package_code_hash(),
            # 注：max_epochs / disable_early_stop 不入身份——延长训练
            # （断点续跑、S-long/D1-cont 预算匹配）是合法恢复场景
        }
        self.run_dir: Path = safe_artifact_dir(run_name)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._check_identity()

    # ------------------------------------------------------------------
    # 身份与 checkpoint（原子 + 恢复核验；weights_only 装载）
    # ------------------------------------------------------------------

    def _identity_path(self) -> Path:
        return self.run_dir / "train_identity.json"

    def _check_identity(self) -> None:
        p = self._identity_path()
        if p.exists():
            stored = json.loads(p.read_text("utf-8"))
            if stored != self.identity:
                raise RuntimeError(
                    f"训练 run 身份不一致：缓存 {stored} ≠ 本次 {self.identity}——"
                    f"不得混写（请改用新 run_name）"
                )
        else:
            tmp = self.run_dir / "train_identity.json.tmp"
            tmp.write_text(
                json.dumps(self.identity, ensure_ascii=False, indent=1), "utf-8")
            tmp.replace(p)

    def _ckpt_path(self, epoch: int | str) -> Path:
        return self.run_dir / f"epoch-{epoch}.pt"

    def _save_ckpt(self, epoch: int, history: list[dict]) -> None:
        payload = {
            "epoch": epoch,
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": history,
            "identity": self.identity,
        }
        if self._lora_named:
            payload["lora"] = self._lora_state()
        tmp = self.run_dir / f"epoch-{epoch}.tmp.pt"
        torch.save(payload, tmp)
        tmp.replace(self._ckpt_path(epoch))
        tmp_last = self.run_dir / "epoch-last.tmp.pt"
        torch.save(payload, tmp_last)
        tmp_last.replace(self._ckpt_path("last"))

    def _lora_state(self) -> dict:
        """LoRA 参数当前值快照（name → CPU tensor）。"""
        return {k: p.detach().cpu().clone() for k, p in self._lora_named.items()}

    def _load_last(self) -> Optional[dict]:
        """恢复 last checkpoint（weights_only=True：载荷仅张量/基本类型）。"""
        p = self._ckpt_path("last")
        if not p.exists():
            return None
        ck = torch.load(p, weights_only=True)
        if ck["identity"] != self.identity:
            raise RuntimeError("last checkpoint 身份不一致，拒绝恢复")
        self.head.load_state_dict(ck["head"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self._lora_named and "lora" in ck:
            with torch.no_grad():
                for k, v in ck["lora"].items():
                    self._lora_named[k].copy_(v.to(self._lora_named[k].device))
        return ck

    # ------------------------------------------------------------------
    # 前向 / 损失
    # ------------------------------------------------------------------

    def _hidden(self, batch: DayTensor) -> torch.Tensor:
        """隐状态：冻结臂 no_grad 在线提取（可用缓存）；D2 走可训路径。"""
        x = batch.x_norm.to(self.device)
        stamp = batch.x_stamp.to(self.device)
        if self.arm == "D2":
            return self.backbone.extract_trainable(x, stamp)
        if self.hidden_cache is not None and batch.date_iso in self.hidden_cache:
            return self.hidden_cache[batch.date_iso]
        with torch.no_grad():
            return self.backbone.extract(x, stamp)

    def _day_losses(self, batch: DayTensor) -> dict[str, torch.Tensor]:
        """单日各损失（FP32；IC 在同日截面内计算；R2 按输出语义还原）。"""
        hidden = self._hidden(batch).to(self.device)
        y_stamp = batch.y_stamp.to(self.device)
        if self.identity["output_space"] == "normalized_close_affine_return":
            pred = self.head(hidden, y_stamp, batch.a.to(self.device),
                             batch.b.to(self.device))
        else:
            pred = self.head(hidden, y_stamp)
        y_real = batch.y_real.to(self.device)
        y_teacher = batch.y_teacher.to(self.device)
        s = normalized_mse(pred, y_real, self.scale_t)
        d = normalized_mse(pred, y_teacher, self.scale_t)
        # 标签方差为 0 的日跳过 IC、保留 MSE（§4.3）
        if float(y_real.mean(dim=-1).var()) > 0:
            i = daily_ic_loss(pred, y_real)
        else:
            i = torch.zeros((), device=self.device)
        nonfinite = int((~torch.isfinite(pred)).sum())
        return {"s": s, "d": d, "i": i, "nonfinite": nonfinite,
                "pred": pred.detach(),
                "task": arm_loss(self.arm, s, d, i)}

    def _clip_params(self) -> list:
        """梯度裁剪参数集：头 + （若 D2）LoRA。"""
        ps = [p for p in self.head.parameters() if p.requires_grad]
        ps += [p for p in self._lora_named.values() if p.requires_grad]
        return ps

    def _epoch(self, epoch: int) -> dict[str, float]:
        """一个 epoch：训练（日序确定性混洗）+ 验证（按日平均）。"""
        self.head.train()
        self.backbone.eval()  # 底座恒 eval（LoRA 可训但 dropout 关闭）
        order = day_order(self.seed, epoch, len(self.train_days))
        tr_loss, grad_norm_sum, nonfinite = 0.0, 0.0, 0
        for oi in order:
            losses = self._day_losses(self.train_days[oi])
            self.optimizer.zero_grad(set_to_none=True)
            losses["task"].backward()
            gn = torch.nn.utils.clip_grad_norm_(
                self._clip_params(), self.cfg.grad_clip)
            grad_norm_sum += float(gn) if torch.isfinite(gn) else float("inf")
            self.optimizer.step()
            tr_loss += float(losses["task"])
            nonfinite += int(losses["nonfinite"])
        tr_loss /= max(len(order), 1)

        self.head.eval()
        val = {"d": 0.0, "s": 0.0, "i": 0.0, "task": 0.0}
        with torch.no_grad():
            for batch in self.val_days:
                losses = self._day_losses(batch)
                for k in val:
                    val[k] += float(losses[k])
                nonfinite += int(losses["nonfinite"])
        n = max(len(self.val_days), 1)
        val = {k: v / n for k, v in val.items()}
        val_task = val["s"] + 0.05 * val["i"]
        return {
            "epoch": epoch, "train_loss": tr_loss,
            "val_d": val["d"], "val_s": val["s"], "val_i": val["i"],
            "val_task": val_task,
            "val_loss": val["d"] if self.arm == "D0" else val_task,
            "grad_norm": grad_norm_sum / max(len(order), 1),
            "nonfinite": nonfinite,
        }

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def fit(self) -> TrainResult:
        """训练到收敛/上限，逐 epoch 落盘 checkpoint，返回选点结果。"""
        ck = self._load_last()
        start_epoch = (ck["epoch"] + 1) if ck else 0
        history: list[dict] = list(ck["history"]) if ck else []
        if history:
            best_val = min(h["val_loss"] for h in history)
            since_best = 0
            for h in history:
                if h["val_loss"] >= best_val:
                    pass
            # 重新计算无改进轮次（从最后一个最优点起算）
            best_idx = min(range(len(history)),
                           key=lambda k: (history[k]["val_loss"], history[k]["epoch"]))
            since_best = len(history) - 1 - best_idx
        else:
            best_val, since_best = float("inf"), 0

        for epoch in range(start_epoch, self.max_epochs):
            t0 = time.time()
            row = self._epoch(epoch)
            row["seconds"] = round(time.time() - t0, 2)
            history.append(row)
            self._save_ckpt(epoch, history)
            logger.info(
                f"[{self.run_name}] epoch {epoch} train={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} ({self.criterion}) "
                f"val_s={row['val_s']:.4f} val_i={row['val_i']:.4f} "
                f"({row['seconds']}s)"
            )
            if row["val_loss"] < best_val - 1e-12:
                best_val, since_best = row["val_loss"], 0
            else:
                since_best += 1
            if not self.disable_early_stop and since_best >= self.cfg.patience:
                logger.info(f"[{self.run_name}] 早停：{self.cfg.patience} 轮无改进")
                break

        best_epoch = select_best_epoch(history)
        result = {
            "arm": self.arm, "seed": self.seed, "best_epoch": best_epoch,
            "criterion": self.criterion, "history": history,
            "identity": self.identity,
        }
        tmp = self.run_dir / "result.json.tmp"
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(self.run_dir / "result.json")
        return TrainResult(
            best_epoch=best_epoch, history=history, run_dir=self.run_dir,
            criterion=self.criterion, arm=self.arm,
        )


__all__ = [
    "TRAIN_ARMS", "UnifiedTrainer", "arm_loss", "select_best_epoch",
    "day_order", "DayTensor", "TrainResult",
]
