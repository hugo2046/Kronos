"""教师重复生成、keyed RNG 与断点恢复（方案 §4.2）。

协议要点：

- 教师固定 G1 s100（tokenizer+predictor 全 eval），权重 SHA256 入 manifest；
- 每样本 3 组独立 N=20 平均 close 曲线：replica 0 = 蒸馏 target，
  replica 1/2 仅用于教师自身波动与独立保真评估，禁止用于挑 checkpoint；
- 每次以**一只股票、固定 N** 调用原 ``predict``；seed =
  SHA256(``protocol_hash|date|code|replica``) 前 8 字节；``torch.random.fork_rng``
  限定 CPU/所用 CUDA 状态——实现顺序、恢复、外层批大小改变后同设备同样本一致；
- run 目录按 ``{profile}-{split}-{清单hash}`` 命名，内部 manifest 记录权重/
  协议/N/T/top_p/top_k/dtype——恢复时不一致直接报错，不混写（§6）；
- 分片按日落盘：同目录临时文件 + rename 原子替换，落盘后**重读核验完成键**，
  不靠文件存在当完成（§7 任务3）。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger

from dhead_distill.data import DayManifest, safe_artifact_dir

#: 教师生成的 6 列（与 KronosPredictor 输出一致）
_PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def teacher_seed(protocol: str, date: str, code: str, replica: int) -> int:
    """keyed RNG seed：SHA256(protocol|date|code|replica) 前 8 字节 → int。"""
    digest = hashlib.sha256(
        f"{protocol}|{date}|{code}|{replica}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


class TeacherRunner:
    """按日期分片的教师目标生成器（可中断、可恢复、可核验）。

    :param manifest: prepare 产出的冻结清单。
    :param predict_fn: 对齐 ``KronosPredictor.predict`` 签名的单股预测函数。
    :param weight_hash: 教师权重（tokenizer+predictor）联合 SHA256；与缓存
        manifest 不一致时拒绝复用（防新旧权重混写）。
    :param n_paths: 每条生成的路径数 N（固定 20）。
    :param replicas: 独立重复组数（固定 3；0 组=蒸馏 target，1/2=保真评估）。
    :param fork_devices: ``torch.random.fork_rng`` 限定设备列表；None=fork 全部。
    """

    def __init__(
        self,
        manifest: DayManifest,
        predict_fn: Callable,
        *,
        weight_hash: str,
        n_paths: int,
        replicas: int,
        teacher_T: float,
        teacher_top_p: float,
        teacher_top_k: int,
        predict_len: int,
        fork_devices: Optional[Sequence[str]] = None,
    ):
        self.manifest = manifest
        self._predict_fn = predict_fn
        self.weight_hash = weight_hash
        self.n_paths = n_paths
        self.replicas = replicas
        self.teacher_T = teacher_T
        self.teacher_top_p = teacher_top_p
        self.teacher_top_k = teacher_top_k
        self.predict_len = predict_len
        self.fork_devices = list(fork_devices) if fork_devices is not None else None
        self.identity = {
            "protocol": manifest.protocol,
            "manifest_content_hash": manifest.content_hash,
            "weight_hash": weight_hash,
            "n_paths": n_paths, "replicas": replicas,
            "T": teacher_T, "top_p": teacher_top_p, "top_k": teacher_top_k,
            "predict_len": predict_len, "dtype": "float32",
        }
        run_name = (
            f"teacher-{manifest.profile}-{manifest.split}"
            f"-{manifest.content_hash[:12]}"
        )
        self.run_dir: Path = safe_artifact_dir(run_name)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._check_or_init_run_manifest()

    # ------------------------------------------------------------------
    # run manifest：身份核验（不一致 → 报错，不混写）
    # ------------------------------------------------------------------

    def _run_manifest_path(self) -> Path:
        return self.run_dir / "teacher_run.json"

    def _check_or_init_run_manifest(self) -> None:
        p = self._run_manifest_path()
        if p.exists():
            stored = json.loads(p.read_text("utf-8"))
            for k, v in self.identity.items():
                if stored.get(k) != v:
                    raise RuntimeError(
                        f"教师缓存身份不一致（{k}：缓存 {stored.get(k)!r} ≠ "
                        f"本次 {v!r}）——权重/协议/清单/dtype 变化后不得混写旧缓存；"
                        f"请人工核验后清理 {self.run_dir} 或改用新 profile/split"
                    )
        else:
            tmp = self.run_dir / "teacher_run.json.tmp"
            tmp.write_text(
                json.dumps(
                    {**self.identity, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "days": {}},
                    ensure_ascii=False, indent=1,
                ),
                "utf-8",
            )
            tmp.replace(p)

    def _update_day_status(self, date_iso: str, n: int) -> None:
        """原子更新 run manifest 的日期完成状态。"""
        p = self._run_manifest_path()
        doc = json.loads(p.read_text("utf-8"))
        doc["days"][date_iso] = {"n": n, "done": True}
        tmp = self.run_dir / "teacher_run.json.tmp"
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(p)

    # ------------------------------------------------------------------
    # 分片读写（原子 + 完成键核验）
    # ------------------------------------------------------------------

    def _shard_path(self, date_iso: str) -> Path:
        return self.run_dir / f"day-{date_iso}.npz"

    def _load_verified_shard(self, date_iso: str) -> Optional[dict]:
        """读分片并核验完成键；不存在或未完成 → None（重新生成）。"""
        p = self._shard_path(date_iso)
        if not p.exists():
            return None
        try:
            z = np.load(p, allow_pickle=False)
            if int(z["done"]) != 1:
                return None
            return {
                "codes": [str(c) for c in z["codes"]],
                "y": z["y_teacher"].astype(np.float32),
                "close_t": z["close_t"].astype(np.float64),
            }
        except Exception:  # noqa: BLE001 - 损坏分片视为未完成，重新生成
            logger.warning(f"教师分片损坏，将重新生成：{p.name}")
            return None

    def _write_shard(self, date_iso: str, codes: list[str],
                     y: np.ndarray, close_t: np.ndarray) -> None:
        """原子落盘 + 重读核验完成键（不靠文件存在当完成）。"""
        p = self._shard_path(date_iso)
        # 临时文件名必须以 .npz 结尾：np.savez 对非 .npz 后缀会再追加 .npz
        tmp = self.run_dir / f"day-{date_iso}.tmp.npz"
        np.savez(
            tmp, codes=np.array(codes), y_teacher=y.astype(np.float32),
            close_t=close_t.astype(np.float64), done=np.int64(1),
        )
        tmp.replace(p)
        # 落盘后核验：重读 + 校验完成键与形状
        z = np.load(p, allow_pickle=False)
        assert int(z["done"]) == 1 and z["y_teacher"].shape == y.shape

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def _generate_sample(
        self, date_iso: str, code: str, replica: int, close_t: float,
        x_raw: np.ndarray, x_cal: np.ndarray, y_cal: np.ndarray,
    ) -> np.ndarray:
        """单样本单 replica：keyed RNG 下单股 N 路径平均 → [H] 收益。"""
        seed = teacher_seed(self.manifest.protocol, date_iso, code, replica)
        x_idx, y_idx = pd.DatetimeIndex(x_cal), pd.DatetimeIndex(y_cal)
        df = pd.DataFrame(x_raw.astype(np.float64), columns=_PRICE_COLS,
                          index=x_idx)
        # KronosPredictor.predict 的 calc_time_stamps 需要 .dt 访问器 → Series
        x_ts = pd.Series(x_idx)
        y_ts = pd.Series(y_idx)
        with torch.random.fork_rng(devices=self.fork_devices):
            torch.manual_seed(seed)
            pred = self._predict_fn(
                df, x_ts, y_ts, self.predict_len,
                T=self.teacher_T, top_k=self.teacher_top_k,
                top_p=self.teacher_top_p, sample_count=self.n_paths,
                verbose=False,
            )
        closes = np.asarray(pred["close"].values, dtype=np.float64)
        if closes.shape != (self.predict_len,):
            raise ValueError(
                f"教师输出 close 形状 {closes.shape} ≠ ({self.predict_len},)"
            )
        return (closes / close_t - 1.0).astype(np.float32)

    def run(self) -> dict[tuple[str, str], np.ndarray]:
        """生成/恢复全部日期分片，返回 {(date_iso, code): [R,H]}。"""
        out: dict[tuple[str, str], np.ndarray] = {}
        by_date: dict[str, list] = {}
        for s in self.manifest.samples:
            by_date.setdefault(s.date.strftime("%Y-%m-%d"), []).append(s)

        t0 = time.time()
        n_generated = 0
        for d_iso, samples in by_date.items():
            shard = self._load_verified_shard(d_iso)
            if shard is None:
                codes = [s.code for s in samples]
                y = np.zeros((len(samples), self.replicas, self.predict_len),
                             dtype=np.float32)
                ct = np.zeros(len(samples), dtype=np.float64)
                for i, s in enumerate(samples):
                    key = (d_iso, s.code)
                    ct[i] = self.manifest.close_t[key]
                    for r in range(self.replicas):
                        y[i, r] = self._generate_sample(
                            d_iso, s.code, r, ct[i],
                            self.manifest.x_raw[key],
                            self.manifest.x_cal[d_iso],
                            self.manifest.y_cal[d_iso],
                        )
                    n_generated += 1
                self._write_shard(d_iso, codes, y, ct)
                self._update_day_status(d_iso, len(samples))
                shard = {"codes": codes, "y": y, "close_t": ct}
            for c, row in zip(shard["codes"], shard["y"]):
                out[(d_iso, c)] = row
        if n_generated:
            dt = time.time() - t0
            logger.info(
                f"教师生成完成：{n_generated} 样本 × {self.replicas} replica，"
                f"{dt:.1f}s（{n_generated * self.replicas / max(dt, 1e-9):.2f} 次/s）"
            )
        return out

    def load_targets_array(self) -> tuple[np.ndarray, list[tuple[str, str]]]:
        """按 manifest.samples 顺序装载 [N, R, H]（须已全部完成）。"""
        rows = []
        keys = []
        for s in self.manifest.samples:
            d_iso = s.date.strftime("%Y-%m-%d")
            shard = self._load_verified_shard(d_iso)
            if shard is None:
                raise RuntimeError(f"教师分片未完成：{d_iso}（先运行 teacher）")
            idx = shard["codes"].index(s.code)
            rows.append(shard["y"][idx])
            keys.append((d_iso, s.code))
        return np.stack(rows), keys


def combined_weight_hash(tokenizer_hash: str, predictor_hash: str) -> str:
    """教师权重联合 hash（tokenizer + predictor）。"""
    return hashlib.sha256(
        f"{tokenizer_hash}|{predictor_hash}".encode("utf-8")
    ).hexdigest()


__all__ = ["TeacherRunner", "teacher_seed", "combined_weight_hash"]
