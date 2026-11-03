"""只读冻结权重的SAM失败机制诊断；不训练、不创建行情provider。"""
from __future__ import annotations

import fcntl
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from state_adapter import base_state_hash, inject_adapters, load_adapter_state, set_state
from state_adapter.train import _capture_rng, _restore_rng, future_token_loss

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/user/workspace/Kronos-sam-adapter/artifacts/experiments/sam_adapter_20261102')
OUT = ROOT / 'artifacts/experiments/sam_mechanism_20261103'


def digest(path: Path) -> str:
    """计算输入身份。\n\n:param path: 文件。\n:returns: SHA256。"""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def save(path: Path, value: Any) -> None:
    """原子落盘诊断。\n\n:param path: 目标。\n:param value: JSON内容。\n:returns: 无。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def bounded_ascent(params: list[Tensor], objective: Callable[[bool], float], *,
                   rho: float, steps: int = 5) -> dict[str, Any]:
    """寻找固定L2邻域中的高损失点，正常/异常均恢复参数、梯度与随机流。

    :param params: 允许临时扰动的参数。
    :param objective: 给出损失；True时累加其梯度。
    :param rho: 全局L2半径。
    :param steps: 包含完整半径梯度步的总步数。
    :returns: 访问路径、最大已观测损失与对应扰动副本。
    :raises ValueError: 非法参数或非有限梯度/损失。
    """
    if not params or not np.isfinite(rho) or rho < 0 or steps < 1:
        raise ValueError('非法扰动配置')
    originals = [p.detach().clone() for p in params]
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in params]
    rng = _capture_rng()
    best_delta = [torch.zeros_like(p) for p in params]
    try:
        with torch.no_grad():
            zero = objective(False)
        if not np.isfinite(zero):
            raise ValueError('损失非有限')
        history = [dict(step=0, loss=zero, norm=0.)]
        best = zero
        for step in range(1, steps + 1):
            if rho == 0:
                break
            for p in params:
                p.grad = None
            objective(True)
            grad = [torch.zeros_like(p) if p.grad is None else p.grad.detach() for p in params]
            norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grad]))
            if not torch.isfinite(norm):
                raise ValueError('梯度非有限')
            if norm.item() == 0:
                break
            with torch.no_grad():
                length = rho if step == 1 else rho / 4
                delta = [p - old + length * g / norm for p, old, g in zip(params, originals, grad)]
                radius = torch.linalg.vector_norm(torch.stack([d.norm() for d in delta]))
                scale = min(1., rho / max(radius.item(), 1e-30))
                for p, old, d in zip(params, originals, delta):
                    p.copy_(old + d * scale)
                actual = float(torch.linalg.vector_norm(torch.stack([(p-old).norm() for p, old in zip(params, originals)])))
                loss = objective(False)
            if not np.isfinite(loss):
                raise ValueError('损失非有限')
            history.append(dict(step=step, loss=loss, norm=actual))
            if loss > best:
                best = loss
                best_delta = [(p.detach() - old).clone() for p, old in zip(params, originals)]
        return dict(zero_loss=zero, best_loss=best, history=history, delta=best_delta)
    finally:
        # 使用copy_，避免浮点加减恢复误差；诊断不留下任何优化器更新。
        with torch.no_grad():
            for p, old, gradient in zip(params, originals, gradients):
                p.copy_(old)
                p.grad = gradient
        _restore_rng(rng)


def configure_state(model: torch.nn.Module, state: Tensor) -> bool:
    """仅为已有适配器设置历史状态，原G1不需要此操作。

    :param model: 原G1或已注入适配器的模型。
    :param state: 历史状态张量。
    :returns: 是否设置了适配器状态。
    """
    adapted = any('state_adapter' in name.split('.') for name, _ in model.named_parameters())
    if adapted:
        set_state(model, state)
    return adapted


class CacheObjective:
    """在完整原批序上计算双头损失，并只对拟合半集反传。

    :param model: 冻结底座及可诊断适配器。
    :param batches: 固定token缓存。
    :param guard: 时间/资源守卫。
    """
    def __init__(self, model: torch.nn.Module, batches: list[dict[str, Tensor]],
                 guard: Callable[[], None]) -> None:
        self.model = model
        self.batches = batches
        self.guard = guard
        self.reference: list[tuple[Tensor, Tensor]] | None = None
        self.last: dict[str, Any] = {}

    def __call__(self, backward: bool, *, capture: bool = False) -> float:
        """按原eval_seed运行完整批序。

        :param backward: 是否只反传偶数批次平均CE。
        :param capture: 是否记录未扰动条件logits分布。
        :returns: 拟合半集平均损失。
        """
        rng = _capture_rng()
        sums = np.zeros((2, 5), dtype=np.float64)
        counts = np.zeros(2, dtype=np.int64)
        reference = []
        fit_count = sum(b['s1'].shape[0] for i, b in enumerate(self.batches) if i % 2 == 0)
        try:
            torch.manual_seed(103100)
            torch.cuda.manual_seed_all(103100)
            for i, batch in enumerate(self.batches):
                self.guard()
                half = i % 2
                b = {k: v.to('cuda:0') for k, v in batch.items()}
                n = b['s1'].shape[0]
                configure_state(self.model, b['state'])
                with torch.set_grad_enabled(backward and half == 0):
                    logits = self.model(b['s1'][:, :-1], b['s2'][:, :-1],
                                        stamp=b['stamp'][:, :-1], use_teacher_forcing=False)
                    losses = future_token_loss(*logits, b['s1'][:, 1:], b['s2'][:, 1:])
                    if backward and half == 0:
                        (losses[0] * n / fit_count).backward()
                with torch.no_grad():
                    logs = tuple(torch.log_softmax(x[:, 89:99], dim=-1) for x in logits)
                    kl = [0., 0.]
                    if self.reference is not None:
                        for h in range(2):
                            previous = self.reference[i][h].to('cuda:0')
                            kl[h] = float((previous.exp() * (previous - logs[h])).sum(-1).mean())
                    if capture:
                        reference.append(tuple(x.cpu() for x in logs))
                    sums[half] += np.array([float(x.detach()) for x in losses] + kl) * n
                    counts[half] += n
                del logits, losses, logs, b
            averages = sums / counts[:, None]
            self.last = {name: dict(zip(['ce', 's1_ce', 's2_ce', 's1_conditional_kl', 's2_conditional_kl'], averages[j].tolist()))
                         for j, name in enumerate(['fit', 'crosscheck'])}
            self.last['full_ce'] = float(sums[:, 0].sum() / counts.sum())
            self.last['counts'] = counts.tolist()
            if capture:
                self.reference = reference
            return float(averages[0, 0])
        finally:
            _restore_rng(rng)


def main() -> None:
    """执行一次有界冻结诊断并保留恢复与身份审计。\n\n:returns: 无。"""
    from model.kronos import Kronos
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clock_path = OUT / 'diagnostic_clock.json'
    if not clock_path.exists():
        save(clock_path, dict(started=time.time(), wall_limit=7200, gpu_limit=1800, gpu_spent=0.))
    clock = json.loads(clock_path.read_text())
    start = time.monotonic()
    def guard() -> None:
        elapsed = time.monotonic() - start
        now = datetime.now(ZoneInfo('Asia/Shanghai'))
        if time.time() > clock['started'] + 7200 or clock['gpu_spent'] + elapsed > 1800:
            raise RuntimeError('新诊断预算已达上限，不能重置')
        if now.hour == 16 and 25 <= now.minute < 45:
            raise RuntimeError('资源保留窗口，保存账本后退出')
    source = json.loads((SOURCE / 'protocol.json').read_text())
    cache_path = Path(source['validation_cache']['path'])
    paths = {'cache': (cache_path, source['validation_cache']['sha256']),
             'MC40': (Path(source['reference_adapter']['path']), source['reference_adapter']['sha256']),
             'SAM': (SOURCE / 'arms/sam/best_adapter.pt', '9ed6b8d808276dc1b4d0de20646ed61f41c91638cac07700e36f23085a60ef28')}
    identities = {}
    for name, (path, expected) in paths.items():
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f'{name}身份漂移')
        identities[name] = dict(path=str(path), sha256=actual)
    save(OUT / 'input_identity.json', identities)
    batches = torch.load(cache_path, map_location='cpu', weights_only=False)
    if isinstance(batches, dict):
        batches = batches['batches']
    assert len(batches) == 32 and sum(b['s1'].shape[0] for b in batches) == 2048
    results = {}
    try:
        for name in ['G1_original', 'MC40', 'SAM']:
            guard()
            checkpoint_path = OUT / f'geometry_{name}.json'
            if checkpoint_path.exists():
                results[name] = json.loads(checkpoint_path.read_text())
                continue
            logger.info('诊断 {}', name)
            model = Kronos.from_pretrained(source['baseline']['model_paths']['predictor']).to('cuda:0').eval()
            if name != 'G1_original':
                inject_adapters(model, 'dynamic', seed=100, rank=32, n_layers=2)
                ck = torch.load(paths[name][0], map_location='cpu', weights_only=False)
                assert ck['base_hash'] == source['reference_adapter']['base_hash']
                load_adapter_state(model, ck['adapter_state'])
            model.eval()
            base_before = base_state_hash(model)
            assert base_before == source['reference_adapter']['base_hash']
            objective = CacheObjective(model, batches, guard)
            with torch.no_grad():
                objective(False, capture=True)
            clean = objective.last
            expected_ce = {'MC40': source['reference_adapter']['val_ce'], 'SAM': 2.4830846562981606}
            if name in expected_ce and abs(clean['full_ce'] - expected_ce[name]) > 1e-6:
                raise RuntimeError('正常CE未复现')
            rows = []
            if name != 'G1_original':
                named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                assert all('state_adapter' in n.split('.') for n, p in named)
                params = [p for n, p in named]
                originals = [p.detach().clone() for p in params]
                for rho in [.01, .05]:
                    attack = bounded_ascent(params, objective, rho=rho)
                    layer_stats = []
                    try:
                        with torch.no_grad():
                            for (n, p), old, delta in zip(named, originals, attack['delta']):
                                p.copy_(old + delta)
                                wn = float(old.norm())
                                dn = float((p-old).norm())
                                layer_stats.append(dict(name=n, weight_norm=wn, delta_norm=dn,
                                                        relative_delta=dn/wn if wn > 0 else None))
                            objective(False)
                        perturbed = objective.last
                    finally:
                        with torch.no_grad():
                            for p, old in zip(params, originals):
                                p.copy_(old)
                    row = dict(rho=rho, path=attack['history'], best_fit_ce=attack['best_loss'],
                               perturbed=perturbed, layers=layer_stats,
                               parameters_restored=all(torch.equal(p, old) for p, old in zip(params, originals)))
                    rows.append(row)
                    logger.info('{} rho={} fit增量={} crosscheck增量={}', name, rho,
                                perturbed['fit']['ce'] - clean['fit']['ce'],
                                perturbed['crosscheck']['ce'] - clean['crosscheck']['ce'])
            assert base_state_hash(model) == base_before
            result = dict(model=name, clean=clean, attacks=rows, base_hash=base_before,
                          base_restored=True, checkpoint_identity=identities.get(name),
                          cache_sha256=identities['cache']['sha256'])
            save(checkpoint_path, result)
            results[name] = result
            del objective, model
            torch.cuda.empty_cache()
        save(OUT / 'geometry.json', results)
        save(OUT / 'gpu_complete.json', dict(complete=True, provider_queries=0, trained_updates=0))
    finally:
        clock['gpu_spent'] += time.monotonic() - start
        save(clock_path, clock)


if __name__ == '__main__':
    main()
