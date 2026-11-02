"""固定2×2机制实验：身份冻结、静态控制复现及可恢复双流推理。"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from state_adapter import run as legacy
from state_adapter.data import WindowCorpus
from state_adapter.date_balanced import DateBalancedCorpus, replay_date_counts
from state_adapter.gpu_budget import GPUBudget
from state_adapter.mc40 import _wide, pool_signals
from state_adapter.train import train_arm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/experiments/state_factorial_20261102'
OLD = Path('/home/user/workspace/Kronos-state-adapter/artifacts/experiments/state_adapter_20261030')
MC = Path('/home/user/workspace/Kronos-state-mc40/artifacts/experiments/state_mc40_20261101')
DATE = Path('/home/user/workspace/Kronos-date-balanced/artifacts/experiments/date_balanced_20261101')
EXPECTED = {'A': '873db507a7ae6109274fca0072a46be06dab583b968e5d83e705eea9ccafad17',
            'B': 'd208a8b9f5359aa81bd6cfff473bf9aae193f0b37542e92b7ae39d00c649c402',
            'D': '2c45c8a38559653c9be72facbd8ca911e621f9448f5919573dbfc9a621aaa13d'}


def bind() -> None:
    """绑定本轮独立输出，避免写回旧实验。

    :returns: 无。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for name, filename in [('PROTOCOL_PATH', 'protocol.json'), ('PROTOCOL_SHA_PATH', 'protocol.json.sha256'),
                           ('STATUS_PATH', 'bot_status.json'), ('GPU_BUDGET_PATH', 'gpu_budget.json')]:
        setattr(legacy, name, OUT / filename)
    legacy.OUT = OUT


def save(name: str, value: Any) -> None:
    """原子保存审计记录。

    :param name: 相对产物路径。
    :param value: JSON对象。
    :returns: 无。
    """
    legacy._atomic_text(OUT / name, json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def freeze() -> None:
    """冻结输入SHA与四格定义；已存在的协议仅核验。

    :returns: 无。
    :raises RuntimeError: 既有身份或预期SHA不符。
    """
    bind()
    if (OUT / 'protocol.json').exists():
        load()
        return
    source = json.loads((DATE / 'protocol.json').read_text())
    p = {k: copy.deepcopy(source[k]) for k in ['baseline', 'corpus', 'training', 'state', 'inference',
           'qlib', 'windows', 'expected_coverage', 'validation_cache']}
    p.update(experiment='state_factorial_20261102', forward_cutoff='2026-07-24',
             code_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             budget=dict(gpu_total_seconds=36000, wall_total_seconds=50400, cumulative=True),
             comparison=dict(block=20, resamples=5000, seed=100, method='paired daily net return; circular blocks; common indices for interaction'),
             attribution='W3 原G1/B/D真实持仓成交对账、日收益成本、回撤、历史20/60日价量暴露；历史行业市值不可得则缺项',
             reuse_check='A100每窗首中末三日逐值重推，atol=1e-10,rtol=0；失败调查后全双流推理',
             selection='最低2025H1 CE，每500步，同分较早；不使用W3/W4选点，不自动采纳',
             static_caveat='仅state输入置零，保留gate；107712名义参数，部分参数无数据梯度，有效自由度不宣称相同',
             arms={})
    p['training']['seeds'] = [100]
    paths = {'A': OLD / 'arms/static/seed_100/best_adapter.pt',
             'B': OLD / 'arms/dynamic/seed_100/best_adapter.pt', 'D': DATE / 'arms/candidate/best_adapter.pt'}
    refs = dict(source['reference_files'])
    for arm, path in paths.items():
        digest = legacy._sha256_file(path)
        if digest != EXPECTED[arm]:
            raise RuntimeError(f'{arm} 权重SHA不符')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        p['arms'][arm] = {k: ck[k] for k in ['family', 'seed', 'protocol_sha', 'config_sha', 'base_hash', 'step', 'val_ce']}
        p['arms'][arm].update(path=str(path), sha256=digest, sampling='date' if arm == 'D' else 'window')
        refs[str(path)] = digest
    p['arms']['C'] = dict(family='static', seed=100, sampling='date', path=str(OUT / 'arms/C/best_adapter.pt'), new_training=True)
    files = [DATE / 'protocol.json', DATE / 'protocol.json.sha256', MC / 'manifest.json', OLD / 'protocol.json',
             OLD / 'arms/static/seed_100/validation_batches.pt', Path(p['validation_cache']['path'])]
    for window in p['windows']:
        files += [OLD / 'signals/State_static_100' / f'{window}{ext}' for ext in ['.parquet', '.meta.json']]
        files += [DATE / f'{prefix}_{window}_Date_balanced_MC40.parquet' for prefix in ['signal', 'report', 'daily']]
        for seed in [42, 100]:
            files += [DATE / f'signals/Date_balanced_train100_infer{seed}' / f'{window}{ext}' for ext in ['.parquet', '.meta.json']]
        for key, path in source['mc40_manifest']['paths'][window].items():
            if legacy._sha256_file(path) != source['mc40_manifest']['sha256'][window][key]:
                raise RuntimeError('MC40 manifest输入漂移')
            files.append(Path(path))
    for path in files:
        refs[str(path)] = legacy._sha256_file(path)
    p['reference_files'] = refs
    p['reuse_dates'] = {}
    for window in p['windows']:
        baseline = _wide(p['baseline']['signal_paths'][window])
        p['reuse_dates'][window] = [str(baseline.index[i].date()) for i in [0, len(baseline)//2, len(baseline)-1]]
    save('protocol.json', p)
    legacy._atomic_text(OUT / 'protocol.json.sha256', legacy._sha256_object(p) + '\n')
    load()


def load() -> tuple[dict[str, Any], str]:
    """校验全部冻结输入并返回协议。

    :returns: 协议及canonical SHA。
    :raises RuntimeError: 任一文件漂移。
    """
    bind()
    p, sha = legacy.load_protocol()
    for path, digest in p['reference_files'].items():
        if legacy._sha256_file(path) != digest:
            raise RuntimeError(f'冻结输入漂移: {path}')
    return p, sha


def signal_identity(path: Path, identity: dict[str, Any], protocol_sha: str, tag: str) -> None:
    """写入或严格核验恢复信号身份。

    :param path: 信号文件。
    :param identity: family、原训练身份、权重、流、窗口、mask哈希。
    :param protocol_sha: 外层协议SHA。
    :param tag: 信号臂名称。
    :returns: 无。
    :raises RuntimeError: 缓存身份不一致或孤立信号。
    """
    meta_path = path.with_suffix('.meta.json')
    expected = dict(identity, protocol_sha=protocol_sha, tag=tag)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if any(meta.get(k) != v for k, v in expected.items()):
            raise RuntimeError('信号缓存身份不匹配')
    else:
        if path.exists():
            raise RuntimeError('信号缓存缺少身份')
        legacy._atomic_text(meta_path, json.dumps(dict(expected, completed=[], forward_cutoff='2026-07-24')))


def audit_cache(p: dict[str, Any]) -> None:
    """逐tensor核对旧静态与动态验证缓存。

    :param p: 冻结协议。
    :returns: 无。
    :raises RuntimeError: 验证内容不一致。
    """
    a = torch.load(OLD / 'arms/static/seed_100/validation_batches.pt', map_location='cpu', weights_only=False)
    b = torch.load(p['validation_cache']['path'], map_location='cpu', weights_only=False)
    def equal(x: Any, y: Any) -> bool:
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(equal(u, v) for u, v in zip(x, y))
        return bool(x == y)
    if not equal(a, b):
        raise RuntimeError('A/B验证缓存内容不一致')
    save('validation_cache_audit.json', dict(tensors_equal=True, container_sha_equal=legacy._sha256_file(OLD / 'arms/static/seed_100/validation_batches.pt') == p['validation_cache']['sha256']))


def require_control(p: dict[str, Any]) -> None:
    """从实际权重重算控制门禁，不能凭文件存在跳过。

    :param p: 协议。
    :returns: 无。
    :raises RuntimeError: 任一控制tensor、最佳CE或步数不一致。
    """
    a = torch.load(OUT / 'arms/A_control/best_adapter.pt', map_location='cpu', weights_only=False)
    b = torch.load(p['arms']['A']['path'], map_location='cpu', weights_only=False)
    diffs = {k: float((a['adapter_state'][k] - b['adapter_state'][k]).abs().max()) for k in b['adapter_state']}
    checks = dict(step_equal=a['step'] == b['step'], ce_equal=a['val_ce'] == b['val_ce'],
                  tensors_equal=a['adapter_state'].keys() == b['adapter_state'].keys() and all(v == 0 for v in diffs.values()),
                  max_abs=max(diffs.values()), new_ce=a['val_ce'], old_ce=b['val_ce'])
    save('control_comparison.json', checks)
    if not all(checks[k] for k in ['step_equal', 'ce_equal', 'tensors_equal']):
        raise RuntimeError('静态控制未完全复现')


def require_sampling() -> None:
    """恢复时重查C与D采样审计，不能因训练记录已存在而绕过。

    :returns: 无。
    :raises RuntimeError: 采样审计缺失或失败。
    """
    path = OUT / 'sampling_audit.json'
    if not path.exists():
        raise RuntimeError('C采样审计缺失')
    audit = json.loads(path.read_text())
    if audit.get('matches_D') is not True or audit.get('draws') != 192000:
        raise RuntimeError('C采样审计未通过')


def train(p: dict[str, Any], sha: str, budget: GPUBudget) -> None:
    """重训A控制与从零初始化C；旧权重只读。

    :param p: 协议。
    :param sha: 协议SHA。
    :param budget: 累计预算。
    :returns: 无。
    """
    audit_cache(p)
    validation = WindowCorpus(p['corpus']['val_path'], target_start='2025-01-01', target_end='2025-06-30', history_path=p['corpus']['train_path'])
    for arm, cls in [('A_control', WindowCorpus), ('C', DateBalancedCorpus)]:
        if arm == 'C':
            require_control(p)
        directory = OUT / 'arms' / arm
        directory.mkdir(parents=True, exist_ok=True)
        cache = directory / 'validation_batches.pt'
        if not cache.exists():
            shutil.copyfile(p['validation_cache']['path'], cache)
        if legacy._sha256_file(cache) != p['validation_cache']['sha256']:
            raise RuntimeError('验证缓存漂移')
        record_path = OUT / f'training_{arm}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if legacy._sha256_file(directory / 'best_adapter.pt') != record['adapter_sha256']:
                raise RuntimeError('已训练权重漂移')
            if arm == 'C':
                require_sampling()
            continue
        corpus = cls(p['corpus']['train_path'], target_end='2024-12-31')
        cfg = legacy.config_sha(p, 100) if arm == 'C' else p['arms']['A']['config_sha']
        def progress(record: dict[str, Any]) -> None:
            with (OUT / f'training_{arm}.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            legacy._write_status('running', f'{arm}训练{record["step"]}/3000', '完成训练后校验A100并推理缺失流', pid=os.getpid())
        with budget.stage('train_' + arm):
            result = train_arm('static', model_path=p['baseline']['model_paths']['predictor'], tokenizer_path=p['baseline']['model_paths']['tokenizer'],
                               train_corpus=corpus, validation_corpus=validation, output_dir=directory,
                               protocol_sha=sha, config_sha=cfg, seed=100, steps=3000, batch_size=64,
                               learning_rate=3e-4, weight_decay=.01, validation_windows=2048,
                               validation_every=500, device='cuda:0', budget_guard_fn=budget.should_stop, progress_fn=progress)
        if result['base_hash'] != p['arms']['A']['base_hash']:
            raise RuntimeError('底座漂移')
        result.update(adapter_sha256=legacy._sha256_file(directory / 'best_adapter.pt'), config_sha=cfg)
        save(record_path.name, result)
        if arm == 'C':
            counts = replay_date_counts(corpus)
            counts.to_csv(OUT / 'C_sample_date_counts.csv')
            prior = pd.read_csv(DATE / 'candidate_sample_date_counts.csv', index_col=0).iloc[:, 0]
            if not np.array_equal(counts.to_numpy(), prior.to_numpy()) or list(counts.index.astype(str)) != list(prior.index):
                raise RuntimeError('C/D逐批日期抽样不一致')
            save('sampling_audit.json', dict(draws=int(counts.sum()), matches_D=True, target_end='2024-12-31'))
        del corpus
        torch.cuda.empty_cache()
    require_control(p)


def checkpoint_identity(p: dict[str, Any], arm: str, sha: str) -> dict[str, Any]:
    """验证四格各自训练身份，外层新协议不覆盖旧checkpoint身份。

    :param p: 四格冻结协议。
    :param arm: A/B/C/D。
    :param sha: 外层协议SHA。
    :returns: 已核验checkpoint。
    :raises RuntimeError: 权重或原训练身份不匹配。
    """
    identity = p['arms'][arm]
    path = Path(identity['path'])
    digest = (json.loads((OUT / 'training_C.json').read_text())['adapter_sha256']
              if arm == 'C' else identity['sha256'])
    if legacy._sha256_file(path) != digest:
        raise RuntimeError('四格权重SHA不匹配')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    expected = dict(family='static' if arm in ['A', 'C'] else 'dynamic', seed=100,
                    base_hash=p['arms']['A']['base_hash'],
                    protocol_sha=sha if arm == 'C' else identity['protocol_sha'],
                    config_sha=legacy.config_sha(p, 100) if arm == 'C' else identity['config_sha'])
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise RuntimeError('四格原训练身份不匹配')
    return checkpoint


def infer(p: dict[str, Any], sha: str, budget: GPUBudget) -> None:
    """先验证A100复用，再补齐A42及C双流。

    :param p: 协议。
    :param sha: 协议SHA。
    :param budget: 累计预算。
    :returns: 无。
    :raises RuntimeError: 身份、覆盖、逐值复用检查失败。
    """
    require_control(p)
    require_sampling()
    for arm in ['A', 'C']:
        ck_path = Path(p['arms'][arm]['path'])
        ck = checkpoint_identity(p, arm, sha)
        digest = legacy._sha256_file(ck_path)
        expected_digest = p['arms'][arm]['sha256'] if arm == 'A' else json.loads((OUT / 'training_C.json').read_text())['adapter_sha256']
        if digest != expected_digest:
            raise RuntimeError('推理权重漂移')
        with budget.stage('infer_' + arm):
            model, tokenizer, _ = legacy._load_g1(torch.device('cuda:0'), 'static', ck_path,
                                                  protocol_sha=ck['protocol_sha'], cfg_sha=ck['config_sha'], seed=100)
            predictor = legacy.StateKronosPredictor(model, tokenizer, device='cuda:0', max_context=512, clip=5.)
            provider = legacy._new_provider()
            for phase in (['validate', 'generate'] if arm == 'A' else ['generate']):
                for window in p['windows']:
                    baseline = _wide(p['baseline']['signal_paths'][window])
                    mask_sha = legacy._sha256_object({'dates': list(baseline.index.astype(str)), 'columns': list(baseline.columns), 'finite': np.isfinite(baseline).astype(int).values.tolist()})
                    def score(seed: int, path: Path, tag: str) -> None:
                        identity = dict(family='static', training_protocol=ck['protocol_sha'], config_sha=ck['config_sha'],
                                        adapter_sha256=digest, seed=seed, window=window, mask_sha=mask_sha)
                        signal_identity(path, identity, sha, tag)
                        legacy.score_window(predictor, provider, tag, window, path, seed=seed, budget=budget)
                    if phase == 'validate':
                        source_path = OLD / 'signals/State_static_100' / f'{window}.parquet'
                        old = _wide(str(source_path))
                        legacy.check_signal_coverage(baseline, old, baseline.index)
                        audits = []
                        for day in p['reuse_dates'][window]:
                            path = OUT / 'reuse_checks' / window / f'{day}.parquet'
                            original_bounds = legacy.WINDOWS[window]
                            try:
                                legacy.WINDOWS[window] = (day, day)
                                score(100, path, f'A100_check_{day}')
                            finally:
                                legacy.WINDOWS[window] = original_bounds
                            actual = _wide(str(path)).reindex(columns=old.columns)
                            reference = old.loc[actual.index]
                            if not np.array_equal(np.isfinite(actual), np.isfinite(reference)):
                                raise RuntimeError('A100复用mask不一致')
                            delta = float(np.nanmax(np.abs(actual.to_numpy() - reference.to_numpy())))
                            audits.append(dict(day=day, max_abs=delta))
                            save(f'reuse_{window}.json', audits)
                            if delta > 1e-10:
                                raise RuntimeError('A100复用逐值校验失败，需要调查')
                    if phase == 'validate':
                        continue
                    for seed in ([42] if arm == 'A' else [42, 100]):
                        path = OUT / 'signals' / f'{arm}_infer{seed}' / f'{window}.parquet'
                        score(seed, path, f'{arm}_infer{seed}')
                        wide = _wide(str(path))
                        legacy.check_signal_coverage(baseline, wide, baseline.index)
                        expected = p['expected_coverage'][window]
                        if len(wide) != expected['dates'] or int(np.isfinite(wide).sum().sum()) != expected['finite']:
                            raise RuntimeError('推理覆盖计数不符')
            del predictor, model, tokenizer
            torch.cuda.empty_cache()
    save('inference_complete.json', dict(complete=True, weight_sha={a: legacy._sha256_file(p['arms'][a]['path']) for a in p['arms']}))


def backtest(p: dict[str, Any]) -> None:
    """四格完整回测并重放B/D日报门禁。

    :param p: 协议。
    :returns: 无。
    :raises RuntimeError: 冻结日报重放或覆盖不一致。
    """
    from qlib_mean_comparison.run import run_qlib_backtest
    gates = {}
    for window, bounds in p['windows'].items():
        baseline = _wide(p['baseline']['signal_paths'][window])
        sources = {'A': [OUT / f'signals/A_infer42/{window}.parquet', OLD / f'signals/State_static_100/{window}.parquet'],
                   'C': [OUT / f'signals/C_infer{s}/{window}.parquet' for s in [42, 100]]}
        signals = {a: pool_signals(*[_wide(str(path)) for path in paths]) for a, paths in sources.items()}
        signals.update(B=_wide(str(MC / f'signal_{window}_State_MC40.parquet')),
                       D=_wide(str(DATE / f'signal_{window}_Date_balanced_MC40.parquet')), G1_original=baseline)
        for arm, wide in signals.items():
            legacy.check_signal_coverage(baseline, wide, baseline.index)
            wide.to_parquet(OUT / f'signal_{window}_{arm}.parquet')
            signal = wide.stack(); signal.index.names = ['datetime', 'instrument']
            report, meta = run_qlib_backtest(signal.swaplevel().sort_index(), *bounds)
            report.to_parquet(OUT / f'report_{window}_{arm}.parquet')
            save(f'backtest_{window}_{arm}.json', meta)
            if arm in ['B', 'D', 'G1_original']:
                source = DATE / f'report_{window}_Date_balanced_MC40.parquet' if arm == 'D' else MC / f'report_{window}_{"State_MC40" if arm == "B" else arm}.parquet'
                old = pd.read_parquet(source)
                if not report.index.equals(old.index) or not report.columns.equals(old.columns):
                    raise RuntimeError('重放日报轴不符')
                delta = float(np.abs(report - old).to_numpy().max())
                gates[f'{window}_{arm}'] = delta
                if delta > 1e-10:
                    raise RuntimeError('重放日报数值不符')
    save('backtest_replay_audit.json', gates)


def main() -> None:
    """互斥运行固定阶段，预算和墙钟跨恢复累计。

    :returns: 无。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['freeze', 'train', 'infer', 'backtest', 'all'], default='all')
    args = parser.parse_args(); bind()
    lock = (OUT / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == 'freeze':
        freeze(); return
    p, sha = load()
    clock = OUT / 'execution_clock.json'
    if not clock.exists():
        save(clock.name, dict(started=time.time(), deadline=time.time() + 50400))
    deadline = json.loads(clock.read_text())['deadline']
    budget = GPUBudget(OUT / 'gpu_budget.json', limit_seconds=36000, cumulative=True)
    stop = budget.should_stop
    budget.should_stop = lambda: stop() or time.time() >= deadline
    legacy._write_status('running', f'执行{args.stage}', 'A控制复现、C训练、完整推理回测与归因', pid=os.getpid())
    try:
        for stage, function in [('train', train), ('infer', infer), ('backtest', backtest)]:
            if args.stage not in [stage, 'all']:
                continue
            if time.time() >= deadline:
                raise RuntimeError('14小时墙钟预算耗尽')
            logger.info('开始阶段 {}', stage)
            if stage == 'train':
                function(p, sha, budget)
            else:
                from state_adapter.market_guard import market_guard
                with market_guard(OUT / f'market_requests_{stage}.jsonl'):
                    function(p) if stage == 'backtest' else function(p, sha, budget)
        legacy._write_status('continue', f'{args.stage}阶段正常退出', '检查完整四格并完成归因报告；交付前不标candidate', pid=os.getpid())
    except BaseException as exc:
        legacy._write_status('continue', f'{type(exc).__name__}: {exc}', '检查日志，修复后从持久化产物恢复；预算不重置', pid=os.getpid())
        raise


if __name__ == '__main__':
    main()
