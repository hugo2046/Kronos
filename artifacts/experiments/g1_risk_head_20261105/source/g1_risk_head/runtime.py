"""固定协议、累计预算、事务身份和唯一有界读取入口。"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from state_adapter import tta_runtime as parent
from state_adapter.tta import atomic_json,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/experiments/g1_risk_head_20261105'
PARENT=Path('/home/user/workspace/Kronos-mc40-tta/artifacts/experiments/mc40_tta_research_20261104/protocol.json')
parent.OUT=OUT


def identity(obj: dict) -> str:
    """计算固定JSON身份。:param obj: 协议。:return: SHA256。"""
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def sources() -> dict[str,str]:
    """绑定本实验及其已审计依赖。:return: 路径到SHA映射。"""
    result=parent.source_hashes()
    for directory in ['g1_risk_head','cross_section_kda','g5_head','tools/risk']:
        result.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/directory).glob('*.py')})
    result.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tests').glob('test_g1_risk*.py')})
    return result


def freeze() -> dict:
    """冻结唯一配置，恢复时禁止改变。:return: 固定协议。"""
    old=json.loads(PARENT.read_text())
    for path,h in old['reference_files'].items():
        if sha(path)!=h:raise RuntimeError('原模型/参照文件身份改变:'+path)
    p=dict(candidate='G1-downside-quantile',reference_files=old['reference_files'],reference_paths=old['reference_paths'],
        windows=old['windows'],expected=old['expected'],cutoff='2026-07-24',
        train=['2014-01-02','2024-12-31'],Q1=['2025-01-01','2025-03-31'],Q2=['2025-04-01','2025-06-30'],
        train_stride=5,lookback=90,horizon=10,label='adjusted close[t+10]/adjusted open[t+1]-1',
        scale='max(std(last20 close returns,ddof=1)*sqrt(10),1e-4)',
        head='Linear(h_norm_last,3); sorted outputs',quantiles=[.1,.5,.9],seed=100,lr=.001,
        betas=[.9,.999],weight_decay=.01,batch=2048,epochs=20,patience=3,
        selection='Q1 raw-Y date-equal pinball; include epoch0; earliest tie',penalty=.25,
        Q2_G1_generation=dict(authorized=True,seed=42,paths=20,chunk=32,T=1,top_p=.9,top_k=0,max_context=512,
            signal='mean(predicted close[1:10])/close[t]-1',scope='Q2 only; W3/W4 immutable'),
        trading=old['trading'],gpu_seconds=21600,wall_seconds=36000,
        availability='current-final assumed; historical vintage unavailable; old sealed-query incident retained')
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=p:raise RuntimeError('固定协议不允许变更')
    atomic_json(path,p)
    return p


def budget_exhausted(state: dict,now: float) -> bool:
    """判断真实累计预算。:param state: 持久化状态。:param now: 当前秒。:return: 是否耗尽。"""
    used=state['gpu_seconds']+(now-state['active_since'] if state['active_since'] is not None else 0)
    return used>=21600 or (state['first_model_at'] is not None and now-state['first_model_at']>=36000)


class Budget(parent.Budget):
    """沿用持久化恢复/资源守卫，固定本轮10小时墙钟。"""
    def check(self) -> None:
        """硬预算检查。:return: 无。"""
        if budget_exhausted(self.state,time.time()):raise RuntimeError('budget_exhausted')

    def watchdog(self):
        """长调用预算和保留窗口硬兜底；退出75由协调器在窗口后恢复。

        :return: 用于结束监督线程的事件。
        """
        import threading
        import os
        import torch
        from state_adapter.run import in_registry_guard
        stop=threading.Event()
        def watch():
            while not stop.wait(.5):
                try:self.check()
                except RuntimeError:
                    status('budget_exhausted');os._exit(76)
                resident=torch.cuda.is_initialized() and torch.cuda.memory_allocated()>0
                if in_registry_guard() and (self.active or resident):
                    # 不在监督线程中并发搬移CUDA对象。进程退出释放GPU，恢复保守计费。
                    status('resource_guard_exit',resume_after='16:45 Asia/Shanghai');os._exit(75)
        threading.Thread(target=watch,daemon=True).start()
        return stop

    @contextmanager
    def compute(self):
        """核算一次真实模型计算区间。:yield: 无。"""
        self.guard(getattr(self,'resident',None));self.begin()
        try:yield
        finally:self.end()


def status(stage: str,**kwargs) -> None:
    """落盘真实执行状态。:param stage: 阶段。:param kwargs: 进度字段。:return: 无。"""
    import os
    atomic_json(OUT/'bot_status.json',dict(stage=stage,pid=os.getpid(),at=time.time(),**kwargs))


def require_gate(p: dict) -> None:
    """核验root独立审查精确身份。:param p: 冻结协议。:return: 无。"""
    gate=json.loads((OUT/'review_gate.json').read_text())
    if not (gate.get('approved') and gate.get('reviewer')=='root' and gate.get('protocol_sha')==identity(p)
            and gate.get('source_hashes')==sources()):raise RuntimeError('独立代码审查门禁未通过')


@contextmanager
def provider():
    """复用受审逐时点原始SQL入口，日志写入本轮独立目录。:yield: 有界provider。"""
    with parent.live_provider() as reader:yield reader
