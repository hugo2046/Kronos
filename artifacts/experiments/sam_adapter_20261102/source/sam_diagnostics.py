"""固定验证集的参数敏感性诊断，不据此选择rho。"""
from __future__ import annotations
from typing import Any, Callable
import numpy as np
import torch
from torch import nn
from state_adapter import base_state_hash
from state_adapter.train import _capture_rng,_restore_rng


def rng_equal(left: Any, right: Any) -> bool:
    """递归比较完整随机状态。:param left: 状态。 :param right: 状态。 :returns: 逐值相同。"""
    if isinstance(left,torch.Tensor):return torch.equal(left,right)
    if isinstance(left,np.ndarray):return np.array_equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(rng_equal(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return len(left)==len(right) and all(rng_equal(a,b) for a,b in zip(left,right))
    return bool(left==right)


def perturbation_diagnostic(model: nn.Module, evaluate: Callable[[],float], *, seed: int=20261102) -> dict[str,Any]:
    """固定4个独立球面方向及其负向，每个非零rho共8次评估。

    :param model: 仅adapter可训练的冻结模型。
    :param evaluate: 固定cache CE评估器，可包含预算门禁。
    :param seed: 独立CPU方向Generator种子。
    :returns: 全部CE增量、实际范数及精确恢复证据。
    :raises RuntimeError: 参数或底座发生漂移。
    """
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not named or any('state_adapter' not in n.split('.') for n,_ in named):raise ValueError('只能扰动adapter')
    params=[p for _,p in named];snapshots=[p.detach().clone() for p in params]
    rng=_capture_rng();base=base_state_hash(model);rows=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    directions=[]
    for _ in range(4):
        parts=[torch.randn(p.shape,generator=generator,dtype=p.dtype).to(p.device) for p in params]
        norm=torch.linalg.vector_norm(torch.stack([part.norm() for part in parts]))
        directions.append([part/norm for part in parts])
    try:
        _restore_rng(rng);zero=evaluate();_restore_rng(rng);repeat=evaluate()
        for rho in [.01,.05]:
            for i,parts in enumerate(directions):
                for sign in [1,-1]:
                    with torch.no_grad():
                        for p,s,d in zip(params,snapshots,parts):p.copy_(s+sign*rho*d)
                    actual=float(torch.linalg.vector_norm(torch.stack([(p.detach()-s).norm() for p,s in zip(params,snapshots)])))
                    _restore_rng(rng);ce=evaluate()
                    rows.append(dict(rho=rho,direction=i,sign=sign,ce=ce,delta_ce=ce-zero,actual_norm=actual))
    finally:
        with torch.no_grad():
            for p,s in zip(params,snapshots):p.copy_(s)
        _restore_rng(rng)
    exact=all(torch.equal(p,s) for p,s in zip(params,snapshots))
    base_after=base_state_hash(model)
    result=dict(seed=seed,independent_directions=4,paired_directions_per_rho=8,zero_ce=zero,repeat_zero_ce=repeat,
                zero_repeat_equal=zero==repeat,directions=rows,parameters_equal=exact,
                base_hash_before=base,base_hash_after=base_after,rng_equal=rng_equal(rng,_capture_rng()))
    if not exact or base!=base_after or not result['rng_equal'] or zero!=repeat:raise RuntimeError('诊断恢复/重复性门禁失败')
    return result
