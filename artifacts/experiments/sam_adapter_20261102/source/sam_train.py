"""仅对冻结G1的状态适配器执行共同随机数的非adaptive SAM。"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn
from state_adapter import set_state
from state_adapter.train import _capture_rng, _restore_rng, future_token_loss


def sam_train_batch(
    model: nn.Module, tokenizer: nn.Module,
    batch: dict[str, np.ndarray | Tensor], optimizer: torch.optim.Optimizer,
    device: torch.device | str, grad_clip: float = 1.0, rho: float = 0.05,
) -> dict[str, float]:
    """两次前向/反向，仅在精确还原参数后执行一次AdamW。

    :param model: 只允许state_adapter参数可训练的Kronos。
    :param tokenizer: 冻结tokenizer，每批只encode一次。
    :param batch: 原WindowCorpus的90+10批次。
    :param optimizer: 仅含全部可训练适配参数的AdamW。
    :param device: 设备。
    :param grad_clip: 第二梯度的全局裁剪阈值。
    :param rho: 非adaptive全局L2半径；零值也执行通用两阶段。
    :returns: 原指标及两前向CE、扰动和第一梯度范数。
    :raises ValueError: 非法配置、批次、参数或梯度。
    """
    if not np.isfinite(rho) or rho < 0 or not np.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError('rho必须非负有限，grad_clip必须正有限')
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params = [p for _,p in named]
    if not params or any('state_adapter' not in n.split('.') for n,_ in named):
        raise ValueError('仅允许已注入的adapter参数可训练')
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if len(optimized)!=len(params) or {id(p) for p in optimized}!={id(p) for p in params}:
        raise ValueError('optimizer必须且仅包含全部可训练adapter参数')
    if not {'x','x_stamp','state'}.issubset(batch):
        raise ValueError('batch缺少字段')
    model.eval(); tokenizer.eval()
    x=torch.as_tensor(batch['x'],dtype=torch.float32,device=device)
    stamps=torch.as_tensor(batch['x_stamp'],dtype=torch.float32,device=device)
    state=torch.as_tensor(batch['state'],dtype=torch.float32,device=device)
    if x.ndim!=3 or x.shape[1:]!=(100,6) or stamps.shape!=(x.shape[0],100,5) or state.shape!=(x.shape[0],3):
        raise ValueError('必须是90+10窗口及batch×3历史状态')
    initial_rng=_capture_rng()
    snapshots=[p.detach().clone() for p in params]
    optimizer.zero_grad(set_to_none=True)
    succeeded=False
    after_first: dict[str, Any] | None = None
    try:
        with torch.no_grad():
            s1,s2=tokenizer.encode(x,half=True)
        set_state(model,state)
        before_forward=_capture_rng()

        def forward_backward() -> tuple[Tensor,Tensor,Tensor]:
            logits=model(s1[:,:-1],s2[:,:-1],stamp=stamps[:,:-1],use_teacher_forcing=False)
            losses=future_token_loss(*logits,s1[:,1:],s2[:,1:])
            if not torch.isfinite(losses[0]).item():
                raise ValueError('训练CE非有限')
            losses[0].backward()
            if not any(p.grad is not None for p in params) or any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in params):
                raise ValueError('适配器梯度缺失或非有限')
            return losses

        first=forward_backward()
        after_first=_capture_rng()
        norm=torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in params if p.grad is not None]),2)
        if not torch.isfinite(norm).item() or norm.item()<=0:
            raise ValueError('适配器梯度范数必须有限且正')
        with torch.no_grad():
            for p in params:
                if p.grad is not None: p.add_(p.grad*(rho/(norm+1e-12)))
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(before_forward)
        second=forward_backward()
        succeeded=True
    finally:
        # 不用减epsilon：浮点加减不能保证逐位还原。
        with torch.no_grad():
            for p,snapshot in zip(params,snapshots): p.copy_(snapshot)
        if not succeeded:
            optimizer.zero_grad(set_to_none=True)
            _restore_rng(initial_rng)
        else:
            assert after_first is not None
            _restore_rng(after_first)
    grad_norm=torch.nn.utils.clip_grad_norm_(params,grad_clip)
    if not torch.isfinite(grad_norm).item() or grad_norm.item()<=0:
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(initial_rng)
        raise ValueError('第二梯度范数必须有限且正')
    optimizer.step()
    return {'loss':float(first[0].detach()),'s1_ce':float(first[1].detach()),
            's2_ce':float(first[2].detach()),'grad_norm':float(grad_norm),
            'first_loss':float(first[0].detach()),'second_loss':float(second[0].detach()),
            'first_grad_norm':float(norm),'rho':rho}
