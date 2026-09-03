"""S1 工程预检（计划 §3，20260903：Blackwell sm_120 上 mamba_ssm/causal_conv1d）。

在生产机（RTX 5090, sm_120, CUDA 13.0, torch 2.13.0+cu130, py3.13）上验证
Mamba-2 块前向/反向（batch 8 × L 512 × d 832，d_inner=1664, headdim=64, nheads=26）：

1. 两个包的安装形态（预编译轮子 vs 源码编译）与版本；
2. causal_conv1d / selective_scan2 CUDA 扩展是否真正加载（非纯 PyTorch 回退）；
3. 前向 + 反向各 5 次计时（首次含编译/缓存预热，取后续均值）。

结果只记录，供 S1 立项时决定学生骨架（Mamba-2 / 纯 PyTorch 回退 / xLSTM 等），
无任何判据。

用法：``/home/user/miniconda3/envs/quant/bin/python -m l1_context.s1_kernel_precheck``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    import torch

    print("=" * 78)
    print("S1 工程预检：Blackwell sm_120 上 mamba_ssm / causal_conv1d")
    print("=" * 78)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} "
          f"capability={torch.cuda.get_device_capability(0)}")

    # —— 1. 安装形态 ——
    import causal_conv1d
    import mamba_ssm

    print(f"causal_conv1d=={causal_conv1d.__version__} @ {Path(causal_conv1d.__file__).parent}")
    print(f"mamba_ssm=={mamba_ssm.__version__} @ {Path(mamba_ssm.__file__).parent}")
    print("安装形态：PyPI 无 cp313/sm120 预编译轮子（pip download 仅解析到 sdist："
          "causal_conv1d-1.7.0.tar.gz / mamba_ssm-2.3.2.post1.tar.gz）；"
          "两者均以 CUDA_HOME=/usr/local/cuda-13.0、TORCH_CUDA_ARCH_LIST=12.0 源码编译成功")

    # —— 2. CUDA 扩展真实加载 ——
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    print(f"causal_conv1d_fn: {causal_conv1d_fn.__module__}")
    print(f"selective_scan_fn: {selective_scan_fn.__module__}")
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update

        print("triton selective_state_update: 可用")
    except Exception as e:  # pragma: no cover
        print(f"triton selective_state_update: 不可用（{e}）")

    # —— 3. Mamba-2 块前向/反向（batch 8 × L 512 × d 832）——
    from mamba_ssm.modules.mamba2 import Mamba2

    device = "cuda:0"
    B, L, D = 8, 512, 832
    torch.manual_seed(42)
    block = Mamba2(d_model=D, d_state=128, d_conv=4, expand=2).to(device)
    n_params = sum(p.numel() for p in block.parameters())
    print(f"Mamba2(d_model={D}, expand=2→d_inner={block.d_inner}, "
          f"headdim={block.headdim}, nheads={block.nheads}) 参数量 {n_params:,}")

    x = torch.randn(B, L, D, device=device, requires_grad=True)
    fwd_times, bwd_times = [], []
    for i in range(6):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = block(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = (y.float() ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i > 0:  # 首次含 kernel JIT/内存池预热，丢弃
            fwd_times.append(t1 - t0)
            bwd_times.append(t2 - t1)
        block.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"前向耗时（5 次均值）：{sum(fwd_times) / len(fwd_times) * 1e3:.2f} ms"
          f"（min {min(fwd_times) * 1e3:.2f} / max {max(fwd_times) * 1e3:.2f}）")
    print(f"反向耗时（5 次均值）：{sum(bwd_times) / len(bwd_times) * 1e3:.2f} ms"
          f"（min {min(bwd_times) * 1e3:.2f} / max {max(bwd_times) * 1e3:.2f}）")
    print(f"输出形状 {tuple(y.shape)}（预期 ({B}, {L}, {D})）| "
          f"loss={float(loss):.4f} | 显存峰值 {peak:.2f} GB")
    print("结论字段：轮子=无（仅 sdist）；编译=成功；前向/反向计时如上；"
          "供 S1 立项决定学生骨架，无判据。")


if __name__ == "__main__":
    main()
