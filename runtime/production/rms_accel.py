"""Explicit fused RMSNorm boundary; record the actual selected kernel backend."""
import atexit
import json
import os
from pathlib import Path

import torch


def install():
    backend = os.environ.get('H3_RMS_BACKEND', '')
    if not backend:
        return
    if backend.startswith('triton-one-pass-'):
        from lightx2v.common.ops.norm import triton_ops
        def kernel(x, weight, eps, enable_pdl=False):
            return triton_ops.rms_norm_kernel(x, weight, eps)
        module_path = triton_ops.__file__
        implementation = 'official LightX2V Triton one-pass RMSNorm via explicit custom op'
    elif backend.startswith('flashinfer-'):
        import flashinfer
        from flashinfer.norm import rmsnorm as kernel
        module_path = flashinfer.__file__
        implementation = 'flashinfer.norm.rmsnorm via explicit torch custom op'
    elif backend.startswith('sgl-kernel-'):
        import sgl_kernel
        kernel = sgl_kernel.rmsnorm
        module_path = sgl_kernel.__file__
        implementation = 'sgl_kernel.rmsnorm via explicit torch custom op'
    else:
        raise ValueError('Unknown H3_RMS_BACKEND: ' + backend)
    from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightSgl, RMSWeightOnePass
    target = RMSWeightOnePass if backend.startswith('triton-one-pass-') else RMSWeightSgl

    if getattr(target, '_h3_explicit_fused_rms', False):
        return
    counter = {'kernel_calls': 0}

    @torch.library.custom_op('h3_deployment::fused_rmsnorm', mutates_args=(), device_types='cuda')
    def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        counter['kernel_calls'] += 1
        return kernel(x, weight, eps, enable_pdl=False)

    @rmsnorm.register_fake
    def fake(x, weight, eps):
        return torch.empty_like(x)

    original = target.apply

    def apply(self, x):
        weight = self._get_actual_weight()
        if (weight is not None and x.device.type == 'cuda'
                and x.dtype in (torch.float16, torch.bfloat16)
                and weight.dtype == x.dtype and self.sensitive_layer_dtype == self.infer_dtype):
            shape = x.shape
            result = rmsnorm(x.contiguous().view(-1, shape[-1]), weight, self.eps)
            return result.view(shape)
        return original(self, x)

    target.apply = apply
    target._h3_explicit_fused_rms = True

    def report():
        path = Path(os.environ['H3_BENCH_JOB']) / f'rms-kernel-rank{os.environ.get("RANK", "0")}.json'
        path.write_text(json.dumps(dict(counter, module=module_path,
            implementation=implementation, backend=backend, enable_pdl=False), indent=2))

    atexit.register(report)
