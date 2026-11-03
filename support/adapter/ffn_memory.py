"""Bound H3 FFN activations by rows; preserve native MM, LoRA and one TP reduction.

Only the fresh SwiGLU projection is mutated. Input/weights remain unchanged.
Normal production never enables profiling. chunk_rows=0 retains the old path.
"""
import atexit
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

DEFAULT_CHUNK_ROWS = 16384
_INSTALLED = False


def validate_chunk_rows(value):
    if type(value) is not int or value < 0:
        raise ValueError('FFN chunk_rows must be a nonnegative integer; 0 disables chunking')
    return value


def _activation(weights, hidden_states):
    projected = weights.in_proj.apply(hidden_states)
    value, gate = projected.chunk(2, dim=-1)
    F.silu(gate, inplace=True)
    value.mul_(gate)
    activated = value.contiguous()
    del value, gate, projected
    return activated


def low_memory_ff(weights, hidden_states):
    """Previous whole-sequence path, retained as the numerical/benchmark reference."""
    return weights.out_proj.apply(_activation(weights, hidden_states))


def chunked_ff(weights, hidden_states, chunk_rows=DEFAULT_CHUNK_ROWS):
    """Chunk only independent tokens; reduce the assembled local output once.

    H3 uses a column-parallel input and row-parallel output projection. Calling
    out_proj.apply per chunk would multiply collective calls. Use its native
    local MM (including LoRA) and perform the same all_reduce/bias epilogue once.
    Unsupported TP wrappers fail rather than silently changing communication.
    """
    validate_chunk_rows(chunk_rows)
    if hidden_states.ndim != 2:
        raise ValueError('H3 FFN requires unbatched [tokens, hidden] activations')
    if torch.is_grad_enabled() and hidden_states.requires_grad:
        raise ValueError('Chunked H3 FFN is inference-only')
    rows = hidden_states.shape[0]
    if chunk_rows == 0 or rows <= chunk_rows:
        return low_memory_ff(weights, hidden_states)

    output_projection = weights.out_proj
    tp = None
    if hasattr(output_projection, '_mm'):
        from lightx2v.common.ops.mm.mm_weight import MMWeightTP
        if not isinstance(output_projection, MMWeightTP) or output_projection.split_dim != 'row':
            raise TypeError('Chunked H3 FFN requires the native row-parallel output wrapper')
        tp = output_projection
        if tp.tp_size > 1 and tp.reduce_output and tp.tp_group is None:
            raise ValueError('TP output reduction requires the native process group')
        output_projection = tp._mm

    output = None
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        activated = _activation(weights, hidden_states[start:stop])
        local = output_projection.apply(activated)
        if output is None:
            output = local.new_empty((rows, local.shape[-1]))
        output[start:stop].copy_(local)
        del local, activated

    if tp is not None and tp.reduce_output and tp.tp_size > 1:
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp.tp_group)
    # Match native MMWeightTP.apply: a saved row bias is added after reduction.
    if tp is not None and tp.reduce_output and tp.tp_size > 1 and tp._row_split_bias is not None:
        output.add_(tp._row_split_bias)
    return output


def install(chunk_rows=DEFAULT_CHUNK_ROWS):
    global _INSTALLED
    validate_chunk_rows(chunk_rows)
    if _INSTALLED:
        raise RuntimeError('FFN memory adapter was already installed in this process')
    from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
    counter = dict(chunk_rows=chunk_rows, calls=0, local_projection_chunks=0, max_rows=0)

    def ff(weights, x):
        counter['calls'] += 1
        counter['max_rows'] = max(counter['max_rows'], x.shape[0])
        counter['local_projection_chunks'] += max(1, (x.shape[0] + chunk_rows - 1) // chunk_rows) if chunk_rows else 1
        return chunked_ff(weights, x, chunk_rows)

    MiniMaxH3TransformerInfer._ff = staticmethod(ff)
    _INSTALLED = True

    def report():
        folder = os.environ.get('H3_BENCH_JOB')
        if folder:
            rank = os.environ.get('RANK', '0')
            (Path(folder) / f'ffn-memory-rank{rank}.json').write_text(json.dumps(dict(
                **counter, algorithm='row-chunked-native-ffn-v1',
                native_base_kernels=True, native_lora=True,
                tp_output_reductions_per_ffn=1, weights_dtype_unchanged=True), indent=2))

    atexit.register(report)
