from decode_memory import install as install_decode_memory
install_decode_memory()
from feihou_adapter import install as install_feihou
install_feihou()
import atexit
import json
import os
from pathlib import Path

import torch
from loguru import logger

from lightx2v.common.ops.attn.sage_attn import SageAttn2Weight, sageattn
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from ref2va_phase_runner import Ref2VAPhaseRunner
from rms_accel import install as install_rms

install_rms()

assert sageattn is not None, "No SDPA fallback is allowed"
original = SageAttn2Weight.apply
counter = {"calls": 0, "bridged": 0, "finite_checked": 0}
job = Path(os.environ["H3_BENCH_JOB"])
rank = int(os.environ.get("RANK", 0))


def traced(self, *args, **kwargs):
    counter["calls"] += 1
    tensors = {
        key: kwargs.get(key, args[index] if len(args) > index else None)
        for index, key in enumerate(["q", "k", "v"])
    }
    current = torch.cuda.current_stream(tensors["q"].device)
    default = torch.cuda.default_stream(tensors["q"].device)
    if current != default:
        counter["bridged"] += 1
        default.wait_stream(current)
        for tensor in tensors.values():
            tensor.record_stream(default)
        with torch.cuda.stream(default):
            output = original(self, *args, **kwargs)
        current.wait_stream(default)
        output.record_stream(current)
    else:
        output = original(self, *args, **kwargs)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError(
            "SAGE_NONFINITE rank={} call={} block={}".format(
                rank, counter["calls"], kwargs.get("block_idx")
            )
        )
    counter["finite_checked"] += 1
    if counter["calls"] == 1:
        logger.info("SAGE2_CONFIRMED rank={} stream_bridge_enabled=true", rank)
    return output


SageAttn2Weight.apply = traced


def summary():
    result = {
        "rank": rank,
        **counter,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
    }
    (job / f"backend-rank{rank}.json").write_text(json.dumps(result, indent=2))
    logger.info("SAGE2_SUMMARY {}", json.dumps(result))


atexit.register(summary)
RUNNER_REGISTER["minimax_h3"] = Ref2VAPhaseRunner

from lightx2v.infer import main


if __name__ == "__main__":
    main()
