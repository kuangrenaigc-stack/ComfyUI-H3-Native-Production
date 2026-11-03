# H3 FFN diagnostics

Opt-in tools for development; no profiler or benchmark runs during normal ComfyUI generation.
The vendored LightX2V tools are pinned in `upstream/UPSTREAM.json`; runtime inference keeps its existing native source. This integration does not upgrade the model, CUDA kernels, sampler or quantization.

## Requirements

Use the installed native H3 venv, two idle GPUs, matching mixed weights and native v1.1 LoRA. Run under **both** installed production locks using `run_locked.py`. Its `--control-dir` must point at the control directory actually used by your node, not a new replacement directory. It refuses busy GPUs and never terminates someone else's work. Bound the invocation with `--timeout`.

```bash
PLATFORM=cuda DTYPE=BF16 CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
H3_NODE_WORKER=/srv/comfyui/ComfyUI/custom_nodes/ComfyUI-H3-FeiHou-Native/worker \
/srv/venvs/h3/bin/python /srv/h3/support/diagnostics/run_locked.py \
  --runtime-root /srv/h3/runtime --control-dir /run/user/1000/h3-native \
  --log /srv/h3/diagnostics/ffn-run.log --timeout 1200 -- \
  /srv/venvs/h3/bin/torchrun --standalone --nproc_per_node=2 \
  /srv/h3/support/diagnostics/ffn_bench.py \
  --runtime-root /srv/h3/runtime --weights-dir /srv/models/feihou-aligned-mixed \
  --lora-file /srv/models/minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16.safetensors \
  --output /srv/h3/diagnostics/ffn-65536 --rows 65536 --profile
```

This measures whole FFNs with actual H3 weights, native mixed INT8/BF16 operators, real LoRA and native TP2. Inputs are explicitly synthetic; it is not a full-video benchmark. Default blocks 0 and 29 cover INT8 and BF16 output projections. All compared cases retain the same input and model factors. `chunk_rows=0` is the previous whole-sequence reference. The numerical check reports bit equality and normalized RMSE; a performance result with `pending_review` is not a correctness approval.

Upstream `PreparedModule/benchmark` times synchronized completion, excludes reset/check/barrier time and uses the slowest rank per iteration. Upstream `Capture` writes one warmup plus one active call per rank. Timing with a profiler is diagnostic and is kept separate from the formal benchmark. Workload statistics are disabled in this integration because the deployed engine does not share all the newer compute-stat hooks; no full FLOP/precision coverage is claimed.

Analyze a matching pair of traces:

```bash
cd /srv/h3/support/diagnostics/upstream
/srv/venvs/h3/bin/python -m tools.profile.parallel_report \
  /srv/h3/diagnostics/ffn-65536/block0-chunk16384/rank0 \
  /srv/h3/diagnostics/ffn-65536/block0-chunk16384/rank1 \
  --output /srv/h3/diagnostics/parallel-block0
```

No hardware peak file is assumed. Do not interpret missing compute-stat coverage as unexecuted work, or logical communication GB/s as physical PCIe bandwidth. Full operator-backend sweeps may require additional upstream runtime hooks and dependencies; this release qualifies the whole-FFN benchmark and rank-trace route above.

## Existing-handoff smoke test

`resample_smoke.py` is a developer-only entry that reads a completed single-segment job and runs its existing learned handoff through the unchanged 8-step/index-3 suffix into a **new** directory. Run it under `run_locked.py` too. It does not publish or renumber a video, alter the source job, or regenerate the first pass. This is not an arbitrary-MP4 upscaling feature.

## Production policy

Only final sampling already using `ffn_inplace_activation` installs the row-chunked adapter. Default chunk size is 16,384 rows. Each chunk retains native linear/LoRA calls. The local outputs are assembled before one original-shape TP all-reduce, preserving collective count. The first pass, attention, weight precision, LoRA scaling and time grid are unchanged. A new worker picks up code updates; no ComfyUI restart is necessary for this backend-only file.

For developer A/B, `ffn_memory.install(chunk_rows=0)` selects the retained reference. Never change code or chunk policy inside an already bound production job; retain its original code snapshot or start a new job.
