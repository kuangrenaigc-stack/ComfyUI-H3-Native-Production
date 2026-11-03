# FFN row chunking validation — 2026-11-03

The optional upstream benchmark/profiler and the FFN memory fix were integrated separately. Normal production does not run a profiler. Both saved workflows keep the native 10-step draft, learned clean-latent 2x and original eight-step/index-three suffix.

## Measured results

Two RTX 5090 32GB GPUs, native mixed INT8/BF16 weights and native Ref2V v1.1 dynamic LoRA (strength 1, alpha 8), TP2. Synthetic inputs are labeled explicitly; weights and native operators are real.

| Workload | Whole-sequence reference | 16,384-row chunks |
| --- | --- | --- |
| 65,536 rows, INT8 block 0 | 4.06 GiB peak; 95.35 ms median | 2.97 GiB peak; 96.44 ms median |
| 65,536 rows, mixed BF16-output block 29 | 4.10 GiB peak; 106.08 ms median | 3.00 GiB peak; 107.03 ms median |
| 235,520 rows, INT8 block 0 | 14.27 GiB peak; 343.87 ms median | 8.08 GiB peak; 347.63 ms median |

These are allocator peaks for the diagnostic FFN process, including its input/reference buffers, **not whole-video memory figures**. Each of the tested 8,192 / 16,384 / 32,768 row cases on blocks 0/29 was bit-exact to its whole-sequence reference. The large-row case was also bit-exact. The trace for each rank contained one FFN output all-reduce; chunking does not multiply collective calls.

A complete 1920x1088, 362-frame final resample (five evaluations) from an existing learned handoff completed in 700.80 seconds without OOM. Its final MP4 was byte-identical to the existing successful reference. The older reference took 710.59 seconds, but one run each is insufficient evidence of an end-to-end speed improvement.

**The whole-process maximum allocated GPU memory was unchanged** (rank 0 approximately 28.07 GiB, rank 1 25.97 GiB). Other phases dominate that peak. This fix reduces the large FFN allocation requests and their fragmentation exposure; it does not establish that every OOM is eliminated. A fresh first-pass-plus-final V2 run was not part of this validation.

## CPU and transport

The same native TP2 diagnostic reported NCCL `SHM/direct/direct` with cuMem shareable host buffers. The system topology is PHB and peer read/write capability reports unsupported on this host. Seeing `Using network Socket` in NCCL initialization does not mean the actual intra-node all-reduce data used Wi-Fi; the per-channel SHM lines identify the selected local transport.

`OMP_NUM_THREADS=4` is configured per rank, not a fixed CPU core assignment. In the 30-iteration FFN benchmark, each rank averaged about 1.00 logical CPU core equivalent (CPU time / wall time), about 2.00 combined. This includes benchmark coordination/warmup and is not isolated NCCL CPU consumption or a whole-video CPU utilization claim. The process threads can move across logical CPUs; thread count is not core usage.

Raw numeric results without private paths or source media: [validation/ffn-20261103.json](../validation/ffn-20261103.json).
