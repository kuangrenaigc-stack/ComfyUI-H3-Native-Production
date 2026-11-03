# Third-party notices

This repository is an independently maintained production integration, not an official MiniMax or LightX2V release. Existing copyright and license notices in vendored files remain applicable.

- `runtime/source/` contains an inference compatibility snapshot derived from [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V), with local H3 integration changes. It is a subset, not a complete upstream training checkout. Its Apache-2.0 license is retained at `runtime/source/LICENSE`, with additional component licenses retained in their directories. The snapshot is identified as `deployed-compatibility-snapshot-20261102`; no official upstream commit equivalence is asserted. Local adaptations cover clean-latent handoff/resampling, model-bound AdaLN cache validation, and production compatibility.
- `support/diagnostics/upstream/` vendors the Apache-2.0 LightX2V benchmark/profile tools and their offline compute-count helper, pinned in `UPSTREAM.json`, without implementation edits. These are optional diagnostics; they do not replace the production inference source.
- `support/architecture/ComfyUI-H3-Latent-Upscaler-Mamad8-main/upscaler.py` is the H3 clean-latent upscaler architecture distributed with the Mamad8 implementation. Its original MIT license and copyright are retained alongside the file. Model weights are not included or relicensed.
- Turbo LoRA files are external assets from [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo). This package does not distribute these files, the base H3 model, FeiHou weights, or any weight license grant.
- `ComfyUI-LightX2V-RemoteH3` in this bundle is the local reference collector retained for workflow compatibility; the historical remote HTTP submitter is omitted.

New integration, packaging and deployment code is provided under the root Apache-2.0 LICENSE. Dependencies such as ComfyUI, PyTorch, CUDA libraries and attention kernels are installed separately and keep their own terms.
