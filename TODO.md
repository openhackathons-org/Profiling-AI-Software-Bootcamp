# Workshop TODO List

## Tomorrow's Priority Tasks

### 1. PyTorch Profiler Integration (shapes downstream notebook work)
- **Goal**: Introduce PyTorch's native profiling alongside the existing nsys + manual NVTX workflow, so students see the recommended three-level funnel (PyTorch Profiler → Nsight Systems → Nsight Compute) instead of only the middle layer.
- **Why early**: Decisions made here (which notebook is the entry point, whether `emit_nvtx()` stays, whether manual NVTX ranges get replaced by `--pytorch=autograd-nvtx`) ripple through every later notebook edit.

- **Current state (audited 2026-05-22)**:
  - Manual `torch.cuda.nvtx.range_push/range_pop` in `source_code/baseline.py`, `ddp_optimize.py`, `fp8/te_transformer_layer_fp8.py`, `fp8/te_unfused_attn_fp8.py`.
  - One vestigial `torch.autograd.profiler.emit_nvtx()` wrapping the training loop in `baseline.py`; not used elsewhere.
  - **Zero** uses of `torch.profiler.profile()`, `torch_tb_profiler`, or `nsys profile --pytorch=autograd-nvtx` anywhere — despite `torch_tb_profiler` being pre-installed in the container.

- **Substeps (do in order)**:
  - **(a) Entry-point notebook**: Pick where students first meet PyTorch Profiler. Candidates: add cells to existing `nsys-introduction.ipynb` (currently markdown-only — good blank canvas) *or* create a new `pytorch-profiler-intro.ipynb` ahead of `nsys-introduction` in the TOC. Decide which before writing any code.
  - **(b) Instrument `baseline.py` with `torch.profiler`**: Add a `torch.profiler.profile()` context with `tensorboard_trace_handler(logdir='/workspace/logs')`. baseline.py is simplest and already has manual NVTX, so students can compare the two views side-by-side. Add a notebook cell that runs it and points students to TensorBoard at `:8889`.
  - **(c) Auto-NVTX via nsys on `ddp_optimize.py`**: Remove the manual `nvtx.range_push/pop` lines, profile via `nsys profile --pytorch=autograd-nvtx --trace=cuda,osrt,nvtx …`. Compare timelines with the manual-NVTX version (`ddp-baseline_nvtx.py`) in a notebook cell to show what auto-annotation produces vs hand-curated ranges.
  - **(d) Combined cell**: In `nsys-application.ipynb` (or a new lab), run a training step under *both* `torch.profiler.profile()` *and* `nsys profile --pytorch=autograd-nvtx` simultaneously. Demonstrates the three-level funnel concretely on one training script.
  - **(e) Decide fate of `emit_nvtx()`**: With `--pytorch=autograd-nvtx` providing automatic PyTorch-op NVTX from nsys' side, the `emit_nvtx()` context in `baseline.py` is largely redundant. Either remove it and rely on the nsys flag, or keep it as a "here's the in-process equivalent" teaching moment. Pick one and document the reasoning.

- **Components reference**:
  - `torch_tb_profiler`: pre-installed; TensorBoard already auto-starts in the container on port 8889.
  - `torch.profiler.profile()` + `tensorboard_trace_handler`: macro-level per-step timing, view in TensorBoard.
  - `nsys profile --pytorch=autograd-nvtx`: nsys flag that auto-emits NVTX ranges for PyTorch ops; no code changes required.
  - `torch.autograd.profiler.emit_nvtx()`: in-process equivalent of the above; older API.

- **Workflow**: PyTorch Profiler (which step is slow?) → Nsight Systems (root cause: I/O? sync?) → Nsight Compute (kernel details)

- **Resources**:
  - [PyTorch Profiler with TensorBoard Tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
  - [Automatic NVTX annotations with --pytorch flag](https://the-dsvolk.github.io/ai-perf/ai-infra/Tale_of_two_profilers.html)
  - [Speed Up PyTorch Training with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)

### 2. Nsight Systems Analysis Recipes (multi-report / cross-rank analysis)
- **Goal**: Introduce the `nsys recipe` system — multi-report statistical analysis that complements opening individual `.nsys-rep` files in the GUI. Cover a built-in recipe, a diagnostic pair, and a custom recipe, all anchored to the existing 4-GPU DDP reports the lab already produces.
- **Why early**: The recipe system is the natural next step after students have learned to read individual nsys timelines. It also gives concrete numbers (overlap %, idle gaps, per-rank stats) to back up qualitative timeline observations.
- **Notebook home**: Extend `nsys-application.ipynb` with a new section at the end. Reuses the `baseline_nvtx.nsys-rep` and `firstOptim.nsys-rep` reports the lab already generates, so the before/after pairing lives in one place.

- **Substeps (do in order)**:
  - **(a) Built-in recipe headline demo — `nccl_gpu_overlap_trace`**: Run the recipe on both `baseline_nvtx.nsys-rep` and `firstOptim.nsys-rep`. Compare the resulting communication-vs-compute overlap percentages per rank. Directly quantifies the DDP optimization the lab teaches qualitatively ("overlap went from X% to Y%"). One CLI invocation per report; the recipe emits CSV + a Jupyter notebook in an `.nsys-analysis` bundle. Decide whether to show the auto-generated notebook inline or pull the CSV into our own cells.
  - **(b) Diagnostic recipe pair — `gpu_gaps` + `cuda_gpu_kern_pace`**: `gpu_gaps` (default threshold 500ms) flags idle periods — surfaces data-loader stalls and sync waits. `cuda_gpu_kern_pace` shows whether kernel cadence is consistent across all 4 ranks; divergence indicates stragglers. Run both on `firstOptim.nsys-rep` (the more interesting one) feeding all 4 GPU traces together. Demonstrates the multi-report value-add that's hard to see in a single timeline.
  - **(c) Custom recipe — per-rank time in NVTX ranges**: Write a bespoke recipe that extracts the manual NVTX ranges already in `ddp_optimize.py` (`"Train"`, `"Data loading"`, `"Copy to device"`, `"Forward pass"`, `"Backward pass"`) and produces a per-rank summary (stacked bar or table). Will likely require copying a built-in recipe (e.g., `nccl_sum.py`) as a template since the public docs are light on the user-defined-recipe API. Side benefit: ties student-written NVTX instrumentation from earlier labs into programmatic downstream analysis.

- **Operational notes**:
  - Reports already exist at `workspace/reports/baseline_nvtx.nsys-rep` and `workspace/reports/firstOptim.nsys-rep` (regenerated whenever the user re-runs `nsys-application.ipynb`).
  - `nsys recipe` is available in the compose container alongside `nsys` itself (`nvcr.io/nvidia/pytorch:26.02-py3`); no new installs required.
  - Built-in recipes live at `<nsys-target-dir>/python/packages/nsys-recipe/recipes/` inside the container — locate via `nsys --version` then `find` from `/usr/local/cuda`. Read one before writing (c).
  - Output `.nsys-analysis` bundles include both raw CSV/Parquet and a generated Jupyter notebook; we can either embed those notebooks or load the CSVs into our own cells.

- **Resources**:
  - [Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
  - "Available Advanced Analysis Recipes" section of the above for the recipe catalog
  - "Tutorial: Create a User-Defined Recipe" section (referenced but light) — fall back to reading built-in recipes as templates

### 3. Add JupyterLab Nsight Extension
- **Goal**: Enable in-browser Nsight Systems profiling
- **Resource**: https://developer.nvidia.com/tools-overview/nsight-jupyterlab
- **Why**: Students can view profiling results directly in JupyterLab without downloading .nsys-rep files
- **Implementation**: Add to docker-compose pip install and configure JupyterLab extension
- **Benefit**: First couple of notebooks can use local Nsight analysis before needing downloads

### 4. Add Nsight Compute Coverage (Future)
- **Goal**: Cover Nsight Compute for detailed kernel analysis
- **Use Case**: Micro-level Python kernel profiling (warp stalls, cache efficiency, occupancy)
- **Note**: Third level after PyTorch Profiler and Nsight Systems
- **When**: After establishing PyTorch Profiler workflow
- **Resources**:
  - [Fix GPU Bottlenecks: PyTorch Profiler + Nsight](https://acecloud.ai/blog/gpu-bottlenecks-pytorch-profiler-nsight/)
  - [Profiling PyTorch with Nsight Compute](https://dev-discuss.pytorch.org/t/how-profiling-pytorch-using-nsight-compute/2530/2)

## Completed Today ✓
- ✓ Single-node migration of all notebooks
- ✓ Removed Slurm commands
- ✓ Fixed hardcoded IPs to localhost
- ✓ Created docker-compose.yaml for Brev
- ✓ Added automated data download and setup
- ✓ Added SYS_ADMIN capability for nsys profiling
- ✓ Tested nsys profiling - working perfectly!

## Implementation Notes

### PyTorch + Nsight Integration Points
1. **Automatic NVTX Annotations**:
   - Update notebook cells to use: `!nsys profile --pytorch=autograd-nvtx --trace=cuda,osrt,nvtx ...`
   - No code changes needed - automatically annotates PyTorch ops

2. **PyTorch Profiler TensorBoard**:
   - Already have `torch_tb_profiler` installed
   - Can use `torch.profiler.profile()` with `tensorboard_trace_handler`
   - View results on TensorBoard (port 8889)

3. **Profiling Workflow** (from research):
   - Level 1: PyTorch Profiler - "Which operator is slow?"
   - Level 2: Nsight Systems - "Why? (I/O bound, sync bound, etc.)"
   - Level 3: Nsight Compute - "Kernel-level details (warp stalls, cache)"

## Notes
- Target hardware: 4× NVIDIA L4 GPUs (23GB each, Ada Lovelace, FP8 supported)
- Base image: nvcr.io/nvidia/pytorch:26.02-py3
- All changes on main branch, pushed to origin
- torch_tb_profiler already installed for TensorBoard integration

## Sources
- [Interactive Guide: Nsys vs PyTorch Profiler](https://the-dsvolk.github.io/ai-perf/ai-infra/Tale_of_two_profilers.html)
- [PyTorch Profiler Tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [Speed Up PyTorch Training 3x with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
- [Fix GPU Bottlenecks Guide](https://acecloud.ai/blog/gpu-bottlenecks-pytorch-profiler-nsight/)
- [NASA AI Profiler Guide](https://www.nas.nasa.gov/hackathon/assets/pdf/AI_Profiler.pdf)
