# Workshop TODO List

## Current Priority: Intro Notebook Polish
**Branch**: `introductory-notebooks`

### 1. New Introductory Notebooks: PyTorch Profiler → Nsight Systems via a Two-Bug Demo

**Status**: Scripts complete and verified. Notebooks written. Remaining: review, images, TOC wiring.

**What's done (2026-05-28):**
- 7 training scripts in `workspace/source_code/intro/`, all verified on L4 (batch=256):
  - `train_v1.py` — Bug 1 (DataLoader stall), ~260 ms/step
  - `train_v1_profile.py` — Bug 1 + torch.profiler
  - `train_v1_fixed.py` — Bug 1 fixed + torch.profiler, ~94 ms/step
  - `train_v2.py` — AMP + Bug 2 (set_detect_anomaly left on), ~66 ms/step
  - `train_v2_profile.py` — AMP + Bug 2 + torch.profiler
  - `train_v2_nvtx.py` — AMP + Bug 2 + manual NVTX ranges (for nsys)
  - `train_v2_fixed.py` — Bug 2 fixed, ~47 ms/step (~2× over FP32)
- Bug 2 design: `set_detect_anomaly(True)` left on after debugging a NaN issue.
  Creates per-op D2H syncs in backward — sawtooth GPU timeline. +38% overhead at batch=256.
  Empirical search in `workspace/source_code/intro/v2_bakeoff.py`.
- 4 intro notebooks written (following nsys-application.ipynb style):
  - `intro-profiling-concepts.ipynb` — profiling loop, tool overview
  - `intro-pytorch-profiler.ipynb` — torch.profiler, Bug 1 discovery + fix
  - `intro-amp.ipynb` — AMP intro, profiler hits its limits
  - `intro-nsys.ipynb` — nsys plain → NVTX → auto-NVTX, Bug 2 fix
- `nsys-intermediate.ipynb` copied from `nsys-introduction.ipynb` (cut point TBD)
- `--pytorch=autograd-nvtx` flag confirmed in nsys 2026.1.1 on target hardware

**Remaining substeps:**
- **(a) Review notebooks in JupyterLab** — check cell output formatting, verify TensorBoard links, confirm nsys commands work end-to-end
- **(b) Add profiler screenshot images** — run train_v1_profile and train_v1_fixed, capture TensorBoard screenshots; run train_v2_nvtx under nsys, capture the sawtooth timeline. Add to `workspace/jupyter_notebook/images/`.
- **(c) Wire into TOC** — update `workspace/start_here.ipynb` to place the four intro notebooks ahead of the existing lab sequence
- **(d) Find nsys-introduction cut point** — review nsys-introduction.ipynb and decide where to split: intro-nsys.ipynb covers the basic concepts; nsys-intermediate.ipynb picks up with multi-GPU / DDP profiling

**Pedagogical arc:**
- Bug 1: DataLoader stall. `torch.profiler` step view makes it obvious. Fix: `num_workers`, `pin_memory`.
- Bug 2: `set_detect_anomaly(True)` left on in AMP training. Profiler says backward is slow. nsys shows sawtooth per-op syncs. Fix: remove the flag.
- Message: profiler tells you *which section* is slow; nsys tells you *why*.

**nsys flag reference (verified):**
- `nsys profile --trace=cuda,nvtx,osrt` — manual NVTX ranges
- `nsys profile --trace=cuda,nvtx,osrt --pytorch=autograd-nvtx` — auto per-op NVTX, no code changes

---

### 2. Nsight Systems Analysis Recipes (multi-report / cross-rank analysis)
- **Goal**: Introduce the `nsys recipe` system — multi-report statistical analysis that complements opening individual `.nsys-rep` files in the GUI.
- **Notebook home**: Extend `nsys-application.ipynb` with a new section at the end.
- **Substeps:**
  - **(a)** Built-in recipe `nccl_gpu_overlap_trace` on `baseline_nvtx.nsys-rep` + `firstOptim.nsys-rep` — quantifies overlap % before/after DDP optimisation
  - **(b)** Diagnostic pair: `gpu_gaps` + `cuda_gpu_kern_pace` on the multi-GPU reports
  - **(c)** Custom recipe extracting per-rank time in NVTX ranges from `ddp_optimize.py`
- **Notes**: Built-in recipes in container at `<nsys-target-dir>/python/packages/nsys-recipe/recipes/`. Reports at `workspace/reports/`.
- **Resources**: [Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

---

### 3. Add JupyterLab Nsight Extension
- **Goal**: Enable in-browser `.nsys-rep` viewing so students don't need to download files
- **Resource**: https://developer.nvidia.com/tools-overview/nsight-jupyterlab
- **Implementation**: Add to docker-compose pip install and configure JupyterLab extension

---

### 4. Add Nsight Compute Coverage (Future)
- **Goal**: Cover Nsight Compute for kernel-level analysis (warp stalls, cache efficiency, occupancy)
- **When**: After intro notebooks are fully reviewed and wired into TOC
- **Resources**:
  - [Fix GPU Bottlenecks: PyTorch Profiler + Nsight](https://acecloud.ai/blog/gpu-bottlenecks-pytorch-profiler-nsight/)
  - [Profiling PyTorch with Nsight Compute](https://dev-discuss.pytorch.org/t/how-profiling-pytorch-using-nsight-compute/2530/2)

---

## Completed ✓
- ✓ Single-node migration of all notebooks
- ✓ Removed Slurm commands, fixed hardcoded IPs to localhost
- ✓ Created docker-compose.yaml for Brev
- ✓ Added automated data download and setup
- ✓ Added SYS_ADMIN capability for nsys profiling
- ✓ Tested nsys profiling — working perfectly
- ✓ Empirical search for Bug 2 candidate (v2_bakeoff.py) — set_detect_anomaly chosen
- ✓ 7 intro training scripts written and verified (v1/v1_profile/v1_fixed/v2/v2_profile/v2_nvtx/v2_fixed)
- ✓ 4 intro notebooks written (profiling-concepts, pytorch-profiler, amp, nsys)
- ✓ nsys-intermediate.ipynb created (copy of nsys-introduction, cut point TBD)

---

## Notes
- Target hardware: 1× NVIDIA L4 GPU (23GB, Ada Lovelace, FP8 supported) — single node
- Base image: `nvcr.io/nvidia/pytorch:26.02-py3` (nsys 2026.1.1, PyTorch 2.11)
- `torch_tb_profiler` pre-installed; TensorBoard auto-starts on port 8889
- `--pytorch=autograd-nvtx` confirmed working in nsys 2026.1.1

## Sources
- [Interactive Guide: Nsys vs PyTorch Profiler](https://the-dsvolk.github.io/ai-perf/ai-infra/Tale_of_two_profilers.html)
- [PyTorch Profiler Tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Speed Up PyTorch Training 3x with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
- [NASA AI Profiler Guide](https://www.nas.nasa.gov/hackathon/assets/pdf/AI_Profiler.pdf)
