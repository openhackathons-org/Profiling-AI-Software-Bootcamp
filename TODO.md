# Workshop TODO List

## Tomorrow's Priority Tasks

### 1. Add JupyterLab Nsight Extension
- **Goal**: Enable in-browser Nsight Systems profiling
- **Resource**: https://developer.nvidia.com/tools-overview/nsight-jupyterlab
- **Why**: Students can view profiling results directly in JupyterLab without downloading .nsys-rep files
- **Implementation**: Add to docker-compose pip install and configure JupyterLab extension
- **Benefit**: First couple of notebooks can use local Nsight analysis before needing downloads

### 2. PyTorch Profiler Integration
- **Goal**: Leverage PyTorch's built-in profiling capabilities
- **Components**:
  - **torch_tb_profiler**: Already installed for TensorBoard visualization
  - **torch.profiler**: PyTorch's native profiler for macro-level step timing
  - **Automatic NVTX**: Use `nsys profile --pytorch=autograd-nvtx` flag for automatic annotations
  - **Manual NVTX**: `torch.autograd.profiler.emit_nvtx()` context manager for custom markers
- **Workflow**: PyTorch Profiler (which step is slow?) → Nsight Systems (root cause: I/O? sync?) → Nsight Compute (kernel details)
- **Resources**:
  - [PyTorch Profiler with TensorBoard Tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
  - [Automatic NVTX annotations with --pytorch flag](https://the-dsvolk.github.io/ai-perf/ai-infra/Tale_of_two_profilers.html)
  - [Speed Up PyTorch Training with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)

### 3. Add Nsight Compute Coverage (Future)
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
- Target hardware: L40S GPUs (48GB) with FP8 support
- Base image: nvcr.io/nvidia/pytorch:26.02-py3
- All changes on main branch, pushed to origin
- torch_tb_profiler already installed for TensorBoard integration

## Sources
- [Interactive Guide: Nsys vs PyTorch Profiler](https://the-dsvolk.github.io/ai-perf/ai-infra/Tale_of_two_profilers.html)
- [PyTorch Profiler Tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [Speed Up PyTorch Training 3x with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
- [Fix GPU Bottlenecks Guide](https://acecloud.ai/blog/gpu-bottlenecks-pytorch-profiler-nsight/)
- [NASA AI Profiler Guide](https://www.nas.nasa.gov/hackathon/assets/pdf/AI_Profiler.pdf)
