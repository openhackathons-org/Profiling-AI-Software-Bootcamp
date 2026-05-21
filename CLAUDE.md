# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a workshop repository for the "Profiling AI Software Bootcamp" covering GPU performance profiling for AI/ML applications using NVIDIA Nsight Systems. The material was originally designed for Slurm cluster environments but is being ported to run on single-node cloud instances.

**Hardware Scaling**:
- Original: Multi-node DGX systems with H100 GPUs (80GB) - tested configuration
- Target: Single-node cloud instances with NVIDIA L40S GPUs (48GB)
- Migration involves both topology changes (multi-node → single-node) and hardware scaling (H100 → L40S)
- L40S chosen for FP8 support (required for Lab 4 - Transformer Engine)

## Environment Setup

### Python Environment (Labs 1-3)
```bash
# Create UV virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download dataset
cd workspace
uv run python source_code/download-data.py
unzip -u data/data-list.zip -d data/
unzip -u source_code/saved_models.zip -d source_code/
```

### Starting Jupyter Lab
```bash
# From repository root with venv activated
jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=./workspace
```

### Container Environment (Lab 4 - Original Multi-Node Setup)
```bash
# Docker
sudo docker build -f Dockerfile --network=host -t tecont:v1 .
docker run --rm -it --gpus all -p 8888:8888 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ./workspace:/workspace tecont:v1 jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace

# Singularity/Apptainer
apptainer build --fakeroot --sandbox tecont.simg Singularity
singularity run --nv -B workspace:/workspace tecont.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace
```

## Repository Structure

### Core Directories
- `workspace/jupyter_notebook/`: All workshop notebook materials organized by lab
- `workspace/source_code/`: Python training scripts (baseline, DDP, model parallelism, FP8)
- `workspace/source_code/slurm/`: Original Slurm batch scripts for multi-node training
- `workspace/data/`: Training datasets (CIFAR-10, models)
- `workspace/reports/`: Nsight Systems profiling outputs

### Key Training Scripts
- `baseline.py`: Single-GPU baseline training with NVTX annotations
- `ddp_baseline.py`, `ddp_optimize.py`: Distributed Data Parallel variants
- `ddp_run_optimize.py`, `ddp_run_optimize_update.py`: DDP with optimizations
- `slurm_ddp-optimize.py`, `slurm_ddp-2nd.py`: Multi-node Slurm scripts
- `source_code/ddp/`, `dp/`, `mp/`: Organized training code by parallelism strategy
- `source_code/fp8/`: FP8 precision and Transformer Engine examples

## Workshop Labs Overview

1. **Lab 1: System Topology** (`system-topology.ipynb`) - Understanding multi-GPU architecture
2. **Lab 2: Distributed Training Strategy**
   - `data-parallelism.ipynb` - DDP training patterns
   - `model-parallelism.ipynb` - Model sharding approaches
3. **Lab 3: Performance Overview**
   - `nsys-introduction.ipynb` - Basic Nsight Systems profiling
   - `nsight_advanced.ipynb` - Advanced profiling techniques
   - `multinode.ipynb` - Multi-node profiling (original cluster setup)
4. **Lab 4: Transformer Engine**
   - `transEng.ipynb` - FP8 and Transformer Engine overview
   - `nsys-fp8.ipynb` - FP8 optimization profiling

Entry point: `workspace/start_here.ipynb`

## Porting from Slurm to Single-Node

### Key Migration Tasks

**Original Setup**: Workshop assumed 2+ nodes with 4 GPUs each (8 total GPUs minimum) managed by Slurm, using H100 GPUs (80GB).

**Target Setup**: Single cloud instance with L4 or L40S GPUs (local execution only).

### Hardware Scaling Considerations

When scaling from H100 to L40S:
- **Memory constraints**: L40S (48GB) has less memory than H100 (80GB); may need to reduce batch sizes
- **Model sizes**: Current scripts use ResNet50d with 336x336 images; should work on L40S
- **FP8 features**: Lab 4 (Transformer Engine) requires FP8 support; L40S supports FP8 (Ada Lovelace architecture)
- **Multi-GPU**: Adjust `WORLD_SIZE` for available GPU count on instance
- **NCCL Topology**: Notebooks show examples of 8-GPU (single node) and 2x4 GPU (multi-node) ring topologies for educational purposes, even if not executing with that configuration

### Common Patterns to Update in Notebooks

1. **Slurm Commands**: Replace `sbatch`, `srun`, `squeue` with direct execution
   - Original: `!sbatch script.slurm`
   - Updated: `!bash script.sh` or direct Python execution

2. **Multi-Node torch.distributed**: Convert to single-node multi-GPU
   - Original: `torchrun` with `--nnodes=2 --nproc_per_node=4`
   - Updated: `torchrun --nnodes=1 --nproc_per_node=<available_gpus>`
   - Or use: `python -m torch.distributed.launch`

3. **Environment Variables**: Update distributed training vars for local execution
   - `MASTER_ADDR` should be `localhost` or `127.0.0.1` (not remote node IP)
   - `MASTER_PORT` can remain arbitrary (e.g., `12355`)

4. **GPU Allocation**: Adjust for available local GPUs
   - Check available GPUs: `torch.cuda.device_count()`
   - Update `WORLD_SIZE`, `NCCL` configuration appropriately

5. **Profiling Commands**: Adjust Nsight Systems profiling for local multi-GPU
   - Use `nsys profile` with appropriate process tracking flags
   - Profile single-node multi-GPU rather than multi-node setups

### Planned Enhancements for Cloud Instance Mode

- **jupyterlab-nvdashboard**: Real-time GPU monitoring dashboard in JupyterLab
- **torch_tb_profiler**: TensorBoard profiler plugin for PyTorch profiling visualization

## Important Implementation Details

### Distributed Training Setup
- Uses PyTorch DDP with NCCL backend
- Training scripts expect `LOCAL_RANK`, `RANK`, `WORLD_SIZE` environment variables
- Typical batch size pattern: `BATCH_SIZE = 256 // WORLD_SIZE`
- Training uses ResNet50d on CIFAR-10 with 336x336 images

### NVTX Annotations
Many scripts use `torch.cuda.nvtx` for profiling markers:
```python
from torch.cuda import nvtx
nvtx.range_push("training_step")
# ... code ...
nvtx.range_pop()
```

### Known Issues

**"Invalid device ordinal" error**: Occurs when Slurm allocates all 8 GPUs from a single node instead of distributing across 2 nodes (4 GPUs each). This is relevant to the original cluster setup but should not occur in single-node porting.

**Multi-node requirement**: Lab 4 originally required 2 nodes minimum. When porting, adjust expectations and code to work with available GPUs on single instance.

## Key Dependencies

- PyTorch 2.6.0 with CUDA support
- Nsight Systems (external tool, must be installed separately)
- NVTX for profiling annotations
- timm (PyTorch Image Models)
- Standard ML stack: torchvision, numpy, pandas, tqdm

Container base: `nvcr.io/nvidia/pytorch:26.02-py3`
