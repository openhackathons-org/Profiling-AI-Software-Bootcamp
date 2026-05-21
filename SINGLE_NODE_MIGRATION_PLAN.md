# Single-Node Migration Plan for Jupyter Notebooks

## Overview
This document outlines the changes needed to convert all Jupyter notebooks from Slurm cluster execution to single-node execution.

## Files Requiring Changes

### High Priority (Core Training Labs)
1. `data-parallelism.ipynb` - Heavy Slurm usage, multi-node examples
2. `model-parallelism.ipynb` - Slurm with model parallelism
3. `nsys-application.ipynb` - Multiple profiling examples with srun
4. `system-topology.ipynb` - GPU topology commands with srun

### Medium Priority (Advanced Topics)
5. `multinode.ipynb` - Entire notebook focused on multi-node profiling
6. `nsight_advanced.ipynb` - Multi-process profiling guidance

### Lower Priority (Context/Overview)
7. `advanced_optimizations.ipynb` - Contains distributed setup code
8. Other notebooks - May have references but less direct execution

---

## Change Patterns Required

### 1. SLURM Command Replacements

#### Pattern A: `srun` with nvidia-smi
**Original:**
```bash
!srun --partition=primary -n1 --gres=gpu:4 nvidia-smi
!srun --partition=primary -n1 --gres=gpu:8 nvidia-smi topo -m
!srun --partition=primary -n2 --gres=gpu:8 nvidia-smi topo -p2p r
```

**Replace with:**
```bash
!nvidia-smi
!nvidia-smi topo -m
!nvidia-smi topo -p2p r
```

**Notes:**
- Remove all srun wrappers for nvidia-smi commands
- GPU count flags (--gres) become irrelevant; command uses all available GPUs
- Partition and task count flags removed

---

#### Pattern B: `srun` with torchrun (single-node DDP/DP)
**Original:**
```bash
!cd ../source_code && srun -p primary -N 1 --gres=gpu:4 torchrun --nnodes=1 --nproc-per-node=4 dp/main.py
!cd ../source_code && srun -p primary -N 1 --gres=gpu:4 torchrun --nproc_per_node=4 --nnodes=1 --standalone --master_addr="localhost" --master_port=1234 ddp_baseline.py
```

**Replace with:**
```bash
!cd ../source_code && torchrun --nnodes=1 --nproc-per-node=4 dp/main.py
!cd ../source_code && torchrun --nproc_per_node=4 --nnodes=1 --standalone --master_addr="localhost" --master_port=1234 ddp_baseline.py
```

**Notes:**
- Simply remove the `srun -p primary -N 1 --gres=gpu:4` prefix
- Keep all torchrun arguments intact
- `--nnodes=1` is redundant but harmless (as user specified)

**Affected Files:**
- `data-parallelism.ipynb` (line 243)
- `model-parallelism.ipynb` (line 191) - uses 3 GPUs
- `nsys-application.ipynb` (lines 81, 350)

---

#### Pattern C: `srun` with nsys profile
**Original:**
```bash
!cd ../source_code && srun -p primary -N 1 --gres=gpu:4 nsys profile --trace cuda,osrt,nvtx --cuda-graph-trace=node -o reports/ddp-baseline_nvtx torchrun --nproc_per_node=4 --nnodes=1 --standalone --master_addr="localhost" --master_port=1234 ddp-baseline_nvtx.py
```

**Replace with:**
```bash
!cd ../source_code && nsys profile --trace cuda,osrt,nvtx --cuda-graph-trace=node -o reports/ddp-baseline_nvtx torchrun --nproc_per_node=4 --nnodes=1 --standalone --master_addr="localhost" --master_port=1234 ddp-baseline_nvtx.py
```

**Notes:**
- Remove srun wrapper, keep full nsys command
- All profiling arguments remain the same
- Output path (-o reports/...) stays unchanged

**Affected Files:**
- `nsys-application.ipynb` (lines 149, 293)
- `nsight_advanced.ipynb` (line 61)

---

#### Pattern D: `sbatch` with multi-node scripts
**Original:**
```bash
!sbatch workspace/source_code/slurm/ddp_multinode.slurm
```

**Replace with:**
```bash
!bash workspace/source_code/slurm/ddp_multinode.sh
```

**Notes:**
- This requires creating shell script versions of .slurm files
- Need to extract the executable commands from SLURM scripts
- Remove SLURM directives (#SBATCH lines)
- Convert SLURM environment variables to direct values

**Affected Files:**
- `data-parallelism.ipynb` (line 606)
- `multinode.ipynb` (referenced but not executed directly)

**Action Required:**
- Create new `.sh` versions of slurm scripts OR
- Replace with direct torchrun execution in notebooks

---

#### Pattern E: Interactive srun sessions (mentioned in text)
**Original (in markdown/text cells):**
```bash
srun -p primary -N 1 --gres=gpu:4 --pty bash
squeue --me
scancel <JOBID>
```

**Replace with (in markdown/text cells):**
```bash
# No longer needed - commands run directly in notebooks
# To run interactively: just execute commands in terminal with venv activated
```

**Notes:**
- These are explanatory text references to Slurm workflow
- Update text to explain direct execution model
- Remove references to job queue management (squeue, scancel)

**Affected Files:**
- `data-parallelism.ipynb` (lines 440, 482, 516, 522)

---

### 2. Multi-Node Configuration Changes

#### Pattern F: Hardcoded IP addresses
**Original:**
```python
os.environ['MASTER_ADDR'] = '172.31.26.15'
os.environ['MASTER_PORT'] = '12345'
```

```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="172.31.26.15" --master_port=1234 test_ddp.py
```

**Replace with:**
```python
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
```

```bash
torchrun --nproc_per_node=4 --nnodes=1 --master_addr="localhost" --master_port=1234 test_ddp.py
```

**Notes:**
- Replace all IP addresses with 'localhost' or '127.0.0.1'
- Change --nnodes=2 to --nnodes=1
- Remove --node_rank argument (not needed for single node)
- Keep port numbers (arbitrary but must be consistent)

**Affected Files:**
- `data-parallelism.ipynb` (lines 314-315, 430, 434)

---

#### Pattern G: Dynamic IP discovery via SLURM
**Original:**
```bash
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
```

**Replace with:**
```bash
export MASTER_ADDR=localhost
```

**Notes:**
- No need for dynamic discovery on single node
- Simply hardcode to localhost

**Affected Files:**
- `data-parallelism.ipynb` (line 583)
- SLURM scripts in `source_code/slurm/`

---

### 3. GPU Count and World Size Adjustments

#### Pattern H: Adjust GPU counts in documentation
**Original text mentions:**
- "two machines/nodes, each with two GPUs" (4 total)
- "two nodes (4 GPUs each)" (8 total)
- `--gres=gpu:4` or `--gres=gpu:8`

**Replace with:**
- "single machine with N GPUs" (where N = available count)
- Update examples to reflect actual available GPU count
- Add note about scaling from original 8-GPU setup

**Notes:**
- Don't hardcode GPU counts where possible
- Use `torch.cuda.device_count()` dynamically
- Update batch size calculations if needed

**Affected Files:**
- `data-parallelism.ipynb` (lines 297-303, 454, 545-556)
- `multinode.ipynb` (line 73)

---

### 4. Special Case: multinode.ipynb

This notebook is entirely focused on multi-node profiling. Options:

**Option A: Adapt to single-node multi-GPU**
- Retitle: "Multi-GPU Profiling" instead of "Multi-Node"
- Show NCCL communication between GPUs on same node
- Profile 2-4 GPU processes instead of 8

**Option B: Mark as reference/advanced**
- Add prominent note at top explaining this was multi-node content
- Show how concepts apply to single-node
- Keep for educational value but mark as "not executable"

**Option C: Remove entirely**
- If content is redundant with other notebooks

**Recommendation:** Option A - adapt to show multi-GPU profiling patterns

**Affected Files:**
- `multinode.ipynb` (entire notebook)

---

### 5. Source Code Script Changes

Some notebooks reference scripts in `source_code/` that may have hardcoded multi-node assumptions:

**Files to check:**
- `source_code/ddp/main.py` - Has hardcoded MASTER_ADDR='10.184.92.71' (line 42)
- `source_code/slurm/*.slurm` - Convert to .sh scripts
- Any other scripts with IP addresses or node counts

**Action:**
- Update hardcoded IPs to 'localhost'
- Remove node-rank logic where present
- Test all training scripts work with single-node

---

## GPU Count Scaling Strategy

Given migration from H100 (80GB) to L4 (24GB) or L40S (48GB):

### Memory-Constrained Adjustments
1. **Batch Sizes**: Current `BATCH_SIZE = 256 // WORLD_SIZE`
   - May need to reduce to fit L4 memory (24GB)
   - Test with smaller batches: 128 or 64 total

2. **Image Sizes**: Current uses 336x336 images
   - Consider reducing to 224x224 if memory issues arise
   - Or use smaller model variant

3. **Model**: Currently ResNet50d from timm
   - This should fit on L4/L40S
   - Monitor memory usage with nvidia-smi

4. **Number of Processes**:
   - If using L4: may need to reduce from 4 to 2 processes
   - L40S: should handle 4 processes
   - Adjust --nproc-per-node accordingly

---

## Implementation Priority

### Phase 1: Core Training Notebooks (Must Work)
1. `system-topology.ipynb` - GPU topology understanding
2. `data-parallelism.ipynb` - Primary DDP training
3. `nsys-application.ipynb` - Basic profiling

**Success Criteria:** Can execute all cells without errors on single-node instance

### Phase 2: Advanced Topics (Should Work)
4. `model-parallelism.ipynb` - Model sharding
5. `nsight_advanced.ipynb` - Advanced profiling
6. `advanced_optimizations.ipynb` - Optimization techniques

**Success Criteria:** Concepts translate to single-node, executable examples work

### Phase 3: Multi-Node Content (Document/Adapt)
7. `multinode.ipynb` - Adapt or document as reference
8. Update all remaining references

**Success Criteria:** Clear documentation of single-node vs multi-node differences

---

## Testing Checklist

For each updated notebook:
- [ ] All code cells execute without Slurm errors
- [ ] GPU commands (nvidia-smi) work correctly
- [ ] Training scripts complete successfully
- [ ] Profiling commands generate reports
- [ ] Output files created in expected locations
- [ ] Explanatory text matches new single-node model
- [ ] No references to unavailable Slurm commands remain

---

## Additional Enhancements (Post-Migration)

Once single-node execution works:
1. Add jupyterlab-nvdashboard for real-time GPU monitoring
2. Add torch_tb_profiler for TensorBoard visualization
3. Add memory usage monitoring code cells
4. Add GPU count detection and adaptive batch sizing
5. Update images/diagrams showing single-node topology

---

## Potential Issues

### Issue 1: FP8 Support (RESOLVED - Using L40S)
- **Decision**: Target L40S GPUs which support FP8 (Ada Lovelace architecture)
- **Impact**: Lab 4 (Transformer Engine) will work as designed
- **Note**: If hardware changes to L4 later, will need to adapt Lab 4

### Issue 2: Insufficient GPU Memory
- **Problem**: Models may not fit with current batch sizes
- **Impact**: OOM errors during training
- **Solution**: Reduce batch sizes, add try-catch with retry logic

### Issue 3: NCCL Ring Construction (Educational Content)
- **Approach**: Keep examples showing 8-GPU (single-node) and 2x4 GPU (multi-node) topologies
- **Rationale**: Educational value in understanding different NCCL ring patterns
- **Implementation**: Show diagrams/outputs as examples even if not executing with that exact configuration
- **Student Execution**: Will run on available GPU count, but learn about larger topologies

### Issue 4: Profiling Report Differences
- **Problem**: Multi-node profiling creates multiple reports (1 per rank)
- **Impact**: Analysis sections expect 8 reports
- **Solution**: Update to analyze N reports where N = GPU count
