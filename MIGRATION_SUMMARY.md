# Single-Node Migration Summary

## Overview
Successfully migrated the Profiling AI Software Bootcamp from Slurm cluster execution to single-node cloud instance execution.

**Branch**: `single-node-migration`
**Target Hardware**: L40S GPUs (48GB) with FP8 support
**Date**: 2026-05-21

## Commits

1. **1b5eb33** - Add documentation for single-node migration
   - Created CLAUDE.md with repository guidance
   - Created SINGLE_NODE_MIGRATION_PLAN.md with detailed migration strategy

2. **fc11934** - Migrate system-topology and data-parallelism notebooks to single-node
   - system-topology.ipynb: Removed srun wrappers from 3 nvidia-smi commands
   - data-parallelism.ipynb: Removed srun wrappers, updated MASTER_ADDR examples to localhost, removed multi-node Lab Activities

3. **92e901e** - Migrate nsys-application notebook to single-node
   - Removed srun wrappers from 4 torchrun and nsys profile commands

4. **9e641d6** - Fix hardcoded IP address in ddp/main.py
   - Changed MASTER_ADDR from 10.184.92.71 to localhost

5. **8079857** - Migrate medium-priority notebooks to single-node
   - model-parallelism.ipynb: Removed srun wrapper
   - multinode.ipynb: Converted to educational reference material
   - nsight_advanced.ipynb: Removed srun, added single-node examples

## Files Modified

### Jupyter Notebooks (6 notebooks)
1. **system-topology.ipynb** - 3 code cells updated
2. **data-parallelism.ipynb** - 2 code cells + 3 markdown sections updated
3. **nsys-application.ipynb** - 4 code cells updated
4. **model-parallelism.ipynb** - 1 code cell updated
5. **multinode.ipynb** - Educational restructure, 1 code cell commented
6. **nsight_advanced.ipynb** - 1 code cell + 1 markdown section updated

### Source Code (1 file)
1. **workspace/source_code/ddp/main.py** - Fixed hardcoded IP

### Documentation (3 new files)
1. **CLAUDE.md** - Repository guidance for future Claude Code instances
2. **SINGLE_NODE_MIGRATION_PLAN.md** - Detailed migration plan
3. **MIGRATION_SUMMARY.md** - This file

## Changes Summary

### Pattern A: Remove srun wrappers from nvidia-smi
**Before:**
```bash
!srun --partition=primary -n1 --gres=gpu:4 nvidia-smi
```

**After:**
```bash
!nvidia-smi
```

**Files affected**: system-topology.ipynb (3 cells)

---

### Pattern B: Remove srun wrappers from torchrun
**Before:**
```bash
!cd ../source_code && srun -p primary -N 1 --gres=gpu:4 torchrun --nnodes=1 --nproc-per-node=4 dp/main.py
```

**After:**
```bash
!cd ../source_code && torchrun --nnodes=1 --nproc-per-node=4 dp/main.py
```

**Files affected**:
- data-parallelism.ipynb (1 cell)
- model-parallelism.ipynb (1 cell)
- nsys-application.ipynb (2 cells)

---

### Pattern C: Remove srun wrappers from nsys profile
**Before:**
```bash
!cd ../source_code && srun -p primary -N 1 --gres=gpu:4 nsys profile --trace cuda,osrt,nvtx ... torchrun ...
```

**After:**
```bash
!cd ../source_code && nsys profile --trace cuda,osrt,nvtx ... torchrun ...
```

**Files affected**:
- nsys-application.ipynb (2 cells)
- nsight_advanced.ipynb (1 cell)

---

### Pattern D: Update hardcoded IPs to localhost
**Before:**
```python
os.environ['MASTER_ADDR'] = '172.31.26.15'
```

**After:**
```python
os.environ['MASTER_ADDR'] = 'localhost'
```

**Files affected**:
- data-parallelism.ipynb (markdown examples)
- workspace/source_code/ddp/main.py (line 42)

---

### Pattern E: Remove or adapt multi-node content
**Approach 1 - Remove entirely**: data-parallelism.ipynb
- Removed "Lab Activity 1" section (multi-node master/worker setup)
- Removed "Lab Activity 2" section (Slurm sbatch execution)
- Removed interactive srun examples from markdown

**Approach 2 - Convert to reference**: multinode.ipynb
- Added prominent educational note at top
- Kept NCCL ring/tree topology explanations
- Converted executable sections to reference material
- Preserved learning value about multi-node patterns

**Files affected**:
- data-parallelism.ipynb (3 markdown cells removed/simplified)
- multinode.ipynb (restructured as educational reference)

---

## Notebooks Not Requiring Changes

The following notebooks had no Slurm commands or multi-node references:
- **nsys-fp8.ipynb** - Lab 4, FP8 profiling
- **nsys-introduction.ipynb** - Basic Nsight Systems intro
- **nsys-trace.ipynb** - Trace analysis
- **transEng.ipynb** - Lab 4, Transformer Engine
- **advanced_optimizations.ipynb** - Already uses single-GPU examples

## Educational Content Preserved

Per user requirements, the following multi-node educational content was retained:
- NCCL ring construction examples (8 GPUs across 2 nodes)
- NCCL tree construction patterns
- Intra-node vs inter-node communication concepts
- Multi-node profiling analysis techniques in multinode.ipynb

Students will learn these concepts even while executing on single-node hardware.

## Testing Checklist

For each updated notebook, verify:
- [ ] All code cells execute without Slurm errors
- [ ] GPU commands (nvidia-smi) work correctly
- [ ] Training scripts complete successfully
- [ ] Profiling commands generate reports
- [ ] Output files created in expected locations
- [ ] Explanatory text matches new single-node model

## Next Steps

1. Test notebooks on actual L40S instance
2. Adjust batch sizes if memory issues arise (L40S has 48GB vs H100's 80GB)
3. Add jupyterlab-nvdashboard integration
4. Add torch_tb_profiler integration
5. Update any remaining references to node counts in explanatory text

## Known Issues

None identified during migration. All Slurm commands successfully removed or adapted.

## Notes

- All `--nnodes=1` flags kept in torchrun commands (harmless redundancy per user preference)
- Multi-node educational content preserved in multinode.ipynb
- FP8 Lab 4 content unchanged (requires L40S or H100)
