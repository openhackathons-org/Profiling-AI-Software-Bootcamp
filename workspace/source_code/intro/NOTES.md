# Intro-Notebook Scripts: Design Notes

## Final script inventory

| File | Task | Bug | Status |
|------|------|-----|--------|
| `train_v1.py` | CIFAR-10 + ResNet18 FP32 | Bug 1: DataLoader stall (`num_workers=0`, `pin_memory=False`) | Verified |
| `train_v1_fixed.py` | Same | Bug 1 fixed (`num_workers=4`, `pin_memory=True`) | Verified |
| `train_v2.py` | CIFAR-10 + ResNet18 AMP | Bug 2: `set_detect_anomaly(True)` left on | Verified |
| `train_v2_fixed.py` | Same | Bug 2 fixed (flag removed) | Verified |

## Measured performance on this instance (1× L4, batch=256)

| Script | Mean step | Throughput | Note |
|--------|-----------|------------|------|
| train_v1 | ~261 ms | ~981 img/s | DataLoader stall dominates |
| train_v1_fixed | ~94 ms | ~2730 img/s | 2.8× win from fixing DataLoader |
| train_v2 | ~65 ms | ~3960 img/s | AMP + bug: only 1.45× over v1_fixed |
| train_v2_fixed | ~47 ms | ~5450 img/s | 2× over v1_fixed — AMP working |

## Pedagogical arc

1. **Notebook 1 (PyTorch Profiler)** — v1 → v1_fixed.
   Profile v1: the step view shows `DataLoader` dominating every step.  Fix
   is two lines (`num_workers`, `pin_memory`).  Profile v1_fixed to confirm.
   Message: *"The profiler tells you which section is slow; you fix that section."*

2. **Notebook 2 (Nsight Systems)** — v2 → v2_fixed.
   Profile v2 with the PyTorch Profiler: step time improved vs v1_fixed but
   AMP delivers only 1.45× instead of the expected ~2×.  The profiler shows
   `backward` is much longer than a clean AMP run; it doesn't say why.
   Run nsys on v2: inside the `backward` NVTX range the GPU timeline is a
   sawtooth — each layer's gradient kernel fires then the GPU idles while
   the CPU checks that gradient for NaN, then the next layer starts.
   ~60 syncs per step, one per backward op.  Fix: remove `set_detect_anomaly`.
   Profile v2_fixed with nsys to confirm the sawtooth is gone.
   Message: *"The profiler shows you WHERE time goes; nsys shows you WHY —
   especially concurrency and synchronisation problems invisible in a
   step-level view."*

## Why set_detect_anomaly instead of non_blocking

The original Bug 2 was `.to(device)` without `non_blocking=True`.  Empirically
tested on the bootcamp instance (1× L4): no measurable wallclock difference at
any of 32×32, 128×128, 224×224 with batch 256.  ResNet18 keeps the GPU busy
enough that CPU prep fits inside the GPU compute window.

`set_detect_anomaly(True)` was chosen as a replacement after an empirical
search (`v2_bakeoff.py`).  Key findings:

- Anomaly detection overhead is ~22 ms/step (batch-size independent — it
  scales with backward op count, not data volume).
- At batch=256 AMP, this yields +38% overhead (65 ms buggy vs 47 ms clean).
- At smaller batches the relative overhead is larger (100–183% at batch 64–128),
  because the same absolute overhead lands on a shorter step.
- Per-step `.item()` calls hit a ceiling of ~10% overhead on L4 even with
  multiple calls — subsequent syncs are free after the first one.

## Realistic bug scenario

Developer was debugging a NaN in gradients.  They added
`torch.autograd.set_detect_anomaly(True)` at module level, found and fixed the
NaN, then forgot to remove the flag.  Training kept working; it just became 38%
slower.  With AMP in the picture the developer might attribute the
underperformance to "AMP not helping on this model" rather than suspecting a
leftover debug flag.
