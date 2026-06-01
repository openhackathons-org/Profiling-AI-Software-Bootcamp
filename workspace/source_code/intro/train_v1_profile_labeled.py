"""Intro lab v1 (profiled + labeled): train_v1_profile.py with record_function.

Same as train_v1_profile.py, but each section of the training step is wrapped
in torch.profiler.record_function() so the profiler can categorize time spent
in DataLoader, Forward, Backward, and Optimizer.

Open the resulting trace in TensorBoard and look at the Trace view: zoom
into a single step and the DataLoader span occupies most of it.  (The
labels only show up in the Trace view; the Overview and Operator panes
group by built-in categories and aten operators, not user annotations.)

Trace lands in /workspace/logs/train_v1_profile_labeled/.
View with TensorBoard on port 8889.

Usage:
    python train_v1_profile_labeled.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

SEED = 0
BATCH_SIZE = 256
NUM_ITERS = 60
WARMUP_ITERS = 5
DATA_ROOT = "/workspace/data"
LOG_ROOT = "/workspace/logs"

NUM_WORKERS = 0      # Bug 1: single-threaded data loading
PIN_MEMORY = False   # Bug 1: pageable memory


def build_loader():
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )


def build_model(device):
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def train():
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    loader = build_loader()
    model = build_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    logdir = Path(LOG_ROOT) / "train_v1_profile_labeled"
    logdir.mkdir(parents=True, exist_ok=True)

    step_times = []
    loader_iter = iter(loader)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=WARMUP_ITERS, active=10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(logdir)),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for step in range(NUM_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # --- labeled sections for profiler ---
            with torch.profiler.record_function("DataLoader"):
                x, y = next(loader_iter)

            with torch.profiler.record_function("H2D"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            with torch.profiler.record_function("Forward"):
                logits = model(x)
                loss = loss_fn(logits, y)

            with torch.profiler.record_function("Backward"):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

            with torch.profiler.record_function("Optimizer"):
                optimizer.step()
            # --- end labeled sections ---

            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            prof.step()

    timed = step_times[WARMUP_ITERS:]
    mean_ms = 1000 * sum(timed) / len(timed)
    print(f"steps timed: {len(timed)}  mean step: {mean_ms:.1f} ms  "
          f"throughput: {BATCH_SIZE / (mean_ms / 1000):.0f} img/s")
    print(f"Trace written to {logdir}")


if __name__ == "__main__":
    train()
