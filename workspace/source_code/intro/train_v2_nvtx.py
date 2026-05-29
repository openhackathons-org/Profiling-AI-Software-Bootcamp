"""Intro lab v2 (NVTX-annotated): train_v2.py with manual NVTX ranges added.

NVTX ranges label the major sections of each training step so that the Nsight
Systems timeline is navigable.  Inside the "backward" range, the sawtooth
pattern of GPU idle gaps (one per backward op) is clearly visible — each gap
is the CPU receiving and checking a gradient tensor for NaN/Inf.

Run under nsys:
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --output /workspace/reports/train_v2_nvtx \
        python train_v2_nvtx.py

Usage:
    python train_v2_nvtx.py     # plain run (NVTX ranges recorded but ignored)
"""

import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.cuda import nvtx
from torch.utils.data import DataLoader

SEED = 0
BATCH_SIZE = 256
NUM_ITERS = 60
WARMUP_ITERS = 5
DATA_ROOT = "/workspace/data"

NUM_WORKERS = 4
PIN_MEMORY = True


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
        persistent_workers=True,
    )


def build_model(device):
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def train():
    torch.manual_seed(SEED)
    torch.autograd.set_detect_anomaly(True)  # left from debugging a NaN issue
    device = torch.device("cuda")
    loader = build_loader()
    model = build_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    step_times = []
    loader_iter = iter(loader)

    for step in range(NUM_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        nvtx.range_push("step")

        nvtx.range_push("data_load")
        x, y = next(loader_iter)
        nvtx.range_pop()

        nvtx.range_push("h2d")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        nvtx.range_pop()

        nvtx.range_push("forward")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = loss_fn(logits, y)
        nvtx.range_pop()

        nvtx.range_push("backward")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nvtx.range_pop()

        nvtx.range_push("optimizer")
        scaler.step(optimizer)
        scaler.update()
        nvtx.range_pop()

        nvtx.range_pop()  # step

        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    timed = step_times[WARMUP_ITERS:]
    mean_ms = 1000 * sum(timed) / len(timed)
    print(f"steps timed: {len(timed)}  mean step: {mean_ms:.1f} ms  "
          f"throughput: {BATCH_SIZE / (mean_ms / 1000):.0f} img/s")


if __name__ == "__main__":
    train()
