"""Intro lab v1 (buggy): CIFAR-10 + ResNet18 FP32, single GPU.

Bug 1 (PyTorch Profiler will surface): DataLoader uses num_workers=0 and
pin_memory=False, so every batch is decoded and augmented on the main thread
before each step, stalling the GPU.  The profiler step view makes the
DataLoader section impossible to miss; the fix is two lines.

Usage:
    python train_v1.py      # plain run, prints step timings
"""

import argparse
import time

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

# Bug 1 lives here:
NUM_WORKERS = 0
PIN_MEMORY = False


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

    step_times = []
    loader_iter = iter(loader)

    for step in range(NUM_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        x, y = next(loader_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    timed = step_times[WARMUP_ITERS:]
    mean_ms = 1000 * sum(timed) / len(timed)
    print(f"steps timed: {len(timed)}  mean step: {mean_ms:.1f} ms  "
          f"throughput: {BATCH_SIZE / (mean_ms / 1000):.0f} img/s")


if __name__ == "__main__":
    train()
