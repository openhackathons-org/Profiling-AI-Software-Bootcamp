# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import argparse
import os
import random
import numpy as np

torch.set_warn_always(False)
import signal
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class CachedDataset(Dataset):
    """Loads an entire dataset onto the target device once at construction time.

    All samples are moved to `device` upfront so that __getitem__ returns
    GPU tensors directly — eliminating per-batch .to(device) transfers in the
    training loop entirely.

    Constraints that come with this approach:
      - num_workers must be 0: CUDA tensors cannot be shared across processes.
      - pin_memory, prefetch_factor, persistent_workers must all be disabled.
      - Stochastic transforms (RandomCrop, RandomHorizontalFlip) are NOT applied
        here; they must be run per-batch on the GPU after loading (see gpu_augment).
    """
    def __init__(self, dataset, device):
        # Stack all samples into two contiguous GPU tensors in one transfer.
        self.images = torch.stack([img for img, _ in dataset]).to(device)
        self.labels = torch.tensor([lbl for _, lbl in dataset], dtype=torch.long).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Both tensors already live on the GPU — no transfer needed.
        return self.images[idx], self.labels[idx]


def gpu_augment(images):
    """Applies stochastic augmentations on a batch of GPU tensors.

    Equivalent to the CPU-side RandomCrop(32, padding=4) + RandomHorizontalFlip
    that would normally run inside the DataLoader workers, but executed on the
    GPU after the batch is loaded from the CachedDataset.
    """
    # Pad then take a random 32×32 crop — mirrors RandomCrop(32, padding=4)
    images = TF.pad(images, padding=4)
    i, j, h, w = transforms.RandomCrop.get_params(images, output_size=(32, 32))
    images = TF.crop(images, i, j, h, w)
    # Random horizontal flip with p=0.5
    if torch.rand(1).item() < 0.5:
        images = TF.hflip(images)
    return images


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # When using CachedDataset, images/labels are already on device.
            # The .to(device) calls below are no-ops in that case, but kept
            # for compatibility if a plain CPU DataLoader is passed instead.
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main():
    num_epochs_default = 25
    batch_size_default = 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "./saved_models"
    model_filename_default = "resnet_distributed.pth"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    if local_rank is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        print("Local rank", local_rank)

    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=random_seed)

    torch.distributed.init_process_group(backend="nccl")

    model = torchvision.models.resnet18(pretrained=False)
    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if resume:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Deterministic transforms only — applied once at cache time.
    # Stochastic transforms (RandomCrop, RandomHorizontalFlip) are handled
    # per-batch on the GPU by gpu_augment() to preserve augmentation diversity.
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    raw_train = torchvision.datasets.CIFAR10(root="../data", train=True, download=False, transform=base_transform)
    raw_test  = torchvision.datasets.CIFAR10(root="../data", train=False, download=False, transform=base_transform)

    # Move the entire dataset to GPU once. After this, __getitem__ returns GPU
    # tensors directly and no per-batch .to(device) transfer is needed.
    if local_rank == 0:
        print("Caching datasets on GPU...")
    train_set = CachedDataset(raw_train, device)
    test_set  = CachedDataset(raw_test, device)

    train_sampler = DistributedSampler(dataset=train_set)

    # num_workers=0: CUDA tensors cannot be shared with worker subprocesses.
    # pin_memory, prefetch_factor, persistent_workers are disabled for the same reason.
    # drop_last=True: keeps all DDP ranks at equal batch counts.
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    fp16_scaler = torch.amp.GradScaler("cuda")

    for epoch in range(num_epochs):
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

        # FIX: set_epoch every epoch — required for DistributedSampler to re-shuffle correctly
        train_sampler.set_epoch(epoch)

        if epoch % 5 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                torch.save(ddp_model.state_dict(), model_filepath)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        ddp_model.train()

        for inputs, labels in train_loader:
            # inputs and labels are already on-device — CachedDataset.__getitem__
            # returns GPU tensors directly, so no .to(device) call is needed here.

            # Apply stochastic augmentations on the GPU batch
            # (RandomCrop + RandomHorizontalFlip, equivalent to the original CPU transforms)
            inputs = gpu_augment(inputs)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)

            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

    if num_epochs == 25:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total elapsed time: {end - start:.2f} seconds")
     
