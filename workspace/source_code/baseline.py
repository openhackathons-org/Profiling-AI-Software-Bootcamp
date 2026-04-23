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


from torch.cuda import nvtx
import os
import time
import torch
import torchvision
import timm

from torchvision import transforms
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group("nccl")

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

BATCH_SIZE = 256 // int(os.environ["WORLD_SIZE"])
EPOCHS = 3
WORKERS = 48
IMG_DIMS = (336, 336)
CLASSES = 10

MODEL_NAME = 'resnet50d'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_DIMS),
])

data = torchvision.datasets.CIFAR10('./',
                                    train=True,
                                    download=True,
                                    transform=transform)

train_kwargs = {'pin_memory': True}

sampler = DistributedSampler(data)
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          sampler=sampler,
                                          num_workers=WORKERS, **train_kwargs)

torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=CLASSES)

model = model.to('cuda:' + str(local_rank))
model = DDP(model, device_ids=[local_rank])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

start = time.perf_counter()
for epoch in range(EPOCHS):
    epoch_start_time = time.perf_counter()
    if epoch == 2:
            torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push("Train")
    model.train()
    nvtx.range_pop() # Train
    batch_idx = 1
    with torch.autograd.profiler.emit_nvtx():
        nvtx.range_push("Data loading");
        for batch in tqdm(data_loader, total=len(data_loader)):
            nvtx.range_pop();# Data loading
            nvtx.range_push("Batch " + str(batch_idx))

            nvtx.range_push("Copy to device")
            features, labels = batch[0].to(local_rank), batch[1].to(local_rank)
            nvtx.range_pop() # Copy to device

            nvtx.range_push("Forward pass")
            optimizer.zero_grad()
            # Enables autocasting for the forward pass
            with torch.cuda.amp.autocast(enabled=True):
                preds = model(features)
                loss = loss_fn(preds, labels)
            nvtx.range_pop() # Forward pass

            nvtx.range_push("Backward pass")
            loss.backward()
            optimizer.step()
            nvtx.range_pop() # Backward pass

            nvtx.range_pop() # Batch
            batch_idx+=1
        epoch_end_time = time.perf_counter()
        if global_rank == 0:
            print(f"Epoch {epoch+1} Time", epoch_end_time - epoch_start_time)
        nvtx.range_push("Data loading");
    if epoch == 2:
            torch.cuda.cudart().cudaProfilerStop()    
end = time.perf_counter()
if global_rank == 0:
    print("Training Took", end - start)