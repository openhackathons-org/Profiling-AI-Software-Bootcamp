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
from torch.cuda import nvtx
#import nvtx as nv
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_warn_always(False)
import signal
import time

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, signum, frame):
    self.kill_now = True

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def get_resources():

    if os.environ.get('OMPI_COMMAND'):
        # from mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    else:
        # from slurm
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])

    return rank, local_rank, world_size


def main():

    num_epochs_default = 3  #50 #10000
    batch_size_default = 256 # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "source_code/saved_models"
    model_filename_default = "resnet_distributed.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
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

    rank, local_rank, world_size = get_resources()
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    
    #if local_rank is None:
        #local_rank = int(os.environ["LOCAL_RANK"])
        #print('Local rank ', local_rank)

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    #torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet50(pretrained=False)

    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="data/", train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="data/", train=False, download=False, transform=transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    fp16_scaler = torch.amp.GradScaler("cuda")

    #torch.cuda.cudart().cudaProfilerStart()
    try:
        # Loop over the dataset multiple times
        for epoch in range(num_epochs):
    
            print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
    
            #Save and evaluate model routinely
            if epoch % 2 == 0:
                torch.cuda.cudart().cudaProfilerStart()
                if local_rank == 0:
                    accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                    torch.save(ddp_model.state_dict(), model_filepath)
                    print("-" * 75)
                    print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                    print("-" * 75)
                    
            nvtx.range_push("Train")
            ddp_model.train()
            nvtx.range_pop() # Train
    
            nvtx.range_push("Data loading");
            for data in train_loader:
                nvtx.range_pop();# Data loading
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    nvtx.range_push("Copy to device")
                    inputs, labels = data[0].to(device), data[1].to(device)
                    nvtx.range_pop() # Train
        
                    nvtx.range_push("Forward pass")
                    optimizer.zero_grad()
                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, labels)
                    nvtx.range_pop() # Train
    
                nvtx.range_push("Backward pass")
                fp16_scaler.scale(loss).backward()    
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                nvtx.range_pop()
    
                nvtx.range_push("Data loading"); # pushing the next data in data loader 
            if epoch == 2:
                torch.cuda.cudart().cudaProfilerStop() 
    except:
        print("exception caught at inner ")           
    #if num_epochs ==50:
    #torch.distributed.destroy_process_group() 
    
    #torch.cuda.cudart().cudaProfilerStop() 

if __name__ == "__main__":
    killer = GracefulKiller()
    while not killer.kill_now:
        time.sleep(1)
        main()
