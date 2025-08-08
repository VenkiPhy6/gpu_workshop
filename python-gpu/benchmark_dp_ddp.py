import argparse

import os

import time
 
import torch

import torch.nn as nn

import torch.optim as optim

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.models import resnet18
 
 
def get_data_loader(batch_size=256):

    # Generate dummy data

    inputs = torch.randn(batch_size, 3, 224, 224)

    targets = torch.randint(0, 1000, (batch_size,))

    dataset = torch.utils.data.TensorDataset(inputs, targets)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
 
 
def benchmark(model, device, mode, rank=None):

    loader = get_data_loader()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
 
    # Benchmark 10 iterations

    model.train()

    start = time.time()

    for batch_idx, (data, target) in enumerate(loader):

        if batch_idx >= 10:

            break

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

    torch.cuda.synchronize()

    end = time.time()

    print(f"[{mode.upper()}] Rank {rank if rank is not None else 0} took {end - start:.3f} seconds")
 
 
def run_dp():

    device = torch.device("cuda:0")

    model = resnet18()

    model = nn.DataParallel(model)

    benchmark(model, device, mode="dp")
 
 
def run_ddp(rank, world_size):

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    model = resnet18().to(device)

    model = DDP(model, device_ids=[rank])

    benchmark(model, device, mode="ddp", rank=rank)

    dist.destroy_process_group()
 
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["dp", "ddp"], required=True)

    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())

    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
 
    if args.mode == "dp":

        run_dp()

    else:

        run_ddp(rank=args.rank, world_size=args.world_size)
 
