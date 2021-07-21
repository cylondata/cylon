##
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
##

"""
Install: PyCylon (Follow: https://cylondata.org/docs/)
Run Program: mpirun -n 4 python cylon_pytorch_demo_distributed.py
"""
import os
import socket
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pycylon import CylonContext
from pycylon import Table
from pycylon.csv import csv_reader
from torch.nn.parallel import DistributedDataParallel as DDP

hostname = socket.gethostname()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'r-003'
    os.environ['MASTER_PORT'] = '8088'
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # initialize the process group
    dist.init_process_group('nccl', init_method="env://", timeout=timedelta(seconds=30))
    print(f"Init Process Groups : => [{hostname}]Demo DDP Rank {rank}")


def cleanup():
    dist.destroy_process_group()


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(4, 1)
        self.hidden2 = nn.Linear(1, 16)
        # self.hidden3 = nn.Linear(1024, 10)
        # self.hidden4 = nn.Linear(10, 1)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        x = self.output(x)
        return x


def demo_basic(rank, world_size):
    print(f"Simple Batch Train => [{hostname}]Demo DDP Rank {rank}")
    setup(rank=rank, world_size=world_size)

    base_path = "/tmp"

    user_devices_file = os.path.join(base_path, f'user_device_tm_{rank + 1}.csv')
    user_usage_file = os.path.join(base_path, f'user_usage_tm_{rank + 1}.csv')

    user_devices_data: Table = csv_reader.read(ctx, user_devices_file, ',')
    user_usage_data: Table = csv_reader.read(ctx, user_usage_file, ',')

    print(f"User Devices Data Rows:{user_devices_data.rows}, Columns: {user_devices_data.columns}")
    print(f"User Usage Data Rows:{user_usage_data.rows}, Columns: {user_usage_data.columns}")

    print("--------------------------------")
    print("Before Join")
    print("--------------------------------")
    user_devices_data.show_by_range(1, 5, 0, 4)
    print("-------------------------------------")
    user_usage_data.show_by_range(1, 5, 0, 4)

    new_tb: Table = user_devices_data.join(ctx, user_usage_data, 'inner', 'sort', 0, 3)
    print("----------------------")
    print("New Table After Join (5 Records)")
    new_tb.show_by_range(0, 5, 0, 8)
    print("----------------------")

    data_ar: np.ndarray = new_tb.to_numpy()

    data_features: np.ndarray = data_ar[:, 2:6]
    data_learner: np.ndarray = data_ar[:, 6:7]

    x_train, y_train = data_features[0:100], data_learner[0:100]
    x_test, y_test = data_features[100:], data_learner[100:]

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    x_train = torch.from_numpy(x_train).to(rank)
    y_train = torch.from_numpy(y_train).to(rank)
    x_test = torch.from_numpy(x_test).to(rank)
    y_test = torch.from_numpy(y_test).to(rank)

    # create model and move it to GPU with id rank

    model = Network().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    if rank == 0:
        print("Training A Dummy Model")
    for t in range(20):
        for x_batch, y_batch in zip(x_train, y_train):
            print(f"Epoch {t}", end='\r')
            prediction = ddp_model(x_batch)
            loss = loss_fn(prediction, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    cleanup()


if __name__ == '__main__':
    ctx: CylonContext = CylonContext('mpi')
    rank = ctx.get_rank()
    world_size = ctx.get_world_size()
    demo_basic(rank=rank, world_size=world_size)
    ctx.finalize()
