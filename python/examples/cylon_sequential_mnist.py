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

import os
import getpass
import numpy as np
import torch
from pyarrow import Table as PyArrowTable
from pyarrow import Tensor as ArrowTensor

from pycylon import Table
from pycylon.csv import csv_reader
from pycylon.util.benchutils import benchmark_with_repitions
from pycylon.util.data.DataManager import MiniBatcher

from torch import Tensor as TorchTensor

'''
## Reference

Get the data from [Here](https://www.kaggle.com/oddrationale/mnist-in-csv/)

'''

'''
Configurations
'''

'''
## Place Data

Place the data in the relative path

Assume the data is your 
/home/<your_username>/data/mnist/full/
'''

username = getpass.getuser()

base_path: str = "/home/{}/data/mnist/".format(username)
train_file_name: str = "mnist_train_small.csv"
test_file_name: str = "mnist_test.csv"
train_file_path: str = os.path.join(base_path, train_file_name)
test_file_path: str = os.path.join(base_path, test_file_name)
delimiter: str = ","

'''
Timing Configurations:

'''
reps: int = 10
time_data_loading: int = 0
time_cntb_to_arrowtb: int = 0
time_pyarwtb_to_numpy: int = 0
time_numpy_to_arrowtn: int = 0
time_numpy_to_torchtn: int = 0
time_type: str = "ms"

'''
Check Data Files
'''

print("Train File Path : {}".format(train_file_path))
print("Test File Path : {}".format(test_file_path))

assert os.path.exists(train_file_path) == True
assert os.path.exists(test_file_path) == True

'''
Global Vars
'''

tb_train: Table = None
tb_test: Table = None
tb_train_arw: PyArrowTable = None
tb_test_arw: PyArrowTable = None
train_npy: np.ndarray = None
test_npy: np.ndarray = None
train_arrow_tensor: ArrowTensor = None
test_arrow_tensor: ArrowTensor = None
train_torch_tensor: TorchTensor = None
test_torch_tensor: TorchTensor = None

'''
load To PyCylon Tables
'''


@benchmark_with_repitions(repititions=reps, time_type=time_type)
def load_data_to_cn_tables():
    # TODO : add an API endpoint
    # Table.from_csv('file.csv')
    tb_train: Table = csv_reader.read(train_file_path, delimiter)

    tb_test: Table = csv_reader.read(test_file_path, delimiter)
    return tb_train, tb_test


'''
If some pre-processing to do, do it here...
Join, shuffle, partition, etc
'''


@benchmark_with_repitions(repititions=reps, time_type=time_type)
def convert_cn_table_to_arrow_table():
    # TODO tb_cn: Table => tb_cn.to_arrow(None)
    tb_train_arw: PyArrowTable = Table.to_arrow(tb_train)
    tb_test_arw: PyArrowTable = Table.to_arrow(tb_test)
    return tb_train_arw, tb_test_arw


@benchmark_with_repitions(repititions=reps, time_type=time_type)
def covert_arrow_table_to_numpy():
    # TODO: Can we get a direct npy
    train_npy: np.ndarray = tb_train_arw.to_pandas().to_numpy()
    test_npy: np.ndarray = tb_test_arw.to_pandas().to_numpy()
    return train_npy, test_npy


@benchmark_with_repitions(repititions=reps, time_type=time_type)
def convert_numpy_to_arrow_tensor():
    train_arrow_tensor = ArrowTensor.from_numpy(train_npy)
    test_arrow_tensor = ArrowTensor.from_numpy(test_npy)
    return train_arrow_tensor, test_arrow_tensor


@benchmark_with_repitions(repititions=reps, time_type=time_type)
def convert_numpy_to_torch_tensor():
    train_torch_tensor: TorchTensor = torch.from_numpy(train_npy)
    test_torch_tensor: TorchTensor = torch.from_numpy(test_npy)
    return train_torch_tensor, test_torch_tensor


########################################################################################################################

time_data_loading, (tb_train, tb_test) = load_data_to_cn_tables()
time_cntb_to_arrowtb, (tb_train_arw, tb_test_arw) = convert_cn_table_to_arrow_table()
time_pyarwtb_to_numpy, (train_npy, test_npy) = covert_arrow_table_to_numpy()
time_numpy_to_arrowtn, (train_arrow_tensor, test_arrow_tensor) = convert_numpy_to_arrow_tensor()
time_numpy_to_torchtn, (train_torch_tensor, test_torch_tensor) = convert_numpy_to_torch_tensor()

print("Data Loading Average Time : {} {}".format(time_data_loading, time_type))
print("PyCylon Table to PyArrow Table Average Time : {} {}".format(time_cntb_to_arrowtb, time_type))
print("Pyarrow Table to Numpy Average Time : {} {}".format(time_pyarwtb_to_numpy, time_type))
print("Numpy to Arrow Tensor Average Time : {} {}".format(time_numpy_to_arrowtn, time_type))
print("Numpy to Torch Tensor Average Time : {} {}".format(time_numpy_to_torchtn, time_type))

print("===========================================================================================")

img_size = 28

'''
Splitting Image Data and Target
'''

train_data = train_npy[:, 1:785]
train_target = train_npy[:, 0]
train_target = np.reshape(train_target, (train_target.shape[0], 1))

test_data = test_npy[:, 1:785]
test_target = test_npy[:, 0]
test_target = np.reshape(test_target, (test_target.shape[0], 1))

'''
Generating Minibatches
'''

train_data = MiniBatcher.generate_minibatches(data=train_data, minibatch_size=100)
train_target = MiniBatcher.generate_minibatches(data=train_target, minibatch_size=100)

test_data = MiniBatcher.generate_minibatches(data=test_data, minibatch_size=100)
test_target = MiniBatcher.generate_minibatches(data=test_target, minibatch_size=100)

'''
Data reshaping to match the network config (using original image size)
'''



train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], img_size, img_size))
train_target = np.reshape(train_target, (train_target.shape[0], train_target.shape[1]))

test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], img_size, img_size))
test_target = np.reshape(test_target, (test_target.shape[0], test_target.shape[1]))

'''
Convert Data from Numpy to Torch.Tensor
'''

train_data = torch.from_numpy(train_data)
train_target = torch.from_numpy(train_target)

test_data = torch.from_numpy(test_data)
test_target = torch.from_numpy(test_target)

#########################################################################################################
'''
Sequential Training Algorithm
'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def save_log(file_path=None, stat=""):
    """
    saving the program timing stats
    :rtype: None
    """
    fp = open(file_path, mode="a+")
    fp.write(stat + "\n")
    fp.close()


def launch(fn,
           train_data=None, train_target=None,
           test_data=None, test_target=None,
           do_log=False):
    """ Initialize the distributed environment.
    :param fn: training function
    :param backend: Pytorch Backend
    :param train_data: training data
    :param train_target: training targets
    :param test_data: testing data
    :param test_target: testing targets
    :param do_log: boolean status to log
    """
    # dist.init_process_group(backend, rank=rank, world_size=size)
    # Setting CUDA FOR TRAINING
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")

    device = torch.device("cpu")

    total_communication_time = 0

    local_training_time = time.time()

    model = fn(train_data=train_data, train_target=train_target, do_log=False)

    local_training_time = time.time() - local_training_time

    local_testing_time = time.time()

    predict(model=model, device=device, test_data=test_data, test_target=test_target, do_log=do_log)

    local_testing_time = time.time() - local_testing_time
    print("Total Training Time : {}".format(local_training_time))
    print("Total Testing Time : {}".format(local_testing_time))
    save_log("/tmp/torch_mnist_seq_stats.csv",
             stat="{},{},{},{}".format(1, local_training_time, total_communication_time, local_testing_time))


def predict(model, device, test_data=None, test_target=None, do_log=False):
    """
    testing the trained model
    :rtype: None return
    """
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    count = 0
    with torch.no_grad():
        for data, target in zip(test_data, test_target):
            # total_samples = total_samples + 1
            count = count + 1
            val1 = len(data)
            val2 = len(test_data)
            total_samples = (val1 * val2)
            data, target = data.to(device), target.to(device)
            data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2])) / 128.0
            output = model(data)
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if (do_log):
                print(count, len(data), len(test_data), data.shape, output.shape, correct, total_samples)

    test_loss /= (total_samples)
    local_accuracy = 100.0 * correct / total_samples

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total_samples,
        local_accuracy))


def train(train_data=None, train_target=None, do_log=False):
    """
    training the MNIST model
    :param int world_rank: current processor rank (MPI rank)
    :param int world_size: number of processes (MPI world size)
    :param tensor train_data: training data as pytorch tensor
    :param tensor train_target: training target as pytorch tensor
    :param boolean do_log: set logging
    :return:
    """
    torch.manual_seed(1234)
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = train_data.shape[1]

    if (do_log):
        print("Started Training")
    total_data = len(train_data)
    epochs = 5
    total_steps = epochs * total_data

    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0
        for data, target in zip(train_data, train_target):
            data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2])) / 128.0
            count = count + 1
            result = '{0:.4g}'.format((count / float(total_steps)) * 100.0)
            print("Progress {}% \r".format(result), end='\r')
            optimizer.zero_grad()
            output = model(data)
            # this comes with data loading mechanism use target or target.long()
            # depending on network specifications.
            target = target.long()
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print('Epoch ', epoch, ': ', epoch_loss / num_batches)
    return model

do_log = False

# initialize training
launch(fn=train,
       train_data=train_data,
       train_target=train_target,
       test_data=test_data,
       test_target=test_target,
       do_log=do_log)
