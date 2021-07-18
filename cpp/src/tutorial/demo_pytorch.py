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
Run Program: python demo_pytorch.py
"""

import os

import numpy as np
import pandas as pd
from pycylon import CylonContext
from pycylon import Table
from pycylon.csv import csv_reader

ctx: CylonContext = CylonContext(config='mpi')

base_path = "/tmp"

rank = ctx.get_rank()

user_devices_file = os.path.join(base_path, f'user_device_tm_{rank+1}.csv')
user_usage_file = os.path.join(base_path, f'user_usage_tm_{rank+1}.csv')

user_devices_data: Table = csv_reader.read(ctx, user_devices_file, ',')
user_usage_data: Table = csv_reader.read(ctx, user_usage_file, ',')

user_devices_df: pd.DataFrame = user_devices_data.to_pandas()
user_usage_df: pd.DataFrame = user_usage_data.to_pandas()

print(f"User Devices Data Rows:{user_devices_data.rows}, Columns: {user_devices_data.columns}")
print(f"User Usage Data Rows:{user_usage_data.rows}, Columns: {user_usage_data.columns}")

print("--------------------------------")
print("Before Join")
print("--------------------------------")
user_devices_data.show_by_range(1, 5, 0, 4)
print("-------------------------------------")
user_usage_data.show_by_range(1, 5, 0, 4)

new_tb: Table = user_devices_data.distributed_join(ctx, table=user_usage_data, join_type='inner', algorithm='sort', left_col=0,
                                       right_col=3)

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

import torch

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)


ctx.finalize()
