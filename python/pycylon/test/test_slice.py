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
Run test:
>> mpirun -n 4 python -m pytest --with-mpi -q python/pycylon/test/test_slice.py
"""

import pytest

from copy import deepcopy
from pycylon import DataFrame, CylonEnv
import random
from pycylon.net import MPIConfig
from python.pycylon.test.utils import assert_eq

def random_local_dataset() -> DataFrame:
    df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])
    df1.set_index([0], inplace=True)
    return df1

def random_distributed_dataset() -> DataFrame:
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)

    df1 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                 random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
    df1.set_index([0], inplace=True)
    return df1, env

def test_slice():
    df1 = random_local_dataset()
    assert df1[1:5]

def test_head():
    df1 = random_local_dataset()
    assert df1[:5]

def test_tail():
    df1 = random_local_dataset()
    assert df1[:-5]

def test_distributed_slice():
    df1, env = random_distributed_dataset()
    df3 = df1[0:9, env]
    assert(df3.notnull)

def test_distributed_head():
    df1, env = random_distributed_dataset()

    df3 = df1[:9, env]
    assert(df3.notnull)

def test_distributed_tail():
    df1, env = random_distributed_dataset()

    df3 = df1[:-9, env]
    assert(df3.notnull)



@pytest.mark.mpi
def test_utils():
    test_slice()
    test_head()
    test_tail()
    test_distributed_slice()
    test_distributed_head()
    test_distributed_tail()

