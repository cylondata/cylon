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
>> mpirun -n 4 python -m pytest --with-mpi -q python/pycylon/test/test_equal.py
"""

import pytest

from copy import deepcopy
from pycylon import DataFrame, CylonEnv
import random
from pycylon.net import MPIConfig


def test_ordered_eq(env=None):
    df1 = DataFrame([random.sample(range(10, 100), 50),
                     random.sample(range(10, 100), 50)])
    assert df1.equals(df1, env=env)


def test_ordered_neq(env=None):
    arr1 = [random.sample(range(10, 100), 50),
            random.sample(range(10, 100), 50)]
    arr2 = deepcopy(arr1)
    arr2[0][0] += 1
    df1 = DataFrame(arr1)
    df2 = DataFrame(arr2)
    assert not df1.equals(df2, env=env)

    arr3 = deepcopy(arr1)
    arr3[0] = arr3[0][::-1]
    arr3[1] = arr3[1][::-1]
    df3 = DataFrame(arr3)
    assert not df1.equals(df3, env=env)


def test_unordered_eq(env=None):
    arr1 = [random.sample(range(10, 100), 50),
            random.sample(range(10, 100), 50)]
    arr2 = deepcopy(arr1)
    arr2[0] = arr2[0][::-1]
    arr2[1] = arr2[1][::-1]
    df1 = DataFrame(arr1)
    df2 = DataFrame(arr2)
    assert df1.equals(df2, False, env=env)

    arr3 = deepcopy(arr1)
    arr3[0] = arr3[0][::-1]
    arr3[1] = arr3[1][::-1]
    df3 = DataFrame(arr3)
    assert df1.equals(df3, False, env=env)
    assert df2.equals(df3, False, env=env)


def test_unordered_neq(env=None):
    arr1 = [random.sample(range(10, 100), 50),
            random.sample(range(10, 100), 50)]
    arr2 = deepcopy(arr1)
    arr2[0] = arr2[0][::-1]
    df1 = DataFrame(arr1)
    df2 = DataFrame(arr2)
    assert not df1.equals(df2, False, env=env)

    arr3 = deepcopy(arr1)
    arr3 = arr3[::-1]
    df3 = DataFrame(arr3)
    assert not df1.equals(df3, False, env=env)


@pytest.mark.mpi
def test_distributed():
    env = CylonEnv(config=MPIConfig(), distributed=True)
    test_ordered_eq(env)
    test_ordered_neq(env)
    test_unordered_eq(env)
    test_ordered_neq(env)
