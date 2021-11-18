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
>> pytest -q python/pycylon/test/test_equal.py
"""

from copy import deepcopy
from pycylon import DataFrame, read_csv
import random

def test_ordered_eq():
    df1 = DataFrame([random.sample(range(10, 100), 50),
                random.sample(range(10, 100), 50)])
    assert df1.equals(df1)

def test_ordered_neq():
    arr1 = [random.sample(range(10, 100), 50),
            random.sample(range(10, 100), 50)]
    arr2 = deepcopy(arr1)
    arr2[0][0] += 1
    df1 = DataFrame(arr1)
    df2 = DataFrame(arr2)
    assert not df1.equals(df2)

    arr3 = deepcopy(arr1)
    arr3[0] = arr3[0][::-1]
    arr3[1] = arr3[1][::-1]
    df3 = DataFrame(arr3)
    assert not df1.equals(df3)

def test_unordered_eq():
    arr1 = [random.sample(range(10, 100), 50),
            random.sample(range(10, 100), 50)]
    arr2 = deepcopy(arr1)
    arr2[0] = arr2[0][::-1]
    arr2[1] = arr2[1][::-1]
    df1 = DataFrame(arr1)
    df2 = DataFrame(arr2)
    assert df1.equals(df2, False)

    arr3 = deepcopy(arr1)
    arr3[0] = arr3[0][::-1]
    arr3[1] = arr3[1][::-1]
    df3 = DataFrame(arr3)
    assert df1.equals(df3, False)
    assert df2.equals(df3, False)

def test_unordered_neq():
    arr1 = [random.sample(range(10, 100), 50),
        random.sample(range(10, 100), 50)]
    arr2 = deepcopy(arr1)
    arr2[0] = arr2[0][::-1]
    df1 = DataFrame(arr1)
    df2 = DataFrame(arr2)
    assert not df1.equals(df2, False)

    arr3 = deepcopy(arr1)
    arr3 = arr3[::-1]
    df3 = DataFrame(arr3)
    assert not df1.equals(df3, False)
