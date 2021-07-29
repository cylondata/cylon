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

'''
Run test:
>> pytest -q python/test/test_data_types.py
'''

from pycylon.data.data_type import Type
from pycylon.data.data_type import Layout

import pyarrow as pa
import pandas as pd
import numpy as np

import pycylon as cn
from pycylon import CylonContext

import datetime


def test_data_types_1():

    # Here just check some types randomly
    assert Type.BOOL.value == 0
    assert Layout.FIXED_WIDTH.value == 1

    assert Type.INT32 == 6
    assert Layout.FIXED_WIDTH == 1


def test_temporal_types():
    ctx: CylonContext = CylonContext()

    ar1 = pa.array([4, 2, 1])
    ar2 = pa.array([datetime.datetime(2020, 5, 7), datetime.datetime(
        2020, 3, 17), datetime.datetime(2020, 1, 17)])
    ar3 = pa.array([4., 2., 1.])

    pa_t: pa.Table = pa.Table.from_arrays(
        [ar1, ar2, ar3], names=['col1', 'col2', 'col3'])

    cn_t = cn.Table.from_arrow(ctx, pa_t)

    def do_sort(col, ascending):
        srt = cn_t.sort(col, ascending)
        arr = srt.to_pydict()[col]
        print(srt)
        for i in range(len(arr) - 1):
            if ascending:
                assert arr[i] <= arr[i + 1]
            else:
                assert arr[i] >= arr[i + 1]

    do_sort('col2', True)
            
