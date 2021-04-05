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
>> pytest -q python/test/test_sorting.py
"""

from utils import create_df,assert_eq
import pyarrow as pa
import pandas as pd
import numpy as np
import random

import pycylon as cn
from pycylon import CylonContext
from pycylon import DataFrame


def test_df_sorting():
    df_c, df_p = create_df([random.sample(range(10, 300), 50),
                            random.sample(range(10, 300), 50),
                            random.sample(range(10, 300), 50)])

    def do_sort(col, ascending):
        srt_c = df_c.sort_values(by=col, ascending=ascending)
        srt_p = df_p.sort_values(by=col, ascending=ascending)
        assert_eq(srt_c, srt_p)

    # single column
    for asc in [True, False]:
        for c in range(0, 3):
            do_sort(c, asc)

    # multi column
    for asc in [True, False]:
        for c1 in range(0, 3):
            for c2 in range(0, 3):
                if c1 != c2:
                    do_sort([c1, c2], asc)
                for c3 in range(0, 3):
                    if c1!=c2 and c1!=c3 and c2!=c3:
                        do_sort([c1, c2, c3], asc)


def test_sorting():
    ctx: CylonContext = CylonContext()

    ar1 = pa.array([4, 2, 1, 4, 3])
    ar2 = pa.array(['ad', 'ac', 'ac', 'ab', 'a'])
    ar3 = pa.array([4., 2., 1., 4., 3.])

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

    for asc in [True, False]:
        for c in ['col1', 'col2', 'col3']:
            do_sort(c, asc)


def test_multicol():
    # cylon
    ctx: CylonContext = CylonContext()

    c1 = np.random.randint(10, size=100)
    c2 = np.random.randint(10, size=100)

    ar1 = pa.array(c1)
    ar2 = pa.array(c2)

    pa_t: pa.Table = pa.Table.from_arrays(
        [ar1, ar2], names=['col1', 'col2'])

    cn_t = cn.Table.from_arrow(ctx, pa_t)

    cn_srt = cn_t.sort(order_by=['col1', 'col2'], ascending=[True, False])

    # pandas

    df = pd.DataFrame({
        'col1': c1,
        'col2': c2
    }, columns=['col1', 'col2'])

    df = df.sort_values(by=['col1', 'col2'], ascending=[True, False])

    assert cn_srt.to_pandas().values.tolist() == df.values.tolist()
