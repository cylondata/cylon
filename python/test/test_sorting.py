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

import pyarrow as pa

import pycylon as cn
from pycylon import CylonContext


def test_sorting():
    ctx: CylonContext = CylonContext()

    ar1 = pa.array([4, 2, 1, 4, 3])
    ar2 = pa.array(['ad', 'ac', 'ac', 'ab', 'a'])
    ar3 = pa.array([4., 2., 1., 4., 3.])

    pa_t: pa.Table = pa.Table.from_arrays([ar1, ar2, ar3], names=['col1', 'col2', 'col3'])

    cn_t = cn.Table.from_arrow(ctx, pa_t)

    def do_sort(col, ascending):
        srt = cn_t.sort(col, ascending)
        arr = srt.to_pydict()[col]
        print(srt.to_pydict())
        for i in range(len(arr) - 1):
            if ascending:
                assert arr[i] <= arr[i + 1]
            else:
                assert arr[i] >= arr[i + 1]

    for asc in [True, False]:
        for c in ['col1', 'col2', 'col3']:
            do_sort(c, asc)


