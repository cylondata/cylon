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


import time
import numpy as np
import pandas as pd
import pyarrow as pa
from pycylon import Table
from pycylon import CylonContext


def benchmark_conversions():
    N = 50_000_000

    t1 = time.time()
    r = np.random.random(size=N)
    t2 = time.time()

    a = pa.array(r)

    t3 = time.time()
    npy = a.to_numpy()
    t4 = time.time()

    print(f"Arrow to Numpy Conversion Time: {t4 - t3} s")


def benchmark_map_numeric():
    N = 100_000_000
    a_rand = np.random.random(size=N)
    b_rand = np.random.random(size=N)

    a = pa.array(a_rand)
    b = pa.array(b_rand)

    tb = pa.Table.from_arrays([a, b], ['c1', 'c2'])
    pdf: pd.DataFrame = tb.to_pandas()

    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cntb: Table = Table.from_arrow(ctx, tb)

    map_func = lambda x: x + x

    t1 = time.time()
    new_ct = cntb.applymap(map_func)
    t2 = time.time()

    t3 = time.time()
    new_pdf = pdf.applymap(map_func)
    t4 = time.time()

    print(f"Time for Cylon Apply Map {t2 - t1} s")
    print(f"Time for Cylon Apply Map {t4 - t3} s")


benchmark_conversions()
benchmark_map_numeric()
