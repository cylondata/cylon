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

import numpy as np
import pandas as pd
import pyarrow as pa
from pycylon import Table
from pycylon import CylonContext


def test_non_numeric_applymap():
    a = pa.array(['Rayan', 'Reynolds', 'Jack', 'Mat'])
    b = pa.array(['Cameron', 'Selena', 'Roger', 'Murphy'])

    tb: pa.Table = pa.Table.from_arrays([a, b], ['c1', 'c2'])
    pdf: pd.DataFrame = tb.to_pandas()
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cntb: Table = Table.from_arrow(ctx, tb)

    map_func = lambda x: "Hello, " + x
    new_cntb = cntb.applymap(map_func)
    new_pdf = pdf.applymap(map_func)

    assert new_cntb.to_pandas().values.tolist() == new_pdf.values.tolist()


def test_numeric_applymap():
    N = 100
    a_rand = np.random.random(size=N)
    b_rand = np.random.random(size=N)

    a = pa.array(a_rand)
    b = pa.array(b_rand)

    tb = pa.Table.from_arrays([a, b], ['c1', 'c2'])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cntb = Table.from_arrow(ctx, tb)
    pdf: pd.DataFrame = tb.to_pandas()

    map_func = lambda x: x + x
    new_cntb = cntb.applymap(map_func)
    new_pdf = pdf.applymap(map_func)

    assert new_cntb.to_pandas().values.tolist() == new_pdf.values.tolist()
