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

import pyarrow as pa
import numpy as np
import pandas as pd
import pycylon as cn
from pycylon import CylonContext
from pycylon import Table
from pycylon.index import RangeIndex
import time

ctx: CylonContext = CylonContext(config=None, distributed=False)
num_rows = 1_000_000
data = np.random.randn(num_rows)

df = pd.DataFrame({'data{}'.format(i): data
                   for i in range(100)})

df['key'] = np.random.randint(0, 100, size=num_rows)
rb = pa.record_batch(df)
t = pa.Table.from_pandas(df)

ct = Table.from_pandas(ctx, df)

##
cmp_num_rows = 1_000
cmp_data = np.random.randn(cmp_num_rows)

cmp_data = cmp_data.tolist()



t1 = time.time()
df.isin(cmp_data)
t2 = time.time()

t3 = time.time()
ct.isin(cmp_data)
t4 = time.time()

print(f"Pandas isin time : {t2-t1} s")
print(f"PyCylon isin time : {t4-t3} s")