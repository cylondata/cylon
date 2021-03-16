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


import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from pycylon import Table
from pycylon import CylonContext
import time

ctx: CylonContext = CylonContext(config=None, distributed=False)
num_rows = 10_000_000
data = np.random.randn(num_rows)

df = pd.DataFrame({'data{}'.format(i): data
                   for i in range(100)})

np_key = np.random.randint(0, 100, size=num_rows)
np_all = df.to_numpy()

df['key'] = np_key

rb = pa.record_batch(df)
t = pa.Table.from_pandas(df)

ct = Table.from_pandas(ctx, df)


t1 = time.time()
np_key + 1
t2 = time.time()
ct['key'] + 1
t3 = time.time()
df['key'] + 1
t4 = time.time()
artb = ct.to_arrow().combine_chunks()
ar_key = ct['key'].to_arrow().combine_chunks().columns[0].chunks[0]
pc.add(ar_key, 1)
t5 = time.time()

print(f"Numpy Time: {t2-t1} s")
print(f"PyCylon Time: {t3-t2} s")
print(f"Pandas Time: {t4-t3} s")
print(f"PyArrow Time: {t5-t4} s")