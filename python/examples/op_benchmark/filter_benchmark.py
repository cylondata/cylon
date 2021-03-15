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
                   for i in range(2)})

np_key = np.random.randint(0, 100, size=num_rows)
np_all = df.to_numpy()

df['key'] = np_key

rb = pa.record_batch(df)
t = pa.Table.from_pandas(df)

ct = Table.from_pandas(ctx, df)

print(ct.shape, df.shape)
pdf_time = []
ct_time = []
rep = 1

t1 = time.time()
ct_filter = ct['key'] > 5
t2 = time.time()
df_filter = df['key'] > 5
t3 = time.time()
ct_res = ct[ct_filter]
t4 = time.time()
df_res = df[df_filter]
t5 = time.time()
np_filter = np_key > 5
t6 = time.time()
np_res = np_all[np_filter]
t7 = time.time()


print(f"PyCylon filter time :  {t2-t1} s")
print(f"Pandas filter time: {t3-t2} s")
print(f"Numpy filter time: {t6-t5} s")
print(f"PyCylon assign time: {t4 -t3} s")
print(f"Pandas assign time: {t5 -t4} s")
print(f"Numpy assign time: {t7 -t6} s")

artb = t

artb_filter = ct_filter.to_arrow().combine_chunks()
artb_array_filter = artb_filter.columns[0].chunks[0]
t_ar_s = time.time()
artb = artb.combine_chunks()
from pyarrow import compute
res = []
for chunk_arr in artb.itercolumns():
    res.append(chunk_arr.filter(artb_array_filter))
t_ar_e = time.time()
res_t = pa.Table.from_arrays(res, ct.column_names)
t_ar_e_2 = time.time()
print(f"PyArrow Filter Time : {t_ar_e - t_ar_s}")
print(f"PyArrow Table Creation : {t_ar_e_2 - t_ar_e}")

print(artb.shape, df_res.shape)









