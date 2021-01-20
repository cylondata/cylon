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
import pandas as pd
import pycylon as cn
import numpy as np

dataset = []
cols = ['a', 'b', 'c', 'd']
records = 1_000_000
duplicate_factor = 0.9
gen_record_size = int(records * duplicate_factor)

for col in cols:
    record = np.random.randint(gen_record_size, size=records)
    dataset.append(record)


ctx = cn.CylonContext(config=None, distributed=False)

tb = cn.Table.from_numpy(ctx, cols, dataset)
pdf = tb.to_pandas()
print(tb.shape, pdf.shape)

for _ in range(5):
    t1 = time.time()
    tb2 = tb.unique(columns=['a', 'b'], keep='first')
    t2 = time.time()
    print("cylon", t2-t1, tb2.row_count)
    del tb2
    time.sleep(1)

for _ in range(5):
    t2 = time.time()
    pdf2 = pdf.drop_duplicates(subset=['a', 'b'], inplace=False)
    t3 = time.time()
    print("pandas", t3-t2, len(pdf2))
    del pdf2
    time.sleep(1)




