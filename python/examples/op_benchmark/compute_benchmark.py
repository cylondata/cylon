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
ct.set_index(range(0, num_rows))
##
cmp_num_rows = 10_000
cmp_data = np.random.randn(cmp_num_rows)

cmp_df = pd.DataFrame({'data{}'.format(i): cmp_data
                       for i in range(100)})

cmp_df['key'] = np.random.randint(0, 100, size=cmp_num_rows)
cmp_ct = Table.from_pandas(ctx, cmp_df)
cmp_ct.set_index(range(0, cmp_num_rows))

cmp_list_vals = {'data1': list(range(0, 10_000))}

t1 = time.time()
df.isin(cmp_df)
t2 = time.time()

t3 = time.time()
ct.isin(cmp_ct)
t4 = time.time()

print(t2 - t1, t4 - t3)
