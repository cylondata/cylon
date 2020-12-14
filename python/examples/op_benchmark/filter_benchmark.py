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

df['key'] = np.random.randint(0, 100, size=num_rows)

rb = pa.record_batch(df)
t = pa.Table.from_pandas(df)

ct = Table.from_pandas(ctx, df)

print(ct.shape, ct.column_names)
pdf_time = []
ct_time = []
rep = 1
for i in range(rep):
    t1 = time.time()
    ct[ct['key'] == 5]
    t2 = time.time()
    df[df.key == 5]
    t3 = time.time()
    ct_time.append(t2-t1)
    pdf_time.append(t3-t2)

print(f"PDF : Mean {sum(pdf_time)/rep}")
print(f"CT : Mean {sum(ct_time)/rep}")




