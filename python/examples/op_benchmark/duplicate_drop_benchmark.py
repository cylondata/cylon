import time
import pandas as pd
import pycylon as cn
import numpy as np

dataset = []
cols = ['a', 'b', 'c', 'd']
records = 50_000_000
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
    pdf2 = pdf.drop_duplicates(subset=['a', 'b'], inplace=False)
    t3 = time.time()

    print(t2-t1, t3-t2, tb2.row_count, len(pdf2))




