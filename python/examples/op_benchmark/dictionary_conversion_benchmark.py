from pycylon import Table
from pycylon import CylonContext
import pandas as pd
import numpy as np
import time

ctx = CylonContext(config=None, distributed=False)

records = 1_000_000
values = []
for col in range(0, 1):
    values.append(np.random.random(records))
pdf_t = pd.DataFrame(values)


tb_t = Table.from_pandas(ctx, pdf_t)

t1 = time.time()
pdf_t.to_dict()
t2 = time.time()
tb_t.to_pydict(with_index=True)
t3 = time.time()

print(f"Pandas Dictionary Conversion Time {t2-t1} s")
print(f"PyCylon Dictionary Conversion Time {t3-t2} s")