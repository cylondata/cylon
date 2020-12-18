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
