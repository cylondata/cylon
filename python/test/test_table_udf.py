import time
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import List
from pyarrow.compute import add


class CylonTable(object):
    def __init__(self, tb):
        self.table = tb

    def applymap(self, func):
        new_chunks = []
        for chunk_array in self.table.itercolumns():
            npr = chunk_array.to_numpy()
            new_ca = list(map(func, npr))
            new_chunks.append(pa.array(new_ca))
        return pa.Table.from_arrays(new_chunks, self.table.column_names)


def benchmark_conversions():
    N = 50_000_000

    t1 = time.time()
    r = np.random.random(size=N)
    t2 = time.time()

    a = pa.array(r)

    t3 = time.time()
    npy = a.to_numpy()
    t4 = time.time()

    print(t2 - t1, t4 - t3)


def benchmark_map_numeric():
    N = 10
    a_rand = np.random.random(size=N)
    b_rand = np.random.random(size=N)

    a = pa.array(a_rand)
    b = pa.array(b_rand)

    tb = pa.Table.from_arrays([a, b], ['c1', 'c2'])
    pdf: pd.DataFrame = tb.to_pandas()

    ct = CylonTable(tb)

    map_func = lambda x: x + x
    map_func = lambda x: len(str(x))

    t1 = time.time()
    new_ct = ct.applymap(map_func)
    t2 = time.time()

    t3 = time.time()
    new_pdf = pdf.applymap(map_func)
    t4 = time.time()

    print(t2 - t1, t4 - t3)

    print(new_ct.to_pandas())

    print(new_pdf)


def benchmark_non_numeric():
    a = pa.array(['Rayan', 'Reynolds', 'Jack', 'Mat'])
    b = pa.array(['Cameron', 'Selena', 'Roger', 'Murphy'])

    tb: pa.Table = pa.Table.from_arrays([a, b], ['c1', 'c2'])
    ct = CylonTable(tb)

    map_func = lambda x: "Hello, " + x
    new_ct = ct.applymap(map_func)

    print(new_ct.to_pandas())


benchmark_non_numeric()
