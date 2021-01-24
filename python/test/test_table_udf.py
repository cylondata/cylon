import numpy as np
import pandas as pd
import pyarrow as pa
from pycylon import Table
from pycylon import CylonContext


def test_non_numeric_applymap():
    a = pa.array(['Rayan', 'Reynolds', 'Jack', 'Mat'])
    b = pa.array(['Cameron', 'Selena', 'Roger', 'Murphy'])

    tb: pa.Table = pa.Table.from_arrays([a, b], ['c1', 'c2'])
    pdf: pd.DataFrame = tb.to_pandas()
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cntb: Table = Table.from_arrow(ctx, tb)

    map_func = lambda x: "Hello, " + x
    new_cntb = cntb.applymap(map_func)
    new_pdf = pdf.applymap(map_func)

    assert new_cntb.to_pandas().values.tolist() == new_pdf.values.tolist()


def test_numeric_applymap():
    N = 100
    a_rand = np.random.random(size=N)
    b_rand = np.random.random(size=N)

    a = pa.array(a_rand)
    b = pa.array(b_rand)

    tb = pa.Table.from_arrays([a, b], ['c1', 'c2'])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cntb = Table.from_arrow(ctx, tb)
    pdf: pd.DataFrame = tb.to_pandas()

    map_func = lambda x: x + x
    new_cntb = cntb.applymap(map_func)
    new_pdf = pdf.applymap(map_func)

    assert new_cntb.to_pandas().values.tolist() == new_pdf.values.tolist()
