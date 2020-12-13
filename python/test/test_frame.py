import numpy as np
import pandas as pd
import pycylon as cn
import pyarrow as pa
from pycylon import Series
from pycylon import DataFrame


def test_initialization_1():
    d1 = [[1, 2, 3], [4, 5, 6]]
    d2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    d3 = {'0': [1, 2, 3], '1': [4, 5, 6]}
    d4 = pd.DataFrame(d3)
    d5 = pa.Table.from_pydict(d3)

    cdf1 = DataFrame(d1)
    cdf2 = DataFrame(d2)
    cdf3 = DataFrame(d3)
    cdf4 = DataFrame(d4)
    cdf5 = DataFrame(d5)

    assert cdf1.shape == cdf2.shape == cdf3.shape == cdf4.shape == cdf5.shape


def test_get_set_item():
    d1 = [[1, 2, 3], [4, 5, 6]]
    cdf1 = DataFrame(d1)
    print(cdf1)

    print(cdf1.columns)

    c1 = cdf1['0']
    print(c1.shape)
    d1 = DataFrame([[10, 20, 30]])

    print(d1.shape)
    print(cdf1)
    cdf1['0'] = d1

    print(cdf1)




