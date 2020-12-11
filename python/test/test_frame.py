import numpy as np
import pandas as pd
import pycylon as cn
import pyarrow as pa
from pycylon import Series

a = np.array([1, 2, 3, 4, 5, 6])
b = [1, 2, 3, 4, 5]

b = a.copy()

a = a * 2

print(b)
print(a)

s = Series('id1', [[1, 2, 3, 4], [3, 4, 5, 6]], cn.int32())

print(s)

print(s.shape)

ps = pd.Series([[1, 2, 3, 4]])

print(ps.shape)

a = pa.array([1, 2, 3])

from copy import copy

b = copy(a)

from pyarrow.compute import add

a = add(a, 1)

print("#############")
print(a)
print("------------")
print(b)

