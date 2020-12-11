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

from pycylon.util.pandas.utils import rename_with_new_column_names

pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

pdf = rename_with_new_column_names(pdf, ['a', 'b', 'c'])

print(pdf)

# with distributed(num_workers=2) as dist:
#     print("This computes in Parallel")
#     with distributed(num_workers=1) as seq:
#         print("This executes sequentially")

from pycylon.frame import DataFrame

d1 = [[1, 2, 3], [4, 5, 6]]
d2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
d3 = pd.DataFrame([[1, 4], [2, 5], [3, 6]])
d4 = {'0': [1, 2, 3], '1': [4, 5, 6]}
d5 = pa.Table.from_pydict(d4)

cdf1 = DataFrame(d1)
cdf2 = DataFrame(d2)
cdf3 = DataFrame(d3)
cdf4 = DataFrame(d4)
cdf5 = DataFrame(d5)
