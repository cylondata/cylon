from pyarrow import Table as ATable
from pycylon import Table
from pycylon import CylonContext
import numpy as np
import pandas as pd

ctx: CylonContext = CylonContext(config=None, distributed=False)

print("Initialize From a List")
data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
columns = ['col-1', 'col-2', 'col-3']
tb: Table = Table.from_list(ctx, columns, data)
print(tb)

print("Initialize From a dictionary")
data_dictionary = {'col-1': [1, 2, 3, 4], 'col-2': [5, 6, 7, 8], 'col-3': [9, 10, 11, 12]}
tb: Table = Table.from_pydict(ctx, data_dictionary)
print(tb)

print("Initialize From a PyArrow Table")
data_dictionary = {'col-1': [1, 2, 3, 4], 'col-2': [5, 6, 7, 8], 'col-3': [9, 10, 11, 12]}
atb: ATable = ATable.from_pydict(data_dictionary)
tb: Table = Table.from_arrow(ctx, atb)
print(atb)
print(tb)

print("Initialize From Numpy")
nd_array_list = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])]
columns = ['c1', 'c2', 'c3']
tb: Table = Table.from_numpy(ctx, columns, nd_array_list)
print(tb)

print("Initialize From Pandas")
data_dictionary = {'col-1': [1, 2, 3, 4], 'col-2': [5, 6, 7, 8], 'col-3': [9, 10, 11, 12]}
df = pd.DataFrame(data_dictionary)
tb: Table = Table.from_pandas(ctx, df)
print(df)
print(tb)
