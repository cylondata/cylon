import numpy as np
import pyarrow as pa
import pycylon as cn

columns = 2

nd_array_list = [np.random.random(1000) for i in range(columns)]

ar_array: pa.array = pa.array(nd_array_list)

ar_table: pa.Table = pa.Table.from_arrays(nd_array_list, names=['x0', 'x1'])

print(ar_table)

