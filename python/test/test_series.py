import numpy as np
import pyarrow as pa
import pycylon as cn
from pycylon.data.data_type import DataType
from pycylon.data.data_type import Type
from pycylon.data.data_type import Layout
from pycylon.data.column import Column
from pycylon.data.series import Series

d = DataType(Type.INT32, Layout.FIXED_WIDTH)
data = pa.array([1, 2, 3])
col = Column('id', d, data)

print("id: ", col.id)
print("dtype: ", col.dtype, col.dtype.type)
print("data: ", col.data)

col1 = Column('id1', cn.int32(), pa.array([1, 2, 3]))

print("id: ", col1.id)
print("dtype: ", col1.dtype, col1.dtype.type)
print("data: ", col1.data)

s = Series('s1', [1, 2, 3, 4], cn.int32())

print(s[2])
