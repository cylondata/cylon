import numpy as np
import pyarrow as pa
import pycylon as cn

a = pa.int32()

print(type(a))

from pycylon.data.data_type import DataType
from pycylon.data.data_type import Type
from pycylon.data.data_type import Layout

d = DataType(Type.INT32, Layout.FIXED_WIDTH)

from pycylon.data.column import Column

col = Column('id', d, pa.array([1, 2, 3]))


print("id: ", col.id)
print("dtype: ", col.dtype, col.dtype.type)
print("data: ", col.data)

col1 = Column('id1', cn.int32(), pa.array([1, 2, 3]))

# def int32():
#     return DataType(Type.INT32, Layout.FIXED_WIDTH)

d1 = cn.int32()

print(d1, d1.type)


print("id: ", col1.id)
print("dtype: ", col1.dtype, col1.dtype.type)
print("data: ", col1.data)

