import numpy as np
import pyarrow as pa
import pycylon as cn
from pycylon.data.data_type import DataType
from pycylon.data.data_type import Type
from pycylon.data.data_type import Layout
from pycylon.data.column import Column
from pycylon.data.series import Series


def test_column():
    col_id = 'id'
    dtype = DataType(Type.INT32, Layout.FIXED_WIDTH)
    data = pa.array([1, 2, 3])
    col = Column(col_id, dtype, data)

    assert col_id == col.id
    assert dtype.type == col.dtype.type
    assert data == col.data


def test_series_with_list():
    ld = [1, 2, 3, 4]
    id = 's1'
    dtype = cn.int32()
    s = Series(id, ld, dtype)

    assert s.id == id
    assert s.data == pa.array(ld)
    assert s.dtype.type == dtype.type


def test_series_with_numpy():
    ld = np.array([1, 2, 3, 4])
    id = 's1'
    dtype = cn.int32()
    s = Series(id, ld, dtype)

    assert s.id == id
    assert s.data == pa.array(ld)
    assert s.dtype.type == dtype.type


def test_series_with_pyarrow():
    ld = pa.array([1, 2, 3, 4])
    id = 's1'
    dtype = cn.int32()
    s = Series(id, ld, dtype)

    assert s.id == id
    assert s.data == ld
    assert s.dtype.type == dtype.type
