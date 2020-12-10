from pycylon.data.data_type import DataType
from pycylon.data.data_type import Type
from pycylon.data.data_type import Layout


def int8():
    return DataType(Type.INT8, Layout.FIXED_WIDTH)


def int16():
    return DataType(Type.INT16, Layout.FIXED_WIDTH)


def int32():
    return DataType(Type.INT32, Layout.FIXED_WIDTH)


def int64():
    return DataType(Type.INT64, Layout.FIXED_WIDTH)
