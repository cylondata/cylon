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


def uint8():
    return DataType(Type.UINT8, Layout.FIXED_WIDTH)


def uint16():
    return DataType(Type.UINT16, Layout.FIXED_WIDTH)


def uint32():
    return DataType(Type.UINT32, Layout.FIXED_WIDTH)


def uint64():
    return DataType(Type.UINT64, Layout.FIXED_WIDTH)


def float():
    return DataType(Type.FLOAT, Layout.FIXED_WIDTH)


def double():
    return DataType(Type.DOUBLE, Layout.FIXED_WIDTH)


def half_float():
    return DataType(Type.HALF_FLOAT, Layout.FIXED_WIDTH)


def string():
    return DataType(Type.STRING, Layout.FIXED_WIDTH)


def binary():
    return DataType(Type.BINARY, Layout.FIXED_WIDTH)


def fixed_sized_binary():
    return DataType(Type.FIXED_SIZE_BINARY, Layout.FIXED_WIDTH)


def double():
    return DataType(Type.DOUBLE, Layout.FIXED_WIDTH)


def bool():
    return DataType(Type.BOOL, Layout.FIXED_WIDTH)


def date32():
    return DataType(Type.DATE32, Layout.FIXED_WIDTH)


def date64():
    return DataType(Type.DATE64, Layout.FIXED_WIDTH)


def timestamp():
    return DataType(Type.TIMESTAMP, Layout.FIXED_WIDTH)


def time32():
    return DataType(Type.TIME32, Layout.FIXED_WIDTH)


def time64():
    return DataType(Type.TIME64, Layout.FIXED_WIDTH)


def interval():
    return DataType(Type.INTERVAL, Layout.FIXED_WIDTH)


def decimal():
    return DataType(Type.DECIMAL, Layout.FIXED_WIDTH)


def list():
    return DataType(Type.LIST, Layout.FIXED_WIDTH)


def fixed_sized_list():
    return DataType(Type.FIXED_SIZED_LIST, Layout.FIXED_WIDTH)


def extension():
    return DataType(Type.EXTENSION, Layout.FIXED_WIDTH)


def duration():
    return DataType(Type.DURATION, Layout.FIXED_WIDTH)
