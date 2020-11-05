# cython: infer_types=True
import numpy as np
cimport cython
from libcpp cimport bool
import pyarrow as pa
from pyarrow.compute import (greater, less, less_equal, greater_equal, equal, not_equal, or_,
                             and_)
from pyarrow import compute
from pycylon.data.table cimport CTable
from pycylon.data.table import Table
import numbers



cdef api c_filter(tb: Table, op):
    # TODO: Supported Added via: https://github.com/cylondata/cylon/issues/211
    pass


ctypedef fused CDType:
    int
    double
    float
    long long

cdef cast_scalar(scalar_value, dtype_id):
    if dtype_id == pa.float16().id or dtype_id == pa.float32().id or dtype_id == pa.float64().id:
        return float(scalar_value)
    elif dtype_id == pa.int16().id or dtype_id == pa.int32().id or dtype_id == pa.int64().id:
        return int(scalar_value)
    elif dtype_id == pa.int8().id:
        return bytes(scalar_value)
    else:
        raise ValueError(f"Unsupported Scalar Type {type(scalar_value)}")

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef comparison_compute_op_iter(CDType[:] array, CDType other, op):

    if CDType is int:
        dtype = np.intc
    elif CDType is double:
        dtype = np.double
    elif CDType is cython.longlong:
        dtype = np.longlong

    cdef Py_ssize_t rows = array.shape[0]
    cdef Py_ssize_t index
    result = np.zeros((rows), dtype='bool')
    cdef bool[:] result_view = result
    for index in range(rows):
        result_view[index] = op(array[index], other)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef comparison_compute_np_op(array, other, op):
    return op(array, other)

cdef _resolve_arrow_op(op):
    arrow_op = None
    if op.__name__ == 'gt':
        arrow_op = greater
    elif op.__name__ == 'lt':
        arrow_op = less
    elif op.__name__ == 'eq':
        arrow_op = equal
    elif op.__name__ == 'ge':
        arrow_op = greater_equal
    elif op.__name__ == 'le':
        arrow_op = less_equal
    elif op.__name__ == 'ne':
        arrow_op = not_equal
    elif op.__name__ == 'or_':
        arrow_op = or_
    elif op.__name__ == 'and_':
        arrow_op = and_
    else:
        raise ValueError(f"Not Implemented Operator {op.__name__}")
    return arrow_op


cpdef table_compute_ar_op(table: Table, other, op):
    arrow_op = _resolve_arrow_op(op)
    if isinstance(other, Table):
        arrays = []
        l_table = table.to_arrow().combine_chunks()
        r_table = other.to_arrow().combine_chunks()
        for l_col, r_col in zip(l_table.columns, r_table.columns):
            l_array = l_col.chunks[0]
            r_array = r_col.chunks[0]
            arrays.append(arrow_op(l_array, r_array))
        return Table.from_arrow(table.context, pa.Table.from_arrays(arrays,
                                                                   names=table.column_names))
    elif isinstance(other, numbers.Number):
        arrays = []
        ar_table = table.to_arrow().combine_chunks()
        for col in ar_table.columns:
          ar_array = col.chunks[0]
          r = arrow_op(ar_array, cast_scalar(other, col.type.id))
          arrays.append(r)
        return Table.from_arrow(table.context, pa.Table.from_arrays(arrays,
                                                                   names=table.column_names))
    else:
        raise ValueError(f"Comparison Operator not supported for type {type(other)}. Only Table "
                         f"and numbers are supported!")



cpdef is_null(table:Table):
    ar_tb = table.to_arrow().combine_chunks()
    is_null_values = []
    for chunk_ar in ar_tb.itercolumns():
        is_null_values.append(compute.is_null(chunk_ar))
    return Table.from_arrow(table.context, pa.Table.from_arrays(is_null_values,
                                                                   names=table.column_names))

cpdef invert(table:Table):
    # NOTE: Only Bool invert is supported by PyArrow APIs
    ar_tb = table.to_arrow().combine_chunks()
    invert_values = []
    for chunk_ar in ar_tb.itercolumns():
        invert_values.append(compute.invert(chunk_ar))
    return Table.from_arrow(table.context, pa.Table.from_arrays(invert_values,
                                                                   names=table.column_names))


