# cython: infer_types=True
import numpy as np
cimport cython
from libcpp cimport bool
import pyarrow as pa
from pyarrow.compute import (greater, less, less_equal, greater_equal, equal, not_equal, or_,
                             and_)
from pyarrow.compute import add as a_add, subtract as a_subtract, multiply as a_multiply, \
    divide as a_divide
from pyarrow import compute as a_compute
from pycylon.data.table cimport CTable
from pycylon.data.table import Table
from pycylon.api.types import FilterType
import numbers
from operator import neg as py_neg

from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompareBool,
)
from typing import Any, List


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

cdef _is_division(op):
    return op.__name__ == 'divide'


cpdef is_null(table:Table):
    ar_tb = table.to_arrow().combine_chunks()
    is_null_values = []
    for chunk_ar in ar_tb.itercolumns():
        is_null_values.append(a_compute.is_null(chunk_ar))
    return Table.from_arrow(table.context, pa.Table.from_arrays(is_null_values,
                                                                   names=table.column_names))

cpdef invert(table:Table):
    # NOTE: Only Bool invert is supported by PyArrow APIs
    ar_tb = table.to_arrow().combine_chunks()
    invert_values = []
    for chunk_ar in ar_tb.itercolumns():
        invert_values.append(a_compute.invert(chunk_ar))
    return Table.from_arrow(table.context, pa.Table.from_arrays(invert_values,
                                                                   names=table.column_names))

cpdef neg(table:Table):
    # NOTE: PyArrow API doesn't provide a neg operator.
    ar_tb = table.to_arrow().combine_chunks()
    neg_array = []
    for chunk_arr in ar_tb.itercolumns():
        updated_col_array = []
        for val in chunk_arr:
            updated_col_array.append(py_neg(val.as_py()))
        neg_array.append(updated_col_array)
    return Table.from_arrow(table.context, pa.Table.from_arrays(neg_array,
                                                                   names=table.column_names))

cpdef division_op(table:Table, op, value):
    ar_tb = table.to_arrow().combine_chunks()
    res_array = []
    if not isinstance(value, numbers.Number):
        raise ValueError("Math operation value must be numerical")
    cast_type = pa.float64()
    for chunk_arr in ar_tb.itercolumns():
        chunk_arr = chunk_arr.cast(cast_type)
        value = cast_scalar(value, cast_type.id)
        res_array.append(op(chunk_arr, value))
    return Table.from_arrow(table.context, pa.Table.from_arrays(res_array,
                                                                   names=table.column_names))


cpdef math_op(table:Table, op, value):
    ar_tb = table.to_arrow().combine_chunks()
    res_array = []
    if not isinstance(value, numbers.Number) or isinstance(value, Table):
        raise ValueError("Math operation value must be numerical or a Numeric Table")
    for chunk_arr in ar_tb.itercolumns():
        if isinstance(value, numbers.Number):
            value = cast_scalar(value, chunk_arr.type.id)
        res_array.append(op(chunk_arr, value))
    return Table.from_arrow(table.context, pa.Table.from_arrays(res_array,
                                                                   names=table.column_names))

cpdef add(table:Table, value):
    return math_op(table, a_add, value)

cpdef subtract(table:Table, value):
    return math_op(table, a_subtract, value)

cpdef multiply(table:Table, value):
    return math_op(table, a_multiply, value)

cpdef divide(table:Table, value):
    return division_op(table, a_divide, value)

cpdef unique(table:Table):
    # TODO: axis=1 implementation (row-wise comparison)
    artb = table.to_arrow().combine_chunks()
    res_array = []
    for chunk_ar in artb.itercolumns():
        res_array.append(len(unique(chunk_ar)))
    return Table.from_arrow(table.context, pa.Table.from_arrays(res_array,
                                                                   names=table.column_names))

cpdef nunique(table:Table):
    pass

cdef _is_in_array_like(table: Table, cmp_val, skip_null):
    ar_tb = table.to_arrow().combine_chunks()
    lookup_opts = a_compute.SetLookupOptions(value_set=cmp_val, skip_null=skip_null)
    is_in_res = []
    for chunk_ar in ar_tb.itercolumns():
        is_in_res.append(a_compute.is_in(chunk_ar, options=lookup_opts))
    return Table.from_arrow(table.context, pa.Table.from_arrays(is_in_res, ar_tb.column_names))

def compare_array_like_values(l_org_ar, l_cmp_ar, skip_null=True):
    s = a_compute.SetLookupOptions(value_set=l_cmp_ar, skip_null=skip_null)
    return a_compute.is_in(l_org_ar, options=s)

cdef _broadcast(ar, broadcast_coefficient=1):
    bcast_ar = []
    for elem in ar:
        bcast_elems = []
        for i in range(broadcast_coefficient):
            bcast_elems.append(elem.as_py())
        bcast_ar.append(pa.array(bcast_elems))
    return bcast_ar

cdef _compare_two_arrays(l_ar, r_ar):
    return a_compute.and_(l_ar, r_ar)

cdef _compare_row_and_column(row, columns):
    comp_res = []
    for column in columns:
        comp_res.append(_compare_two_arrays(l_ar=row, r_ar=column))
    return comp_res

cdef _populate_column_with_single_value(value, row_count):
    column_values = []
    for i in range(row_count):
        column_values.append(value)
    return column_values

cdef _tb_compare_dict_values(tb, tb_cmp, skip_null=True):
    return _tb_compare_values(tb, tb_cmp, skip_null=skip_null, index_check=False)

cdef _tb_compare_values(tb, tb_cmp, skip_null=True, index_check=True):

    col_names = tb.column_names
    comp_col_names = tb_cmp.column_names

    row_indices = tb.index.index_values
    row_indices_cmp = tb_cmp.index.index_values

    rows = tb.row_count
    '''
    Compare table column name against comparison-table column names
    '''
    col_comp_res = compare_array_like_values(l_org_ar=pa.array(col_names), l_cmp_ar=pa.array(
        comp_col_names))
    '''
    Compare table row-indices against comparison-table row-indices
    '''
    row_comp_res = compare_array_like_values(l_org_ar=pa.array(row_indices), l_cmp_ar=pa.array(
        row_indices_cmp))
    '''
    broadcast column data to match with row data size to do vectorized array comparisons
    Here the is_in only validates if the row-indices and column-indices are matching in both tables.
    The & operation between row validity vs column_broadcasted validity gives the resultant 
    validity. With this resultant validity the data validity is compared to get the final validity. 
    '''
    bcast_col_comp_res = _broadcast(ar=col_comp_res, broadcast_coefficient=rows)

    tb_ar = tb.to_arrow().combine_chunks()
    tb_cmp_ar = tb_cmp.to_arrow().combine_chunks()
    col_data_map = {}

    if index_check:
        '''
        For tabular data comparison
        '''
        row_col_comp = _compare_row_and_column(row=row_comp_res, columns=bcast_col_comp_res)

        for col_name, validity, row_col_validity in zip(col_names, col_comp_res, row_col_comp):
            if validity.as_py():
                chunk_ar_org = tb_ar.column(col_name)
                chunk_ar_cmp = tb_cmp_ar.column(col_name)
                s = a_compute.SetLookupOptions(value_set=chunk_ar_cmp, skip_null=skip_null)
                col_data_map[col_name] = _compare_two_arrays(a_compute.is_in(chunk_ar_org,
                                                                             options=s), row_col_validity)
            else:
                col_data_map[col_name] = pa.array(_populate_column_with_single_value(False, rows))

        is_in_values = list(col_data_map.values())
        return Table.from_list(tb.context, col_names, is_in_values)

    else:
        '''
        For dictionary comparison
        Note: In dictionary comparison index validity is not required. 
        '''
        for col_name, validity in zip(col_names, col_comp_res):
            if validity.as_py():
                chunk_ar_org = tb_ar.column(col_name)
                chunk_ar_cmp = tb_cmp_ar.column(col_name)
                s = a_compute.SetLookupOptions(value_set=chunk_ar_cmp, skip_null=skip_null)
                col_data_map[col_name] = a_compute.is_in(chunk_ar_org, options=s)
            else:
                col_data_map[col_name] = pa.array(_populate_column_with_single_value(False, rows))

        is_in_values = list(col_data_map.values())
        return Table.from_list(tb.context, col_names, is_in_values)


cpdef is_in(table:Table, comparison_values, skip_null):
    if isinstance(comparison_values, List):
        cmp_val = pa.array(comparison_values)
        return _is_in_array_like(table, cmp_val, skip_null)
    elif isinstance(comparison_values, dict):
        tb_cmp = Table.from_pydict(table.context, comparison_values)
        return _tb_compare_dict_values(table, tb_cmp, skip_null)
    elif isinstance(comparison_values, Table):
        return _tb_compare_values(table, comparison_values, skip_null)
    else:
        raise ValueError(f'Unsupported comparison value type {type(comparison_values)}')


cpdef drop_na(table:Table, how:str, axis=0):
    ar_tb = table.to_arrow().combine_chunks()
    if axis == 0:
        """
        Column-wise computation
        """
        is_null_values = []
        column_names = ar_tb.column_names
        drop_columns = []
        for col_id, chunk_ar in enumerate(ar_tb.itercolumns()):
            res = a_compute.cast(a_compute.is_null(chunk_ar), pa.int32())
            sum_val = a_compute.sum(res).as_py()
            if sum_val > 0 and how == FilterType.ANY.value:
                drop_columns.append(column_names[col_id])
            elif sum_val == len(res) and how == FilterType.ALL.value:
                drop_columns.append(column_names[col_id])
        return Table.from_arrow(table.context, ar_tb.drop(drop_columns))
    elif axis == 1:
        """
        Row-wise computation
        """
        is_null_responses = []
        for chunk_ar in ar_tb.itercolumns():
            is_null_responses.append(a_compute.is_null(chunk_ar))
        '''
        Here the row-major Null check has to be done. 
        Column-wise null check is done and the obtained boolean array
        is then casted to int32 array and sum of all columns is taken. 
        
        Heuristic 1: 
        
        If the resultant sum-array contains an element with value equals to
        0. It implies that all elements in that row are not None. 
        
        Heuristic 2:
        
        If the resultant sum-array contains an element with value equals to
        the number of columns. It implies that all elements in that row are None.
        
        Selection Criterion:
        
        For the criteria on how the dropping is done, when 'any' is selected, the
        sum-array value in corresponding row is greater than 0 that row will be dropped.
        For 'all' criteria that value has to be equal to the number of columns. 
        '''
        column_count = len(is_null_responses)
        sum_res = a_compute.cast(is_null_responses[0], pa.int32())

        for i in range(1, column_count):
            sum_res = a_compute.add(sum_res, a_compute.cast(is_null_responses[i], pa.int32()))

        filtered_indices = []

        for index, value in enumerate(sum_res):
            if value.as_py() == 0 and how == FilterType.ANY.value:
                filtered_indices.append(index)
            elif value.as_py() != column_count and how == FilterType.ALL.value:
                filtered_indices.append(index)

        if filtered_indices:
            return Table.from_arrow(table.context, ar_tb.take(filtered_indices))
        else:
            return None
    else:
        raise ValueError(f"Invalid index {axis}, it must be 0 or 1 !")
