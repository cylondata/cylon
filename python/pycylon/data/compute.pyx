# cython: infer_types=True
import operator

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
from operator import add as py_add
from operator import sub as py_sub
from operator import mul as py_mul
from operator import truediv as py_truediv

from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompareBool,
)
cimport numpy as np
from cython import Py_ssize_t
from numpy cimport ndarray
np.import_array()

from typing import Any, List


ctypedef np.int_t DTYPE_t


cdef api c_filter(tb: Table, op):
    # TODO: Supported Added via: https://github.com/cylondata/cylon/issues/211
    pass


ctypedef fused CDType:
    int
    double
    float
    long long


cdef cast_scalar(scalar_value, dtype_id):
    """
    This method casts a Arrow scalar value into a Python scalar value
    Args:
        scalar_value: Arrow Scalar
        dtype_id: Arrow data type

    Returns: Python scalar

    """
    if dtype_id == pa.float16().id or dtype_id == pa.float32().id or dtype_id == pa.float64().id:
        return float(scalar_value)
    elif dtype_id == pa.int16().id or dtype_id == pa.int32().id or dtype_id == pa.int64().id:
        return int(scalar_value)
    elif dtype_id == pa.int8().id:
        return bytes(scalar_value)
    elif dtype_id == pa.string().id:
        return str(scalar_value)
    else:
        raise ValueError(f"Unsupported Scalar Type {type(scalar_value)}")

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef comparison_compute_op_iter(CDType[:] array, CDType other, op):
    """
    This is a helper function for comparison op using Cythonized approach
    Used as an customizable feature based on use-cases
    Args:
        array: Typed array of CDType
        other: comparison value of type CDType
        op: comparison operator, i.e <,>,<=,>=,!=

    Returns: NdArray

    """
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


_resolve_arrow_op = {
    operator.__gt__: greater,
     operator.__lt__ : less,
     operator.__eq__ : equal,
     operator.__ge__ : greater_equal,
     operator.__le__ : less_equal,
     operator.__ne__ : not_equal,
     operator.__or__ : or_,
     operator.__and__ :and_,
    operator.add : a_add,
    operator.mul : a_multiply,
    operator.sub : a_subtract,
    operator.truediv: a_divide
}

cpdef table_compare_ar_op(table: Table, other, op):
    """
    This method compare a PyCylon table with a scalar or with another PyCylon table and returns 
    filtered values in bool
    Args:
        table: PyCylon table
        other: comparison value, a scalar or a PyCylon table
        op: comparison operation, i.e <,>,<=,>=,!=

    Returns: PyCylon table with bool values

    """
    arrow_op = _resolve_arrow_op[op]
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
    elif np.isscalar(other):
        arrays = []
        ar_table = table.to_arrow().combine_chunks()
        for col in ar_table.columns:
          ar_array = col.chunks[0]
          r = arrow_op(ar_array, pa.scalar(other, col.type))
          arrays.append(r)
        return Table.from_arrow(table.context, pa.Table.from_arrays(arrays,
                                                                   names=table.column_names))
    else:
        raise ValueError(f"Comparison Operator not supported for type {type(other)}. Only Table "
                         f"and numbers are supported!")


cpdef table_compare_np_op(table: Table, other, op):
    """
    This method compare a PyCylon table with a scalar or with another PyCylon table and returns 
    filtered values in bool
    Args:
        table: PyCylon table
        other: comparison value, a scalar or a PyCylon table
        op: comparison operation, i.e <,>,<=,>=,!=

    Returns: PyCylon table with bool values

    """
    if isinstance(other, Table):
        arrays = []
        l_table = table.to_arrow().combine_chunks()
        r_table = other.to_arrow().combine_chunks()
        for l_col, r_col in zip(l_table.columns, r_table.columns):
            l_array = l_col.chunks[0].to_numpy()
            r_array = r_col.chunks[0].to_numpy()
            arrays.append(op(l_array, r_array))
        return Table.from_numpy(table.context, table.column_names, arrays)
    elif np.isscalar(other):
        arrays = []
        ar_table = table.to_arrow().combine_chunks()
        for col in ar_table.columns:
          ar_array = col.chunks[0].to_numpy()
          r = op(ar_array, other)
          arrays.append(r)
        return Table.from_numpy(table.context, table.column_names, arrays)
    else:
        raise ValueError(f"Comparison Operator not supported for type {type(other)}. Only Table "
                         f"and numbers are supported!")


cpdef table_compare_op(table: Table, other, op, engine='arrow'):
    if engine == 'arrow':
        return table_compare_ar_op(table, other, op)
    elif engine == 'numpy':
        return table_compare_np_op(table, other, op)
    else:
        raise ValueError(f"Unsupported engine : {engine}")

cdef _is_division(op):
    return op.__name__ == 'divide'


cpdef is_null(table:Table):
    """
    Compute function to check if null values are present in a PyCylon Table
    Args:
        table: PyCylon Table

    Returns: PyCylon table with bool values. True for null and False for not null values

    """
    ar_tb = table.to_arrow().combine_chunks()
    is_null_values = []
    for chunk_ar in ar_tb.itercolumns():
        is_null_values.append(a_compute.is_null(chunk_ar))
    return Table.from_arrow(table.context, pa.Table.from_arrays(is_null_values,
                                                                   names=table.column_names))

cpdef invert(table:Table):
    """
    Inverts a bool array. Other types are not supported.
    NOTE: Only Bool invert is supported by PyArrow APIs
    Args:
        table: PyCylon table

    Returns: Bool valued PyCylon table

    """
    ar_tb = table.to_arrow().combine_chunks()
    invert_values = []
    for chunk_ar in ar_tb.itercolumns():
        if chunk_ar.type == pa.bool_():
            invert_values.append(a_compute.invert(chunk_ar))
        else:
            raise ValueError(f"Invert only support for bool types, but found {chunk_ar.type}")
    return Table.from_arrow(table.context, pa.Table.from_arrays(invert_values,
                                                                   names=table.column_names))

cpdef neg(table:Table):
    """
    Negative operation over a PyCylon Table. 
    Note: Only support numeric tables
    Args:
        table: PyCylon Table

    Returns: PyCylon Table

    """
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
    """
    Division by scalar on a PyCylon Table
    Args:
        table: PyCylon table
        op: division op
        value: division value (scalar numeric)

    Returns: PyCylon table

    """
    if not isinstance(value, numbers.Number) and not isinstance(value,
                                                                Table) and value.column_count() != 1:
        raise ValueError("Div operation value must be numerical or a Numeric Table with 1 column")

    ar_tb = table.to_arrow().combine_chunks()
    res_array = []

    if isinstance(value, numbers.Number):
        for chunk_arr in ar_tb.itercolumns():
            # scalar casting is inexpensive, hence there's no check!
            res_array.append(op(chunk_arr.cast(pa.float64()), pa.scalar(value, pa.float64())))
    elif isinstance(value, Table):
        if value.row_count != table.row_count:
            raise ValueError("Math operation table lengths do not match")
        value_col = value.to_arrow()[0]  # get first column
        for chunk_arr in ar_tb.itercolumns():
            res_array.append(op(chunk_arr.cast(pa.float64()), value_col.cast(pa.float64())))

    return Table.from_arrow(table.context, pa.Table.from_arrays(res_array,
                                                                names=table.column_names))

cpdef math_op_numpy(table:Table, op, value):
    """
    Math operations for PyCylon table against a non-scalar value (including strings). 
    Generic function to execute addition, subtraction and multiplication.

    Args:
        table: PyCylon table
        op: math operator (except division)
        value: scalar value 

    Returns:

    """
    # if not isinstance(value, numbers.Number) and not isinstance(value, Table) and value.column_count != 1:
    #     raise ValueError("Math operation value must be numerical or a Numeric Table with 1 column")
    ar_tb = table.to_arrow().combine_chunks()
    res_array = []
    if isinstance(value, Table):
        # Case Type Table Addition
        value_tb = value.to_arrow().combine_chunks()
        if value.shape == table.shape:
            # Case 1: Adding two tables with same shape
            for chunk_arr_1, chunk_arr_2 in zip(ar_tb.itercolumns(), value_tb.itercolumns()):
                np_ar_1 = chunk_arr_1.to_numpy()
                np_ar_2 = chunk_arr_2.to_numpy()
                res_array.append(op(np_ar_1, np_ar_2))
            return Table.from_numpy(table.context, table.column_names, res_array)
        elif value.shape[1] == 1:
            # Case 2: Adding a single column to a multi-column table
            chunk_arr_2 = next(value_tb.itercolumns())
            for chunk_arr_1 in ar_tb.itercolumns():
                np_ar_1 = chunk_arr_1.to_numpy()
                np_ar_2 = chunk_arr_2.to_numpy()
                res_array.append(op(np_ar_1, np_ar_2))
            return Table.from_numpy(table.context, table.column_names, res_array)
        else:
            raise ValueError("Left Table shapes must match or right table or right table must have a single column")
    elif np.isscalar(value):
        # Case Type adding a scalar to a table
        for chunk_arr in ar_tb.itercolumns():
            np_ar = chunk_arr.to_numpy()
            res_array.append(op(np_ar, value))
        return Table.from_numpy(table.context, table.column_names, res_array)
    else:
        raise ValueError("Addition must be with either a table with matching shape or a scalar")



cpdef math_op_arrow(table:Table, op, value):
    """
    Math operations for PyCylon table against a scalar value (exclude strings).
    Note: Arrow doesn't support operators on strings (concat ==> +) 
    Generic function to execute addition, subtraction and multiplication.

    Args:
        table: PyCylon table
        op: math operator (except division)
        value: scalar value 

    Returns:

    """
    ar_tb = table.to_arrow().combine_chunks()
    res_array = []
    for chunk_arr in ar_tb.itercolumns():
        value = cast_scalar(value, chunk_arr.type.id)
        res_array.append(op(chunk_arr, value))
    return Table.from_arrow(table.context, pa.Table.from_arrays(res_array,
                                                                names=table.column_names))



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef math_op_c_numpy(table: Table, op, value):
    # NOTE: Type aware Numpy math operations using Cython
    # At the moment the the existing implementation using plain numpy has optimum performance
    # Assigning types dynamically the performance could be further enhanced.
    DTYPE = np.int
    import time
    tp1 = time.time()
    cdef:
        Py_ssize_t n, i, rows
        ndarray[object] result
        object val
        ndarray[DTYPE_t] npr_l
        ndarray[DTYPE_t] npr_r
        int scalar_value
    n = table.column_count
    rows = table.row_count
    ar_list = []
    i = 0
    result = np.empty(n, dtype=object)
    npr_l = np.empty(rows, dtype=DTYPE)
    npr_r = np.empty(rows, dtype=DTYPE)
    artb = table.to_arrow().combine_chunks()
    tp2 = time.time()
    print("prep time : ", tp2-tp1)
    if isinstance(value, Table):
        if value.shape == table.shape:
            artb_value = value.to_arrow().combine_chunks()
            for chunk_array_l, chunk_array_r in zip(artb.itercolumns(), artb_value.itercolumns()):
                npr_l = chunk_array_l.to_numpy()
                npr_r = chunk_array_r.to_numpy()
                val = op(npr_l, npr_r)
                result[i] = val
                i = i + 1
        elif value.shape[1] == 1:
            artb_value = value.to_arrow().combine_chunks()
            for chunk_array_l in artb.itercolumns():
                npr_l = chunk_array_l.to_numpy()
                val = None
                for chunk_array_r in  artb_value.itercolumns():
                    npr_r = chunk_array_r.to_numpy()
                    val = op(npr_l, npr_r)
                result[i] = val
                i = i + 1
    elif np.isscalar(value):
        scalar_value = value
        import time
        t1 = time.time()
        for chunk_array in artb.itercolumns():
            npr_l = chunk_array.to_numpy()
            val = op(npr_l, scalar_value)
            result[i] = val
            i = i + 1
        t2 = time.time()
        print("comp time taken: ",t2-t1)
    i = 0
    tc1 = time.time()
    for i in range(n):
        ar_list.append(result[:][i])
    tc2 = time.time()
    print("recreation t: ", tc2 - tc1)
    tn1 = time.time()
    new_tb = Table.from_arrow(table.context,
                            pa.Table.from_arrays(ar_list, table.column_names))
    tn2 = time.time()
    print("tb cr time : ", tn2- tn1)
    return new_tb


cpdef math_op(table: Table, op, value, engine):
    if engine == 'arrow':
        if np.isscalar(value) and isinstance(value, numbers.Number):
            op = _resolve_arrow_op[op]
            return math_op_arrow(table, op, value)
        else:
            return math_op_numpy(table, op, value)
    elif engine == 'numpy':
        return math_op_numpy(table, op, value)
    else:
        raise ValueError(f"Unsupported engine : {engine}")


cpdef unique(table:Table):
    # Sequential and Distributed Kernels are written in LibCylon
    artb = table.to_arrow().combine_chunks()
    res_array = []
    for chunk_ar in artb.itercolumns():
        res_array.append(len(unique(chunk_ar)))
    return Table.from_arrow(table.context, pa.Table.from_arrays(res_array,
                                                                   names=table.column_names))

cpdef nunique(table:Table):
    pass

cdef _is_in_array_like_np(table: Table, cmp_val):
    '''
    Isin helper function for array-like comparisons, where a table is compared against a list of 
    values. 
    Args:
        table: PyCylon table
        cmp_val: comparison value set 
        skip_null: skip null value option for PyArrow compute kernel

    Returns: PyCylon Table

    '''
    ar_tb = table.to_arrow().combine_chunks()
    is_in_res = []
    for chunk_ar in ar_tb.itercolumns():
        chunk_ar_np = chunk_ar.to_numpy()
        is_in_res.append(np.isin(chunk_ar, np.array(cmp_val)))
    tb_new = Table.from_numpy(table.context, table.column_names, is_in_res)
    tb_new.set_index(table.index)
    return tb_new


cdef _is_in_array_like(table: Table, cmp_val, skip_null):
    '''
    Isin helper function for array-like comparisons, where a table is compared against a list of 
    values. 
    Args:
        table: PyCylon table
        cmp_val: comparison value set 
        skip_null: skip null value option for PyArrow compute kernel

    Returns: PyCylon Table

    '''
    ar_tb = table.to_arrow().combine_chunks()
    lookup_opts = a_compute.SetLookupOptions(value_set=cmp_val, skip_null=skip_null)
    is_in_res = []
    for chunk_ar in ar_tb.itercolumns():
        is_in_res.append(a_compute.is_in(chunk_ar, options=lookup_opts))
    tb_new = Table.from_arrow(table.context, pa.Table.from_arrays(is_in_res, ar_tb.column_names))
    tb_new.set_index(table.index)
    return tb_new

def compare_array_like_values(l_org_ar, l_cmp_ar, skip_null=True):
    '''
    Compare array like values
    Args:
        l_org_ar: PyArrow array
        l_cmp_ar: PyArrow array
        skip_null: skip_null option for isin check in PyArrow compute isin

    Returns: PyArrow array

    '''
    s = a_compute.SetLookupOptions(value_set=l_cmp_ar, skip_null=skip_null)
    return a_compute.is_in(l_org_ar, options=s)

cdef _broadcast(ar, broadcast_coefficient=1):
    # TODO: this method must be efficiently written using Cython
    '''
    A compute helper function to broadcast an array into expected dimensionality
    Args:
        ar: PyArrow array
        broadcast_coefficient: expected broadcast size 

    Returns: PyArrow array

    '''
    bcast_ar = []
    cdef int i
    cdef int bc = broadcast_coefficient
    for elem in ar:
        bcast_elems = []
        for i in range(bc):
            bcast_elems.append(elem.as_py())
        bcast_ar.append(pa.array(bcast_elems))
    return bcast_ar

cdef _compare_row_vector_and_column_vectors(col_vector, row_vector):
    '''
    Helper compute function to compare a vector against a set of vectors
    Args:
        col_vector: list of PyArrow arrays
        row_vector: PyArrow array

    Returns: List of PyArrow arrays

    '''
    row_col = []
    for col in col_vector:
        row = a_compute.cast(row_vector, pa.int32())
        col = a_compute.cast(col, pa.int32())
        row_col.append(a_compute.cast(a_compute.multiply(row, col), pa.bool_()))
    return row_col

cdef _compare_two_arrays(l_ar, r_ar):
    '''
    Compare two arrays for and operation. 
    Args:
        l_ar: PyArrow array
        r_ar: PyArrow array

    Returns: PyArrow array bool array

    '''
    return a_compute.and_(l_ar, r_ar)

cdef _compare_row_and_column(row, columns):
    '''
    Helper function to compare row and column
    Args:
        row: Pyarrow array
        columns: Pyarrow array 

    Returns: Pyarrow array of type bool

    '''
    comp_res = []
    for column in columns:
        comp_res.append(_compare_two_arrays(l_ar=row, r_ar=column))
    return comp_res

cdef _populate_column_with_single_value(value, row_count):
    '''
    Helper function to populate a column with a single value
    Args:
        value: Python scalar value
        row_count: number of rows

    Returns: List

    '''
    column_values = []
    cdef int i
    cdef int rc = row_count
    for i in range(rc):
        column_values.append(value)
    return column_values

cdef _tb_compare_dict_values(tb, tb_cmp, skip_null=True):
    # Similar in functionality table compare method internal row-col validation logic is not same
    col_names = tb.column_names
    comp_col_names = tb_cmp.column_names

    row_indices = tb.index.index_values
    row_indices_cmp = row_indices

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

    tb_ar = tb.to_arrow().combine_chunks()
    tb_cmp_ar = tb_cmp.to_arrow().combine_chunks()
    col_data_map = {}

    # Pyarrow compute API doesn't have broadcast enabled computations
    # Here the row-col validity vectors are casted to int32 and a multiply
    # operation is done to get the equivalent response and cast back to boolean
    row = a_compute.cast(row_comp_res, pa.int32())

    for col_name, col_validity in zip(col_names, col_comp_res):
        col = a_compute.cast(col_validity, pa.int32())
        row_col_validity = a_compute.cast(a_compute.multiply(row, col), pa.bool_())
        if col_validity.as_py():
            chunk_ar_org = tb_ar.column(col_name)
            chunk_ar_cmp = tb_cmp_ar.column(col_name)
            s = a_compute.SetLookupOptions(value_set=chunk_ar_cmp, skip_null=skip_null)
            col_data_map[col_name] = _compare_two_arrays(a_compute.is_in(chunk_ar_org,
                                                                         options=s), row_col_validity)
        else:
            col_data_map[col_name] = row_col_validity
    tb_new = Table.from_pydict(tb.context, col_data_map)
    tb_new.set_index(tb.index)
    return tb_new

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

    tb_ar = tb.to_arrow().combine_chunks()
    tb_cmp_ar = tb_cmp.to_arrow().combine_chunks()
    col_data_map = {}

    '''
    For tabular data comparison
    '''
    row = a_compute.cast(row_comp_res, pa.int32())
    for col_name, col in zip(col_names, col_comp_res):
        col = a_compute.cast(col, pa.int32())
        row_col_validity = a_compute.cast(a_compute.multiply(row, col), pa.bool_())
        chunk_ar_org = tb_ar.column(col_name)
        chunk_ar_cmp = tb_cmp_ar.column(col_name)
        s = a_compute.SetLookupOptions(value_set=chunk_ar_cmp, skip_null=skip_null)
        col_data_map[col_name] = _compare_two_arrays(a_compute.is_in(chunk_ar_org,
                                                                     options=s), row_col_validity)
    tb_new = Table.from_pydict(tb.context, col_data_map)
    tb_new.set_index(tb.index)
    return tb_new

cpdef is_in(table:Table, comparison_values, skip_null, engine):
    '''
    PyCylon Tabular is_in function abstraction using PyArrow is_in compute
    Args:
        table: PyCylon table
        comparison_values: comparison values as a List
        skip_null: skip null values upon user response (required for a_compute.isin)

    Returns: PyCylon table of type bool

    '''
    if isinstance(comparison_values, List):
        if engine == "arrow":
            cmp_val = pa.array(comparison_values)
            return _is_in_array_like(table, cmp_val, skip_null)
        elif engine == "numpy":
            return _is_in_array_like_np(table, comparison_values)
    elif isinstance(comparison_values, dict):
        tb_cmp = Table.from_pydict(table.context, comparison_values)
        return _tb_compare_dict_values(table, tb_cmp, skip_null)
    elif isinstance(comparison_values, Table):
        return _tb_compare_values(table, comparison_values, skip_null)
    else:
        raise ValueError(f'Unsupported comparison value type {type(comparison_values)}')


cpdef drop_na(table:Table, how:str, axis=0):
    '''
    Drops not applicable values like nan, None
    Args:
        table: PyCylon table
        how: 'any' or 'all', i.e drop the column or row based on any or all not applicable value 
        presence.
        axis: 0 for column and 1 for row. 

    Returns: PyCylon table

    '''
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


cpdef infer_map(table, func):
    cdef:
        Py_ssize_t n, i
        ndarray[object] result
        object val
    n = table.column_count
    print("N : ", n, i)
    ar_list = []
    i = 0
    result = np.empty(n, dtype=object)
    artb = table.to_arrow().combine_chunks()
    for chunk_array in artb.itercolumns():
        npr = chunk_array.to_numpy()
        val = func(npr)
        result[i] = val
        i = i + 1
    i = 0
    for i in range(n):
        ar_list.append(result[:][i])

    return Table.from_arrow(table.context,
                            pa.Table.from_arrays(ar_list, table.column_names))
