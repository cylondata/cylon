# cython: infer_types=True
import numpy as np
cimport cython
from libcpp cimport bool
from pycylon.data.table cimport CTable
from pycylon.data.table import Table


# TODO: Supported Added via: https://github.com/cylondata/cylon/issues/211

cdef api c_filter(tb: Table, op):
    pass


ctypedef fused my_type:
    int
    double
    long long


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef comparison_compute_op(my_type[:] array, my_type other, op):

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    cdef Py_ssize_t rows = array.shape[0]
    cdef Py_ssize_t index
    result = np.zeros((rows), dtype='bool')
    cdef bool[:] result_view = result
    for index in range(rows):
        result_view[index] = op(array[index], other)

    return result
