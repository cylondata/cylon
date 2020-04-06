import cython
from libcpp.string cimport string

cdef extern from "../../../cpp/src/twisterx/data/table_builder.h" namespace "twisterx::data":
    cdef extern string get_id();
    cdef extern int get_rows();
    cdef extern int get_columns();

@cython.boundscheck(False)
@cython.wraparound(False)
def id():
    return get_id()

@cython.boundscheck(False)
@cython.wraparound(False)
def rows():
    return get_rows()

@cython.boundscheck(False)
@cython.wraparound(False)
def columns():
    return get_columns()



