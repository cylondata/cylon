import cython
from libcpp.string cimport string
from twisterx.common.status cimport _Status
from pytwisterx.common.status import Status

cdef extern from "../../../cpp/src/twisterx/data/table_builder.hpp" namespace "twisterx::data":
    cdef extern string get_id();
    cdef extern int get_rows();
    cdef extern int get_columns();
    cdef extern _Status read_csv();


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

@cython.boundscheck(False)
@cython.wraparound(False)
def read_from_csv(path:str, delimiter:str) ->Status:
    s = Status(0, b'', -1)
    return s


