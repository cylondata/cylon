#from twisterx.table.table_api cimport _table_api
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string
from twisterx.common.code cimport _Code
from twisterx.common.status cimport _Status
import cython


cdef extern from "../../../cpp/src/twisterx/table_api.hpp" namespace "twisterx":
    #cdef extern _Status read_csv(const string, const string)
    cdef extern int column_count(const string)
    cdef extern int row_count(const string)
    #cdef extern _Status show(const string, int, int, int, int)



# @cython.boundscheck(False)
# @cython.wraparound(False)
# @staticmethod
# def csv(string path, string id):
#     read_csv(path, id)

@cython.boundscheck(False)
@cython.wraparound(False)
def column_count_(string id) -> int:
    return 10#column_count(id)

@cython.boundscheck(False)
@cython.wraparound(False)
def row_count_(string id) -> int:
    return 20#row_count(id)





