#from twisterx.table.table_api cimport _table_api
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string
from twisterx.common.code cimport _Code
from twisterx.common.status cimport _Status
import cython


# cdef extern from "../../../cpp/src/twisterx/table_api.hpp" namespace "twisterx":
#     cdef extern _Status read_csv(string, string)
#     cdef extern int column_count(string id)
#     cdef extern int row_count(string id)
#     cdef extern _Status show(string, int, int, int, int)

cdef class Simple:

    def __init__(self):
        pass

    def get_val(self):
        return 10;


# cdef class TableAPI:
#
#     def __init__(self):
#         pass
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @staticmethod
#     def csv(string path, string id):
#         read_csv(path, id)
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @staticmethod
#     def column_count() -> int:
#         return 1;
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @staticmethod
#     def row_count() -> int:
#         return 1;
