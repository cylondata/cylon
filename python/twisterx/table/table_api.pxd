from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string

cdef extern from "../../../cpp/src/table_api.h" namespace "twisterx":
    cdef cppclass table_api:
        int column_count(string id)
        int row_count(string id)