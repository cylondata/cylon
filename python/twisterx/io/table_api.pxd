from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string


cdef extern from "../../../cpp/src/io/table_api.h" namespace "twisterx::io":
    cdef cppclass table_api:
        table_api()
        ctypedef shared_ptr[table_api] table_api_ptr
        read_csv(string, string)