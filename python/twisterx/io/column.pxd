from libcpp.string cimport string

cdef extern from "../../../cpp/src/io/Column.h" namespace "twisterx::io":
    cdef cppclass Column:
        Column (const string)
        string get_id()