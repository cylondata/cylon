from libcpp.string cimport string
from libcpp cimport bool
from twisterx.common.code cimport _Code

cdef extern from "../../../cpp/src/twisterx/status.hpp" namespace "twisterx":
    cdef cppclass _Status "twisterx::Status":
        _Status()
        _Status(int, string)
        _Status(int)
        _Status(_Code)
        _Status(_Code, string)
        int get_code()
        bool is_ok()
        string get_msg()