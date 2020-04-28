from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string
from twisterx.common.status cimport _Status
from pytwisterx.common.status import Status
from libcpp.memory cimport unique_ptr

# cdef extern from "../../../cpp/src/twisterx/table.hpp" namespace "twisterx":
#     cdef cppclass _Table "twisterx::Table":
#         _Table()
#         _Table(string)
#         int columns()
#         int rows()
#         void clear()
#         void tb_print()
#         _Status from_csv(const string, const char)

