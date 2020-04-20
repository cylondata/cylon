from libcpp.string cimport string
from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr, shared_ptr, make_unique
from twisterx.common.status cimport _Status
from pytwisterx.common.status import Status
from libc.stdlib cimport malloc, free
import uuid

cdef extern from "../../../cpp/src/twisterx/table.hpp" namespace "twisterx":
    cdef cppclass _Table "twisterx::Table":
        _Table()
        _Table(string)
        string get_id()


cdef extern from "../../../cpp/src/twisterx/table.hpp" namespace "twisterx::Table":
    #cdef cppclass _Table "twisterx::Table":
    cdef extern int columns()
    cdef extern int rows()
    cdef extern void clear()
    cdef extern void tb_print()
    cdef extern _Status from_csv(const string, const char, const string)


cdef class PyTable:
    cdef _Table *thisPtr
    cdef unique_ptr[_Table] tablePtr

    # def __cinit__(self):
    #     id = uuid.uuid4()
    #     id = id.__str__().encode()
    #     self.thisPtr = new _Table(id)
    #     #self.tablePtr = make_unique[_Table]()
    #
    # def __init__(self):
    #     pass

    # def get_table_ptr(self, id: uuid.UUID):
    #     self.thisPtr = new _Table(id.__str__().encode())
    #     self.tablePtr.reset(self.thisPtr)
        #val = deref(self.thisptr)
        #print(val)


    @staticmethod
    def read_csv(path: str, delimiter: str):
        cdef string spath = path.encode()
        cdef string sdelm = delimiter.encode()
        id = uuid.uuid4()
        id_buf = id.__str__().encode()
        from_csv(spath, sdelm[0], id_buf)

    @staticmethod
    def columns() -> int:
        return 1;

    @staticmethod
    def rows() -> int:
        return 1;
