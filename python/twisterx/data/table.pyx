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
        int columns()
        int rows()
        void show()
        void show(int, int, int, int)




cdef extern from "../../../cpp/src/twisterx/table.hpp" namespace "twisterx::Table":
    cdef extern _Status from_csv(const string, const char, const string)


cdef class Table:
    cdef _Table *thisPtr
    def __cinit__(self, string id):
        self.thisPtr = new _Table(id)
        #self.tablePtr = make_unique[_Table]()

    @property
    def id(self) -> str:
        return self.thisPtr.get_id().decode()

    @property
    def columns(self) -> str:
        return self.thisPtr.columns()

    @property
    def rows(self) -> str:
        return self.thisPtr.rows()

    def show(self):
        self.thisPtr.show()

    def show_by_range(self, row1:int, row2:int, col1: int, col2: int):
        self.thisPtr.show(row1, row2, col1, col2)

cdef class TableUtil:
    cdef _Table *thisPtr
    cdef unique_ptr[_Table] tablePtr

    @staticmethod
    def read_csv(path: str, delimiter: str) -> str:
        cdef string spath = path.encode()
        cdef string sdelm = delimiter.encode()
        id = uuid.uuid4()
        id_str = id.__str__()
        id_buf = id_str.encode()
        from_csv(spath, sdelm[0], id_buf)
        return id_str

