import cython
from libcpp.string cimport string
from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr, shared_ptr, make_unique
from twisterx.common.status cimport _Status
from pytwisterx.common.status import Status
from libc.stdlib cimport malloc, free
import uuid
from twisterx.common.join_config cimport _JoinType
from twisterx.common.join_config cimport _JoinAlgorithm

cdef extern from "../../../cpp/src/twisterx/python/table_cython.h" namespace "twisterx::python::table":
    cdef cppclass CTable "twisterx::python::table::CTable":
        CTable()
        CTable(string)
        string get_id()
        int columns()
        int rows()
        void show()
        void show(int, int, int, int)
        _Status to_csv(const string)
        string join(const string, _JoinType, _JoinAlgorithm, int, int)

cdef extern from "../../../cpp/src/twisterx/python/table_cython.h" namespace "twisterx::python::table::CTable":
    cdef extern _Status from_csv(const string, const char, const string)


cdef class Table:
    cdef CTable *thisPtr
    def __cinit__(self, string id):
        self.thisPtr = new CTable(id)
        #self.tablePtr = make_unique[CTable]()

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

    def show_by_range(self, row1: int, row2: int, col1: int, col2: int):
        self.thisPtr.show(row1, row2, col1, col2)

    def to_csv(self, path: str) -> Status:
        cdef _Status status = self.thisPtr.to_csv(path.encode())
        s = Status(status.get_code(), b"", -1)
        return s

    def join(self, table: Table, join_type: str, algorithm: str, left_col: int, right_col: int) -> Table:
        cdef string table_out_id = self.thisPtr.join(table.id.encode(), _JoinType.RIGHT, _JoinAlgorithm.HASH, left_col,
                                                     right_col)
        if table_out_id.size() == 0:
            raise Exception("Join Failed !!!")
        return Table(table_out_id)

cdef class csv:
    #cdef _Table *thisPtr
    #cdef unique_ptr[_Table] tablePtr

    @staticmethod
    def read(path: str, delimiter: str) -> Table:
        cdef string spath = path.encode()
        cdef string sdelm = delimiter.encode()
        id = uuid.uuid4()
        id_str = id.__str__()
        id_buf = id_str.encode()
        from_csv(spath, sdelm[0], id_buf)
        id_buf = id_str.encode()
        return Table(id_buf)
