##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

from libcpp.string cimport string
from pytwisterx.common.status cimport _Status
from pytwisterx.common.status import Status
import uuid
from pytwisterx.common.join_config cimport CJoinType
from pytwisterx.common.join_config cimport CJoinAlgorithm
from pytwisterx.common.join_config cimport CJoinConfig
from pytwisterx.common.join_config import PJoinType
from pytwisterx.common.join_config import PJoinAlgorithm
from pyarrow.lib cimport CTable
from pyarrow.lib cimport pyarrow_unwrap_table
from pyarrow.lib cimport pyarrow_wrap_table
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref

from pytwisterx.ctx.context cimport CTwisterXContextWrap
from pytwisterx.ctx.context import TwisterxContext

'''
TwisterX Table definition mapping 
'''

cdef extern from "../../../cpp/src/twisterx/python/table_cython.h" namespace "twisterx::python::table":
    cdef cppclass CxTable "twisterx::python::table::CxTable":
        CxTable()
        CxTable(string)
        string get_id()
        int columns()
        int rows()
        void show()
        void show(int, int, int, int)
        _Status to_csv(const string)
        string join(const string, CJoinType, CJoinAlgorithm, int, int)
        string join(const string, CJoinConfig)
        string distributed_join(const string &table_id, CJoinConfig join_config);
        string distributed_join(const string &table_id, CJoinType type,
							   CJoinAlgorithm algorithm,
							   int left_column_index,
							   int right_column_index);

cdef extern from "../../../cpp/src/twisterx/python/table_cython.h" namespace "twisterx::python::table::CxTable":
    cdef extern _Status from_csv(const string, const char, const string)
    cdef extern string from_pyarrow_table(shared_ptr[CTable] table)
    cdef extern shared_ptr[CTable] to_pyarrow_table(const string table_id)

cdef class Table:
    cdef CxTable *thisPtr
    cdef CJoinConfig *jcPtr
    cdef CTwisterXContextWrap *ctx_wrap

    def __cinit__(self, string id):
        '''
        Initializes the PyTwisterX Table
        :param id: unique id for the Table
        :return: None
        '''
        self.thisPtr = new CxTable(id)

    cdef __get_join_config(self, join_type: str, join_algorithm: str, left_column_index: int,
                           right_column_index: int):
        if left_column_index is None or right_column_index is None:
            raise Exception("Join Column index not provided")

        if join_algorithm is None:
            join_algorithm = PJoinAlgorithm.HASH.value

        if join_algorithm == PJoinAlgorithm.HASH.value:

            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index, right_column_index,
                                             CJoinAlgorithm.CHASH)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index, right_column_index,
                                             CJoinAlgorithm.CHASH)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index, right_column_index,
                                             CJoinAlgorithm.CHASH)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index, right_column_index,
                                             CJoinAlgorithm.CHASH)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))

        elif join_algorithm == PJoinAlgorithm.SORT.value:

            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index, right_column_index,
                                             CJoinAlgorithm.CSORT)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index, right_column_index,
                                             CJoinAlgorithm.CSORT)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index, right_column_index,
                                             CJoinAlgorithm.CSORT)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index, right_column_index,
                                             CJoinAlgorithm.CSORT)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))
        else:
            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index, right_column_index)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index, right_column_index)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index, right_column_index)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index, right_column_index)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))

    @property
    def id(self) -> str:
        '''
        Table Id is extracted from the TwisterX C++ API
        :return: table id
        '''
        return self.thisPtr.get_id().decode()

    @property
    def columns(self) -> int:
        '''
        Column count is extracted from the TwisterX C++ Table API
        :return: number of columns in PyTwisterX table
        '''
        return self.thisPtr.columns()

    @property
    def rows(self) -> int:
        '''
        Rows count is extracted from the TwisterX C++ Table API
        :return: number of rows in PyTwisterX table
        '''
        return self.thisPtr.rows()

    def show(self):
        '''
        prints the table in console from the TwisterX C++ Table API
        :return: None
        '''
        self.thisPtr.show()

    def show_by_range(self, row1: int, row2: int, col1: int, col2: int):
        '''
        prints the table in console from the TwisterX C++ Table API
        uses row range and column range
        :param row1: starting row number as int
        :param row2: ending row number as int
        :param col1: starting column number as int
        :param col2: ending column number as int
        :return: None
        '''
        self.thisPtr.show(row1, row2, col1, col2)

    def to_csv(self, path: str) -> Status:
        '''
        writes a PyTwisterX table to CSV file
        :param path: passed as a str, the path of the csv file
        :return: Status of the process (SUCCESS or FAILURE)
        '''
        cdef _Status status = self.thisPtr.to_csv(path.encode())
        s = Status(status.get_code(), b"", -1)
        return s

    def join(self, ctx: TwisterxContext, table: Table, join_type: str, algorithm: str, left_col: int, right_col: int) -> Table:
        '''
        Joins two PyTwisterX tables
        :param table: PyTwisterX table on which the join is performed (becomes the left table)
        :param join_type: Join Type as str ["inner", "left", "right", "outer"]
        :param algorithm: Join Algorithm as str ["hash", "sort"]
        :param left_col: Join column of the left table as int
        :param right_col: Join column of the right table as int
        :return: Joined PyTwisterX table
        '''
        self.__get_join_config(join_type=join_type, join_algorithm=algorithm, left_column_index=left_col,
                               right_column_index=right_col)
        cdef CJoinConfig *jc1 = self.jcPtr        
        cdef string table_out_id = self.thisPtr.join(table.id.encode(), jc1[0])
        if table_out_id.size() == 0:
            raise Exception("Join Failed !!!")
        return Table(table_out_id)


    def distributed_join(self, ctx: TwisterxContext, table: Table, join_type: str, algorithm: str, left_col: int, right_col: int) -> Table:
        '''
        Joins two PyTwisterX tables
        :param table: PyTwisterX table on which the join is performed (becomes the left table)
        :param join_type: Join Type as str ["inner", "left", "right", "outer"]
        :param algorithm: Join Algorithm as str ["hash", "sort"]
        :param left_col: Join column of the left table as int
        :param right_col: Join column of the right table as int
        :return: Joined PyTwisterX table
        '''
        self.__get_join_config(join_type=join_type, join_algorithm=algorithm, left_column_index=left_col,
                               right_column_index=right_col)
        cdef CJoinConfig *jc1 = self.jcPtr        
        cdef string table_out_id = self.thisPtr.distributed_join(table.id.encode(), jc1[0])
        if table_out_id.size() == 0:
            raise Exception("Join Failed !!!")
        return Table(table_out_id)


    @staticmethod
    def from_arrow(obj) -> Table:       
        '''
        creating a PyTwisterX table from PyArrow Table
        :param obj: PyArrow table
        :return: PyTwisterX table
        '''
        cdef shared_ptr[CTable] artb = pyarrow_unwrap_table(obj)
        cdef string table_id
        if artb.get() == NULL:
            raise TypeError("not an table")
        table_id = from_pyarrow_table(artb)
        return Table(table_id)

    @staticmethod
    def to_arrow(tx_table: Table) :
        '''
        creating PyArrow Table from PyTwisterX table
        :param tx_table: PyTwisterx Table
        :return: PyArrow Table
        '''
        table = to_pyarrow_table(tx_table.id.encode())
        py_arrow_table = pyarrow_wrap_table(table)
        return py_arrow_table




cdef class csv_reader:


    @staticmethod
    def read(ctx: TwisterxContext, path: str, delimiter: str) -> Table:    
        cdef string spath = path.encode()
        cdef string sdelm = delimiter.encode()
        id = uuid.uuid4()
        id_str = id.__str__()
        id_buf = id_str.encode()
        from_csv(spath, sdelm[0], id_buf)
        id_buf = id_str.encode()
        return Table(id_buf)



