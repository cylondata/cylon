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
from pycylon.common.status cimport CStatus
from pycylon.common.status import Status
from pycylon.common.join_config cimport CJoinType
from pycylon.common.join_config cimport CJoinAlgorithm
from pycylon.common.join_config cimport CJoinConfig
from pycylon.common.join_config import PJoinType
from pycylon.common.join_config import PJoinAlgorithm
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pycylon.io.csv_read_config import CSVReadOptions
from pycylon.io.csv_read_config cimport CSVReadOptions
from pycylon.io.csv_write_config cimport CCSVWriteOptions
from pycylon.io.csv_write_config import CSVWriteOptions
from pycylon.io.csv_write_config cimport CSVWriteOptions

from pyarrow.lib cimport CTable as CArrowTable
from pycylon.data.table cimport CTable
from pyarrow.lib cimport (pyarrow_unwrap_table, pyarrow_wrap_table)
from libcpp.memory cimport shared_ptr, make_shared

from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.api.lib cimport (pycylon_unwrap_context,
pycylon_unwrap_table_out_ptr,
pycylon_wrap_table,
pycylon_unwrap_csv_read_options,
pycylon_unwrap_csv_write_options)

import pyarrow as pa
import numpy as np
import pandas as pd
from typing import List
import warnings

'''
Cylon Table definition mapping 
'''

cdef class Table:
    def __cinit__(self, pyarrow_table=None, context=None, columns=None):
        cdef shared_ptr[CArrowTable] c_arrow_tb_shd_ptr
        cdef shared_ptr[CCylonContext] ctx_shd_ptr
        if self._is_pycylon_context(context) and self._is_pyarrow_table(pyarrow_table):
            c_arrow_tb_shd_ptr = pyarrow_unwrap_table(pyarrow_table)
            ctx_shd_ptr = pycylon_unwrap_context(context)
            self.table_shd_ptr = make_shared[CTable](c_arrow_tb_shd_ptr, ctx_shd_ptr)

    cdef void init(self, const shared_ptr[CTable]& table):
        self.table_shd_ptr = table

    @staticmethod
    def _is_pyarrow_table(pyarrow_table):
        return isinstance(pyarrow_table, pa.Table)

    @staticmethod
    def _is_pycylon_context(context):
        return isinstance(context, CylonContext)

    @staticmethod
    def from_arrow(context, pyarrow_table) -> Table:
        cdef shared_ptr[CCylonContext] ctx = pycylon_unwrap_context(context)
        cdef shared_ptr[CArrowTable] arw_table = pyarrow_unwrap_table(pyarrow_table)
        cdef shared_ptr[CTable] cn_table
        cdef CStatus status = CTable.FromArrowTable(ctx, arw_table, &cn_table)

        if status.is_ok():
            return pycylon_wrap_table(cn_table)
        else:
            raise Exception("Table couldn't be created from PyArrow Table")

    @staticmethod
    def from_csv(context, path, csv_read_options) -> Table:
        cdef shared_ptr[CCylonContext] ctx = pycylon_unwrap_context(context)
        cdef string cpath = path.encode()
        cdef CCSVReadOptions c_csv_read_options = pycylon_unwrap_csv_read_options(csv_read_options)
        cdef shared_ptr[CTable] cn_table
        cdef CStatus status = CTable.FromCSV(ctx, cpath, cn_table, c_csv_read_options)
        if status.is_ok():
            return pycylon_wrap_table(cn_table)
        else:
            raise Exception("Table couldn't be created from CSV")

    def to_csv(self, path, csv_write_options):
        cdef string cpath = path.encode()
        cdef CCSVWriteOptions c_csv_write_options = pycylon_unwrap_csv_write_options(
            csv_write_options)
        self.table_shd_ptr.get().WriteCSV(cpath, c_csv_write_options)

    def show(self):
        self.table_shd_ptr.get().Print()

    def to_arrow(self) -> pa.Table:
        cdef shared_ptr[CArrowTable] converted_tb
        cdef CStatus status = self.table_shd_ptr.get().ToArrowTable(converted_tb)
        if status.is_ok():
            return pyarrow_wrap_table(converted_tb)
        else:
            raise Exception("Table couldn't be converted to a PyArrow Table")

    def sort(self, index) -> Table:
        cdef shared_ptr[CTable] output
        sort_index = -1
        if isinstance(index, str):
            sort_index = self._resolve_column_index_from_column_name(index)
        else:
            sort_index = index

        cdef CStatus status = self.table_shd_ptr.get().Sort(sort_index, output)
        if status.is_ok():
            return pycylon_wrap_table(output)
        else:
            raise Exception("Table couldn't be sorted")

    @property
    def column_names(self):
        return pyarrow_wrap_table(pyarrow_unwrap_table(self.to_arrow())).column_names

    def _resolve_column_index_from_column_name(self, column_name) -> int:
        index = None
        for idx, col_name in enumerate(self.column_names):
            if column_name == col_name:
                index = idx
        if not index:
            raise ValueError(f"Column {column_name} does not exist in the table")
        return index


    def _resolve_join_column_indices_from_column_names(self, column_names: List[
    str], op_column_names: List[str]) -> List[int]:
        resolve_col_ids = []
        for op_col_id, op_column_name in enumerate(op_column_names):
            for col_id, column_name in enumerate(column_names):
                if op_column_name == column_name:
                    resolve_col_ids.append(col_id)
        return resolve_col_ids

    def _get_join_column_indices(self, table: Table, **kwargs):
        ## Check if Passed values are based on left and right column names or indices
        left_cols = kwargs.get('left_on')
        right_cols = kwargs.get('right_on')
        column_names = kwargs.get('on')

        table_col_names_list = [self.column_names, table.column_names]

        if left_cols and right_cols and isinstance(left_cols, List) and isinstance(right_cols,
                                                                                   List):
            if isinstance(left_cols[0], str) and isinstance(right_cols[0], str):
                left_cols = self._resolve_join_column_indices_from_column_names(self.column_names,
                                                                           left_cols)
                right_cols = self._resolve_join_column_indices_from_column_names(table.column_names,
                                                                            right_cols)
                self._check_column_names_viable(left_cols, right_cols)

                return left_cols, right_cols
            elif isinstance(left_cols[0], int) and isinstance(right_cols[0], int):
                return left_cols, right_cols
        ## Check if Passed values are based on common column names in two tables
        elif column_names and isinstance(column_names, List):
            if isinstance(column_names[0], str):
                left_cols = self._resolve_join_column_indices_from_column_names(self.column_names,
                                                                           column_names)
                right_cols = self._resolve_join_column_indices_from_column_names(table.column_names,
                                                                            column_names)
                self._check_column_names_viable(left_cols, right_cols)
                return left_cols, right_cols
            if isinstance(column_names[0], int):
                return column_names, column_names
        else:
            raise TypeError("kwargs 'on' or 'left_on' and 'right_on' must be provided")

        if not (left_cols and isinstance(left_cols[0], int)) and not (right_cols and isinstance(
                right_cols[0], int)):
            raise TypeError("kwargs 'on' or 'left_on' and 'right_on' must be type List and contain "
                            "int type or str type and cannot be None")

    def _is_column_indices_viable(self, left_cols, right_cols):
        return left_cols and right_cols

    def _check_column_names_viable(self, left_cols, right_cols):
        if not self._is_column_indices_viable(left_cols, right_cols):
                    raise ValueError("Provided Column Names or Column Indices not valid.")

# cdef class Table:
#     def __cinit__(self, string id, context):
#         '''
#         Initializes the PyCylon Table
#         :param id: unique id for the Table
#         :return: None
#         '''
#         self.thisPtr = new CxTable(id)
#         self.ctx = context
#
#     def __get_join_config(self, join_type: str, join_algorithm: str, left_column_index: int,
#                           right_column_index: int):
#         if left_column_index is None or right_column_index is None:
#             raise Exception("Join Column index not provided")
#
#         if join_algorithm is None:
#             join_algorithm = PJoinAlgorithm.HASH.value
#
#         if join_algorithm == PJoinAlgorithm.HASH.value:
#
#             if join_type == PJoinType.INNER.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index,
#                                              right_column_index, CJoinAlgorithm.CHASH)
#             elif join_type == PJoinType.LEFT.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index,
#                                              right_column_index, CJoinAlgorithm.CHASH)
#             elif join_type == PJoinType.RIGHT.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index,
#                                              right_column_index, CJoinAlgorithm.CHASH)
#             elif join_type == PJoinType.OUTER.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index,
#                                              right_column_index, CJoinAlgorithm.CHASH)
#             else:
#                 raise ValueError("Unsupported Join Type {}".format(join_type))
#
#         elif join_algorithm == PJoinAlgorithm.SORT.value:
#
#             if join_type == PJoinType.INNER.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index,
#                                              right_column_index, CJoinAlgorithm.CSORT)
#             elif join_type == PJoinType.LEFT.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index,
#                                              right_column_index, CJoinAlgorithm.CSORT)
#             elif join_type == PJoinType.RIGHT.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index,
#                                              right_column_index, CJoinAlgorithm.CSORT)
#             elif join_type == PJoinType.OUTER.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index,
#                                              right_column_index, CJoinAlgorithm.CSORT)
#             else:
#                 raise ValueError("Unsupported Join Type {}".format(join_type))
#         else:
#             if join_type == PJoinType.INNER.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index,
#                                              right_column_index)
#             elif join_type == PJoinType.LEFT.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index, right_column_index)
#             elif join_type == PJoinType.RIGHT.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index,
#                                              right_column_index)
#             elif join_type == PJoinType.OUTER.value:
#                 self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index,
#                                              right_column_index)
#             else:
#                 raise ValueError("Unsupported Join Type {}".format(join_type))
#
#     @property
#     def id(self) -> str:
#         '''
#         Table Id is extracted from the Cylon C++ API
#         :return: table id
#         '''
#         return self.thisPtr.get_id().decode()
#
#     @property
#     def columns(self) -> int:
#         '''
#         Column count is extracted from the Cylon C++ Table API
#         :return: number of columns in PyCylon table
#         '''
#         return self.thisPtr.columns()
#
#     @property
#     def rows(self) -> int:
#         '''
#         Rows count is extracted from the Cylon C++ Table API
#         :return: number of rows in PyCylon table
#         '''
#         return self.thisPtr.rows()
#
#     @property
#     def context(self) -> CylonContext:
#         return self.ctx
#
#     def show(self):
#         '''
#         prints the table in console from the Cylon C++ Table API
#         :return: None
#         '''
#         self.thisPtr.show()
#
#     def show_by_range(self, row1: int, row2: int, col1: int, col2: int):
#         '''
#         prints the table in console from the Cylon C++ Table API
#         uses row range and column range
#         :param row1: starting row number as int
#         :param row2: ending row number as int
#         :param col1: starting column number as int
#         :param col2: ending column number as int
#         :return: None
#         '''
#         self.thisPtr.show(row1, row2, col1, col2)
#
#     def to_csv(self, path: str) -> Status:
#         '''
#         writes a PyCylon table to CSV file
#         :param path: passed as a str, the path of the csv file
#         :return: Status of the process (SUCCESS or FAILURE)
#         '''
#         cdef CStatus status = self.thisPtr.to_csv(path.encode())
#         s = Status(status.get_code(), b"", -1)
#         return s
#
#     def _is_pycylon_table(self, obj):
#         return isinstance(obj, Table)
#
#     def _resolve_column_indices_from_column_names(self, column_names: List[
#         str], op_column_names: List[str]) -> List[int]:
#         resolve_col_ids = []
#         for op_col_id, op_column_name in enumerate(op_column_names):
#             for col_id, column_name in enumerate(column_names):
#                 if op_column_name == column_name:
#                     resolve_col_ids.append(col_id)
#         return resolve_col_ids
#
#     def _get_join_column_indices(self, table: Table, **kwargs):
#         ## Check if Passed values are based on left and right column names or indices
#         left_cols = kwargs.get('left_on')
#         right_cols = kwargs.get('right_on')
#         column_names = kwargs.get('on')
#
#         table_col_names_list = [self.column_names, table.column_names]
#
#         if left_cols and right_cols and isinstance(left_cols, List) and isinstance(right_cols,
#                                                                                    List):
#             if isinstance(left_cols[0], str) and isinstance(right_cols[0], str):
#                 left_cols = self._resolve_column_indices_from_column_names(self.column_names,
#                                                                            left_cols)
#                 right_cols = self._resolve_column_indices_from_column_names(table.column_names,
#                                                                             right_cols)
#                 self._check_column_names_viable(left_cols, right_cols)
#
#                 return left_cols, right_cols
#             elif isinstance(left_cols[0], int) and isinstance(right_cols[0], int):
#                 return left_cols, right_cols
#         ## Check if Passed values are based on common column names in two tables
#         elif column_names and isinstance(column_names, List):
#             if isinstance(column_names[0], str):
#                 left_cols = self._resolve_column_indices_from_column_names(self.column_names,
#                                                                            column_names)
#                 right_cols = self._resolve_column_indices_from_column_names(table.column_names,
#                                                                             column_names)
#                 self._check_column_names_viable(left_cols, right_cols)
#                 return left_cols, right_cols
#             if isinstance(column_names[0], int):
#                 return column_names, column_names
#         else:
#             raise TypeError("kwargs 'on' or 'left_on' and 'right_on' must be provided")
#
#         if not (left_cols and isinstance(left_cols[0], int)) and not (right_cols and isinstance(
#                 right_cols[0], int)):
#             raise TypeError("kwargs 'on' or 'left_on' and 'right_on' must be type List and contain "
#                             "int type or str type and cannot be None")
#     def _is_column_indices_viable(self, left_cols, right_cols):
#         return left_cols and right_cols
#
#     def _check_column_names_viable(self, left_cols, right_cols):
#         if not self._is_column_indices_viable(left_cols, right_cols):
#                     raise ValueError("Provided Column Names or Column Indices not valid.")
#
#     def join(self, ctx: CylonContext, table: Table, join_type: str,
#              algorithm: str, **kwargs) -> Table:
#         '''
#         Joins two PyCylon tables
#         :param table: PyCylon table on which the join is performed (becomes the left table)
#         :param join_type: Join Type as str ["inner", "left", "right", "outer"]
#         :param algorithm: Join Algorithm as str ["hash", "sort"]
#         :kwargs left_on: Join column of the left table as List[int] or List[str], right_on:
#         Join column of the right table as List[int] or List[str], on: Join column in common with
#         both tables as a List[int] or List[str].
#         :return: Joined PyCylon table
#         '''
#
#         left_cols, right_cols = self._get_join_column_indices(table=table, **kwargs)
#
#         # Cylon only supports join by one column and retrieve first left and right column when
#         # resolving join configs
#         self.__get_join_config(join_type=join_type, join_algorithm=algorithm,
#                                left_column_index=left_cols[0],
#                                right_column_index=right_cols[0])
#         cdef CJoinConfig *jc1 = self.jcPtr
#         cdef string table_out_id = self.thisPtr.join(pycylon_unwrap_context(ctx), table.id.encode(),
#                                                      jc1[0])
#         if table_out_id.size() == 0:
#             raise Exception("Join Failed !!!")
#         return Table(table_out_id, ctx)
#
#     def distributed_join(self, ctx: CylonContext, table: Table, join_type: str,
#                          algorithm: str, **kwargs) -> Table:
#         '''
#         Joins two PyCylon tables
#         :param table: PyCylon table on which the join is performed (becomes the left table)
#         :param join_type: Join Type as str ["inner", "left", "right", "outer"]
#         :param algorithm: Join Algorithm as str ["hash", "sort"]
#         :kwargs left_on: Join column of the left table as List[int] or List[str], right_on:
#         Join column of the right table as List[int] or List[str], on: Join column in common with
#         both tables as a List[int] or List[str].
#         :return: Joined PyCylon table
#         '''
#
#         left_cols, right_cols = self._get_join_column_indices(table=table, **kwargs)
#
#         # Cylon only supports join by one column and retrieve first left and right column when
#         # resolving join configs
#         self.__get_join_config(join_type=join_type, join_algorithm=algorithm,
#                                left_column_index=left_cols[0],
#                                right_column_index=right_cols[0])
#         cdef CJoinConfig *jc1 = self.jcPtr
#         cdef string table_out_id = self.thisPtr.distributed_join(pycylon_unwrap_context(ctx),
#                                                                  table.id.encode(),
#                                                                  jc1[0])
#         if table_out_id.size() == 0:
#             raise Exception("Join Failed !!!")
#         return Table(table_out_id, ctx)
#
#     def union(self, ctx: CylonContext, table: Table) -> Table:
#         '''
#         Union two PyCylon tables
#         :param table: PyCylon table on which the join is performed (becomes the left table)
#         :return: Union PyCylon table
#         '''
#         cdef string table_out_id = self.thisPtr.Union(pycylon_unwrap_context(ctx),
#                                                       table.id.encode())
#         if table_out_id.size() == 0:
#             raise Exception("Union Failed !!!")
#         return Table(table_out_id, ctx)
#
#     def distributed_union(self, ctx: CylonContext, table: Table) -> Table:
#         '''
#         Union two PyCylon tables
#         :param table: PyCylon table on which the join is performed (becomes the left table)
#         :return: Union PyCylon table
#         '''
#         cdef string table_out_id = self.thisPtr.DistributedUnion(pycylon_unwrap_context(ctx),
#                                                                  table.id.encode())
#         if table_out_id.size() == 0:
#             raise Exception("Distributed Union Failed !!!")
#         return Table(table_out_id, ctx)
#
#     def intersect(self, ctx: CylonContext, table: Table) -> Table:
#         '''
#         Union two PyCylon tables
#         :param table: PyCylon table on which the join is performed (becomes the left table)
#         :return: Intersect PyCylon table
#         '''
#         cdef string table_out_id = self.thisPtr.Intersect(pycylon_unwrap_context(ctx),
#                                                           table.id.encode())
#         if table_out_id.size() == 0:
#             raise Exception("Intersect Failed !!!")
#         return Table(table_out_id, ctx)
#
#     def distributed_intersect(self, ctx: CylonContext, table: Table) -> Table:
#         '''
#         Union two PyCylon tables
#         :param table: PyCylon table on which the join is performed (becomes the left table)
#         :return: Intersect PyCylon table
#         '''
#         cdef string table_out_id = self.thisPtr.DistributedIntersect(pycylon_unwrap_context(ctx),
#                                                                      table.id.encode())
#         if table_out_id.size() == 0:
#             raise Exception("Distributed Union Failed !!!")
#         return Table(table_out_id, ctx)
#
#     def subtract(self, ctx: CylonContext, table: Table) -> Table:
#         '''
#         Union two PyCylon tables
#         :param table: PyCylon table on which the join is performed (becomes the left table)
#         :return: Subtract PyCylon table
#         '''
#         cdef string table_out_id = self.thisPtr.Subtract(pycylon_unwrap_context(ctx),
#                                                          table.id.encode())
#         if table_out_id.size() == 0:
#             raise Exception("Subtract Failed !!!")
#         return Table(table_out_id, ctx)
#
#     def distributed_subtract(self, ctx: CylonContext, table: Table) -> Table:
#         '''
#         Union two PyCylon tables
#         :param table: PyCylon table on which the join is performed (becomes the left table)
#         :return: Subtract PyCylon table
#         '''
#
#         cdef string table_out_id = self.thisPtr.DistributedSubtract(pycylon_unwrap_context(ctx),
#                                                                     table.id.encode())
#         if table_out_id.size() == 0:
#             raise Exception("Distributed Subtract Failed !!!")
#         return Table(table_out_id, ctx)
#
#     @staticmethod
#     def from_arrow(obj, ctx: CylonContext) -> Table:
#         '''
#         creating a PyCylon table from PyArrow Table
#         :param obj: PyArrow table
#         :return: PyCylon table
#         '''
#         cdef shared_ptr[CTable] artb = pyarrow_unwrap_table(obj)
#         cdef string table_id
#         if artb.get() == NULL:
#             raise TypeError("not an table")
#         if ctx.get_config() == ''.encode():
#             table_id = from_pyarrow_table(pycylon_unwrap_context(ctx), artb)
#         else:
#             table_id = from_pyarrow_table(pycylon_unwrap_context(ctx), artb)
#         return Table(table_id, ctx)
#
#     # @staticmethod
#     # def from_pandas(obj, ctx: CylonContext) -> Table:
#     #     """
#     #     creating a PyCylon table from Pandas DataFrame
#     #     :param obj: Pandas DataFrame
#     #     :rtype: PyCylon Table
#     #     """
#     #     table = pa.Table.from_pandas(obj)
#     #     return Table.from_arrow(table, ctx)
#
#     def to_arrow(self) -> pa.Table:
#         '''
#         creating PyArrow Table from PyCylon table
#         :param self: PyCylon Table
#         :return: PyArrow Table
#         '''
#         table = to_pyarrow_table(self.id.encode())
#         py_arrow_table = pyarrow_wrap_table(table)
#         return py_arrow_table
#
#     def to_pandas(self) -> pd.DataFrame:
#         """
#         creating Pandas Dataframe from PyCylon Table
#         :param self:
#         :return: a Pandas DataFrame
#         """
#         table = to_pyarrow_table(self.id.encode())
#         py_arrow_table = pyarrow_wrap_table(table)
#         return py_arrow_table.to_pandas()
#
#     def to_numpy(self, order='F') -> np.ndarray:
#         """
#         [Experimental]
#         This method converts a Cylon Table to a 2D numpy array.
#         In the conversion we stack each column in the Table and create a numpy array.
#         For Heterogeneous Tables, use the generated array with Caution.
#         :param order: numpy array order. 'F': Fortran Style F_Contiguous or 'C' C Style C_Contiguous
#         :return: ndarray
#         """
#         table = to_pyarrow_table(self.id.encode())
#         py_arrow_table = pyarrow_wrap_table(table)
#         ar_lst = []
#         _dtype = None
#         for col in py_arrow_table.columns:
#             npr = col.chunks[0].to_numpy()
#             if None == _dtype:
#                 _dtype = npr.dtype
#             if _dtype != npr.dtype:
#                 warnings.warn(
#                     "Heterogeneous Cylon Table Detected!. Use Numpy operations with Caution.")
#             ar_lst.append(npr)
#         return np.array(ar_lst, order=order).T
#
#     @property
#     def column_names(self):
#         table = to_pyarrow_table(self.id.encode())
#         py_arrow_table = pyarrow_wrap_table(table)
#         return py_arrow_table.column_names
#
#     @property
#     def schema(self):
#         table = to_pyarrow_table(self.id.encode())
#         py_arrow_table = pyarrow_wrap_table(table)
#         return py_arrow_table.schema
#
#     def __getitem__(self, key):
#         table = to_pyarrow_table(self.id.encode())
#         py_arrow_table = pyarrow_wrap_table(table)
#         if isinstance(key, slice):
#             start, stop, step = key.indices(self.columns)
#             return self.from_arrow(py_arrow_table.slice(start, stop),
#                                    self.context)
#         else:
#             return self.from_arrow(py_arrow_table.slice(key, 1), self.context)
#
#     def __repr__(self):
#         table = to_pyarrow_table(self.id.encode())
#         py_arrow_table = pyarrow_wrap_table(table)
#         table_str = ''
#         for i in range(self.rows):
#             row = py_arrow_table.slice(i, 1)
