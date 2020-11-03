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
from pycylon.data.table cimport *
from pyarrow.lib cimport (pyarrow_unwrap_table, pyarrow_wrap_table)
from libcpp.memory cimport shared_ptr, make_shared

from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.api.lib cimport (pycylon_wrap_context,
pycylon_unwrap_context,
pycylon_unwrap_table,
pycylon_wrap_table,
pycylon_unwrap_csv_read_options,
pycylon_unwrap_csv_write_options)

from pycylon.data.aggregates cimport (Sum, Count, Min, Max)
from pycylon.data.aggregates cimport CGroupByAggregationOp
from pycylon.data.aggregates import AggregationOp
from pycylon.data.groupby cimport (GroupBy, PipelineGroupBy)
from pycylon.data.compute cimport (c_filter)

import math
import pyarrow as pa
import numpy as np
import pandas as pd
from typing import List
import warnings
import operator
from enum import Enum

'''
Cylon Table definition mapping 
'''


cdef class Table:
    def __cinit__(self, pyarrow_table=None, context=None, columns=None):
        """
        PyClon constructor
        @param pyarrow_table: PyArrow Table
        @param context: PyCylon Context
        @param columns: columns TODO: add support
        """
        cdef shared_ptr[CArrowTable] c_arrow_tb_shd_ptr
        if self._is_pycylon_context(context) and self._is_pyarrow_table(pyarrow_table):
            c_arrow_tb_shd_ptr = pyarrow_unwrap_table(pyarrow_table)
            self.sp_context = pycylon_unwrap_context(context)
            self.table_shd_ptr = make_shared[CTable](c_arrow_tb_shd_ptr, self.sp_context)

    cdef void init(self, const shared_ptr[CTable]& table):
        self.table_shd_ptr = table

    @staticmethod
    def _is_pyarrow_table(pyarrow_table):
        return isinstance(pyarrow_table, pa.Table)

    @staticmethod
    def _is_pycylon_table(pycylon_table):
        return isinstance(pycylon_table, Table)

    @staticmethod
    def _is_pycylon_context(context):
        return isinstance(context, CylonContext)

    def show(self, row1=-1, row2=-1, col1=-1, col2=-1):
        '''
            prints the table in console from the Cylon C++ Table API
            uses row range and column range
            :param row1: starting row number as int
            :param row2: ending row number as int
            :param col1: starting column number as int
            :param col2: ending column number as int
            :return: None
        '''
        if row1 == -1 and row2 == -1 and col1 == -1 and col2 == -1:
            self.table_shd_ptr.get().Print()
        else:
            self.table_shd_ptr.get().Print(row1, row2, col1, col2)

    def sort(self, index) -> Table:
        cdef shared_ptr[CTable] output
        sort_index = -1
        if isinstance(index, str):
            sort_index = self._resolve_column_index_from_column_name(index)
        else:
            sort_index = index

        cdef CStatus status = Sort(self.table_shd_ptr, sort_index, output)
        if status.is_ok():
            return pycylon_wrap_table(output)
        else:
            raise Exception(f"Table couldn't be sorted: {status.get_msg().decode()}")

    def clear(self):
        """
        Clear PyCylon table
        """
        self.table_shd_ptr.get().Clear()

    def retain_memory(self, retain):
        """
        Retain  memory for PyCylon table
        @param retain: bool
        """
        self.table_shd_ptr.get().retainMemory(retain)

    def is_retain(self) -> bool:
        """
        Checks if memory is retained in PyCylon Table
        """
        self.table_shd_ptr.get().IsRetain()

    @staticmethod
    def merge(ctx, tables: List[Table]) -> Table:
        """
        Merging Two PyCylon tables
        @param ctx: PyCylon context
        @param tables: PyCylon table
        @return: PyCylon table
        """
        cdef vector[shared_ptr[CTable]] ctables
        cdef shared_ptr[CTable] curTable
        cdef shared_ptr[CTable] output
        cdef CStatus status
        cdef shared_ptr[CCylonContext] sp_ctx = pycylon_unwrap_context(ctx)
        if tables:
            for table in tables:
                curTable = pycylon_unwrap_table(table)
                ctables.push_back(curTable)
            status = Merge(sp_ctx, ctables, output)
            if status.is_ok():
                return pycylon_wrap_table(output)
            else:
                raise Exception(f"Tables couldn't be merged: {status.get_msg().decode()}")
        else:
            raise ValueError("Tables are not parsed for merge")

    @property
    def column_names(self):
        """
        Produces column names for PyCylon Table
        @return:
        """
        return pyarrow_wrap_table(pyarrow_unwrap_table(self.to_arrow())).column_names

    def _resolve_column_index_from_column_name(self, column_name) -> int:
        index = None
        for idx, col_name in enumerate(self.column_names):
            if column_name == col_name:
                return idx
        if index is None:
            raise ValueError(f"Column {column_name} does not exist in the table")

    @property
    def column_count(self) -> int:
        """
        Produces column count
        @return: int
        """
        return self.table_shd_ptr.get().Columns()

    @property
    def row_count(self) -> int:
        """
        Produces row count
        @return: int
        """
        return self.table_shd_ptr.get().Rows()

    @property
    def context(self) -> CylonContext:
        """
        Get the CylonContext from PyCylon Table
        @rtype: CylonContext
        """
        return pycylon_wrap_context(self.table_shd_ptr.get().GetContext())

    @property
    def column_names(self) -> List[str]:
        """
        Produces column names
        @return: list
        """
        column_names = []
        cdef vector[string] c_column_names = self.table_shd_ptr.get().ColumnNames()
        for col_name in c_column_names:
            column_names.append(col_name.decode())
        return column_names

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

    def __get_join_config(self, join_type: str, join_algorithm: str, left_column_index: int,
                          right_column_index: int):
        if left_column_index is None or right_column_index is None:
            raise Exception("Join Column index not provided")

        if join_algorithm is None:
            join_algorithm = PJoinAlgorithm.HASH.value

        if join_algorithm == PJoinAlgorithm.HASH.value:

            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index,
                                             right_column_index, CJoinAlgorithm.CHASH)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index,
                                             right_column_index, CJoinAlgorithm.CHASH)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index,
                                             right_column_index, CJoinAlgorithm.CHASH)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index,
                                             right_column_index, CJoinAlgorithm.CHASH)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))

        elif join_algorithm == PJoinAlgorithm.SORT.value:

            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index,
                                             right_column_index, CJoinAlgorithm.CSORT)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index,
                                             right_column_index, CJoinAlgorithm.CSORT)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index,
                                             right_column_index, CJoinAlgorithm.CSORT)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index,
                                             right_column_index, CJoinAlgorithm.CSORT)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))
        else:
            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.CINNER, left_column_index,
                                             right_column_index)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CLEFT, left_column_index, right_column_index)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.CRIGHT, left_column_index,
                                             right_column_index)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.COUTER, left_column_index,
                                             right_column_index)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))

    cdef shared_ptr[CTable] init_join_ra_params(self, table, join_type, algorithm, kwargs):
        left_cols, right_cols = self._get_join_column_indices(table=table, **kwargs)

        # Cylon only supports join by one column and retrieve first left and right column when
        # resolving join configs
        self.__get_join_config(join_type=join_type, join_algorithm=algorithm,
                               left_column_index=left_cols[0],
                               right_column_index=right_cols[0])
        cdef shared_ptr[CTable] right = pycylon_unwrap_table(table)
        return right

    cdef _get_join_ra_response(self, op_name, shared_ptr[CTable] output, CStatus status):
        if status.is_ok():
            return pycylon_wrap_table(output)
        else:
            raise ValueError(f"{op_name} operation failed: : {status.get_msg().decode()}")

    cdef _get_ra_response(self, table, ra_op_name):
        cdef shared_ptr[CTable] output
        cdef shared_ptr[CTable] right = pycylon_unwrap_table(table)
        cdef CStatus status
        # TODO: add callable for Cython functions via FPointers

        if ra_op_name == 'union':
            status = Union(self.table_shd_ptr, right, output)
        elif ra_op_name == 'distributed_union':
            status = DistributedUnion(self.table_shd_ptr, right, output)
        elif ra_op_name == 'intersect':
            status = Intersect(self.table_shd_ptr, right, output)
        elif ra_op_name == 'distributed_intersect':
            status = DistributedIntersect(self.table_shd_ptr, right, output)
        elif ra_op_name == 'subtract':
            status = Subtract(self.table_shd_ptr, right, output)
        elif ra_op_name == 'distributed_subtract':
            status = DistributedSubtract(self.table_shd_ptr, right, output)
        else:
            raise ValueError(f"Unsupported relational algebra operator: {ra_op_name}")

        if status.is_ok():
            return pycylon_wrap_table(output)
        else:
            raise ValueError(f"{ra_op_name} operation failed : {status.get_msg().decode()}")

    def join(self, table: Table, join_type: str,
             algorithm: str, **kwargs) -> Table:
        '''
        Joins two PyCylon tables
        :param table: PyCylon table on which the join is performed (becomes the left table)
        :param join_type: Join Type as str ["inner", "left", "right", "outer"]
        :param algorithm: Join Algorithm as str ["hash", "sort"]
        :kwargs left_on: Join column of the left table as List[int] or List[str], right_on:
        Join column of the right table as List[int] or List[str], on: Join column in common with
        both tables as a List[int] or List[str].
        :return: Joined PyCylon table
        '''
        cdef shared_ptr[CTable] output
        cdef shared_ptr[CTable] right = self.init_join_ra_params(table, join_type, algorithm,
                                                                 kwargs)
        cdef CJoinConfig *jc1 = self.jcPtr
        cdef CStatus status = Join(self.table_shd_ptr, right, jc1[0], output)
        return self._get_join_ra_response("Join", output, status)

    def distributed_join(self, table: Table, join_type: str,
                         algorithm: str, **kwargs) -> Table:
        '''
         Joins two PyCylon tables in distributed memory
        :param table: PyCylon table on which the join is performed (becomes the left table)
        :param join_type: Join Type as str ["inner", "left", "right", "outer"]
        :param algorithm: Join Algorithm as str ["hash", "sort"]
        :kwargs left_on: Join column of the left table as List[int] or List[str], right_on:
        Join column of the right table as List[int] or List[str], on: Join column in common with
        both tables as a List[int] or List[str].
        :return: Joined PyCylon table
        '''
        cdef shared_ptr[CTable] output
        cdef shared_ptr[CTable] right = self.init_join_ra_params(table, join_type, algorithm,
                                                                 kwargs)
        cdef CJoinConfig *jc1 = self.jcPtr
        cdef CStatus status = DistributedJoin(self.table_shd_ptr, right, jc1[0], output)
        return self._get_join_ra_response("Distributed Join", output, status)

    def union(self, table: Table) -> Table:
        '''
        Union two PyCylon tables
        :param table: PyCylon table on which the union is performed (becomes the left table)
        :return: PyCylon table
        '''
        return self._get_ra_response(table, 'union')

    def distributed_union(self, table: Table) -> Table:
        '''
        Union two PyCylon tables in distributed memory
        :param table: PyCylon table on which the union is performed (becomes the left table)
        :return: PyCylon table
        '''
        return self._get_ra_response(table, 'distributed_union')

    def subtract(self, table: Table) -> Table:
        '''
        Subtract two PyCylon tables
        :param table: PyCylon table on which the subtract is performed (becomes the left table)
        :return: PyCylon table
        '''
        return self._get_ra_response(table, 'subtract')

    def distributed_subtract(self, table: Table) -> Table:
        '''
        Subtract two PyCylon tables in distributed memory
        :param table: PyCylon table on which the subtract is performed (becomes the left table)
        :return: PyCylon table
        '''
        return self._get_ra_response(table, 'distributed_subtract')

    def intersect(self, table: Table) -> Table:
        '''
        Intersect two PyCylon tables
        :param table: PyCylon table on which the intersect is performed (becomes the left table)
        :return: PyCylon table
        '''
        return self._get_ra_response(table, 'intersect')

    def distributed_intersect(self, table: Table) -> Table:
        '''
        Intersect two PyCylon tables in distributed memory
        :param table: PyCylon table on which the join is performed (becomes the left table)
        :return: PyCylon table
        '''
        return self._get_ra_response(table, 'distributed_intersect')

    def project(self, columns: List):
        '''
        Project a PyCylon table
        :param columns: List of columns to be projected
        :return: PyCylon table
        '''
        cdef vector[long] c_columns
        cdef shared_ptr[CTable] output
        cdef CStatus status
        if columns:
            if isinstance(columns[0], int) or isinstance(columns[0], str):
                for column in columns:
                    if isinstance(column, str):
                        column = self._resolve_column_index_from_column_name(column)
                    c_columns.push_back(column)
                status = Project(self.table_shd_ptr, c_columns, output)
                if status.is_ok():
                    return pycylon_wrap_table(output)
                else:
                    raise ValueError(f"Project operation failed : {status.get_msg().decode()}")
            else:
                raise ValueError("Invalid column list, it must be column names in string or "
                                 "column indices in int")
        else:
            raise ValueError("Columns not passed.")

    def _agg_op(self, column, op):
        cdef shared_ptr[CTable] output
        cdef CStatus status
        agg_index = -1
        if isinstance(column, str):
            agg_index = self._resolve_column_index_from_column_name(column)
        elif isinstance(column, int):
            agg_index = column
        else:
            raise ValueError("column must be str or int")

        if op == AggregationOp.SUM:
            status = Sum(self.table_shd_ptr, agg_index, output)
        elif op == AggregationOp.COUNT:
            status = Count(self.table_shd_ptr, agg_index, output)
        elif op == AggregationOp.MIN:
            status = Min(self.table_shd_ptr, agg_index, output)
        elif op == AggregationOp.MAX:
            status = Max(self.table_shd_ptr, agg_index, output)
        else:
            raise ValueError(f"Unsupported aggregation type {op}")

        if status.is_ok():
            return pycylon_wrap_table(output)
        else:
            raise Exception(f"Aggregate op {op.name} failed: {status.get_msg().decode()}")

    def sum(self, column):
        return self._agg_op(column, AggregationOp.SUM)

    def count(self, column):
        return self._agg_op(column, AggregationOp.COUNT)

    def min(self, column):
        return self._agg_op(column, AggregationOp.MIN)

    def max(self, column):
        return self._agg_op(column, AggregationOp.MAX)

    def groupby(self, index_col: int, aggregate_cols: List,
                aggregate_ops: List[AggregationOp]):
        cdef CStatus status
        cdef shared_ptr[CTable] output
        cdef vector[long] caggregate_cols
        cdef vector[CGroupByAggregationOp] caggregate_ops

        if not aggregate_cols and not aggregate_ops:
            raise ValueError("Aggregate columns and Aggregate operations cannot be empty")
        else:
            # set aggregate col to c-vector
            for aggregate_col in aggregate_cols:
                col_idx = -1
                if isinstance(aggregate_col, str):
                    col_idx = self._resolve_column_index_from_column_name(aggregate_col)
                elif isinstance(aggregate_col, int):
                    col_idx = aggregate_col
                else:
                    raise ValueError("Aggregate column must be either column name (str) or column "
                                     "index (int)")
                caggregate_cols.push_back(col_idx)

            for aggregate_op in aggregate_ops:
                caggregate_ops.push_back(aggregate_op)

            status = GroupBy(self.table_shd_ptr, index_col, caggregate_cols, caggregate_ops,
                             output)
            if status.is_ok():
                return pycylon_wrap_table(output)
            else:
                raise Exception(f"Groupby operation failed {status.get_msg().decode()}")

    @staticmethod
    def from_arrow(context, pyarrow_table) -> Table:
        '''
            creating a PyCylon table from PyArrow Table
            :param obj: PyArrow table
            :return: PyCylon table
        '''
        cdef shared_ptr[CCylonContext] ctx = pycylon_unwrap_context(context)
        cdef shared_ptr[CArrowTable] arw_table = pyarrow_unwrap_table(pyarrow_table)
        cdef shared_ptr[CTable] cn_table
        cdef CStatus status = CTable.FromArrowTable(ctx, arw_table, cn_table)

        if status.is_ok():
            return pycylon_wrap_table(cn_table)
        else:
            raise Exception(
                f"Table couldn't be created from PyArrow Table: {status.get_msg().decode()}")

    @staticmethod
    def from_numpy(context: CylonContext, col_names: List[str], ar_list: List[np.ndarray]) -> Table:
        """
        creating a PyCylon table from numpy arrays
        @param context: CylonContext
        @param col_names: Column names
        @param ar_list: numpy arrays as a list to form the column data of all columns
        @return: PyCylon table
        """
        return Table.from_arrow(context, pa.Table.from_arrays(ar_list, names=col_names))

    @staticmethod
    def from_list(context: CylonContext, col_names: List[str], data_list: List) -> Table:
        """
        creating a PyCylon table from a list
        @param context: CylonContext
        @param col_names: Column names
        @param data_list: data as a list
        @return: PyCylon table
        """
        ar_list = []
        if len(col_names) == len(data_list):
            for data in data_list:
                ar_list.append(data)
            return Table.from_arrow(context, pa.Table.from_arrays(ar_list, names=col_names))
        else:
            raise ValueError("Column Names count doesn't match data columns count")

    @staticmethod
    def from_pydict(context: CylonContext, dictionary: dict) -> Table:
        """
        creating a PyCylon table from a dictionary
        @param context: CylonContext
        @param dictionary: dict with table data
        @return: PyCylon table
        """
        return Table.from_arrow(context, pa.Table.from_pydict(dictionary))

    @staticmethod
    def from_pandas(context: CylonContext = None, df: pd.DataFrame = None, preserve_index=False,
                    nthreads=None, columns=None, safe=False) -> Table:
        """
            creating a PyCylon table from Pandas DataFrame
            :param obj: Pandas DataFrame
            :rtype: PyCylon Table
        """
        return Table.from_arrow(context,
                                pa.Table.from_pandas(df=df, schema=None,
                                                     preserve_index=preserve_index,
                                                     nthreads=nthreads, columns=columns, safe=safe)
                                )

    def to_pandas(self):
        """
         creating Pandas Dataframe from PyCylon Table
         :param self:
         :return: a Pandas DataFrame
         """
        return self.to_arrow().to_pandas()

    def to_numpy(self, order: str = 'F'):
        """
         [Experimental]
         This method converts a Cylon Table to a 2D numpy array.
         In the conversion we stack each column in the Table and create a numpy array.
         For Heterogeneous Tables, use the generated array with Caution.
         :param order: numpy array order. 'F': Fortran Style F_Contiguous or 'C' C Style C_Contiguous
         :return: ndarray
         """
        ar_lst = []
        _dtype = None
        for col in self.to_arrow().combine_chunks().columns:
            npr = col.chunks[0].to_numpy()
            if None == _dtype:
                _dtype = npr.dtype
            if _dtype != npr.dtype:
                warnings.warn(
                    "Heterogeneous Cylon Table Detected!. Use Numpy operations with Caution.")
            ar_lst.append(npr)
        return np.array(ar_lst, order=order).T

    def to_pydict(self):
        """
        creating a dictionary from PyCylon table
        @return: dict
        """
        return self.to_arrow().to_pydict()

    def to_csv(self, path, csv_write_options):
        """
        creating a csv file with PyCylon table data
        @param path: str
        @param csv_write_options: CSVWriteOptions
        """
        cdef string cpath = path.encode()
        cdef CCSVWriteOptions c_csv_write_options = pycylon_unwrap_csv_write_options(
            csv_write_options)
        self.table_shd_ptr.get().WriteCSV(cpath, c_csv_write_options)

    def to_arrow(self) -> pa.Table:
        '''
         creating PyArrow Table from PyCylon table
         :param self: PyCylon Table
         :return: PyArrow Table
         '''
        cdef shared_ptr[CArrowTable] converted_tb
        cdef CStatus status = self.table_shd_ptr.get().ToArrowTable(converted_tb)
        if status.is_ok():
            return pyarrow_wrap_table(converted_tb)
        else:
            raise Exception(
                f"Table couldn't be converted to a PyArrow Table : {status.get_msg().decode()}")

    @property
    def column_names(self):
        """
        Produces column names from PyCylon Table
        @return: list
        """
        return self.to_arrow().column_names

    # @property
    # def schema(self):
    #     """
    #     Produces schema from PyCylon Table
    #     @return: schema
    #     """
    #     pass
    def filter(self, statement):
        # TODO: Supported Added via: https://github.com/cylondata/cylon/issues/211
        return statement


    def _table_from_mask(self, mask: Table) -> Table:
        mask_dict = mask.to_pydict()
        list_of_mask_values = list(mask_dict.values())
        if len(list_of_mask_values) == 1:
            # Handle when masking is done based on a single column data
            filter_row_indices = []
            for idx, mask_value in enumerate(list_of_mask_values[0]):
                if mask_value:
                    filter_row_indices.append(idx)
            filtered_rows = []
            for row_id in filter_row_indices:
                filtered_rows.append(self[row_id])
            return Table.merge(self.context, filtered_rows)
        else:
            # Handle when masking is done on whole table
            filtered_all_data = []
            table_dict = self.to_pydict()
            list_of_table_values = list(table_dict.values())
            for mask_col_data,  table_col_data in zip(list_of_mask_values, list_of_table_values):
                filtered_data = []
                for mask_value, table_value in zip(mask_col_data, table_col_data):
                    if mask_value:
                        filtered_data.append(table_value)
                    else:
                        filtered_data.append(math.nan)
                filtered_all_data.append(filtered_data)
            return Table.from_list(self.context, self.column_names, filtered_all_data)


    def _aggregate_filters(self, filter: Table, op) -> Table:
        filter1_dict = filter.to_pydict()
        filter2_dict = self.to_pydict()
        aggregated_filter_response = []
        for key1, key2 in zip(filter1_dict, filter2_dict):
            values1, values2 = filter1_dict[key1], filter2_dict[key2]
            column_data = []
            for value1, value2 in zip(values1, values2):
                column_data.append(op(value1, value2))
            aggregated_filter_response.append(column_data)
        return Table.from_list(self.context, self.column_names, aggregated_filter_response)


    def __getitem__(self, key) -> Table:
        py_arrow_table = self.to_arrow().combine_chunks()
        if isinstance(key, slice):
            return self.from_arrow(self.context, py_arrow_table.slice(key.start, key.stop))
        elif isinstance(key, int):
            return self.from_arrow(self.context, py_arrow_table.slice(key, 1))
        elif isinstance(key, str):
            index = self._resolve_column_index_from_column_name(key)
            chunked_arr = py_arrow_table.column(index)
            return self.from_arrow(self.context, pa.Table.from_arrays([chunked_arr.chunk(0)],
                                                                      [key]))
        elif self._is_pycylon_table(key):
            return self._table_from_mask(key)
        else:
            raise ValueError(f"Unsupported Key Type in __getitem__ {type(key)}")


    def _comparison_operation(self, other, op):
        selected_data = []
        for col in self.to_arrow().combine_chunks().columns:
            col_data = []
            for val in col.chunks[0]:
                col_data.append(op(val.as_py(), other))
            selected_data.append(col_data)
        return Table.from_list(self.context, self.column_names, selected_data)


    def __eq__(self, other) -> Table:
        return self._comparison_operation(other, operator.__eq__)

    def __ne__(self, other) -> Table:
        return self._comparison_operation(other, operator.__ne__)

    def __lt__(self, other) -> Table:
        return self._comparison_operation(other, operator.__lt__)

    def __gt__(self, other) -> Table:
        return self._comparison_operation(other, operator.__gt__)

    def __le__(self, other) -> Table:
        return self._comparison_operation(other, operator.__le__)

    def __ge__(self, other) -> Table:
        return self._comparison_operation(other, operator.__ge__)

    def __or__(self, other) -> Table:
        return self._aggregate_filters(other, operator.__or__)

    def __and__(self, other) -> Table:
        return self._aggregate_filters(other, operator.__and__)
