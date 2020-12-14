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
pycylon_unwrap_csv_write_options,
pycylon_unwrap_sort_options)

from pycylon.data.aggregates cimport (Sum, Count, Min, Max)
from pycylon.data.aggregates cimport CGroupByAggregationOp
from pycylon.data.aggregates import AggregationOp
from pycylon.data.groupby cimport (GroupBy, PipelineGroupBy)
from pycylon.data import compute

from pycylon.index import RangeIndex, NumericIndex, range_calculator, process_index_by_value

import math
import pyarrow as pa
import numpy as np
import pandas as pd
from typing import List, Any
import warnings
import operator

'''
Cylon Table definition mapping 
'''

cdef class Table:
    def __init__(self, pyarrow_table=None, context=None):
        self.initialize(pyarrow_table, context)
        self._index = None

    def __cinit__(self, pyarrow_table=None, context=None, columns=None):
        """
        PyClon constructor
        @param pyarrow_table: PyArrow Table
        @param context: PyCylon Context
        @param columns: columns TODO: add support
        """
        self.initialize(pyarrow_table, context)
        self._index = None

    def initialize(self, pyarrow_table=None, context=None):
        cdef shared_ptr[CArrowTable] c_arrow_tb_shd_ptr
        if self._is_pycylon_context(context) and self._is_pyarrow_table(pyarrow_table):
            c_arrow_tb_shd_ptr = pyarrow_unwrap_table(pyarrow_table)
            self.sp_context = pycylon_unwrap_context(context)
            self.table_shd_ptr = make_shared[CTable](c_arrow_tb_shd_ptr, self.sp_context)

    cdef void init(self, const shared_ptr[CTable]& table):
        self.table_shd_ptr = table
        self._index = None

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

    def distributed_sort(self, sort_column=None, sort_options: SortOptions = None)-> Table:
        '''
        Does a distributed sort on the table by re-partitioning the data to maintain the sort
        order across all processes
        Args:
            sort_column: str or int
            sort_options: SortOption

        Returns: PyCylon Table

        Examples
        --------

        >>> from pycylon.data.table import SortOptions
        >>> s = SortOptions(ascending=True, num_bins=0, num_samples=0)
        >>> tb1.distributed_sort(sort_column='use_id', sort_options=s)

        '''
        cdef shared_ptr[CTable] output
        cdef CSortOptions *csort_options
        col_index = 0
        if isinstance(sort_column, str):
            col_index = self._resolve_column_index_from_column_name(sort_column)
        elif isinstance(sort_column, int):
            col_index = sort_column
        else:
            raise ValueError("Sort column must be column index or column name")

        if sort_options:
            csort_options = pycylon_unwrap_sort_options(sort_options)
        else:
            csort_options = pycylon_unwrap_sort_options(SortOptions(True, 0, 0))
        cdef CStatus status = DistributedSort(self.table_shd_ptr, col_index, output,
                                              csort_options[0])
        if status.is_ok():
            return pycylon_wrap_table(output)
        else:
            raise ValueError(f"Operation failed: : {status.get_msg().decode()}")

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
        Creating a PyCylon table from PyArrow Table
        Args:
            context: pycylon.CylonContext
            pyarrow_table: PyArrow Table

        Returns: PyCylon Table

        Examples
        --------
        >>> atb
            pyarrow.Table
            col-1: int64
            col-2: int64
            col-3: int64

        >>> Table.from_arrow(ctx, atb)
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12
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
        '''
        Creating a PyCylon table from numpy arrays
        Args:
            context: pycylon.CylonContext
            col_names: column names as a List
            ar_list: Numpy ndarrays as a list (one 1D array per column)

        Returns: PyCylon Table

        Examples
        --------

        >>> Table.from_numpy(ctx, ['c1', 'c2', 'c3'], [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])])
               c1  c2  c3
            0   1   5   9
            1   2   6  10
            2   3   7  11
            3   4   8  12
        '''

        return Table.from_arrow(context, pa.Table.from_arrays(ar_list, names=col_names))

    @staticmethod
    def from_list(context: CylonContext, col_names: List[str], data_list: List) -> Table:
        '''
        Creating a PyCylon table from a list
        Args:
            context: pycylon.CylonContext
            col_names: Column names as a List[str]
            data_list: data as a List of List, (List per column)

        Returns: PyCylon Table

        Examples
        --------

        >>> Table.from_list(ctx, ['col-1', 'col-2', 'col-3'], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12
        '''
        ar_list = []
        if len(col_names) == len(data_list):
            for data in data_list:
                ar_list.append(data)
            return Table.from_arrow(context, pa.Table.from_arrays(ar_list, names=col_names))
        else:
            raise ValueError("Column Names count doesn't match data columns count")

    @staticmethod
    def from_pydict(context: CylonContext, dictionary: dict) -> Table:
        '''
        Creating a PyCylon table from a dictionary
        Args:
            context: pycylon.CylonContext
            dictionary: dict object with key as column names and values as a List

        Returns: PyCylon Table

        Examples
        --------

        >>> Table.from_pydict(ctx, {'col-1': [1, 2, 3, 4], 'col-2': [5, 6, 7, 8], 'col-3': [9, 10, 11, 12]})
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        '''
        return Table.from_arrow(context, pa.Table.from_pydict(dictionary))

    @staticmethod
    def from_pandas(context: CylonContext = None, df: pd.DataFrame = None, preserve_index=False,
                    nthreads=None, columns=None, safe=False) -> Table:
        '''
        Creating a PyCylon table from Pandas DataFrame
        Args:
            context: cylon.CylonContext
            df: pd.DataFrame
            preserve_index: keep indexes as same as in original DF
            nthreads: number of threads for the operation
            columns: column names, if updated
            safe: safe operation

        Returns: PyCylon Table

        Examples
        --------

        >>> Table.from_pandas(ctx, df)
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12
        '''
        return Table.from_arrow(context,
                                pa.Table.from_pandas(df=df, schema=None,
                                                     preserve_index=preserve_index,
                                                     nthreads=nthreads, columns=columns, safe=safe)
                                )

    def to_pandas(self):
        '''
        Creating Pandas Dataframe from PyCylon Table
        Returns: pd.DataFrame

        '''
        return self.to_arrow().to_pandas()

    def to_numpy(self, order: str = 'F', zero_copy_only: bool = True, writable: bool = False):
        '''
        [Experimental]
         This method converts a Cylon Table to a 2D numpy array.
         In the conversion we stack each column in the Table and create a numpy array.
         For Heterogeneous Tables, use the generated array with Caution.
         :param order:
        Args:
            order: numpy array order. 'F': Fortran Style F_Contiguous or 'C' C Style C_Contiguous
            zero_copy_only: bool to enable zero copy, and default is True
            writable: config writable

        Returns: Numpy NDArray
        '''
        ar_lst = []
        _dtype = None
        for col in self.to_arrow().combine_chunks().columns:
            npr = col.chunks[0].to_numpy(zero_copy_only=zero_copy_only, writable=writable)
            if None == _dtype:
                _dtype = npr.dtype
            if _dtype != npr.dtype:
                warnings.warn(
                    "Heterogeneous Cylon Table Detected!. Use Numpy operations with Caution.")
            ar_lst.append(npr)
        npy = np.array(ar_lst).T
        array = np.asfortranarray(npy) if order == 'F' else np.ascontiguousarray(npy)
        return array

    def to_pydict(self):
        '''
        Creating a dictionary from PyCylon table
        Returns: dict object

        '''
        return self.to_arrow().to_pydict()

    def to_csv(self, path, csv_write_options):
        '''
        Creating a csv file with PyCylon table data
        Args:
            path: path to file
            csv_write_options: pycylon.io.CSVWriteOptions

        Returns: None

        Examples
        --------
        >>> from pycylon.io import CSVWriteOptions
        >>> csv_write_options = CSVWriteOptions().with_delimiter(',')
        >>> tb.to_csv('/tmp/data.csv', csv_write_options)


        '''
        cdef string cpath = path.encode()
        cdef CCSVWriteOptions c_csv_write_options = pycylon_unwrap_csv_write_options(
            csv_write_options)
        WriteCSV(self.table_shd_ptr, cpath, c_csv_write_options)

    def to_arrow(self) -> pa.Table:
        '''
         Creating PyArrow Table from PyCylon table
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

    @property
    def shape(self):
        return self.to_arrow().shape

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
        '''
        Creates a PyCylon Table from a mask of type PyCylon Table.
        Args:
            mask: PyCylon Table

        Returns: PyCylon Table

        '''
        mask_batches = mask.to_arrow().combine_chunks().to_batches()

        if mask.column_count == 1:
            # Handle when masking is done based on a single column data
            return self.from_arrow(self.context, self.to_arrow().filter(mask_batches[0][0]))
        else:
            # Handle when masking is done on whole table
            filtered_all_data = []
            table_record_batches = self.to_arrow().combine_chunks().to_batches()
            for mask_batch, table_batch in zip(mask_batches[0], table_record_batches[0]):
                filtered_data = []
                for mask_value, table_value in zip(mask_batch, table_batch):
                    if mask_value.as_py():
                        filtered_data.append(table_value.as_py())
                    else:
                        filtered_data.append(math.nan)
                filtered_all_data.append(filtered_data)
            return Table.from_list(self.context, self.column_names, filtered_all_data)

    def _aggregate_filters(self, filter: Table, op) -> Table:
        return compute.table_compute_ar_op(self, filter, op)

    def __getitem__(self, key) -> Table:
        """
        This method allows to retrieve a subset of a Table by means of a key
        Args:
            key: a key can be the following
                 1. slice i.e table[1:5], rows 1:5
                 2. int i.e a row index
                 3. str i.e extract the data column-wise by column-name
                 4. List of columns are extracted
                 5. PyCylon Table
        Returns: PyCylon Table

        Examples
        --------
        >>> ctx: CylonContext = CylonContext(config=None, distributed=False)
        >>> data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        >>> columns = ['col-1', 'col-2', 'col-3']

        >>> tb: Table = Table.from_list(ctx, columns, data)

        >>> tb1 = tb[1:3]
            col-1  col-2  col-3
                0      2      6     10
                1      3      7     11
                2      4      8     12

        >>> tb2 = tb['col-1']
               col-1
            0      1
            1      2
            2      3
            3      4

        >>> tb3 = tb[['col-1', 'col-2']]
               col-1  col-2
            0      1      5
            1      2      6
            2      3      7
            3      4      8

        >>> tb4 = tb > 3
                 col-1  col-2  col-3
            0    False   True   True
            1    False   True   True
            2    False   True   True
            3     True   True   True

        >>> tb5 = tb[tb4]
                col-1  col-2  col-3
            0    NaN      5      9
            1    NaN      6     10
            2    NaN      7     11
            3    4.0      8     12

        >>> tb8 = tb['col-1'] > 2
               col-1  col-2  col-3
            0      3      7     11
            1      4      8     12

        """
        py_arrow_table = self.to_arrow().combine_chunks()
        if isinstance(key, slice):
            return self.from_arrow(self.context, py_arrow_table.slice(key.start, key.stop))
        elif isinstance(key, int):
            return self.from_arrow(self.context, py_arrow_table.slice(key, 1))
        elif isinstance(key, str):
            index = self._resolve_column_index_from_column_name(key)
            chunked_arr = py_arrow_table.column(index)
            tb_filtered = self.from_arrow(self.context, pa.Table.from_arrays([chunked_arr.chunk(0)],
                                                                             [key]))
            return tb_filtered
        elif isinstance(key, List):
            chunked_arrays = []
            selected_columns = []
            column_headers = self.column_names
            for column_name in key:
                index = -1
                if isinstance(column_name, str):
                    index = self._resolve_column_index_from_column_name(column_name)
                elif isinstance(column_name, int):
                    index = key
                chunked_arrays.append(py_arrow_table.column(index).chunk(0))
                selected_columns.append(column_headers[index])
            return self.from_arrow(self.context, pa.Table.from_arrays(chunked_arrays,
                                                                      selected_columns))
        elif self._is_pycylon_table(key):
            return self._table_from_mask(key)
        else:
            raise ValueError(f"Unsupported Key Type in __getitem__ {type(key)}")

    def __setitem__(self, key, value):
        '''
        Sets values for a existing table by means of a column
        Args:
            key: (str) column-name
            value: (Table) data as a single column table

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12


        >>> tb['col-3'] = Table.from_list(ctx, ['x'], [[90, 100, 110, 120]])
               col-1  col-2  col-3
            0      1      5     90
            1      2      6    100
            2      3      7    110
            3      4      8    120

        >>> tb['col-4'] = Table.from_list(ctx, ['x'], [[190, 1100, 1110, 1120]])
                col-1  col-2  col-3  col-4
            0      1      5     90    190
            1      2      6    100   1100
            2      3      7    110   1110
            3      4      8    120   1120
        '''

        if isinstance(key, str) and isinstance(value, Table):
            if value.column_count == 1:
                value_arrow_table = value.to_arrow().combine_chunks()
                chunk_arr = value_arrow_table.columns[0].chunks[0]
                current_ar_table = self.to_arrow()
                if key in self.column_names:
                    index = self._resolve_column_index_from_column_name(key)
                    # A new Column is replacing an existing column
                    self.initialize(current_ar_table.set_column(index, key, chunk_arr),
                                    self.context)
                else:
                    self.initialize(current_ar_table.append_column(key, chunk_arr),
                                    self.context)
        else:
            raise ValueError(f"Not Implemented __setitem__ option for key Type {type(key)} and "
                             f"value type {type(value)}")

    def _comparison_operation(self, other, op):
        return compute.table_compute_ar_op(self, other, op)

    def __eq__(self, other) -> Table:
        '''
        Equal operator for Table
        Args:
            other: can be a numeric scalar or a Table

        Returns: PyCylon Table

        Examples
        --------

        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb['col-1'] == 2
               col-1
            0  False
            1   True
            2  False
            3  False

        >>> tb == 2
               col-1  col-2  col-3
            0  False  False  False
            1   True  False  False
            2  False  False  False
            3  False  False  False

        '''
        return self._comparison_operation(other, operator.__eq__)

    def __ne__(self, other) -> Table:
        '''
        Not equal operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb3 = tb['col-1'] != 2
               col-1
            0   True
            1  False
            2   True
            3   True

        >>> tb4 = tb != 2
               col-1  col-2  col-3
            0   True   True   True
            1  False   True   True
            2   True   True   True
            3   True   True   True
        '''

        return self._comparison_operation(other, operator.__ne__)

    def __lt__(self, other) -> Table:
        '''
        Lesser than operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb3 = tb['col-1'] < 2
               col-1
            0   True
            1  False
            2  False
            3  False

        >>> tb4 = tb < 2
               col-1  col-2  col-3
            0   True  False  False
            1  False  False  False
            2  False  False  False
            3  False  False  False
        '''

        return self._comparison_operation(other, operator.__lt__)

    def __gt__(self, other) -> Table:
        '''
        Greater than operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb3 = tb['col-1'] > 2
                col-1
            0  False
            1  False
            2   True
            3   True

        >>> tb4 = tb > 2
               col-1  col-2  col-3
            0  False   True   True
            1  False   True   True
            2   True   True   True
            3   True   True   True
        '''

        return self._comparison_operation(other, operator.__gt__)

    def __le__(self, other) -> Table:
        '''
        Lesser than or equal operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb3 = tb['col-1'] <= 2
                col-1
            0   True
            1   True
            2  False
            3  False

        >>> tb4 = tb <= 2
               col-1  col-2  col-3
            0   True  False  False
            1   True  False  False
            2  False  False  False
            3  False  False  False
        '''
        return self._comparison_operation(other, operator.__le__)

    def __ge__(self, other) -> Table:
        '''
        Greater than or equal operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12


        >>> tb3 = tb['col-1'] >= 2
               col-1
            0  False
            1   True
            2   True
            3   True

        >>> tb4 = tb >= 2
               col-1  col-2  col-3
            0  False   True   True
            1   True   True   True
            2   True   True   True
            3   True   True   True
        '''

        return self._comparison_operation(other, operator.__ge__)

    def __or__(self, other) -> Table:
        '''
        Or operator for Table
        Args:
            other: PyCylon Table

        Returns: PyCylon Table

        Examples
        --------
        >>> tb1
               col-1  col-2
            0  False   True
            1   True   True
            2  False  False
            3   True  False

        >>> tb2
                col-1  col-2
            0   True  False
            1   True   True
            2  False  False
            3  False   True

        >>> tb_or = tb1 | tb2
               col-1  col-2
            0   True   True
            1   True   True
            2  False  False
            3   True   True
        '''

        return self._aggregate_filters(other, operator.__or__)

    def __and__(self, other) -> Table:
        '''
        And operator for Table
        Args:
            other: PyCylon Table

        Returns: PyCylon Table

        Examples
        --------
        >>> tb1
               col-1  col-2
            0  False   True
            1   True   True
            2  False  False
            3   True  False

        >>> tb2
                col-1  col-2
            0   True  False
            1   True   True
            2  False  False
            3  False   True

        >>> tb_or = tb1 & tb2
               col-1  col-2
            0  False  False
            1   True   True
            2  False  False
            3  False  False
        '''

        return self._aggregate_filters(other, operator.__and__)

    def __invert__(self):
        '''
         Invert operator for Table

         Returns: PyCylon Table

         Examples
         --------
         >>> tb
                col-1  col-2
            0  False   True
            1   True   True
            2  False  False
            3   True  False

        >>> ~tb
               col-1  col-2
            0   True  False
            1  False  False
            2   True   True
            3  False   True
         '''

        return compute.invert(self)

    def __neg__(self):
        '''
         Negation operator for Table

         Returns: PyCylon Table

         Examples
         --------
         >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

         >>> -tb
               col-1  col-2  col-3
            0     -1     -5     -9
            1     -2     -6    -10
            2     -3     -7    -11
            3     -4     -8    -12
         '''

        return compute.neg(self)

    def __add__(self, other):
        '''
         Add operator for Table
         Args:
             other: scalar numeric

         Returns: PyCylon Table

         Examples
         --------
        >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb + 2
               col-1  col-2  col-3
            0      3      7     11
            1      4      8     12
            2      5      9     13
            3      6     10     14
         '''
        return compute.add(self, other)

    def __sub__(self, other):
        '''
         Subtract operator for Table
         Args:
             other: scalar numeric

         Returns: PyCylon Table

         Examples
         --------
        >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb - 2
               col-1  col-2  col-3
            0     -1      3      7
            1      0      4      8
            2      1      5      9
            3      2      6     10
         '''
        return compute.subtract(self, other)

    def __mul__(self, other):
        '''
         Multiply operator for Table
         Args:
             other: scalar numeric

         Returns: PyCylon Table

         Examples
         --------
        >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb * 2
               col-1  col-2  col-3
            0      2     10     18
            1      4     12     20
            2      6     14     22
            3      8     16     24
         '''

        return compute.multiply(self, other)

    def __truediv__(self, other):
        '''
         Element-wise division operator for Table
         Args:
             other: scalar numeric

         Returns: PyCylon Table

         Examples
         --------
        >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb / 2
               col-1  col-2  col-3
            0    0.5    2.5    4.5
            1    1.0    3.0    5.0
            2    1.5    3.5    5.5
            3    2.0    4.0    6.0
         '''

        return compute.divide(self, other)

    def __repr__(self):
        return self.to_string()

    def to_string(self, row_limit: int = 10):
        # TODO: Need to improve this method with more features:
        #  https://github.com/cylondata/cylon/issues/219

        row_limit = row_limit if row_limit % 2 == 0 else row_limit + 1
        str1 = self.to_pandas().to_string()
        if self.row_count > row_limit:
            printable_rows = []
            rows = str1.split("\n")
            len_mid_line = len(rows[self.row_count])
            dot_line = ""
            for i in range(len_mid_line):
                dot_line += "."
            dot_line += "\n"
            printable_rows = rows[:row_limit // 2] + [dot_line] + rows[-row_limit // 2:]
            row_strs = ""
            len_row = 0
            for row_id, row_str in enumerate(printable_rows):
                row_strs += row_str + "\n"
            return row_strs
        else:
            return str1

    def drop(self, column_names: List[str]):
        '''
        drop a column or list of columns from a Table
        Args:
            column_names: List[str]

        Returns: PyCylon Table

        Examples
        --------

        >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb.drop(['col-1'])
               col-2  col-3
            0      5      9
            1      6     10
            2      7     11
            3      8     12
        '''

        return self.from_arrow(self.context, self.to_arrow().drop(column_names))

    def fillna(self, fill_value):
        '''
        Fill not applicable values with a given value
        Args:
            fill_value: scalar

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> tb.fillna(0)
               col-1  col-2  col-3
            0      1      5      9
            1      0      6     10
            2      3      0     11
            3      4      8      0
        '''
        # Note: Supports numeric types only
        filtered_arrays = []
        for col in self.to_arrow().combine_chunks().columns:
            for val in col.chunks:
                filtered_arrays.append(val.fill_null(fill_value))
        return self.from_arrow(self.context,
                               pa.Table.from_arrays(filtered_arrays, self.column_names))

    def where(self, condition not None, other=None):
        '''
        Experimental version of Where operation.
        Replace values where condition is False
        Args:
            condition: bool Table
            other: Scalar

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb.where(tb > 2)
                col-1  col-2  col-3
            0    NaN      5      9
            1    NaN      6     10
            2    3.0      7     11
            3    4.0      8     12

        >>> tb.where(tb > 2, 10)
               col-1  col-2  col-3
            0     10      5      9
            1     10      6     10
            2      3      7     11
            3      4      8     12
        '''
        # TODO: need to improve and overlap with filter functions
        filtered_all_data = []
        list_of_mask_values = list(condition.to_pydict().values())
        table_dict = self.to_pydict()
        list_of_table_values = list(table_dict.values())
        for mask_col_data, table_col_data in zip(list_of_mask_values, list_of_table_values):
            filtered_data = []
            for mask_value, table_value in zip(mask_col_data, table_col_data):
                if mask_value:
                    filtered_data.append(table_value)
                else:
                    if other:
                        filtered_data.append(other)
                    else:
                        filtered_data.append(math.nan)
            filtered_all_data.append(filtered_data)
        return Table.from_list(self.context, self.column_names, filtered_all_data)

    def isnull(self):
        '''
        Checks for null elements and returns a bool Table
        Returns: PyCylon Table

        Examples
        --------

        >>> tb
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> tb.isnull()
                col-1  col-2  col-3
            0  False  False  False
            1   True  False  False
            2  False   True  False
            3  False  False   True

        '''
        return compute.is_null(self)

    def isna(self):
        '''
        Check for not applicable values and returns a bool Table
        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> tb.isna()
                col-1  col-2  col-3
            0  False  False  False
            1   True  False  False
            2  False   True  False
            3  False  False   True



        '''
        return compute.is_null(self)

    def notnull(self):
        '''
        Check the not null values and returns a bool Table
        Returns: PyCylon Table

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> tb.notnull()
               col-1  col-2  col-3
            0   True   True   True
            1  False   True   True
            2   True  False   True
            3   True   True  False
        '''

        return ~compute.is_null(self)

    def notna(self):
        '''
        Checks for not NA values and returns a bool Table
        Returns: PyCylon Table

        >>> tb
                col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> tb.notna()
               col-1  col-2  col-3
            0   True   True   True
            1  False   True   True
            2   True  False   True
            3   True   True  False
        '''

        return ~compute.is_null(self)

    def rename(self, column_names):
        '''
        Rename a Table with a column name or column names
        Args:
            column_names: dictionary or full list of new column names

        Returns: PyCylon Table

        Examples
        --------
        >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb.rename({'col-1': 'col_1'})
               col_1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb.rename(['c1', 'c2', 'c3'])
               c1  c2  c3
            0   1   5   9
            1   2   6  10
            2   3   7  11
            3   4   8  12
        '''

        if isinstance(column_names, dict):
            table_col_names = self.column_names
            for key in column_names.keys():
                if key not in table_col_names:
                    raise ValueError("Column name doesn't exist in the table")
                else:
                    table_col_names[table_col_names.index(key)] = column_names[key]
            self.initialize(self.to_arrow().rename_columns(table_col_names), self.context)
        elif isinstance(column_names, list):
            if len(column_names) == self.column_count:
                self.initialize(self.to_arrow().rename_columns(column_names), self.context)
        else:
            raise ValueError("Input Column names must be a dictionary or list")

    def add_prefix(self, prefix: str) -> Table:
        '''
        Adding a prefix to column names
        Args:
            prefix: str

        Returns: PyCylon Table with prefix updated

        Examples
        --------

        >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb.add_prefix('old_')
               old_c1  old_c2  old_c3
            0       1       5       9
            1       2       6      10
            2       3       7      11
            3       4       8      12

        '''
        new_column_names = [prefix + col for col in self.column_names]
        return Table.from_arrow(self.context, self.to_arrow().rename_columns(new_column_names))

    def add_suffix(self, suffix: str) -> Table:
        '''
        Adding a prefix to column names
        Args:
            prefix: str

        Returns: PyCylon Table with prefix updated

        Examples
        --------

        >>> tb
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> tb.add_suffix('_old')
               c1_old  c2_old  c3_old
            0       1       5       9
            1       2       6      10
            2       3       7      11
            3       4       8      12

        '''
        new_column_names = [col + suffix for col in self.column_names]
        return Table.from_arrow(self.context, self.to_arrow().rename_columns(new_column_names))

    def _is_index_and_range_validity(self, index_range):
        if isinstance(index_range, range):
            return range_calculator(index_range) == self.row_count
        else:
            raise ValueError("Invalid object, expected range")

    def _is_index_list_and_valid(self, index):
        if isinstance(index, List):
            return len(index) == self.row_count
        else:
            raise ValueError("Invalid object, expected List")

    def _is_index_list_of_columns(self, index):
        for index_item in index:
            if index_item not in self.column_names:
                return False
        return True

    def _get_index_list_from_columns(self, index):
        # multi-column indexing
        index_columns = []
        ar_tb = self.to_arrow().combine_chunks()
        if isinstance(index, List):
            for index_item in index:
                index_columns.append(ar_tb.column(index_item))
            return index_columns
        elif isinstance(index, str):
            index_columns.append(ar_tb.column(index))
            return index_columns
        else:
            return NotImplemented("Not Supported index pattern")

    def _is_index_str_and_valid(self, index):
        if isinstance(index, str):
            if index in self.column_names:
                return True
        return False

    def _get_column_by_name(self, column_name):
        artb = self.to_arrow().combine_chunks()
        return artb.column(column_name)

    @property
    def index(self):
        '''
        Retrieve index if exists or provide a range index as default
        Returns: Index object

        Examples:

        >>> tb.index
            <pycylon.index.RangeIndex object at 0x7f58bde8e040>

        '''
        if self._index == None:
            self._index = RangeIndex(range(0, self.row_count))
        return self._index

    def set_index(self, key):
        '''
        Set Index
        Args:
            key: pycylon.Index Object or an object extended from pycylon.Index

        Returns: None

        Examples
        --------

        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12


        >>> tb.set_index(['a', 'b', 'c', 'd'])

        >>> tb.index
            <pycylon.index.CategoricalIndex object at 0x7fa72c2b6ca0>

        >>> tb.index.index_values
            ['a', 'b', 'c', 'd']
        '''

        # TODO: Multi-Indexing support: https://github.com/cylondata/cylon/issues/233
        # TODO: Enhancing: https://github.com/cylondata/cylon/issues/235
        self._index = process_index_by_value(key, self)

    def reset_index(self, key) -> Table:
        # TODO: Enhance to support move existing index to data column and drop
        self._index = RangeIndex(range(0, self.row_count))

    def dropna(self, axis=0, how='any', inplace=False):
        '''
        Drop not applicable values from a Table
        Args:
            axis: 0 for column and 1 for row and only do dropping on the specified axis
            how: any or all, any refers to drop if any value is NA and drop only if all values
            are NA in the considered axis
            inplace: do the operation on the existing Table itself when set to True, the default
            is False and it produces a new Table with the drop update

        Returns: PyCylon Table

        Examples
        --------

        >>> tb
               col-1  col-2  col-3
            0    1.0      5    9.0
            1    NaN      6   10.0
            2    3.0      7   11.0
            3    4.0      8    NaN

        >>> tb_na.dropna(how='any')
               col-2
            0      5
            1      6
            2      7
            3      8

        >>> tb_na.dropna(how='all')
               col-1  col-2  col-3
            0    1.0      5    9.0
            1    NaN      6   10.0
            2    3.0      7   11.0
            3    4.0      8    NaN

        >>> tb_na.dropna(axis=1, how='any')
               col-1  col-2  col-3
            0      1      5      9
            1      3      7     11

        >>> tb_na.dropna(axis=1, how='all')
               col-1  col-2  col-3
            0    1.0      5    9.0
            1    NaN      6   10.0
            2    3.0      7   11.0
            3    4.0      8    NaN

        >>> tb_na
               col-1  col-2  col-3
            0    1.0      5    9.0
            1    NaN      6   10.0
            2    3.0      7   11.0
            3    4.0      8    NaN

        >>> tb_na.dropna(axis=1, how='any', inplace=True)
               col-1  col-2  col-3
            0      1      5      9
            1      3      7     11

        >>> tb_na
               col-1  col-2  col-3
            0      1      5      9
            1      3      7     11
        '''

        new_tb = compute.drop_na(self, how, axis)
        if inplace:
            self.initialize(new_tb.to_arrow(), self.context)
        else:
            return new_tb

    def isin(self, value, skip_null=True) -> Table:
        return compute.is_in(self, value, skip_null)


class EmptyTable(Table):
    '''
    Empty Table definition required in returning an Empty Table when an operation reduces a None
    object after an operation on a Table. As a standard, we return an Empty Table to symbolize
    this. This helps to validate an operation followed by this.
    TODO: Add validity for checking Empty Table when executing ops
    '''

    def __init__(self, context: CylonContext, index: RangeIndex):
        self.ctx = context
        self.idx = index
        self._empty_initialize()
        self.set_index(index)

    def _empty_initialize(self):
        empty_data = []
        self.initialize(pa.Table.from_arrays([], []), self.ctx)


cdef class SortOptions:
    '''
    Sort Operations for Distribtued Sort
    '''
    def __cinit__(self, ascending: bool = True, num_bins: int=0, num_samples: int=0):
        '''
        Initializes the CSortOptions struct
        Args:
            ascending: bool
            num_bins: int
            num_samples: int

        Returns: None

        '''
        self.thisPtr = new CSortOptions()
        self.thisPtr.ascending = ascending
        self.thisPtr.num_bins = num_bins
        self.thisPtr.num_samples = num_samples

    cdef void init(self, CSortOptions *csort_options):
        self.thisPtr = csort_options
