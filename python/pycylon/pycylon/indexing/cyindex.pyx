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

from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.string cimport string
from pyarrow.lib cimport CArray as CArrowArray
from pyarrow.lib cimport CScalar as CArrowScalar
from pycylon.indexing.cyindex cimport CIndexingType
# from pycylon.indexing.cyindex cimport CArrowLocIndexer
from pycylon.indexing.cyindex cimport CBaseArrowIndex
from pyarrow.lib cimport (pyarrow_unwrap_table, pyarrow_wrap_table, pyarrow_wrap_array,
pyarrow_unwrap_array, pyarrow_wrap_scalar, pyarrow_unwrap_scalar)

from pycylon.api.lib cimport (pycylon_wrap_context, pycylon_unwrap_context, pycylon_unwrap_table,
pycylon_wrap_table)

from pycylon.data.table cimport CTable
from pycylon.data.table import Table
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
import pyarrow as pa
import numpy as np

from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref, preincrement as inc

from typing import List, Tuple
from pycylon.data.compute import compare_array_like_values

'''
Cylon Indexing is done with the following enums. 
'''
ctypedef long LONG
ctypedef double DOUBLE
ctypedef float FLOAT
ctypedef int INT

cpdef enum IndexingType:
    RANGE = CIndexingType.CRANGE
    LINEAR = CIndexingType.CLINEAR
    HASH = CIndexingType.CHASH
    BINARYTREE = CIndexingType.CBINARYTREE
    BTREE = CIndexingType.CBTREE


def _resolve_column_index_from_column_name(column_name, cn_table) -> int:
    index = None
    if isinstance(column_name, str):
        for idx, col_name in enumerate(cn_table.column_names):
            if column_name == col_name:
                return idx
    elif isinstance(column_name, int):
        assert column_name < cn_table.column_count
        return column_name
    else:
        raise ValueError("column_name must be str or int")


cdef class BaseArrowIndex:
    cdef void init(self, const shared_ptr[CBaseArrowIndex]& index):
        self.bindex_shd_ptr = index

    def get_index_array(self) -> pa.array:
        cdef shared_ptr[CArrowArray] index_arr = self.bindex_shd_ptr.get().GetIndexArray()
        py_arw_index_arr = pyarrow_wrap_array(index_arr)
        return py_arw_index_arr

    def get_type(self) -> IndexingType:
        return IndexingType(self.bindex_shd_ptr.get().GetIndexingType())

    @property
    def index_values(self):
        return self.get_index_array().tolist()

    @property
    def values(self):
        npr = None
        try:
            npr = self.get_index_array().to_numpy()
        except ValueError:
            npr = self.get_index_array().to_numpy(zero_copy_only=False)
        return npr

    def isin(self, values, skip_null=True, zero_copy_only=False):
        res_isin = None
        if isinstance(values, List) or isinstance(values, np.ndarray):
            values_arw_ar = pa.array(values)
            res_isin = compare_array_like_values(self.get_index_array(), values_arw_ar, skip_null)
            return res_isin.to_numpy(zero_copy_only=zero_copy_only)
        else:
            raise ValueError("values must be List or np.ndarray")


cdef class ArrowLocIndexer:
    def __cinit__(self, CIndexingType indexing_type):
        # self.indexer_shd_ptr = make_shared[CArrowLocIndexer](indexing_type)
        pass

    def _fix_partial_slice_inidices(self, start_index, end_index, index):
        if start_index and end_index:
            return start_index, end_index
        elif start_index is None and end_index is None:
            start_index = index.get_index_array()[0].as_py()  # first element of index
            end_index = index.get_index_array()[-1].as_py()  # last element of the index
        elif start_index and end_index is None:
            end_index = index.get_index_array()[-1].as_py()  # last element of the index
        elif start_index is None and end_index:
            start_index = index.get_index_array()[0].as_py()  # first element of index
        return start_index, end_index

    def _resolve_column_indices_vector(self, columns : List, table):
        resolved_columns = []
        if isinstance(columns, List):
            for column in columns:
                resolved_columns.append(_resolve_column_index_from_column_name(column, table))
            return resolved_columns
        else:
            raise ValueError("columns must be input as a List")

    def _resolve_column_index_slice(self, column:slice, table):
        if isinstance(column, slice):
            start_col_idx, end_col_idx = column.start, column.stop
            if start_col_idx:
                start_col_idx = _resolve_column_index_from_column_name(start_col_idx, table)
            else:
                start_col_idx = 0
            if end_col_idx:
                end_col_idx = _resolve_column_index_from_column_name(end_col_idx, table)
            else:
                end_col_idx = table.column_count - 1
        else:
            raise ValueError("column must be passed as a slice")
        return start_col_idx, end_col_idx


    def loc_with_index_range(self, start_index, end_index, column, table):
        cdef:
            shared_ptr[CTable] output
            shared_ptr[CTable] input
            shared_ptr[CArrowScalar] start
            shared_ptr[CArrowScalar] end
            int c_column_index
            int c_start_column_index
            int c_end_column_index
            vector[int] c_column_vector
        # cast indices to appropriate scalar types
        index = table.get_index()
        start_index, end_index = self._fix_partial_slice_inidices(start_index, end_index, index)
        index_array = index.get_index_array()
        start_scalar = pa.scalar(start_index, index_array.type)
        end_scalar = pa.scalar(end_index, index_array.type)
        start = pyarrow_unwrap_scalar(start_scalar)
        end = pyarrow_unwrap_scalar(end_scalar)
        input = pycylon_unwrap_table(table)
        if np.isscalar(column):
            # single column
            c_column_index = self._resolve_column_indices_vector([column], table)[0]
            # self.indexer_shd_ptr.get().loc(start, end, c_column_index, input, output)
            Loc(input, start, end, c_column_index, &output)
            return pycylon_wrap_table(output)
        elif isinstance(column, slice):
            # range of columns
            start_index, end_index = self._resolve_column_index_slice(column, table)
            c_start_column_index = start_index
            c_end_column_index = end_index
            # self.indexer_shd_ptr.get().loc(start, end, c_start_column_index, c_end_column_index, input, output)
            Loc(input, start, end, c_start_column_index, c_end_column_index, & output)
            return pycylon_wrap_table(output)
        elif isinstance(column, List):
            # list of columns
            column = self._resolve_column_indices_vector(column, table)
            for col in column:
                c_column_vector.push_back(col)
            # self.indexer_shd_ptr.get().loc(start, end, c_column_vector, input, output)
            Loc(input, start, end, c_column_vector, &output)
            return pycylon_wrap_table(output)

    def loc_with_indices(self, indices, column, table):
        cdef:
            shared_ptr[CTable] output
            shared_ptr[CTable] input
            shared_ptr[CArrowArray] c_indices
            int c_column_index
            int c_start_column_index
            int c_end_column_index
            vector[int] c_column_vector
        # cast indices to appropriate scalar types
        index_array = table.get_index().get_index_array()
        indices_array = pa.array(indices, index_array.type)
        c_indices = pyarrow_unwrap_array(indices_array)
        input = pycylon_unwrap_table(table)
        if np.isscalar(column):
            # single column
            c_column_index = self._resolve_column_indices_vector([column], table)[0]
            # self.indexer_shd_ptr.get().loc(c_indices, c_column_index, input, output)
            Loc(input, c_indices, c_column_index, &output)
            return pycylon_wrap_table(output)
        elif isinstance(column, slice):
            # range of columns
            start_index, end_index = self._resolve_column_index_slice(column, table)
            c_start_column_index = start_index
            c_end_column_index = end_index
            # self.indexer_shd_ptr.get().loc(c_indices, c_start_column_index, c_end_column_index, input, output)
            Loc(input, c_indices, c_start_column_index, c_end_column_index, &output)
            return pycylon_wrap_table(output)
        elif isinstance(column, List):
            # list of columns
            column = self._resolve_column_indices_vector(column, table)
            for col in column:
                c_column_vector.push_back(col)
            # self.indexer_shd_ptr.get().loc(c_indices, c_column_vector, input, output)
            Loc(input, c_indices, c_column_vector, &output)
            return pycylon_wrap_table(output)


cdef class ArrowILocIndexer:
    def __cinit__(self, CIndexingType indexing_type):
        # self.indexer_shd_ptr = make_shared[CArrowILocIndexer](indexing_type)
        pass

    def _fix_partial_slice_inidices(self, start_index, end_index, index):
        if start_index is not None and end_index is not None:
            # (excluding the boundary value)
            return start_index, end_index - 1
        elif start_index is None and end_index is None:
            start_index = 0
            end_index = len(index.get_index_array()) - 1
            return start_index, end_index
        elif start_index and end_index is None:
            end_index = len(index.get_index_array()) - 1
            return start_index, end_index
        elif start_index is None and end_index:
            start_index = 0
            end_index = end_index - 1
            return start_index, end_index

    def _fix_partial_slice_column_inidices(self, start_index, end_index, num_columns):
        if start_index and end_index:
            # (excluding the boundary value)
            return start_index, end_index - 1
        elif start_index is None and end_index is None:
            start_index = 0
            end_index = num_columns - 1
        elif start_index and end_index is None:
            end_index = num_columns - 1
        elif start_index is None and end_index:
            start_index = 0
        return start_index, end_index

    def _resolve_column_indices_vector(self, columns : List, table):
        resolved_columns = []
        if isinstance(columns, List):
            for column in columns:
                resolved_columns.append(_resolve_column_index_from_column_name(column, table))
            return resolved_columns
        else:
            raise ValueError("columns must be input as a List")

    def _resolve_column_index_slice(self, column:slice, table):
        if isinstance(column, slice):
            start_col_idx, end_col_idx = column.start, column.stop
            if start_col_idx:
                start_col_idx = _resolve_column_index_from_column_name(start_col_idx, table)
            else:
                start_col_idx = 0
            if end_col_idx:
                end_col_idx = _resolve_column_index_from_column_name(end_col_idx, table)
                if end_col_idx > 0:
                    end_col_idx = end_col_idx - 1
            else:
                end_col_idx = table.column_count - 1
        else:
            raise ValueError("column must be passed as a slice")
        return start_col_idx, end_col_idx

    def loc_with_index_range(self, start_index, end_index, column, table):
        cdef:
            shared_ptr[CTable] output
            shared_ptr[CTable] input
            shared_ptr[CArrowScalar] start
            shared_ptr[CArrowScalar] end
            int c_column_index
            int c_start_column_index
            int c_end_column_index
            vector[int] c_column_vector
        # cast indices to appropriate scalar types
        index = table.get_index()
        start_index, end_index = self._fix_partial_slice_inidices(start_index, end_index, index)
        index_array = index.get_index_array()
        start_scalar = pa.scalar(start_index, pa.int64())
        end_scalar = pa.scalar(end_index, pa.int64())
        start = pyarrow_unwrap_scalar(start_scalar)
        end = pyarrow_unwrap_scalar(end_scalar)
        input = pycylon_unwrap_table(table)
        if np.isscalar(column):
            # single column
            c_column_index = self._resolve_column_indices_vector([column], table)[0]
            # self.indexer_shd_ptr.get().loc(start, end, c_column_index, input, output)
            iLoc(input, start, end, c_column_index, &output)
            return pycylon_wrap_table(output)
        elif isinstance(column, slice):
            # range of columns
            start_index, end_index = self._resolve_column_index_slice(column, table)
            c_start_column_index = start_index
            c_end_column_index = end_index
            # self.indexer_shd_ptr.get().loc(start, end, c_start_column_index, c_end_column_index, input, output)
            iLoc(input, start, end, c_start_column_index, c_end_column_index, &output)
            return pycylon_wrap_table(output)
        elif isinstance(column, List):
            # list of columns
            column = self._resolve_column_indices_vector(column, table)
            for col in column:
                c_column_vector.push_back(col)
            # self.indexer_shd_ptr.get().loc(start, end, c_column_vector, input, output)
            iLoc(input, start, end, c_column_vector, & output)
            return pycylon_wrap_table(output)

    def loc_with_indices(self, indices, column, table):
        cdef:
            shared_ptr[CTable] output
            shared_ptr[CTable] input
            shared_ptr[CArrowArray] c_indices
            int c_column_index
            int c_start_column_index
            int c_end_column_index
            vector[int] c_column_vector
        # cast indices to appropriate scalar types
        index_array = table.get_index().get_index_array()
        indices_array = pa.array(indices, pa.int64())
        c_indices = pyarrow_unwrap_array(indices_array)
        input = pycylon_unwrap_table(table)
        if np.isscalar(column):
            # single column
            c_column_index = self._resolve_column_indices_vector([column], table)[0]
            # self.indexer_shd_ptr.get().loc(c_indices, c_column_index, input, output)
            iLoc(input, c_indices, c_column_index, &output)
            return pycylon_wrap_table(output)
        elif isinstance(column, slice):
            # range of columns
            start_index, end_index = self._resolve_column_index_slice(column, table)
            c_start_column_index = start_index
            c_end_column_index = end_index
            # self.indexer_shd_ptr.get().loc(c_indices, c_start_column_index, c_end_column_index, input, output)
            iLoc(input, c_indices, c_start_column_index, c_end_column_index, &output)
            return pycylon_wrap_table(output)
        elif isinstance(column, List):
            # list of columns
            column = self._resolve_column_indices_vector(column, table)
            for col in column:
                c_column_vector.push_back(col)
            # self.indexer_shd_ptr.get().loc(c_indices, c_column_vector, input, output)
            iLoc(input, c_indices, c_column_vector, & output)
            return pycylon_wrap_table(output)


class PyLocIndexer:

    def __init__(self, cn_table, mode):
        self._cn_table = cn_table
        if mode == "loc":
            self._loc_indexer = ArrowLocIndexer(cn_table.get_index().get_type())
        elif mode == "iloc":
            self._loc_indexer = ArrowILocIndexer(cn_table.get_index().get_type())

    def _resolve_column_index_from_column_name(self, column_name) -> int:
        index = None
        for idx, col_name in enumerate(self._cn_table.column_names):
            if column_name == col_name:
                return idx
        if index is None:
            raise ValueError(f"Column {column_name} does not exist in the table")

    def __getitem__(self, item):
        if isinstance(item, Tuple):
            # with both column and index option
            if len(item) != 2:
                raise ValueError("Invalid number of arguments for LocIndexer, expected 2")

            index_values, column_values = item[0], item[1]

            if isinstance(index_values, slice):
                start_idx = index_values.start
                end_idx = index_values.stop
                return self._loc_indexer.loc_with_index_range(start_idx, end_idx, column_values,
                                                              self._cn_table)
            elif isinstance(index_values, List):
                return self._loc_indexer.loc_with_indices(index_values, column_values, self._cn_table)
        elif item:
            column_index = slice(None, None)
            if isinstance(item, slice):
                start_idx = item.start
                end_idx = item.stop
                return self._loc_indexer.loc_with_index_range(start_idx, end_idx, column_index,
                                                              self._cn_table)
            elif isinstance(item, List):
                return self._loc_indexer.loc_with_indices(item, column_index, self._cn_table)
            elif np.isscalar(item):
                return self._loc_indexer.loc_with_indices([item], column_index, self._cn_table)
            else:
                raise ValueError("Index values must be either slice or List")
        else:
            raise ValueError("No values passed for loc operation")
