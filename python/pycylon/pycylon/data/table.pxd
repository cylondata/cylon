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
from libcpp cimport bool
from pycylon.common.status cimport CStatus
from pycylon.common.status import Status
import uuid
from pycylon.common.join_config cimport CJoinType
from pycylon.common.join_config cimport CJoinAlgorithm
from pycylon.common.join_config cimport CJoinConfig
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pycylon.io.csv_write_config cimport CCSVWriteOptions
from pyarrow.lib cimport CTable as CArrowTable
from pyarrow.lib cimport CArray as CArrowArray
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.indexing.cyindex cimport CBaseArrowIndex
from pycylon.indexing.cyindex import BaseArrowIndex


cdef extern from "../../../../cpp/src/cylon/table.hpp" namespace "cylon":
    cdef cppclass CTable "cylon::Table":
        CTable(const shared_ptr[CCylonContext] & ctx, shared_ptr[CArrowTable] tab)

        @ staticmethod
        CStatus FromArrowTable(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CArrowTable] & table,
                               shared_ptr[CTable] & tableOut)

        CStatus ToArrowTable(shared_ptr[CArrowTable] & output)

        int Columns() const

        int Rows() const

        void Print()

        void Print(int row1, int row2, int col1, int col2)

        const shared_ptr[CCylonContext] & GetContext()

        vector[string] ColumnNames()

        void retainMemory(bool retain)

        bool IsRetain() const

        CStatus SetArrowIndex(shared_ptr[CBaseArrowIndex] & index, bool drop)

        shared_ptr[CBaseArrowIndex] GetArrowIndex()

        CStatus ResetArrowIndex(bool drop)

        CStatus AddColumn(int position, string column_name, shared_ptr[CArrowArray] input_column)


cdef extern from "../../../../cpp/src/cylon/table.hpp" namespace "cylon":
    CStatus WriteCSV(shared_ptr[CTable] & table, const string & path,
                     const CCSVWriteOptions & options)

    CStatus Sort(shared_ptr[CTable] & table, const vector[int] sort_columns,
                 shared_ptr[CTable] & output, const vector[bool] & sort_direction)

    CStatus Project(shared_ptr[CTable] & table, const vector[int] & project_columns, shared_ptr[
            CTable] & output)

    CStatus Merge(vector[shared_ptr[CTable]] & tables, shared_ptr[CTable] output)

    CStatus Join(shared_ptr[CTable] & left, shared_ptr[CTable] & right,
                 const CJoinConfig& join_config, shared_ptr[CTable] & output)

    CStatus DistributedJoin(shared_ptr[CTable] & left, shared_ptr[CTable] & right,
                            const CJoinConfig & join_config, shared_ptr[CTable] & output);

    CStatus Union(shared_ptr[CTable] & first, shared_ptr[CTable] & second, shared_ptr[CTable]
    & output)

    CStatus DistributedUnion(shared_ptr[CTable] & first, shared_ptr[CTable] & second, shared_ptr[
            CTable]
    & output)

    CStatus Subtract(shared_ptr[CTable] & first, shared_ptr[CTable] & second, shared_ptr[CTable]
    & output)

    CStatus DistributedSubtract(shared_ptr[CTable] & first, shared_ptr[CTable] & second,
                                shared_ptr[CTable] & output)

    CStatus Intersect(shared_ptr[CTable] & first, shared_ptr[CTable] & second, shared_ptr[CTable]
    & output)

    CStatus DistributedIntersect(shared_ptr[CTable] & first, shared_ptr[CTable] & second,
                                 shared_ptr[CTable] & output)

    CStatus DistributedSort(shared_ptr[CTable] & table, const vector[int] sort_columns,
                            shared_ptr[CTable] & output, const vector[bool] & sort_direction,
                            CSortOptions sort_options)

    CStatus Shuffle(shared_ptr[CTable] & table, const vector[int] & hash_columns, shared_ptr[CTable]
    & output)

    CStatus Unique(shared_ptr[CTable] & input_table, const vector[int] & columns, shared_ptr[CTable]
    & output, bool first)

    CStatus DistributedUnique(shared_ptr[CTable] & input_table, const vector[int] & columns,
                              shared_ptr[CTable]& output)

    CStatus Equals(shared_ptr[CTable] & a, shared_ptr[CTable] & b, bool& result, bool ordered)
    
    CStatus DistributedEquals(shared_ptr[CTable] & a, shared_ptr[CTable] & b, bool& result, bool ordered)

    # CStatus Repartition(shared_ptr[CTable] & table, const vector[int] & rows_per_partition, const vector[int] & receive_build_rank_order, shared_ptr[CTable]* output)
    
    # CStatus Repartition(shared_ptr[CTable] & table, const vector[int] & rows_per_partition, shared_ptr[CTable]* output)

    # CStatus Repartition(shared_ptr[CTable] & table, shared_ptr[CTable]* output)

cdef extern from "../../../../cpp/src/cylon/table.hpp" namespace "cylon":
    cdef cppclass CSortOptions "cylon::SortOptions":
        int num_bins
        long num_samples
        @ staticmethod
        CSortOptions Defaults()


cdef class SortOptions:
    cdef:
        CSortOptions *thisPtr
        void init(self, CSortOptions *csort_options)

cdef class Table:
    cdef:
        shared_ptr[CTable] table_shd_ptr
        shared_ptr[CTable] *table_out_shd_ptr
        shared_ptr[CCylonContext] sp_context

        dict __dict__

        void init(self, const shared_ptr[CTable]& table)

        _get_join_ra_response(self, op_name, shared_ptr[CTable] output, CStatus status)

        _get_ra_response(self, table, ra_op_name)
