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
import uuid
from pycylon.common.join_config cimport CJoinType
from pycylon.common.join_config cimport CJoinAlgorithm
from pycylon.common.join_config cimport CJoinConfig
from pycylon.common.join_config import PJoinType
from pycylon.common.join_config import PJoinAlgorithm
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pycylon.io.csv_write_config cimport CCSVWriteOptions
from pyarrow.lib cimport CTable as CArrowTable
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext

cdef extern from "../../../cpp/src/cylon/table.hpp" namespace "cylon":
    cdef cppclass CTable "cylon::Table":
        CTable(shared_ptr[CArrowTable] &tab, shared_ptr[CCylonContext] &ctx)

        @staticmethod
        CStatus FromArrowTable(shared_ptr[CCylonContext] &ctx, shared_ptr[CArrowTable] &table,
                               shared_ptr[CTable] *tableOut)

        CStatus ToArrowTable(shared_ptr[CArrowTable] &output)

        @staticmethod
        CStatus FromCSV(shared_ptr[CCylonContext] &ctx, const string &path, shared_ptr[CTable]
                        &tableOut, const CCSVReadOptions &options)

        CStatus WriteCSV(const string &path, const CCSVWriteOptions &options)

        CStatus Sort(int sort_column, shared_ptr[CTable] &output)

        @staticmethod
        CStatus Merge(shared_ptr[CCylonContext] &ctx, vector[shared_ptr[CTable]] &tables,
                      shared_ptr[CTable] output)

        @staticmethod
        CStatus Join(shared_ptr[CTable] &left, shared_ptr[CTable] &right,  CJoinConfig
        join_config, shared_ptr[CTable] *output)

        @staticmethod
        CStatus DistributedJoin(shared_ptr[CTable] &left, shared_ptr[CTable] &right,  CJoinConfig
        join_config, shared_ptr[CTable] *output);

        @staticmethod
        CStatus Union(shared_ptr[CTable] &first, shared_ptr[CTable] &second, shared_ptr[CTable]
        &output)

        @staticmethod
        CStatus DistributedUnion(shared_ptr[CTable] &first, shared_ptr[CTable] &second, shared_ptr[
                CTable]
        &output)

        @staticmethod
        CStatus Subtract(shared_ptr[CTable] &first, shared_ptr[CTable] &second, shared_ptr[CTable]
        &output)

        @staticmethod
        CStatus DistributedSubtract(shared_ptr[CTable] &first, shared_ptr[CTable] &second,
                                   shared_ptr[CTable] &output)

        @staticmethod
        CStatus Intersect(shared_ptr[CTable] &first, shared_ptr[CTable] &second, shared_ptr[CTable]
        &output)

        @staticmethod
        CStatus DistributedIntersect(shared_ptr[CTable] &first, shared_ptr[CTable] &second,
                                   shared_ptr[CTable] &output)

        int Columns()

        int Rows()

        void Print()

        shared_ptr[CCylonContext] GetContext()


cdef class Table:
    cdef:
        shared_ptr[CTable] table_shd_ptr
        shared_ptr[CTable] *table_out_shd_ptr
        shared_ptr[CCylonContext] sp_context
        CJoinConfig *jcPtr
        dict __dict__

        void init(self, const shared_ptr[CTable]& table)

        shared_ptr[CTable] init_join_ra_params(self, table, join_type, join_algorithm, kwargs)

        _get_join_ra_response(self, op_name, shared_ptr[CTable] output, CStatus status)

        _get_ra_response(self, table, ra_op_name)
