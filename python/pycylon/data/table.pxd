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
from pyarrow.lib cimport CTable as CArrowTable
from pyarrow.lib cimport pyarrow_unwrap_table
from pyarrow.lib cimport pyarrow_wrap_table
from libcpp.memory cimport shared_ptr

from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext

cdef extern from "../../../cpp/src/cylon/table.hpp" namespace "cylon":
    cdef cppclass CTable "cylon::Table":
        CTable(shared_ptr[CArrowTable] &tab, shared_ptr[CCylonContext] &ctx)

        @staticmethod
        CStatus FromArrowTable(shared_ptr[CCylonContext] &ctx, shared_ptr[CArrowTable] &table,
                               shared_ptr[CTable] *tableOut)



        int Columns()

        int Rows()

        void Print()


cdef class Table:
    cdef:
        shared_ptr[CTable] table_shd_ptr
        shared_ptr[CTable] *table_out_shd_ptr
        shared_ptr[CArrowTable] c_arrow_tb_shd_ptr
        shared_ptr[CCylonContext] ctx_shd_ptr
        dict __dict__

# cdef extern from "../../../cpp/src/cylon/python/table_cython.h" namespace "cylon::python::table":
#     cdef cppclass CxTable "cylon::python::table::CxTable":
#         CxTable()
#         CxTable(string)
#         string get_id()
#         int columns()
#         int rows()
#         void show()
#         void show(int, int, int, int)
#         CStatus to_csv(const string)
#
#         string join(CCylonContextWrap *ctx_wrap, const string, CJoinConfig)
#
#         string distributed_join(CCylonContextWrap *ctx_wrap, const string & table_id,
#                                 CJoinConfig join_config);
#
#         string Union(CCylonContextWrap *ctx_wrap, const string & table_right);
#
#         string DistributedUnion(CCylonContextWrap *ctx_wrap, const string & table_right);
#
#         string Intersect(CCylonContextWrap *ctx_wrap, const string & table_right);
#
#         string DistributedIntersect(CCylonContextWrap *ctx_wrap, const string & table_right);
#
#         string Subtract(CCylonContextWrap *ctx_wrap, const string & table_right);
#
#         string DistributedSubtract(CCylonContextWrap *ctx_wrap, const string & table_right);
#
#         #string Project(const vector[int64_t]& project_columns);

# cdef extern from "../../../cpp/src/cylon/python/table_cython.h" namespace "cylon::python::table::CxTable":
#     cdef extern string from_pyarrow_table(CCylonContextWrap *ctx_wrap, shared_ptr[CArrowTable] table)
#     cdef extern shared_ptr[CArrowTable] to_pyarrow_table(const string table_id)
#
# cdef class XTable:
#     cdef:
#         CxTable *thisPtr
#         CJoinConfig *jcPtr
#         CCylonContextWrap *ctx_wrap
#         dict __dict__


