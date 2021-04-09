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
from pyarrow.lib cimport CTable as CArrowTable
from pyarrow.lib cimport CArray as CArrowArray
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.table cimport CTable
from pycylon.data.table import Table
from pycylon.indexing.index cimport CIndexingSchema
from pycylon.indexing.index import IndexingSchema
from pycylon.indexing.index_utils cimport CIndexUtil

from pycylon.api.lib cimport (pycylon_wrap_context, pycylon_unwrap_context, pycylon_unwrap_table,
                                pycylon_wrap_table)
from pyarrow.lib cimport (pyarrow_wrap_array, pyarrow_unwrap_array)
from typing import List
import pyarrow as pa

cdef class IndexUtil:

    @staticmethod
    def build_arrow_index(indexing_schema: IndexingSchema, table: Table, column: int, drop: bool):
        cdef shared_ptr[CTable] output
        cdef shared_ptr[CTable] input = pycylon_unwrap_table(table)
        CIndexUtil.BuildArrowIndex(indexing_schema, input, column, drop, output)
        cn_table = pycylon_wrap_table(output)
        #cn_table.indexing_schema = indexing_schema
        return cn_table

    @staticmethod
    def build_arrow_index_from_list(indexing_schema: IndexingSchema, table: Table, index_arr: List):
        cdef shared_ptr[CTable] output
        cdef shared_ptr[CTable] input = pycylon_unwrap_table(table)
        arrow_index_array = pa.array(index_arr)
        cdef shared_ptr[CArrowArray] c_index_array = pyarrow_unwrap_array(arrow_index_array)
        CIndexUtil.BuildArrowIndexFromArray(indexing_schema, input, c_index_array, output)
        return pycylon_wrap_table(output)