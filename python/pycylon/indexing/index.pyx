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
from pyarrow.lib cimport CArray as CArrowArray
from pycylon.indexing.index cimport CIndexingSchema
from pycylon.indexing.index cimport CLocIndexer
from pycylon.indexing.index cimport CBaseIndex
from pyarrow.lib cimport (pyarrow_unwrap_table, pyarrow_wrap_table, pyarrow_wrap_array)

from pycylon.api.lib cimport (pycylon_wrap_context, pycylon_unwrap_context, pycylon_unwrap_table,
                                pycylon_wrap_table)

from pycylon.data.table cimport CTable
from pycylon.data.table import Table
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
import pyarrow as pa

from cpython.ref cimport PyObject
'''
Cylon Indexing is done with the following enums. 
'''

cpdef enum IndexingSchema:
    RANGE = CIndexingSchema.CRANGE
    LINEAR = CIndexingSchema.CLINEAR
    HASH = CIndexingSchema.CHASH
    BINARYTREE = CIndexingSchema.CBINARYTREE
    BTREE = CIndexingSchema.CBTREE


cdef class BaseIndex:

    cdef void init(self, const shared_ptr[CBaseIndex]& index):
        self.bindex_shd_ptr = index

cdef class LocIndexer:

    def __cinit__(self, CIndexingSchema indexing_schema):
        self.indexer_shd_ptr = make_shared[CLocIndexer](indexing_schema)

    def loc(self, start_index, end_index, column_index, table):

        cdef shared_ptr[CTable] output
        cdef long a = <long> start_index
        cdef long b = <long> end_index
        #cdef const void* c_start_index = <const void *> a
        #cdef const void* c_end_index = <const void *> b
        cdef int c_column_index = <int>column_index
        cdef shared_ptr[CTable] input = pycylon_unwrap_table(table)





        self.indexer_shd_ptr.get().loc(&a, &b, c_column_index, input,
                                       output)
        return pycylon_wrap_table(output)


