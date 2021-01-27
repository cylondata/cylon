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
from pyarrow.lib cimport CArray as CArrowArray
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.table cimport CTable


cdef extern from "../../../cpp/src/cylon/indexing/index.hpp" namespace "cylon":

    cdef enum CIndexingSchema 'cylon::IndexingSchema':
        CRANGE 'cylon::IndexingSchema::Range'
        CLINEAR 'cylon::IndexingSchema::Linear'
        CHASH 'cylon::IndexingSchema::Hash'
        CBINARYTREE 'cylon::IndexingSchema::BinaryTree'
        CBTREE 'cylon::IndexingSchema::BTree'


cdef extern from "../../../cpp/src/cylon/indexing/index.hpp" namespace "cylon":
    cdef cppclass CBaseIndex "cylon::BaseIndex":
        CBaseIndex (int col_id, int size, shared_ptr[CCylonContext] & ctx)

        shared_ptr[CArrowArray] GetIndexArray()


cdef class BaseIndex:
    cdef:
        shared_ptr[CBaseIndex] bindex_shd_ptr
        shared_ptr[CCylonContext] ctx_shd_ptr
        int column_id
        int size
        dict __dict__

        void init(self, const shared_ptr[CBaseIndex]& index)


cdef extern from "../../../cpp/src/cylon/indexing/indexer.hpp" namespace "cylon":
    cdef cppclass CLocIndexer "cylon::LocIndexer":
        CLocIndexer(CIndexingSchema indexing_schema)

        CStatus loc(const void *start_index, const void *end_index, const int column_index,
                    const shared_ptr[CTable] &input_table, shared_ptr[CTable] &output)

cdef class PyObjectToCObject:
    cdef:
        dict __dict__
        void * c_ptr
        void * get_ptr(self)

cdef class LocIndexer:

    cdef:
        shared_ptr[CLocIndexer] indexer_shd_ptr
        CIndexingSchema c_indexing_schema

        void resolve_ctype_from_python_object(self, py_object, index, void* c_ptr)

