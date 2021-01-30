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

        CStatus loc(const void *start_index, const void *end_index, const vector[int]
                    &column_indices, const shared_ptr[CTable] & input_table,
                    shared_ptr[CTable] & output)

        CStatus loc(const void *index, const int column_index,
                    const shared_ptr[CTable] & input_table, shared_ptr[CTable] &output)

        CStatus loc(const void *index, const vector[int] &column_indices,
                    const shared_ptr[CTable] & input_table, shared_ptr[CTable] & output)

        CStatus loc(const vector[void*] &indices, const int column,
                    const shared_ptr[CTable] & input_table, shared_ptr[CTable] & output)

        CStatus loc(const vector[void*] & indices, const vector[int] &column_indices,
                    const shared_ptr[CTable] & input_table, shared_ptr[CTable] & output)



cdef class PyObjectToCObject:
    cdef:
        dict __dict__
        void * c_ptr
        void * get_cptr_from_object(self, py_object)
        vector[void*] get_vector_ptrs_from_list(self, py_list)
        long to_long(self, py_object)
        bool to_bool(self, py_object)
        signed char to_int8(self, py_object)
        signed short to_int16(self, py_object)
        signed int to_int32(self, py_object)
        signed long long to_int64(self, py_object)
        unsigned char to_uint8(self, py_object)
        unsigned short to_uint16(self, py_object)
        unsigned int to_uint32(self, py_object)
        unsigned long long to_uint64(self, py_object)
        long to_long(self, py_object)
        float to_float(self, py_object)
        double to_double(self, py_object)
        unsigned short to_half_float(self, py_object)
        string to_string(self, py_object)





cdef class LocIndexer:

    cdef:
        shared_ptr[CLocIndexer] indexer_shd_ptr
        CIndexingSchema c_indexing_schema
