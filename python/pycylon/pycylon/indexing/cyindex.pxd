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
from pyarrow.lib cimport CScalar as CArrowScalar
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.table cimport CTable


cdef extern from "../../../../cpp/src/cylon/indexing/index.hpp" namespace "cylon":
    cdef enum CIndexingType 'cylon::IndexingType':
        CRANGE 'cylon::IndexingType::Range'
        CLINEAR 'cylon::IndexingType::Linear'
        CHASH 'cylon::IndexingType::Hash'
        CBINARYTREE 'cylon::IndexingType::BinaryTree'
        CBTREE 'cylon::IndexingType::BTree'


cdef extern from "../../../../cpp/src/cylon/indexing/index.hpp" namespace "cylon":
    cdef cppclass CBaseArrowIndex "cylon::BaseArrowIndex":

        const shared_ptr[CArrowArray] & GetIndexArray()

        CIndexingType GetIndexingType()


cdef class BaseArrowIndex:
    cdef:
        shared_ptr[CBaseArrowIndex] bindex_shd_ptr
        shared_ptr[CCylonContext] ctx_shd_ptr
        int column_id
        int size
        dict __dict__

        void init(self, const shared_ptr[CBaseArrowIndex]& index)



cdef extern from "../../../../cpp/src/cylon/indexing/indexer.hpp" namespace "cylon::indexing":
    CStatus Loc(const shared_ptr[CTable] & input_table,
                const shared_ptr[CArrowScalar] & start_index,
                const shared_ptr[CArrowScalar] & end_index,
                const int column_index,
                shared_ptr[CTable] * output)

    CStatus Loc(const shared_ptr[CTable] & input_table,
                const shared_ptr[CArrowScalar] & start_index,
                const shared_ptr[CArrowScalar] & end_index,
                const int start_column_index,
                const int end_column_index,
                shared_ptr[CTable] * output)

    CStatus Loc(const shared_ptr[CTable] & input_table,
                const shared_ptr[CArrowScalar] & start_index,
                const shared_ptr[CArrowScalar] & end_index,
                const vector[int] & columns,
                shared_ptr[CTable] * output)

    CStatus Loc(const shared_ptr[CTable] & input_table,
                const shared_ptr[CArrowArray] & indices,
                const int column_index,
                shared_ptr[CTable] * output)

    CStatus Loc(const shared_ptr[CTable] & input_table,
                const shared_ptr[CArrowArray] & indices,
                const int start_column_index,
                const int end_column_index, shared_ptr[CTable] * output)

    CStatus Loc(const shared_ptr[CTable] & input_table,
                const shared_ptr[CArrowArray] & indices,
                const vector[int] & columns,
                shared_ptr[CTable] * output)

    CStatus iLoc(const shared_ptr[CTable] & input_table,
                 const shared_ptr[CArrowScalar] & start_index,
                 const shared_ptr[CArrowScalar] & end_index,
                 const int column_index,
                 shared_ptr[CTable] * output)

    CStatus iLoc(const shared_ptr[CTable] & input_table,
                 const shared_ptr[CArrowScalar] & start_index,
                 const shared_ptr[CArrowScalar] & end_index,
                 const int start_column_index, const int end_column_index,
                 shared_ptr[CTable] * output)

    CStatus iLoc(const shared_ptr[CTable] & input_table,
                 const shared_ptr[CArrowScalar] & start_index,
                 const shared_ptr[CArrowScalar] & end_index,
                 const vector[int] & columns, shared_ptr[CTable] * output)

    CStatus iLoc(const shared_ptr[CTable] & input_table,
                 const shared_ptr[CArrowArray] & indices,
                 const int column_index,
                 shared_ptr[CTable] * output)

    CStatus iLoc(const shared_ptr[CTable] & input_table,
                 const shared_ptr[CArrowArray] & indices,
                 const int start_column_index,
                 const int end_column_index, shared_ptr[CTable] * output)

    CStatus iLoc(const shared_ptr[CTable] & input_table,
                 const shared_ptr[CArrowArray] & indices,
                 const vector[int] & columns,
                 shared_ptr[CTable] * output)


cdef class ArrowLocIndexer:
    cdef:
        # shared_ptr[CArrowLocIndexer] indexer_shd_ptr
        CIndexingType c_indexing_type

cdef class ArrowILocIndexer:
    cdef:
        # shared_ptr[CArrowILocIndexer] indexer_shd_ptr
        CIndexingType c_indexing_type
