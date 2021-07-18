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
from pycylon.indexing.index cimport CIndexingType


cdef extern from "../../../cpp/src/cylon/indexing/index_utils.hpp" namespace "cylon":

    cdef cppclass CIndexUtil 'cylon::IndexUtil':

        @staticmethod
        CStatus BuildArrowIndex(const CIndexingType schema, const shared_ptr[CTable] & input,
                           const int index_column, const bool drop, shared_ptr[CTable] & output)

        @ staticmethod
        CStatus BuildArrowIndexFromArray(const CIndexingType schema, const shared_ptr[CTable] & input,
                                         const shared_ptr[CArrowArray] & index_array)


cdef class IndexUtil:
    cdef:
        dict __dict__