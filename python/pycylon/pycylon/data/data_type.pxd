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
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.ctype cimport CType
from pycylon.data.layout cimport CLayout



cdef extern from "../../../../cpp/src/cylon/data_types.hpp" namespace "cylon":
    cdef cppclass CDataType "cylon::DataType":
        CDataType()

        CDataType(CType.ctype t)

        CDataType(CType.ctype t, CLayout.clayout l)

        CType.ctype getType()

        CLayout.clayout getLayout()

        @staticmethod
        shared_ptr[CDataType] Make(CType.ctype t, CLayout.clayout l)


cdef extern from "../../../../cpp/src/cylon/data_types.hpp" namespace "cylon":
    shared_ptr[CDataType] TYPE_FACTORY(NAME, TYPE, LAYOUT)


cdef class DataType:
    cdef:
        CDataType *thisPtr
        shared_ptr[CDataType] sp_data_type
        void init(self, const shared_ptr[CDataType] &cdata_type)
