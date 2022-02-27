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

# distutils: language = c++

from libcpp.memory cimport shared_ptr
from pycylon.data.data_type cimport CDataType
from pyarrow.lib cimport CScalar as ArrowCScalar

cdef extern from "../../../../cpp/src/cylon/scalar.hpp" namespace "cylon":
    cdef cppclass CScalar "cylon::Scalar":
        CScalar(shared_ptr[ArrowCScalar] data_)

        const shared_ptr[ArrowCScalar]& data() const

        const shared_ptr[CDataType]& type() const

cdef class Scalar:
    cdef:
        shared_ptr[CScalar] thisPtr
        void init(self, shared_ptr[CScalar] data)