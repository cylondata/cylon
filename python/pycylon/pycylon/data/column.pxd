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
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.common.status cimport CStatus
from pycylon.data.data_type cimport CDataType
from pyarrow.lib cimport CArray as ArrowCAarray



cdef extern from "../../../../cpp/src/cylon/column.hpp" namespace "cylon":
    ctypedef fused T:
        signed char
        signed short
        signed int
        signed long
        signed long long

        unsigned char
        unsigned short
        unsigned int
        unsigned long
        unsigned long long

        float
        double
        long double

    cdef cppclass CColumn "cylon::Column":
        CColumn(shared_ptr[CDataType] type_, shared_ptr[ArrowCAarray] data_)

        const shared_ptr[ArrowCAarray]& data() const

        const shared_ptr[CDataType]& type() const

        @ staticmethod
        shared_ptr[CColumn] Make(shared_ptr[CDataType] type_, shared_ptr[ArrowCAarray] data_)

        @ staticmethod
        CStatus FromVector[T](const shared_ptr[CCylonContext] & ctx,
                              const shared_ptr[CDataType] & type,
                              const vector[T] & data_vector,
                              shared_ptr[CColumn] & output)


cdef class Column:
    cdef:
        CColumn *thisPtr
        shared_ptr[CColumn] sp_column
