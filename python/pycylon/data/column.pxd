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

from libcpp.string cimport string
from libcpp cimport bool
from pycylon.common.status cimport CStatus
from pycylon.common.status import Status
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.data_type cimport CDataType
from pycylon.data.data_type cimport CType
from pycylon.data.data_type cimport CLayout
from pyarrow.lib cimport CArray as ArrowCAarray
from pyarrow.lib cimport CChunkedArray as ArrowCChunkedAarray
from pyarrow.lib cimport pyarrow_unwrap_array


cdef extern from "../../../cpp/src/cylon/column.hpp" namespace "cylon":
    cdef cppclass CColumn "cylon::Column":
        CColumn(const string &id, const shared_ptr[CDataType] &type, const shared_ptr[
                ArrowCChunkedAarray] &data_)

        CColumn(const string &id, const shared_ptr[CDataType] &type, const shared_ptr[
                ArrowCAarray] &data_)

        shared_ptr[ArrowCChunkedAarray] GetColumnData() const

        string GetId() const

        shared_ptr[CDataType] GetDataType() const

        @staticmethod
        shared_ptr[CColumn] Make(const string &id, const shared_ptr[CDataType] &type,
                                const shared_ptr[ArrowCChunkedAarray] &data_)

        @staticmethod
        shared_ptr[CColumn] Make(const string &id, const shared_ptr[CDataType] &type,
                                const shared_ptr[ArrowCAarray] &data_)



cdef extern from "../../../cpp/src/cylon/column.hpp" namespace "cylon":

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

    cdef cppclass CVectorColumn "cylon::VectorColumn"[T]:
        CVectorColumn(const string &id, const shared_ptr[CDataType] &type, const shared_ptr[
                vector[T]] &data_vector)

        @staticmethod
        shared_ptr[CVectorColumn[T]] Make(const string &id, const shared_ptr[CDataType] &type,
                                          const shared_ptr[vector[T]] &data_vector)


cdef class Column:
    cdef:
        CColumn *thisPtr
        shared_ptr[CColumn] sp_column

