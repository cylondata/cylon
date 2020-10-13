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


cdef extern from "../../../cpp/src/cylon/data_types.hpp" namespace "cylon":
    cdef cppclass CType 'cylon::Type':
        enum ctype:
            # Boolean as 1 bit, LSB bit-packed ordering
            CBOOL 'cylon::Type::BOOL'
            # Unsigned 8-bit little-endian integer
            CUINT8 'cylon::Type::UINT8'
            # Signed 8-bit little-endian integer
            CINT8 'cylon::Type::INT8'
            # Unsigned 16-bit little-endian integer
            CUINT16 'cylon::Type::UINT16'
            # Signed 16-bit little-endian integer
            CINT16 'cylon::Type::INT16'
            # Unsigned 32-bit little-endian integer
            CUINT32 'cylon::Type::UINT32'
            # Signed 32-bit little-endian integer
            CINT32 'cylon::Type::INT32'
            # Unsigned 64-bit little-endian integer
            CUINT64 'cylon::Type::UINT64'
            # Signed 64-bit little-endian integer
            CINT64 'cylon::Type::INT64'
            # 2-byte floating point value
            CHALF_FLOAT 'cylon::Type::HALF_FLOAT'
            # 4-byte floating point value
            CFLOAT 'cylon::Type::FLOAT'
            # 8-byte floating point value
            CDOUBLE 'cylon::Type::DOUBLE'
            # UTF8 variable-length string as List<Char>
            CSTRING 'cylon::Type::STRING'
            # Variable-length bytes (no guarantee of UTF8-ness)
            CBINARY 'cylon::Type::BINARY'
            # Fixed-size binary. Each value occupies the same number of bytes
            CFIXED_SIZE_BINARY 'cylon::Type::FIXED_SIZE_BINARY'
            # int32_t days since the UNIX epoch
            CDATE32 'cylon::Type::DATE32'
            # int64_t milliseconds since the UNIX epoch
            CDATE64 'cylon::Type::DATE64'
            # Exact timestamp encoded with int64 since UNIX epoch
            # Default unit millisecond
            CTIMESTAMP 'cylon::Type::TIMESTAMP'
            # Time as signed 32-bit integer, representing either seconds or
            # milliseconds since midnight
            CTIME32 'cylon::Type::TIME32'
            # Time as signed 64-bit integer, representing either microseconds or
            # nanoseconds since midnight
            CTIME64 'cylon::Type::TIME64'
            # YEAR_MONTH or DAY_TIME interval in SQL style
            CINTERVAL 'cylon::Type::INTERVAL'
            # Precision- and scale-based decimal type. Storage type depends on the
            # parameters.
            CDECIMAL 'cylon::Type::DECIMAL'
            # A list of some logical data type
            CLIST 'cylon::Type::LIST'
            # Custom data type, implemented by user
            CEXTENSION 'cylon::Type::EXTENSION'
            # Fixed size list of some logical type
            CFIXED_SIZE_LIST 'cylon::Type::FIXED_SIZE_LIST'
            # or nanoseconds.
            CDURATION 'cylon::Type::DURATION'


cdef extern from "../../../cpp/src/cylon/data_types.hpp" namespace "cylon":
    cdef cppclass CLayout 'cylon::Layout':
        enum clayout:
            CFIXED_WIDTH 'cylon::Layout::FIXED_WIDTH'
            CVARIABLE_WIDTH 'cylon::Layout::VARIABLE_WIDTH'


cdef extern from "../../../cpp/src/cylon/data_types.hpp" namespace "cylon":
    cdef cppclass CDataType "cylon::DataType":
        DataType()

        DataType(CType.ctype t)

        DataType(CType.ctype t, CLayout.clayout l)

        CType.ctype getType()

        CLayout.clayout getLayout()

        @staticmethod
        shared_ptr[CDataType] Make(CType.ctype t, CLayout.clayout l)

cdef class DataType:
    pass