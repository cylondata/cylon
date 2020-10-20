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

from typing import List
from libcpp.string cimport string
from libcpp cimport bool
from pycylon.common.status cimport CStatus
from pycylon.common.status import Status
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.data_type cimport CDataType
from pycylon.data.data_type cimport DataType
from pycylon.data.data_type cimport CType
from pycylon.data.data_type cimport CLayout
from pyarrow.lib cimport CArray as ArrowCAarray
from pyarrow.lib cimport CChunkedArray as ArrowCChunkedAarray
from pyarrow.lib cimport pyarrow_unwrap_array
from pycylon.data.column cimport CColumn
from pycylon.data.column cimport CVectorColumn
from pycylon.api.lib cimport pycylon_wrap_data_type


cdef class Column:

    def __cinit__(self):
        # TODO: Implement if required
        pass



cdef class VectorColumn:

    def __cinit__(self, id, type, data_list):
        # TODO: Implement if required
       pass

