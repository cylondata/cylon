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
from pycylon.api.lib cimport pycylon_unwrap_data_type, pycylon_wrap_data_type
from pycylon.data.data_type import DataType
from pyarrow.lib cimport pyarrow_unwrap_array, pyarrow_wrap_array


cdef class Column:

    def __cinit__(self, id:str, dtype: DataType, array):
       cdef string cid = id.encode()
       cdef shared_ptr[CDataType] cdt = pycylon_unwrap_data_type(dtype)
       cdef shared_ptr[ArrowCAarray] ca = pyarrow_unwrap_array(array)
       self.thisPtr = new CColumn(cid, cdt, ca)

    @property
    def id(self):
        return self.thisPtr.GetID().decode()

    @property
    def data(self):
        cdef shared_ptr[ArrowCChunkedAarray] ca = self.thisPtr.GetColumnData()
        cdef shared_ptr[ArrowCAarray] ar = ca.get().chunk(0)
        return pyarrow_wrap_array(ar)

    @property
    def dtype(self):
        cdef shared_ptr[CDataType] cdtype = self.thisPtr.GetDataType()
        return pycylon_wrap_data_type(cdtype)


