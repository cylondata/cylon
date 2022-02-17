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
from pycylon.data.data_type cimport DataType
from pyarrow.lib cimport CArray as ArrowCAarray
from pycylon.data.column cimport CColumn
from pycylon.api.lib cimport pycylon_unwrap_data_type, pycylon_wrap_data_type
from pycylon.data.data_type import DataType
from pyarrow.lib cimport pyarrow_unwrap_array, pyarrow_wrap_array


cdef class Column:

    def __cinit__(self, dtype: DataType, array):
       cdef shared_ptr[CDataType] cdt = pycylon_unwrap_data_type(dtype)
       cdef shared_ptr[ArrowCAarray] ca = pyarrow_unwrap_array(array)
       self.thisPtr = new CColumn(cdt, ca)

    @property
    def data(self):
        cdef shared_ptr[ArrowCAarray] ar = self.thisPtr.data()
        return pyarrow_wrap_array(ar)

    @property
    def dtype(self):
        cdef shared_ptr[CDataType] cdtype = self.thisPtr.type()
        return pycylon_wrap_data_type(cdtype)


