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

import numpy
from libcpp.memory cimport shared_ptr, make_shared
from pycylon.data.data_type cimport CDataType
from pyarrow.lib cimport CArray as ArrowCAarray
from pycylon.data.column cimport CColumn
from pycylon.api.lib cimport pycylon_wrap_data_type
from pyarrow.lib cimport pyarrow_unwrap_array, pyarrow_wrap_array
import pyarrow as pa
import pycylon as pc
import numpy as np

cdef class Column:
    def __cinit__(self, array = None):
        cdef shared_ptr[ArrowCAarray] carray
        if array is not None:
            if isinstance(array, (List, np.ndarray)):
                carray = pyarrow_unwrap_array(pa.array(array))
            elif isinstance(array, pa.Array):
                carray = pyarrow_unwrap_array(array)
            else:
                raise ValueError(f'Invalid type {type(array)}, data must be List, Numpy NdArray or '
                                 f'PyArrow array')
            self.thisPtr = make_shared[CColumn](carray)

    cdef void init(self, const shared_ptr[CColumn] & data_):
        self.thisPtr = data_

    @property
    def data(self)-> pa.Array:
        cdef shared_ptr[ArrowCAarray] ar = self.thisPtr.get().data()
        return pyarrow_wrap_array(ar)

    @property
    def dtype(self)-> pc.DataType:
        cdef shared_ptr[CDataType] cdtype = self.thisPtr.get().type()
        return pycylon_wrap_data_type(cdtype)

    def __len__(self) -> int:
        cdef int length = self.thisPtr.get().length()
        return length
