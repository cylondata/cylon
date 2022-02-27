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

from libcpp.memory cimport shared_ptr, make_shared
from pycylon.data.scalar cimport CScalar
from pycylon.api.lib cimport pycylon_wrap_data_type
from pyarrow.lib cimport CScalar as ArrowCScalar
from pyarrow.lib cimport pyarrow_unwrap_scalar, pyarrow_wrap_scalar
import pyarrow as pa
import pycylon as pc

cdef class Scalar:
    def __cinit__(self, scalar: pa.Scalar = None):
        cdef shared_ptr[ArrowCScalar] cscalar
        if scalar:
            cscalar = pyarrow_unwrap_scalar(scalar)
            self.thisPtr = make_shared[CScalar](cscalar)

    cdef void init(self, const shared_ptr[CScalar] & data_):
        self.thisPtr = data_

    @property
    def data(self) -> pa.Scalar:
        cdef shared_ptr[ArrowCScalar] scalar = self.thisPtr.get().data()
        return pyarrow_wrap_scalar(scalar)

    @property
    def dtype(self) -> pc.DataType:
        cdef shared_ptr[CDataType] cdtype = self.thisPtr.get().type()
        return pycylon_wrap_data_type(cdtype)
