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


from libcpp.memory cimport shared_ptr, make_shared
from pycylon.data.data_type cimport CDataType
from pycylon.data.ctype cimport CType
from pycylon.data.ctype import Type
from pycylon.data.layout cimport CLayout
from pycylon.data.layout import Layout

import pyarrow as pa







cdef class DataType:

    # TODO: Implement if Required
    def __cinit__(self, type=None, layout=None):
        if type is None and layout is None:
            #self.thisPtr = new CDataType()
            self.sp_data_type = make_shared[CDataType]()
        # elif type is not None and layout is None:
        #     if isinstance(type, Type):
        #         #self.thisPtr = new CDataType(type)
        #         #self.sp_data_type = make_shared[CDataType](type)
        elif type is not None and layout is not None:
            if isinstance(type, Type) and isinstance(layout,Layout):
                #self.thisPtr = new CDataType(type, layout)
                self.sp_data_type = self.thisPtr.Make(type, layout)
        else:
            raise ValueError("Invalid arguments!")

    cdef void init(self, const shared_ptr[CDataType] &cdata_type):
        self.sp_data_type = cdata_type

    @property
    def type(self):
        return self.sp_data_type.get().getType()

    @property
    def layout(self):
        return self.sp_data_type.get().getLayout()
