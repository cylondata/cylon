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
from pycylon.common.status cimport _Status
import uuid


from pycylon import Table
from pycylon.ctx.context cimport CCylonContextWrap
from pycylon.ctx.context import CylonContext


'''
Cylon CSV Utils 
'''

cdef extern from "../../../cpp/src/cylon/python/table_cython.h" namespace "cylon::python::table::CxTable":
    cdef extern _Status from_csv(CCylonContextWrap *ctx_wrap, const string, const char, const string)

cdef class csv_reader:

    @staticmethod
    def read(ctx: CylonContext, path: str, delimiter: str) -> Table:
        cdef string spath = path.encode()
        cdef string sdelm = delimiter.encode()
        id = uuid.uuid4()
        id_str = id.__str__()
        id_buf = id_str.encode()
        # this condition is checked with the initialization in csv.pyx for None config
        if ctx.get_config() is ''.encode():
            from_csv(new CCylonContextWrap(''.encode()), spath, sdelm[0], id_buf)
        else:
            from_csv(new CCylonContextWrap(ctx.get_config()), spath, sdelm[0], id_buf)
        id_buf = id_str.encode()
        return Table(id_buf)



