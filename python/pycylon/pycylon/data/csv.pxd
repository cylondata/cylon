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
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pyarrow.lib cimport CTable as CArrowTable
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pycylon.data.table cimport CTable
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext

cdef extern from "../../../../cpp/src/cylon/table.hpp" namespace "cylon":

    CStatus FromCSV(shared_ptr[CCylonContext] &ctx, const string &path, shared_ptr[CTable]
                        &tableOut, const CCSVReadOptions &options)

