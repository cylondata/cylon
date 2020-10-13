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

from libcpp.memory cimport shared_ptr
from pycylon.data.table cimport Table
from pycylon.data.table import Table
from pycylon.data.table cimport CTable
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context cimport CylonContext
from pycylon.ctx.context import CylonContext
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.mpi_config cimport CMPIConfig
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pycylon.io.csv_read_config import CSVReadOptions
from pycylon.io.csv_read_config cimport CSVReadOptions
from pycylon.io.csv_write_config cimport CCSVWriteOptions
from pycylon.io.csv_write_config import CSVWriteOptions
from pycylon.io.csv_write_config cimport CSVWriteOptions

cdef api bint pyclon_is_context(object context)

#cdef api shared_ptr[CCommConfig] pycylon_unwrap_comm_config(object comm_config)

cdef api shared_ptr[CCylonContext] pycylon_unwrap_context(object context)

cdef api shared_ptr[CMPIConfig] pycylon_unwrap_mpi_config(object config)

cdef api shared_ptr[CTable] pycylon_unwrap_table (object table)

cdef api CCSVReadOptions pycylon_unwrap_csv_read_options(object csv_read_options)

cdef api CCSVWriteOptions pycylon_unwrap_csv_write_options(object csv_write_options)

cdef api object pycylon_wrap_table(const shared_ptr[CTable] &ctable)

cdef api object pycylon_wrap_context(const shared_ptr[CCylonContext] &ctx)
