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
from pycylon.data.column cimport CColumn
from pycylon.data.scalar cimport CScalar
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context cimport CylonContext
from pycylon.ctx.context import CylonContext
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.mpi_config cimport CMPIConfig

IF CYTHON_GLOO:
	from pycylon.net.gloo_config cimport CGlooConfig
IF CYTHON_UCX & CYTHON_UCC:
	from pycylon.net.ucx_config cimport CUCXConfig
	from pycylon.net.ucc_config cimport CUCCConfig
	from pycylon.net.ucc_ucx_communicator cimport CUCXUCCCommunicator
IF CYTHON_UCX:
	from pycylon.net.ucx_communicator cimport CUCXCommunicator
from pycylon.net.mpi_communicator cimport CMPICommunicator
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pycylon.io.csv_read_config import CSVReadOptions
from pycylon.io.csv_read_config cimport CSVReadOptions
from pycylon.io.csv_write_config cimport CCSVWriteOptions
from pycylon.io.csv_write_config import CSVWriteOptions
from pycylon.io.csv_write_config cimport CSVWriteOptions
from pycylon.data.ctype cimport CType
from pycylon.data.ctype import Type
from pycylon.data.layout cimport CLayout
from pycylon.data.layout import Layout
from pycylon.data.data_type cimport CDataType
from pycylon.data.data_type import DataType
from pycylon.common.status cimport CStatus
from pycylon.common.status import Status
from pycylon.common.status cimport Status
from pycylon.data.table cimport CSortOptions
from pycylon.data.table import SortOptions
from pycylon.data.table cimport SortOptions
from pycylon.common.join_config cimport CJoinConfig
from pycylon.indexing.cyindex import BaseArrowIndex
from pycylon.indexing.cyindex cimport CBaseArrowIndex
from pycylon.indexing.cyindex cimport BaseArrowIndex

cdef api bint pyclon_is_context(object context)

#cdef api shared_ptr[CCommConfig] pycylon_unwrap_comm_config(object comm_config)

cdef api shared_ptr[CCylonContext] pycylon_unwrap_context(object context)

cdef api shared_ptr[CMPIConfig] pycylon_unwrap_mpi_config(object config)



IF CYTHON_GLOO:
	cdef api shared_ptr[CGlooConfig] pycylon_unwrap_gloo_config(object config)

IF CYTHON_UCX & CYTHON_UCC:
	cdef api shared_ptr[CUCXConfig] pycylon_unwrap_ucx_config(object config)
	cdef api shared_ptr[CUCCConfig] pycylon_unwrap_ucc_config(object config)

	cdef api object pycylon_wrap_ucc_ucx_communicator(const shared_ptr[CUCXUCCCommunicator] & ccommunicator)

IF CYTHON_UCX:
	cdef api object pycylon_wrap_ucx_communicator(const shared_ptr[CUCXCommunicator] & ccommunicator)

cdef api object pycylon_wrap_mci_communicator(const shared_ptr[CMPICommunicator] & ccomunicator)

cdef api shared_ptr[CTable] pycylon_unwrap_table(object table)

cdef api shared_ptr[CDataType] pycylon_unwrap_data_type(object data_type)

cdef api CCSVReadOptions pycylon_unwrap_csv_read_options(object csv_read_options)

cdef api CCSVWriteOptions pycylon_unwrap_csv_write_options(object csv_write_options)

cdef api shared_ptr[CSortOptions] pycylon_unwrap_sort_options(object sort_options)

cdef api shared_ptr[CBaseArrowIndex] pycylon_unwrap_base_arrow_index(object base_arrow_index)

cdef api CType pycylon_unwrap_type(object type)

cdef api CLayout pycylon_unwrap_layout(object layout)

cdef api CJoinConfig * pycylon_unwrap_join_config(object config)

cdef api object pycylon_wrap_table(const shared_ptr[CTable] & ctable)

cdef api object pycylon_wrap_context(const shared_ptr[CCylonContext] & ctx)

cdef api object pycylon_wrap_type(const CType & type)

cdef api object pycylon_wrap_layout(const CLayout & layout)

cdef api object pycylon_wrap_data_type(const shared_ptr[CDataType] & data_type)

cdef api object pycylon_wrap_sort_options(const shared_ptr[CSortOptions] &sort_options)

cdef api object pycylon_wrap_base_arrow_index(const shared_ptr[CBaseArrowIndex] & base_arrow_index)

cdef api shared_ptr[CColumn] pycylon_unwrap_column(object column)

cdef api object pycylon_wrap_column(const shared_ptr[CColumn] & ccolumn)

cdef api shared_ptr[CScalar] pycylon_unwrap_scalar(object scalar)

cdef api object pycylon_wrap_scalar(const shared_ptr[CScalar] & cscalar)
