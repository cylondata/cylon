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

from pycylon.data.table cimport Table
from pycylon.data.table import Table
from pycylon.data.table cimport CTable
from pycylon.data.column cimport Column
from pycylon.data.column cimport CColumn
from pycylon.data.scalar cimport Scalar
from pycylon.data.scalar cimport CScalar
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context cimport CylonContext
from pycylon.ctx.context import CylonContext
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.mpi_config cimport CMPIConfig
from pycylon.net.mpi_config import MPIConfig
from pycylon.net.mpi_config cimport MPIConfig
IF CYTHON_GLOO:
    from pycylon.net.gloo_config import GlooMPIConfig, GlooStandaloneConfig
    from pycylon.net.gloo_config cimport CGlooConfig, GlooMPIConfig, GlooStandaloneConfig
IF CYTHON_UCX & CYTHON_UCC:
    from pycylon.net.ucx_config import UCXConfig
    from pycylon.net.ucx_config cimport CUCXConfig, UCXConfig
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pycylon.io.csv_read_config import CSVReadOptions
from pycylon.io.csv_read_config cimport CSVReadOptions
from pycylon.io.csv_write_config cimport CCSVWriteOptions
from pycylon.io.csv_write_config import CSVWriteOptions
from pycylon.io.csv_write_config cimport CSVWriteOptions
from pycylon.data.data_type cimport CType
from pycylon.data.data_type import Type
from pycylon.data.data_type cimport CLayout
from pycylon.data.data_type import Layout
from pycylon.data.data_type cimport CDataType
from pycylon.data.data_type import DataType
from pycylon.data.data_type cimport DataType
from pycylon.common.status cimport CStatus
from pycylon.common.status import Status
from pycylon.common.status cimport Status
from pycylon.data.table cimport CSortOptions
from pycylon.data.table import SortOptions
from pycylon.data.table cimport SortOptions
from pycylon.common.join_config cimport CJoinConfig
from pycylon.common.join_config import JoinConfig
from pycylon.common.join_config cimport JoinConfig
from pycylon.indexing.cyindex import BaseArrowIndex
from pycylon.indexing.cyindex cimport CBaseArrowIndex
from pycylon.indexing.cyindex cimport BaseArrowIndex


cdef api bint pyclon_is_context(object context):
    return isinstance(context, CylonContext)

cdef api bint pyclon_is_table(object table):
    return isinstance(table, Table)

cdef api bint pyclon_is_mpi_config(object mpi_config):
    return isinstance(mpi_config, MPIConfig)

cdef api bint pyclon_is_csv_read_options(object csv_read_options):
    return isinstance(csv_read_options, CSVReadOptions)

cdef api bint pyclon_is_csv_write_options(object csv_write_options):
    return isinstance(csv_write_options, CSVWriteOptions)

cdef api bint pyclon_is_type(object type):
    return isinstance(type, Type)

cdef api bint pyclon_is_layout(object layout):
    return isinstance(layout, Layout)

cdef api bint pyclon_is_data_type(object data_type):
    return isinstance(data_type, DataType)

cdef api bint pyclon_is_sort_options(object sort_options):
    return isinstance(sort_options, SortOptions)

cdef api bint pyclon_is_join_config(object config):
    return isinstance(config, JoinConfig)

cdef api bint pyclon_is_base_arrow_index(object base_arrow_index):
    return isinstance(base_arrow_index, BaseArrowIndex)

cdef api shared_ptr[CCylonContext] pycylon_unwrap_context(object context):
    cdef CylonContext ctx
    if pyclon_is_context(context):
        ctx = <CylonContext> context
        return ctx.ctx_shd_ptr
    return CCylonContext.Init()

cdef api shared_ptr[CTable] pycylon_unwrap_table(object table):
    cdef Table tb
    if pyclon_is_table(table):
        tb = <Table> table
        return tb.table_shd_ptr
    else:
        raise ValueError('Passed object is not a Cylon Table')

cdef api shared_ptr[CMPIConfig] pycylon_unwrap_mpi_config(object config):
    cdef MPIConfig mpi_config
    if pyclon_is_mpi_config(config):
        mpi_config = <MPIConfig> config
        return mpi_config.mpi_config_shd_ptr
    else:
        raise ValueError('Passed object is not an instance of MPIConfig')

IF CYTHON_GLOO:
    cdef api shared_ptr[CGlooConfig] pycylon_unwrap_gloo_config(object config):
        if isinstance(config, GlooStandaloneConfig):
            return (<GlooStandaloneConfig> config).gloo_config_shd_ptr
        elif isinstance(config, GlooMPIConfig):
            return (<GlooMPIConfig> config).gloo_config_shd_ptr
        else:
            raise ValueError('Passed object is not an instance of GlooConfig')

IF CYTHON_UCX & CYTHON_UCC:
    cdef api shared_ptr[CUCXConfig] pycylon_unwrap_ucx_config(object config):
        if isinstance(config, UCXConfig):
            return (<UCXConfig> config).ucx_config_shd_ptr
        else:
            raise ValueError('Passed object is not an instance of UcxConfig')

cdef api CCSVReadOptions pycylon_unwrap_csv_read_options(object csv_read_options):
    cdef CSVReadOptions csvrdopt
    if pyclon_is_csv_read_options(csv_read_options):
        csvrdopt = <CSVReadOptions> csv_read_options
        return csvrdopt.thisPtr[0]
    else:
        raise ValueError('Passed object is not an instance of CSVReadOptions')

cdef api CCSVWriteOptions pycylon_unwrap_csv_write_options(object csv_write_options):
    cdef CSVWriteOptions csvwdopt
    if pyclon_is_csv_write_options(csv_write_options):
        csvwdopt = <CSVWriteOptions> csv_write_options
        return csvwdopt.thisPtr[0]
    else:
        raise ValueError('Passed object is not an instance of CSVWriteOptions')

cdef api shared_ptr[CDataType] pycylon_unwrap_data_type(object data_type):
    cdef DataType dt
    if pyclon_is_data_type(data_type):
        dt = <DataType> data_type
        return dt.sp_data_type
    else:
        raise ValueError('Passed object is not an instance of DataType')

cdef api shared_ptr[CSortOptions] pycylon_unwrap_sort_options(object sort_options):
    cdef SortOptions so
    if pyclon_is_sort_options(sort_options):
        so = <SortOptions> sort_options
        return so.thisPtr
    else:
        raise ValueError('Passed object is not an instance of SortOptions')

cdef api shared_ptr[CBaseArrowIndex] pycylon_unwrap_base_arrow_index(object base_arrow_index):
    cdef BaseArrowIndex bi
    if pyclon_is_base_arrow_index(base_arrow_index):
        bi = <BaseArrowIndex> base_arrow_index
        return bi.bindex_shd_ptr
    else:
        raise ValueError('Passed object is not an instance of DataType')

cdef api CType pycylon_unwrap_type(object type):
    pass

cdef api CLayout pycylon_unwrap_layout(object layout):
    pass

cdef api CJoinConfig * pycylon_unwrap_join_config(object config):
    cdef JoinConfig jc
    if pyclon_is_join_config(config):
        jc = <JoinConfig> config
        return jc.jcPtr
    else:
        raise ValueError('Passed object is not an instance of JoinConfig')

cdef api object pycylon_wrap_table(const shared_ptr[CTable]& ctable):
    cdef Table table = Table.__new__(Table)
    table.init(ctable)
    return table

cdef api object pycylon_wrap_context(const shared_ptr[CCylonContext] & ctx):
    cdef CylonContext context = CylonContext.__new__(CylonContext)
    context.init(ctx)
    return context

cdef api object pycylon_wrap_type(const CType & type):
    pass

cdef api object pycylon_wrap_layout(const CLayout & layout):
    pass

cdef api object pycylon_wrap_data_type(const shared_ptr[CDataType] & cdata_type):
    cdef DataType data_type = DataType.__new__(DataType)
    data_type.init(cdata_type)
    return data_type

cdef api object pycylon_wrap_sort_options(const shared_ptr[CSortOptions] &csort_options):
    cdef SortOptions sort_options = SortOptions.__new__(SortOptions)
    sort_options.init(csort_options)
    return sort_options

cdef api object pycylon_wrap_base_arrow_index(
        const shared_ptr[CBaseArrowIndex] & cbase_arrow_index):
    cdef BaseArrowIndex base_arrow_index = BaseArrowIndex.__new__(BaseArrowIndex)
    base_arrow_index.init(cbase_arrow_index)
    return base_arrow_index

cdef api bint pyclon_is_column(object column):
    return isinstance(column, Column)

cdef api shared_ptr[CColumn] pycylon_unwrap_column(object column):
    cdef Column col
    if pyclon_is_column(column):
        col = <Column> column
        return col.thisPtr
    else:
        raise ValueError('Passed object is not a Cylon Column')

cdef api object pycylon_wrap_column(const shared_ptr[CColumn]& ccol):
    cdef Column col = Column()
    col.init(ccol)
    return col

cdef api bint pyclon_is_scalar(object scalar):
    return isinstance(scalar, Scalar)

cdef api shared_ptr[CScalar] pycylon_unwrap_scalar(object scalar):
    cdef Scalar s
    if pyclon_is_scalar(scalar):
        s = <Scalar> scalar
        return s.thisPtr
    else:
        raise ValueError('Passed object is not a Cylon scalar')

cdef api object pycylon_wrap_scalar(const shared_ptr[CScalar]& cscalar):
    cdef Scalar s = Scalar()
    s.init(cscalar)
    return s
