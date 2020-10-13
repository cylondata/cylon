from pycylon.data.table cimport Table
from pycylon.data.table import Table
from pycylon.data.table cimport CTable
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context cimport CylonContext
from pycylon.ctx.context import CylonContext
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.mpi_config cimport CMPIConfig
from pycylon.net.mpi_config import MPIConfig
from pycylon.net.mpi_config cimport MPIConfig
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pycylon.io.csv_read_config import CSVReadOptions
from pycylon.io.csv_read_config cimport CSVReadOptions
from pycylon.io.csv_write_config cimport CCSVWriteOptions
from pycylon.io.csv_write_config import CSVWriteOptions
from pycylon.io.csv_write_config cimport CSVWriteOptions

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

cdef api object pycylon_wrap_table(const shared_ptr[CTable]& ctable):
    cdef Table table = Table.__new__(Table)
    table.init(ctable)
    return table

cdef api object pycylon_wrap_context(const shared_ptr[CCylonContext] &ctx):
    cdef CylonContext context = CylonContext.__new__(CylonContext)
    context.init(ctx)
    return context
