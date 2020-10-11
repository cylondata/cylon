#from pycylon.data.table cimport Table
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context cimport CylonContext
from pycylon.ctx.context import CylonContext
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.mpi_config cimport CMPIConfig
from pycylon.net.mpi_config import MPIConfig
from pycylon.net.mpi_config cimport MPIConfig


cdef api bint pyclon_is_context(object context):
    return isinstance(context, CylonContext)

cdef api bint pyclon_is_mpi_config(object mpi_config):
    return isinstance(mpi_config, MPIConfig)

cdef api shared_ptr[CCylonContext] pycylon_unwrap_context(object context):
    cdef CylonContext ctx
    if pyclon_is_context(context):
        ctx = <CylonContext> context
        return ctx.ctx_shd_ptr
    else:
        raise ValueError('Passed object is not a CylonContext')


cdef api shared_ptr[CMPIConfig] pycylon_unwrap_mpi_config(object config):
    cdef MPIConfig mpi_config
    if pyclon_is_mpi_config(config):
        mpi_config = <MPIConfig> config
        return mpi_config.mpi_config_shd_ptr
    else:
        raise ValueError('Passed object is not a MPI Config')
