from libcpp.memory cimport shared_ptr
from pycylon.net.comm_type cimport CCommType
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.comm_config import CommConfig
from pycylon.net.comm_config cimport CommConfig


cdef extern from "../../../cpp/src/cylon/net/mpi/mpi_communicator.hpp" namespace "cylon::net":
    cdef cppclass CMPIConfig "cylon::net::MPIConfig" (CCommConfig):

        @staticmethod
        shared_ptr[CMPIConfig] Make();


cdef class MPIConfig(CommConfig):
    cdef:
        shared_ptr[CMPIConfig] mpi_config_shd_ptr