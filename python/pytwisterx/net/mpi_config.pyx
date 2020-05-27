from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from pytwisterx.net.comm_type cimport _CommType
from pytwisterx.net.comms.types import CommType


cdef extern from "../../../cpp/src/twisterx/net/mpi/mpi_communicator.h" namespace "twisterx::net":
    cdef cppclass CMPIConfig "twisterx::MPIConfig":
        _CommType Type()