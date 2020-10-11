from libcpp.memory cimport shared_ptr
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.comm_type cimport CCommType


cdef extern from "../../../cpp/src/cylon/net/communicator.hpp" namespace "cylon::net":
    cdef cppclass CCommunicator "cylon::net":
        void Init(const shared_ptr[CCommConfig] &config)
        # TODO: add create Channel
        int GetRank()
        int GetWorldSize()
        void Finalize()
        void Barrier()
        CCommType GetCommType()

cdef extern from "../../../cpp/src/cylon/net/mpi/mpi_communicator.hpp" namespace "cylon::net":
    cdef cppclass CMPICommunicator "cylon::net::MPICommunicator":
        void Init(const shared_ptr[CCommConfig] &config)
        int GetRank()
        int GetWorldSize()
        void Finalize()
        void Barrier()
        CCommType GetCommType()


