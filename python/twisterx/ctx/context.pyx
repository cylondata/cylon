from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from pytwisterx.net.comms.types import CommType
from twisterx.net.comm_type cimport _CommType

cdef extern from "../../../cpp/src/twisterx/ctx/twisterx_context.h" namespace "twisterx":
    cdef cppclass CTwisterXContext "twisterx::TwisterXContext":
        void Finalize();
        void AddConfig(const string &key, const string &value);
        string GetConfig(const string &key, const string &defn);
        #net::Communicator *GetCommunicator() const;
        int GetRank();
        int GetWorldSize();
        vector[int] GetNeighbours(bool include_self);

cdef extern from "../../../cpp/src/twisterx/net/comm_config.h" namespace "twisterx::net":
    cdef cppclass CCommConfig "twisterx::net::CommConfig":
        _CommType Type()

cdef extern from "../../../cpp/src/twisterx/ctx/twisterx_context.h" namespace "twisterx::TwisterXContext":
    cdef extern CTwisterXContext *Init()
    cdef extern CTwisterXContext *InitDistributed(CCommConfig *config);

cdef extern from "../../../cpp/src/twisterx/net/mpi/mpi_communicator.h" namespace "twisterx::net":
    cdef cppclass CMPIConfig "twisterx::MPIConfig":
        _CommType Type()

cdef class CommConfig:
    @staticmethod
    def type() -> CommType:
        return CommType.MPI.value

cdef class TwisterxContext:
    cdef CTwisterXContext *thisPtr;

    def __cinit__(self, config):
        if config is None:
            self.thisPtr = Init()
        else:
            print("Distributed Config")
            pass










