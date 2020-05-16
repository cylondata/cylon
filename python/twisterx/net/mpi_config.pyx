from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "../../../cpp/src/twisterx/net/mpi/mpi_communicator.h" namespace "twisterx::net":
    cdef cppclass CMPIConfig "twisterx::MPIConfig":
        void Finalize();
        void AddConfig(const string &key, const string &value);
        string GetConfig(const string &key, const string &defn);
        #net::Communicator *GetCommunicator() const;
        int GetRank();
        int GetWorldSize();
        vector[int] GetNeighbours(bool include_self);