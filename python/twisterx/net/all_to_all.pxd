from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector


cdef extern from "../../../cpp/src/twisterx/net/all_to_all.hpp" namespace "twisterx":
    cdef cppclass CReceiveCallback "twisterx::ReceiveCallback":
        CReceiveCallback()
        bool onReceive(int source, void *buffer, int length) except +
        bool onReceiveHeader(int source, int finished, int *buffer, int length) except +

cdef extern from "../../../cpp/src/twisterx/net/all_to_all.hpp" namespace "twisterx":
    cdef cppclass CAllToAll "twisterx::AllToAll":
        CAllToAll(int worker_id, const vector[int] &source, const vector[int] &targets, int edgeId,
           CReceiveCallback *callback)