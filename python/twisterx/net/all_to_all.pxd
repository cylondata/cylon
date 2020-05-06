from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector


cdef extern from "../../../cpp/src/twisterx/python/net/comm/Callback.h" namespace "twisterx::net::comms":
    cdef cppclass CReceiveCallback "twisterx::net::comms::Callback":
        CReceiveCallback()
        bool onReceive(int source, void *buffer, int length) except +
        bool onReceiveHeader(int source, int finished, int *buffer, int length) except +

cdef extern from "../../../cpp/src/twisterx/net/ops/all_to_all.hpp" namespace "twisterx":
    cdef cppclass CAllToAll "twisterx::AllToAll":
        CAllToAll(int worker_id, const vector[int] &source, const vector[int] &targets, int edgeId,
           CReceiveCallback *callback)
        int insert(void *buffer, int length, int target, int *header, int headerLength)
        int insert(void *buffer, int length, int target)
        bool isComplete()
        void finish()
        void close()