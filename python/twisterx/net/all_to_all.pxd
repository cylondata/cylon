from libcpp cimport bool

cdef extern from "../../../cpp/src/twisterx/net/all_to_all.hpp" namespace "twisterx":
    cdef cppclass CReceiveCallback "twisterx::ReceiveCallback":
        CReceiveCallback()
        bool onReceive(int source, void *buffer, int length) except +
        bool onReceiveHeader(int source, int finished, int *buffer, int length) except +

