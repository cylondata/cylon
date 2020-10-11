from pycylon.net.comm_type cimport CCommType

cdef extern from "../../../cpp/src/cylon/net/comm_config.hpp" namespace "cylon::net":
    cdef cppclass CCommConfig "cylon::net::CommConfig":
        CCommType Type()

cdef class CommConfig:
    cdef:
        CCommConfig *thisPtr