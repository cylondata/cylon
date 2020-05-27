from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from pytwisterx.net.comms.types import CommType
from pytwisterx.net.comm_type cimport _CommType

cdef extern from "../../../cpp/src/twisterx/net/comm_config.h" namespace "twisterx::net":
    cdef cppclass CCommConfig "twisterx::net::CommConfig":
        _CommType Type()


cdef class CommConfig:
    cdef CCommConfig *thisPtr

    def type(self):
        return self.thisPtr.Type()
