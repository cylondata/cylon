from twisterx.net.comm_type cimport _CommType

cpdef enum CommType:
    MPI = _CommType._MPI
    TCP = _CommType._TCP
    UCX = _CommType._UCX