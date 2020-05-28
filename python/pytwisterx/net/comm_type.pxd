cdef extern from "../../../cpp/src/twisterx/net/comm_type.h" namespace "twisterx::net":

    cdef enum _CommType 'twisterx::net::CommType':
        _MPI 'twisterx::net::CommType::MPI'
        _TCP 'twisterx::net::CommType::TCP'
        _UCX 'twisterx::net::CommType::UCX'