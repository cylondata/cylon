cdef extern from "../../../cpp/src/twisterx/python/net/distributed.h" namespace "twisterx::net":
    cdef extern void cdist_init()
    cdef extern void cdist_finalize()
    cdef extern int cget_rank()
    cdef extern int cget_size()


def dist_init():
    cdist_init()

def dist_finalize():
    cdist_finalize()

def rank() -> int:
    return cget_rank()

def size() -> int:
    return cget_size()