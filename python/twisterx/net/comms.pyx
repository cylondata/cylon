import numpy as np
cimport numpy as np
from twisterx.net.comms cimport CAll_to_all_wrap
from libcpp.vector cimport vector

cdef class Communication:
    cdef vector[int] c_sources
    cdef vector[int] c_targets
    cdef CAll_to_all_wrap *thisPtr
    cdef public buf_val_type

    def __cinit__(self, int worker_id, sources:list, targets: list, int edge_id):
        print("===> Init AlltoAll")
        self.c_sources = sources
        self.c_targets = targets
        self.thisPtr = new CAll_to_all_wrap(worker_id, self.c_sources, self.c_targets, edge_id)

    def insert(self, np.ndarray buffer, length:int, target:int, np.ndarray[int, ndim=1, mode="c"] header, header_length: int):
        print("==> Insert")
        self.buf_val_type = buffer.dtype
        cdef double[:] buf_val_double
        if self.buf_val_type == np.double:
            print("====@insert")
            buf_val_double = buffer
            self.thisPtr.insert(&buf_val_double[0], length, target, &header[0], header_length)

    def __dealloc__(self):
        del self.thisPtr

    def finish(self):
        self.thisPtr.finish()

    def wait(self):
        self.thisPtr.wait()