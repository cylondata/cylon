import numpy as np
cimport numpy as np
from twisterx.net.all_to_all cimport CReceiveCallback
from twisterx.net.all_to_all cimport CAllToAll
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

cdef class ReceiveCallBack:
    cdef CReceiveCallback *thisPtr
    cdef public int[:] buf_val_int
    cdef public int[:] buf_header_val_int
    cdef public float[:] buf_val_float
    cdef public char[:] buf_val_char
    cdef public bytes[:] buf_val_bytes
    cdef public double[:] buf_val_double
    cdef public long[:] buf_val_long
    cdef public buf_val_type

    def __cinit__(self):
        print("==> Init ReceiveCallback")
        #self.thisPtr = new CReceiveCallback()

    def on_receive(self, int source, np.ndarray buf, int length):
        print("=====OnReceive")
        self.buf_val_type = buf.dtype
        cdef double[:] buf_val_double
        if buf is None or source == -1 or length == 0:
            raise ValueError("Invalid input")
        elif source != -1 and length != 0 and buf is not None:
            # if self.buf_val_type == np.int16:
            #     self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_int[0]
            #
            # if self.buf_val_type == np.int32:
            #     self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_int[0]
            #
            # if self.buf_val_type == np.int64:
            #     self.buf_val_long = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_long[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_long[0]
            #
            # if self.buf_val_type == np.float32:
            #     self.buf_val_float = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_float[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_float[0]

            if self.buf_val_type == np.double:
                buf_val_double = buf
                print("@OnReceive")
                self.thisptr.onReceive(source, buf_val_double[0], length)
                #self.thisptr.buffer = &self.buf_val_double[0]

            # if self.buf_val_type == np.float64:
            #     self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_double[0]
            #
            # if self.buf_val_type == np.double:
            #     self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_double[0]
            #
            # if self.buf_val_type == np.long:
            #     self.buf_val_long = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_long[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_long[0]

    def on_receive_header(self, int source, int finish, np.ndarray buf, int length):
        print("=====OnReceiveHeader")
        self.buf_val_type = buf.dtype
        cdef int[:] buf_header_val_int
        if buf is None or source == -1 or length == 0 or finish == -1:
            raise ValueError("Invalid input")
        elif source != -1 and source != -1 and length != 0 and buf is not None:
            # if self.buf_val_type == np.int16:
            #     self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_int[0]
            #
            # if self.buf_val_type == np.int32:
            #     self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_int[0]
            #
            # if self.buf_val_type == np.int64:
            #     self.buf_val_long = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_long[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_long[0]
            #
            # if self.buf_val_type == np.float32:
            #     self.buf_val_float = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_float[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_float[0]

            if self.buf_val_type == np.int32:
                print("==> @onReceiveHeader")
                buf_header_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr.onReceiveHeader(source, finish, buf_header_val_int[0], length)
                #self.thisptr.buffer = &self.buf_val_double[0]

            # if self.buf_val_type == np.float64:
            #     self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_double[0]
            #
            # if self.buf_val_type == np.double:
            #     self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_double[0]
            #
            # if self.buf_val_type == np.long:
            #     self.buf_val_long = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_long[0], len, &head[0], hLength)
            #     self.thisptr.buffer = &self.buf_val_long[0]

cdef class AllToAll:

    cdef vector[int] c_sources
    cdef vector[int] c_targets
    cdef CReceiveCallback *callback
    cdef CAllToAll *thisPtr
    cdef public buf_val_type

    def __cinit__(self, int worker_id, sources:list, targets: list, int edge_id):
        print("===> Init AlltoAll")
        self.c_sources = sources
        self.c_targets = targets
        self.callback = new CReceiveCallback()
        self.thisPtr = new CAllToAll(worker_id, self.c_sources, self.c_targets, edge_id, self.callback)

    def insert(self, np.ndarray buffer, length:int, target:int, np.ndarray[int, ndim=1, mode="c"] header, header_length: int):
        print("==> Insert")
        self.buf_val_type = buffer.dtype
        cdef double[:] buf_val_double
        if self.buf_val_type == np.double:
            print("====@insert")
            buf_val_double = buffer
            self.thisPtr.insert(&buf_val_double[0], length, target, &header[0], header_length)

    def insert(self, np.ndarray buffer, length:int, target:int):
        self.buf_val_type = buffer.dtype
        cdef double[:] buf_val_double
        if self.buf_val_type == np.double:
            buf_val_double = buffer
            self.thisPtr.insert(&buf_val_double[0], length, target)

    def __dealloc__(self):
        del self.thisPtr

    def is_complete(self) -> bool:
        return self.thisPtr.isComplete()

    def finish(self):
        self.thisPtr.finish()

    def close(self):
        self.thisPtr.close()

