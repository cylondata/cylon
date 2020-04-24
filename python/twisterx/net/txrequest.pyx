from libcpp.string cimport string

import numpy as np
cimport numpy as np
from twisterx.net.txrequest cimport _TxRequest

ctypedef np.int_t DTYPE_int
ctypedef np.float_t DTYPE_float

cdef class TxRequest:
    cdef _TxRequest *thisptr
    cdef public int[:] buf_val_int
    cdef public float[:] buf_val_float
    cdef public char[:] buf_val_char
    cdef public bytes[:] buf_val_bytes
    cdef public double[:] buf_val_double
    cdef public long[:] buf_val_long
    cdef public buf_val_type

    cdef public np_buf_val
    cdef public np_head_val

    def __cinit__(self, int tgt, np.ndarray buf, int len,
                  np.ndarray[int, ndim=1, mode="c"] head, int hLength):
        if tgt != -1 and buf is None and len == -1 and head is None and hLength == -1:
            self.thisptr = new _TxRequest(tgt)
            self.thisptr.target = tgt
        if tgt != -1 and buf is not None and len != -1 and head is not None and hLength != -1:

            self.np_head_val = head
            self.buf_val_type = buf.dtype

            if self.buf_val_type == np.int16:
                self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len, &head[0], hLength)
                self.thisptr.buffer = &self.buf_val_int[0]

            if self.buf_val_type == np.int32:
                self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len, &head[0], hLength)
                self.thisptr.buffer = &self.buf_val_int[0]

            if self.buf_val_type == np.int64:
                self.buf_val_long = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_long[0], len, &head[0], hLength)
                self.thisptr.buffer = &self.buf_val_long[0]

            if self.buf_val_type == np.float32:
                self.buf_val_float = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_float[0], len, &head[0], hLength)
                self.thisptr.buffer = &self.buf_val_float[0]

            if self.buf_val_type == np.float:
                self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len, &head[0], hLength)
                self.thisptr.buffer = &self.buf_val_double[0]

            if self.buf_val_type == np.float64:
                self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len, &head[0], hLength)
                self.thisptr.buffer = &self.buf_val_double[0]

            if self.buf_val_type == np.double:
                self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len, &head[0], hLength)
                self.thisptr.buffer = &self.buf_val_double[0]

            if self.buf_val_type == np.long:
                self.buf_val_long = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_long[0], len, &head[0], hLength)
                self.thisptr.buffer = &self.buf_val_long[0]

            # if self.buf_val_type == np.char:
            #     self.buf_val_char = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_char[0], len, &head[0], hLength)

            # if self.buf_val_type == np.bytes:
            #     self.buf_val_bytes = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_bytes[0], len, &head[0], hLength)

            self.np_buf_val = buf

            self.thisptr.target = tgt
            # if self.buf_val_int:
            #     self.thisptr.buffer = &self.buf_val_int[0]
            # if self.buf_val_float:
            #     self.thisptr.buffer = &self.buf_val_float[0]
            self.thisptr.length = len
            self.thisptr.header = &head[0]
            self.thisptr.headerLength = hLength

        if tgt != -1 and buf is not None and len != -1 and head is None and hLength == -1:

            self.np_head_val = head
            self.buf_val_type = buf.dtype
            self.np_buf_val = buf

            if self.buf_val_type == np.int16:
                self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len)
                self.thisptr.buffer = &self.buf_val_int[0]

            if self.buf_val_type == np.int32:
                self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len)
                self.thisptr.buffer = &self.buf_val_int[0]

            if self.buf_val_type == np.int64:
                self.buf_val_int = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_int[0], len)
                self.thisptr.buffer = &self.buf_val_int[0]

            if self.buf_val_type == np.float32:
                self.buf_val_float = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_float[0], len)
                self.thisptr.buffer = &self.buf_val_float[0]

            if self.buf_val_type == np.float:
                self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len)
                self.thisptr.buffer = &self.buf_val_double[0]

            if self.buf_val_type == np.float64:
                self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len)
                self.thisptr.buffer = &self.buf_val_double[0]

            if self.buf_val_type == np.double:
                self.buf_val_double = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_double[0], len)
                self.thisptr.buffer = &self.buf_val_double[0]

            if self.buf_val_type == np.long:
                self.buf_val_long = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
                self.thisptr = new _TxRequest(tgt, &self.buf_val_long[0], len)
                self.thisptr.buffer = &self.buf_val_long[0]

            # if self.buf_val_type == np.char:
            #     self.buf_val_char = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_char[0], len)

            # if self.buf_val_type == np.bytes_:
            #     self.buf_val_bytes = buf#np.ndarray(buffer=buf.data, dtype=buf.dtype, ndim=ndim)
            #     self.thisptr = new _TxRequest(tgt, &self.buf_val_bytes[0], len)

            self.thisptr.target = tgt
            self.thisptr.length = len
            self.np_buf_val = buf

    def __dealloc__(self):
        del self.thisptr

    @property
    def target(self):
        return self.thisptr.target

    @property
    def length(self):
        return self.thisptr.length

    @property
    def buf(self):
        return self.np_buf_val

    @property
    def header(self):
        return self.np_head_val

    @property
    def headerLength(self):
        return self.thisptr.headerLength

    def to_string(self, data_type, depth):
        self.thisptr.to_string(data_type, depth)
