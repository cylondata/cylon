from libcpp.string cimport string
from libcpp cimport bool
cimport numpy as np

cdef extern from "../../../cpp/src/twisterx/net/TxRequest.h" namespace "twisterx":
    cdef cppclass _TxRequest "twisterx::TxRequest":
        void *buffer;
        int length;
        int target;
        int header[6];
        int headerLength;
        _TxRequest(int)
        _TxRequest(int, void *, int)
        _TxRequest(int, void *, int, int *, int)
        void to_string()





