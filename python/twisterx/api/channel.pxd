from libcpp.string cimport string
from libcpp cimport bool
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from twisterx.api.txrequest cimport _TxRequest


cdef extern from "../../../cpp/src/twisterx/net/channel.hpp" namespace "twisterx":
    cdef cppclass _ChannelSendCallback "twisterx::ChannelSendCallback":
        void sendComplete(shared_ptr[_TxRequest])
        void sendFinishComplete(shared_ptr[_TxRequest])

    cdef cppclass _ChannelReceiveCallback "twisterx::ChannelReceiveCallback":
        void receivedData(int, void *, int)
        void receivedHeader(int, int, int *, int)
