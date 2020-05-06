from libcpp.string cimport string
from libcpp cimport bool
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from twisterx.net.txrequest cimport CTxRequest


cdef extern from "../../../cpp/src/twisterx/net/channel.hpp" namespace "twisterx":
    cdef cppclass CChannelSendCallback "twisterx::ChannelSendCallback":
        void sendComplete(shared_ptr[CTxRequest])
        void sendFinishComplete(shared_ptr[CTxRequest])

    cdef cppclass CChannelReceiveCallback "twisterx::ChannelReceiveCallback":
        void receivedData(int, void *, int)
        void receivedHeader(int, int, int *, int)
