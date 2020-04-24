from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from twisterx.net.channel cimport _ChannelReceiveCallback
from twisterx.net.channel cimport _ChannelSendCallback
from twisterx.net.txrequest cimport _TxRequest
from pytwisterx.net.comms.request import TxRequest


cdef class ChannelSendCallback:

    cdef _ChannelSendCallback *thisPtr
    cdef shared_ptr[_TxRequest] txreqPtr


    def sendComplete(self, tx: TxRequest):
        self.txreqPtr.reset()
        #self.thisPtr.sendFinishComplete()
        pass

    def sendFinishComplete(self):
        #self.thisPtr.sendFinishComplete()
        pass

cdef class ChannelReceiveCallback:

    cdef _ChannelReceiveCallback *thisPtr

    def receiveData(self):
        #self.thisPtr.receivedData()
        pass

    def receiveHeader(self):
        #self.thisPtr.receivedHeader()
        pass