from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from pytwisterx.net.channel cimport CChannelReceiveCallback
from pytwisterx.net.channel cimport CChannelSendCallback
from pytwisterx.net.txrequest cimport CTxRequest
from pytwisterx.net.txrequest import TxRequest


cdef class ChannelSendCallback:

    cdef CChannelSendCallback *thisPtr
    cdef shared_ptr[CTxRequest] txreqPtr


    def sendComplete(self, tx: TxRequest):
        self.txreqPtr.reset()
        #self.thisPtr.sendFinishComplete()
        pass

    def sendFinishComplete(self):
        #self.thisPtr.sendFinishComplete()
        pass

cdef class ChannelReceiveCallback:

    cdef CChannelReceiveCallback *thisPtr

    def receiveData(self):
        #self.thisPtr.receivedData()
        pass

    def receiveHeader(self):
        #self.thisPtr.receivedHeader()
        pass