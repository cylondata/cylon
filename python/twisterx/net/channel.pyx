from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from twisterx.net.channel cimport CChannelReceiveCallback
from twisterx.net.channel cimport CChannelSendCallback
from twisterx.net.txrequest cimport CTxRequest
from pytwisterx.net.comms.request import TxRequest


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