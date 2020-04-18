from twisterx.api.channel cimport _ChannelReceiveCallback
from twisterx.api.channel cimport _ChannelSendCallback
from twisterx.api.channel cimport _TxRequest


cdef class ChannelSendCallback:

    cdef _ChannelSendCallback *thisPtr

    def sendComplete(self):
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