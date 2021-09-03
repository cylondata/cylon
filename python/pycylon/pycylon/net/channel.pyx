##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

from libcpp.memory cimport shared_ptr
from pycylon.net.channel cimport CChannelReceiveCallback
from pycylon.net.channel cimport CChannelSendCallback
from pycylon.net.txrequest cimport CTxRequest
from pycylon.net.txrequest import TxRequest

'''
Implementation In Progress
'''

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