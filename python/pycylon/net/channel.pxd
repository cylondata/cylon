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

'''
Cython API Mapping for TwisterX C++ TxRequest for communication channels
'''

from libcpp.memory cimport shared_ptr
from pycylon.net.txrequest cimport CTxRequest


cdef extern from "../../../cpp/src/cylon/net/channel.hpp" namespace "cylon":
    cdef cppclass CChannelSendCallback "cylon::ChannelSendCallback":
        void sendComplete(shared_ptr[CTxRequest])
        void sendFinishComplete(shared_ptr[CTxRequest])

    cdef cppclass CChannelReceiveCallback "cylon::ChannelReceiveCallback":
        void receivedData(int, void *, int)
        void receivedHeader(int, int, int *, int)
