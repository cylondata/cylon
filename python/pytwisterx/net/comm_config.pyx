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

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from pytwisterx.net.comms.types import CommType
from pytwisterx.net.comm_type cimport _CommType

'''
Communication Config Mapping from TwisterX C++ 
'''

cdef extern from "../../../cpp/src/twisterx/net/comm_config.h" namespace "twisterx::net":
    cdef cppclass CCommConfig "twisterx::net::CommConfig":
        _CommType Type()


cdef class CommConfig:
    cdef CCommConfig *thisPtr

    def type(self):
        return self.thisPtr.Type()
