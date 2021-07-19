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
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.comm_type cimport CCommType


cdef extern from "../../../cpp/src/cylon/net/communicator.hpp" namespace "cylon::net":
    cdef cppclass CCommunicator "cylon::net":
        void Init(const shared_ptr[CCommConfig] &config)
        # TODO: add create Channel
        int GetRank()
        int GetWorldSize()
        void Finalize()
        void Barrier()
        CCommType GetCommType()

cdef extern from "../../../cpp/src/cylon/net/mpi/mpi_communicator.hpp" namespace "cylon::net":
    cdef cppclass CMPICommunicator "cylon::net::MPICommunicator":
        void Init(const shared_ptr[CCommConfig] &config)
        int GetRank()
        int GetWorldSize()
        void Finalize()
        void Barrier()
        CCommType GetCommType()


