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
Cython Interface for CylonContext
'''

from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.mpi_config cimport CMPIConfig
from pycylon.net.communicator cimport CCommunicator
from pycylon.net.comm_type cimport CCommType

#
cdef extern from "../../../cpp/src/cylon/ctx/cylon_context.hpp" namespace "cylon":
    cdef cppclass CCylonContext "cylon::CylonContext":

        CCylonContext(bool distributed)

        @staticmethod
        shared_ptr[CCylonContext] Init()

        @staticmethod
        shared_ptr[CCylonContext] InitDistributed(const shared_ptr[CCommConfig] &config)

        void Finalize()

        void AddConfig(const string &key, const string &value)

        string GetConfig(const string &key, const string &defn)

        shared_ptr[CCommunicator] GetCommunicator() const

        int GetRank()

        int GetWorldSize()

        vector[int] GetNeighbors(bool include_self)

        # TODO: add MemoryPool if necessary
        int GetNextSequence()

        CCommType GetCommType()

        void Barrier()


cdef class CylonContext:
    cdef:
        CCylonContext *ctx_ptr
        shared_ptr[CCylonContext] ctx_shd_ptr
        void init(self, const shared_ptr[CCylonContext] &ctx)
        shared_ptr[CCommConfig] init_dist(self, config)
        dict __dict__
