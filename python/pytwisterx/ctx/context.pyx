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
from pytwisterx.net.comm_type import CommType
from pytwisterx.net.comm_type cimport _CommType
from pytwisterx.ctx.context cimport CTwisterXContext
from pytwisterx.ctx.context cimport CTwisterXContextWrap

# cdef extern from "../../../cpp/src/twisterx/ctx/twisterx_context.h" namespace "twisterx":
#     cdef cppclass CTwisterXContext "twisterx::TwisterXContext":
#         void Finalize();
#         void AddConfig(const string &key, const string &value);
#         string GetConfig(const string &key, const string &defn);
#         #net::Communicator *GetCommunicator() const;
#         int GetRank();
#         int GetWorldSize();
#         vector[int] GetNeighbours(bool include_self);
#
# cdef extern from "../../../cpp/src/twisterx/net/comm_config.h" namespace "twisterx::net":
#     cdef cppclass CCommConfig "twisterx::net::CommConfig":
#         _CommType Type()
#
# cdef extern from "../../../cpp/src/twisterx/ctx/twisterx_context.h" namespace "twisterx::TwisterXContext":
#     cdef extern CTwisterXContext *Init()
#     cdef extern CTwisterXContext *InitDistributed(CCommConfig *config);
#
# cdef extern from "../../../cpp/src/twisterx/net/mpi/mpi_communicator.h" namespace "twisterx::net":
#     cdef cppclass CMPIConfig "twisterx::MPIConfig":
#         _CommType Type()
#
# cdef class CommConfig:
#     @staticmethod
#     def type() -> CommType:
#         return CommType.MPI.value
#
cdef class TwisterxContext:
    cdef CTwisterXContextWrap *thisPtr;
    cdef string config;

    def __cinit__(self, config: str):
        '''
        Initializing the TwisterX Context based on the distributed or non-distributed context
        :param config: passed as a str => "mpi" (currently MPI is the only supported distributed backend)
        :return: None
        '''
        self.config = config.encode()
        if config is None:
            #print("Single Thread Config Loaded")
            self.thisPtr = new CTwisterXContextWrap()
        else:
            #print("Distributed Config Loaded")
            self.thisPtr = new CTwisterXContextWrap(config.encode())

    def get_rank(self) -> int:
        '''
        this is the process id (unique per process)
        :return: an int as the rank (0 for non distributed mode)
        '''
        return self.thisPtr.GetRank()

    def get_world_size(self) -> int:
        '''
        this is the total number of processes joined for the distributed task
        :return: an int as the world size  (1 for non distributed mode)
        '''
        return self.thisPtr.GetWorldSize()

    def finalize(self):
        '''
        gracefully shuts down the context by closing any distributed processes initialization ,etc
        :return: None
        '''
        self.thisPtr.Finalize()

    def get_config(self):
        return self.config






