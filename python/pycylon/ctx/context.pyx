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
from cython.operator cimport dereference as deref
from pycylon.net.comm_type import CommType
from pycylon.net.comm_type cimport _CommType
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context cimport CCylonContextWrap

cdef class CylonContext:
    cdef CCylonContextWrap* thisPtr;
    cdef string cconfig;

    def __cinit__(self, config):
        '''
        Initializing the Cylon Context based on the distributed or non-distributed context
        :param config: passed as a str => "mpi" (currently MPI is the only supported distributed backend)
        :return: None
        '''
        if config is None:
            print("Single Thread Config Loaded")
            self.cconfig = ''.encode()
            self.thisPtr = new CCylonContextWrap(''.encode())
        else:
            self.cconfig = config.encode()
            print("Distributed Config Loaded")
            self.thisPtr = new CCylonContextWrap(config.encode())

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

    def barrier(self):
        '''
        calling barrier to sync workers
        '''
        self.thisPtr.Barrier()
    
    def get_config(self):
        return self.cconfig







