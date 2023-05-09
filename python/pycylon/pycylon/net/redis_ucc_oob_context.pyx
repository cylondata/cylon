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

IF CYTHON_UCX & CYTHON_UCC & CYTHON_REDIS:

    from pycylon.net.redis_ucc_oob_context cimport CUCCRedisOOBContext
    from pycylon.net.redis_ucx_oob_context cimport CUCXRedisOOBContext
    from pycylon.net.ucx_oob_context cimport CUCXOOBContext
    from libcpp.memory cimport make_shared

    '''
    UCCRedisOOBContext Mapping from Cylon C++ 
    '''

    cdef class UCCRedisOOBContext:

        def __cinit__(self,  world_size: int,  redis_addr : str ):

            if world_size !=-1 :
                self.ucc_redis_oob_context_shd_ptr = CUCCRedisOOBContext.Make(world_size, redis_addr.encode())
            else:
                self.ucc_redis_oob_context_shd_ptr = make_shared[CUCCRedisOOBContext]()

            self.ucx_redis_oob_context_shd_ptr = self.ucc_redis_oob_context_shd_ptr.get().makeUCXOOBContext()

        @property
        def oob_type(self):
            return self.ucc_redis_oob_context_shd_ptr.get().Type()

        @property
        def oob_worldsize(self):
            return self.ucc_redis_oob_context_shd_ptr.get().getWorldSize()

        @property
        def oob_rank(self):
            return self.ucc_redis_oob_context_shd_ptr.get().getRank()

