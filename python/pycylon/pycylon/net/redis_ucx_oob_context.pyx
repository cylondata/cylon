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

    from pycylon.net.redis_ucx_oob_context cimport CUCXRedisOOBContext
    from libcpp.memory cimport make_shared


    cdef class UCXRedisOOBContext:


        def __cinit__(self, int world_size, string redis_addr):
            self.ucx_redis_oob_context_shd_ptr = make_shared[CUCXRedisOOBContext](world_size, redis_addr)