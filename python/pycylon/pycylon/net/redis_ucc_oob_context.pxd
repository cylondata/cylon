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


    from libcpp.memory cimport shared_ptr
    from pycylon.net.oob_type cimport COOBType
    from pycylon.net.ucx_oob_context cimport CUCXOOBContext
    from pycylon.net.ucc_oob_context cimport UCCOOBContext
    from pycylon.net.redis_ucx_oob_context cimport CUCXRedisOOBContext
    from libcpp.string cimport string

    cdef extern from "../../../../cpp/src/cylon/net/ucx/redis_ucx_ucc_oob_context.hpp" namespace "cylon::net":
        cdef cppclass CUCCRedisOOBContext "cylon::net::UCCRedisOOBContext":
            COOBType Type()

            CUCCRedisOOBContext(int world_size, string redis_addr)

            CUCCRedisOOBContext()

            shared_ptr[CUCXOOBContext] makeUCXOOBContext()

            int getWorldSize()

            int getRank()

            void clearDB()
            
            @ staticmethod
            shared_ptr[CUCCRedisOOBContext] Make(int world_size, string redis_addr);

    cdef class UCCRedisOOBContext(UCCOOBContext):
        cdef:

            shared_ptr[CUCCRedisOOBContext] ucc_redis_oob_context_shd_ptr




