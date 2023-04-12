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

    from pycylon.common.status cimport CStatus
    from pycylon.net.ucx_oob_context cimport UCXOOBContext

    cdef extern from "../../../../cpp/src/cylon/net/ucx/redis_ucx_ucc_oob_context.hpp" namespace "cylon::net":
        cdef cppclass CUCXRedisOOBContext "cylon::net::UCXRedisOOBContext":
            CStatus getWorldSizeAndRank(int &world_size, int &rank)

            CUCXRedisOOBContext(int &world_size, int &rank)
    cdef class UCXRedisOOBContext(UCXOOBContext):
        pass