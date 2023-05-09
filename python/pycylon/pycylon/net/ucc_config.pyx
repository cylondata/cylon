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

IF CYTHON_UCC:
    from pycylon.net.comm_config cimport CommConfig
    from pycylon.net.ucc_config cimport CUCCConfig
    from pycylon.net.ucc_oob_context cimport CUCCOOBContext
    IF CYTHON_REDIS:
        from pycylon.net.redis_ucc_oob_context cimport UCCRedisOOBContext

    from pycylon.net.ucc_oob_context cimport UCCOOBContext
    from libcpp.memory cimport shared_ptr, dynamic_pointer_cast



    cdef class UCCConfig(CommConfig):
        """
        UCCConfig Type mapping from libCylon to PyCylon
        """
        def __cinit__(self, object context):
            IF CYTHON_REDIS:
                if isinstance(context, UCCRedisOOBContext):
                    self.ucc_oob_context_ptr = <shared_ptr[CUCCOOBContext]>(<UCCRedisOOBContext> context).ucc_redis_oob_context_shd_ptr

                    self.ucc_config_shd_ptr = CUCCConfig.Make(self.ucc_oob_context_ptr)

                    print("initialized Redis UCC Config context")

                else:
                    raise ValueError('Passed object is not an instance of UCCOOBContext')


        @property
        def comm_type(self):
            return self.ucc_config_shd_ptr.get().Type()