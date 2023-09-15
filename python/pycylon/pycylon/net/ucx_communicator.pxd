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

IF CYTHON_UCX:

    from libcpp.memory cimport shared_ptr
    from pycylon.net.comm_type cimport CCommType
    from pycylon.net.reduce_op cimport CReduceOp
    from pycylon.common.status cimport CStatus
    from pycylon.data.scalar cimport CScalar
    from pycylon.net.communicator cimport Communicator


    cdef extern from "../../../../cpp/src/cylon/net/ucx/ucx_communicator.hpp" namespace "cylon::net":
        cdef cppclass CUCXCommunicator "cylon::net::UCXCommunicator":

            int GetRank()
            int GetWorldSize()
            void Finalize()
            void Barrier()
            CCommType GetCommType()

            CStatus AllReduce(const shared_ptr[CScalar] & value,
                          CReduceOp reduce_op,
                          shared_ptr[CScalar] *output)


    cdef class UCXCommunicator(Communicator):
        cdef:
            shared_ptr[CUCXCommunicator] ucc_cucx_comm_shd_ptr

            void init(self, const shared_ptr[CUCXCommunicator] & communicator)