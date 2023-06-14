
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


    from pycylon.net.communicator cimport Communicator
    from pycylon.net.ucx_communicator cimport CUCXCommunicator
    from pycylon.data.scalar cimport Scalar
    from pycylon.data.scalar cimport CScalar
    from libcpp.memory cimport shared_ptr
    from pycylon.net.reduce_op import ReduceOp
    import pyarrow as pa
    from pyarrow.lib cimport pyarrow_wrap_scalar


    cdef class UCXCommunicator(Communicator):

        def __cinit__(self):
            pass

        cdef void init(self, const shared_ptr[CUCXCommunicator]& communicator):
            self.ucc_cucx_comm_shd_ptr = communicator


        def allreduce(self, value, reduce_op: ReduceOp):
            cdef shared_ptr[CScalar] cresult
            scalarv = Scalar(pa.scalar(value))

            self.ucc_cucx_comm_shd_ptr.get().AllReduce(scalarv.thisPtr, reduce_op, &cresult)

            return pyarrow_wrap_scalar(cresult.get().data()).as_py()