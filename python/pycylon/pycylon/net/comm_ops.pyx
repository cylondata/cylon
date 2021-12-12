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

from libc.stdint cimport uint8_t
from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from pyarrow.lib cimport CBuffer as ArrowCBuffer
from pyarrow.lib cimport pyarrow_unwrap_buffer, pyarrow_wrap_buffer

from pycylon.common.status cimport CStatus
from pycylon.ctx.context cimport CCylonContext
from pycylon.api.lib cimport pycylon_unwrap_context
from pycylon.net.comm_ops cimport AllGatherArrowBuffer

import pyarrow as pa
'''
Communication operations from Cylon C++ 
'''


def allgather_buffer(
        object buf,
        context,
):
    cdef shared_ptr[ArrowCBuffer] c_buf = pyarrow_unwrap_buffer(buf)
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)
    cdef vector[shared_ptr[ArrowCBuffer]] c_all_buffers
    cdef CStatus c_status

    c_status = AllGatherArrowBuffer(c_buf, c_ctx_ptr, c_all_buffers)
    if c_status.is_ok():
        all_buffers = []
        for i in range(c_all_buffers.size()):
            bf = pyarrow_wrap_buffer(c_all_buffers[i])
            all_buffers.append(bf)
        return all_buffers
    else:
        raise ValueError(f"allgather_buffer operation failed : {c_status.get_msg().decode()}")


def gather_buffer(
        object buf,
        root,
        context,
):
    cdef shared_ptr[ArrowCBuffer] c_buf = pyarrow_unwrap_buffer(buf)
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)
    cdef vector[shared_ptr[ArrowCBuffer]] c_all_buffers
    cdef CStatus c_status
    cdef int32_t c_root = root

    c_status = GatherArrowBuffer(c_buf, c_root, c_ctx_ptr, c_all_buffers)
    if c_status.is_ok():
        all_buffers = []
        if root == context.get_rank():
            for i in range(c_all_buffers.size()):
                bf = pyarrow_wrap_buffer(c_all_buffers[i])
                all_buffers.append(bf)
        return all_buffers
    else:
        raise ValueError(f"gather_buffer operation failed : {c_status.get_msg().decode()}")
