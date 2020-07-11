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

'''
LibCylon Communication mapping with PyCylon
'''

import numpy as np
cimport numpy as np
from pycylon.net.comms cimport CAll_to_all_wrap
from libcpp.vector cimport vector

cdef class Communication:
    cdef vector[int] c_sources
    cdef vector[int] c_targets
    cdef CAll_to_all_wrap *thisPtr
    cdef public buf_val_type

    def __cinit__(self, int worker_id, sources:list, targets: list, int edge_id):
        print("===> Init AlltoAll")
        self.c_sources = sources
        self.c_targets = targets
        self.thisPtr = new CAll_to_all_wrap(worker_id, self.c_sources, self.c_targets, edge_id)

    def insert(self, np.ndarray buffer, length:int, target:int, np.ndarray[int, ndim=1, mode="c"] header, header_length: int):
        print("==> Insert")
        self.buf_val_type = buffer.dtype
        cdef double[:] buf_val_double
        if self.buf_val_type == np.double:
            print("====@insert")
            buf_val_double = buffer
            self.thisPtr.insert(&buf_val_double[0], length, target, &header[0], header_length)

    def __dealloc__(self):
        del self.thisPtr

    def finish(self):
        self.thisPtr.finish()

    def wait(self):
        self.thisPtr.wait()