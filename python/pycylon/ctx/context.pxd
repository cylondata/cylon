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
Cython Interface for CylonContext
'''

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
#
cdef extern from "../../../cpp/src/cylon/ctx/cylon_context.hpp" namespace "cylon":
    cdef cppclass CCylonContext "cylon::cylon_context":
        pass
#         void Finalize();
#         #CCylonContext *InitDistributed(net::CommConfig *config);
#         void AddConfig(const string &key, const string &value);
#         string GetConfig(const string &key, const string &defn);
#         #net::Communicator *GetCommunicator() const;
#         int GetRank();
#         int GetWorldSize();
#         vector[int] GetNeighbours(bool include_self);


cdef extern from "../../../cpp/src/cylon/python/cylon_context_wrap.h" namespace "cylon::python":
    cdef cppclass CCylonContextWrap "cylon::python::cylon_context_wrap":
        CCylonContextWrap(string config)
        int GetRank()
        int GetWorldSize()
        void Barrier()
        void Finalize()

cdef class CylonContext:
    cdef:
        CCylonContextWrap *thisPtr;
        string cconfig;