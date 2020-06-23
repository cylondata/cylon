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
Cython Interface for TwisterXContext
'''

from libcpp.vector cimport vector
from libcpp.string cimport string
#
cdef extern from "../../../cpp/src/twisterx/ctx/twisterx_context.h" namespace "twisterx":
    cdef cppclass CTwisterXContext "twisterx::twisterx_context":
        pass
#         void Finalize();
#         #CTwisterXContext *InitDistributed(net::CommConfig *config);
#         void AddConfig(const string &key, const string &value);
#         string GetConfig(const string &key, const string &defn);
#         #net::Communicator *GetCommunicator() const;
#         int GetRank();
#         int GetWorldSize();
#         vector[int] GetNeighbours(bool include_self);


cdef extern from "../../../cpp/src/twisterx/python/twisterx_context_wrap.h" namespace "twisterx::python":
    cdef cppclass CTwisterXContextWrap "twisterx::python::twisterx_context_wrap":
        CTwisterXContextWrap()
        CTwisterXContextWrap(string config)
        CTwisterXContext *getInstance()
        int GetRank()
        int GetWorldSize()
        void Barrier()
        void Finalize()