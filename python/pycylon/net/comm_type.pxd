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
Communication Type mapping from Cylon C++ API
'''

cdef extern from "../../../cpp/src/cylon/net/comm_type.hpp" namespace "cylon::net":

    cdef enum _CommType 'cylon::net::CommType':
        _MPI 'cylon::net::CommType::MPI'
        _TCP 'cylon::net::CommType::TCP'
        _UCX 'cylon::net::CommType::UCX'