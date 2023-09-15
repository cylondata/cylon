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
OOB Type mapping from Cylon C++ API
'''

cdef extern from "../../../../cpp/src/cylon/net/comm_operations.hpp" namespace "cylon::net":

    cdef enum CReduceOp 'cylon::net::ReduceOp':
        _SUM 'cylon::net::ReduceOp::SUM'
        _MIN 'cylon::net::ReduceOp::MIN'
        _MAX 'cylon::net::ReduceOp::MAX'
        _PROD 'cylon::net::ReduceOp::PROD'
        _LAND 'cylon::net::ReduceOp::LAND'
        _LOR 'cylon::net::ReduceOp::LOR'
        _BAND 'cylon::net::ReduceOp::BAND'
        _BOR 'cylon::net::ReduceOp::BOR'