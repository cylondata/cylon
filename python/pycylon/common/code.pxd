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
cdef extern from "../../../cpp/src/cylon/code.cpp" namespace "cylon":

    cdef enum _Code 'cylon::Code':
        _OK 'cylon::Code::OK'
        _OutOfMemory 'cylon::Code::OutOfMemory'
        _KeyError 'cylon::Code::KeyError'
        _TypeError 'cylon::Code::TypeError'
        _Invalid 'cylon::Code::Invalid'
        _IOError 'cylon::Code::IOError'
        _CapacityError 'cylon::Code::CapacityError'
        _IndexError 'cylon::Code::IndexError'
        _UnknownError 'cylon::Code::UnknownError'
        _NotImplemented 'cylon::Code::NotImplemented'
        _SerializationError 'cylon::Code::SerializationError'
        _RError 'cylon::Code::RError'
        _CodeGenError 'cylon::Code::CodeGenError'
        _ExpressionValidationError 'cylon::Code::ExpressionValidationError'
        _ExecutionError 'cylon::Code::ExecutionError'
        _AlreadyExists 'cylon::Code::AlreadyExists'