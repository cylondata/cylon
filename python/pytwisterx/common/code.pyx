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

from libcpp.string cimport string
from libcpp cimport bool
from pytwisterx.common.code cimport _Code

# cdef extern from "../../../cpp/src/twisterx/code.cpp" namespace "twisterx":
#
#     cdef enum _Code 'twisterx::Code':
#         _OK 'twisterx::Code::OK'
#         _OutOfMemory 'twisterx::Code::OutOfMemory'
#         _KeyError 'twisterx::Code::KeyError'
#         _TypeError 'twisterx::Code::TypeError'
#         _Invalid 'twisterx::Code::Invalid'
#         _IOError 'twisterx::Code::IOError'
#         _CapacityError 'twisterx::Code::CapacityError'
#         _IndexError 'twisterx::Code::IndexError'
#         _UnknownError 'twisterx::Code::UnknownError'
#         _NotImplemented 'twisterx::Code::NotImplemented'
#         _SerializationError 'twisterx::Code::SerializationError'
#         _RError 'twisterx::Code::RError'
#         _CodeGenError 'twisterx::Code::CodeGenError'
#         _ExpressionValidationError 'twisterx::Code::ExpressionValidationError'
#         _ExecutionError 'twisterx::Code::ExecutionError'
#         _AlreadyExists 'twisterx::Code::AlreadyExists'

'''
TwisterX C++ Error Tracing is done with the following enums. 
'''

cpdef enum Code:
    OK = _Code._OK
    OutOfMemory = _Code._OutOfMemory
    KeyError = _Code._KeyError
    TypeError = _Code._TypeError
    Invalid = _Code._Invalid
    IOError = _Code._IOError
    CapacityError = _Code._CapacityError
    IndexError = _Code._IndexError
    UnknownError = _Code._UnknownError
    NotImplemented = _Code._NotImplemented
    SerializationError = _Code._SerializationError
    RError = _Code._RError
    CodeGenError = _Code._CodeGenError
    ExpressionValidationError = _Code._ExpressionValidationError
    ExecutionError = _Code._ExecutionError
    AlreadyExists = _Code._AlreadyExists


