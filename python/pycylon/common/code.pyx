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

from pycylon.common.code cimport CCode

'''
Cylon C++ Error Tracing is done with the following enums. 
'''

cpdef enum Code:
    OK = CCode._OK
    OutOfMemory = CCode._OutOfMemory
    KeyError = CCode._KeyError
    TypeError = CCode._TypeError
    Invalid = CCode._Invalid
    IOError = CCode._IOError
    CapacityError = CCode._CapacityError
    IndexError = CCode._IndexError
    UnknownError = CCode._UnknownError
    NotImplemented = CCode._NotImplemented
    SerializationError = CCode._SerializationError
    RError = CCode._RError
    CodeGenError = CCode._CodeGenError
    ExpressionValidationError = CCode._ExpressionValidationError
    ExecutionError = CCode._ExecutionError
    AlreadyExists = CCode._AlreadyExists


