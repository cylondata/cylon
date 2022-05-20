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
TwisterX Status codes carrying responses related to SUCCESS and FAIL report, etc.
'''

from libcpp.string cimport string
from libcpp cimport bool
from pycylon.common.code cimport CCode

cdef extern from "../../../../cpp/src/cylon/status.hpp" namespace "cylon":
    cdef cppclass CStatus "cylon::Status":
        CStatus()
        # CStatus(int, string)
        # CStatus(int)
        CStatus(CCode)
        CStatus(CCode, string)
        int get_code()
        bool is_ok()
        const string & get_msg()

cdef class Status:
    cdef:
        CStatus *thisptr
