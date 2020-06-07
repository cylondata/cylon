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

cdef extern from "../../../cpp/src/twisterx/status.hpp" namespace "twisterx":
    cdef cppclass _Status "twisterx::Status":
        _Status()
        _Status(int, string)
        _Status(int)
        _Status(_Code)
        _Status(_Code, string)
        int get_code()
        bool is_ok()
        string get_msg()