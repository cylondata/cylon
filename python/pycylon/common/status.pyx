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
from pycylon.common.code cimport CCode
from pycylon.common.status cimport CStatus


cdef class Status:
    cdef CStatus *thisptr
    cdef CCode _code
    cdef string msg
    cdef int code

    def __cinit__(self, int code, string msg, CCode _code):
        '''
        Initializes the Status to wrap the C++ object in Cython
        :param code: passes as an int to represent the status code.
        :param msg: passes a str to convery the status message
        :param _code: Cython correpondence to C++ Code object
        :return: None
        '''
        if _code != -1 and msg.size() == 0 and code == -1:
            #print("Status(_Code)")
            self.thisptr = new CStatus(_code)
            self._code = _code

        if msg.size() != 0 and code != -1:
            #print("Status(code, msg)")
            self.thisptr = new CStatus(code, msg)
            self.msg = msg
            self.code = code

        if msg.size() == 0 and _code == -1 and code != -1:
            #print("Status(code)")
            self.thisptr = new CStatus(code)
            self.code = code

        if msg.size() != 0 and _code != -1 and code == -1:
            #print("Status(_Code, msg)")
            self.thisptr = new CStatus(_code, msg)
            self._code = _code
            self.msg = msg

    def get_code(self):
        '''

        :return: the code
        '''
        return self.thisptr.get_code()

    def is_ok(self):
        '''

        :return: OK status from Status
        '''
        return self.thisptr.is_ok()

    def get_msg(self):
        '''

        :return: Message from Status
        '''
        return self.thisptr.get_msg()
